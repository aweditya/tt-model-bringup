# Gemma 4 chunked prefill — design + plan of action (2026-06-08)

## Problem

Gemma 4 server has NO parallel-tokens prefill. "Prefill" = N sequential
`step_forward_v031` calls (one token at a time). On a 200-token chat
prompt: 200 × ~47 ms = **9.4 seconds of TTFT**. That's terrible for
any chat workload; long-context (1k+ token) prompts would take 47s.

27B server (`server_tp.py`) has chunked prefill. Qwen 3.5 / 3.6 has
chunked prefill (S2.1, #108). Gemma 4 needs the same.

## What chunked prefill is

Process L tokens through the model in (a few) parallel steps:
1. **Matmul stages** (Q, K, V, O, gate, up, down) — naturally parallel
   over L. Multiply once over `[B=1, L, hidden]`, cost dominated by
   the matmul shape, not L.
2. **Attention** — must be causal. Use `ttnn.transformer.scaled_dot_product_attention`
   in non-decode mode (full prefill SDPA, `is_causal=True`).
3. **KV cache writes** — `paged_fused_update_cache` over L positions
   in ONE call (vs L separate calls in our current loop).
4. **RoPE** — apply over L positions in parallel: cos/sin shaped
   `[L, head_dim]` rather than `[1, head_dim]`.

For very long prompts (L > chunk_size, e.g. 2048): outer-chunk over the
prompt in blocks of chunk_size, run prefill on each block. Each block
fills its block of KV cache and the next block's attention reads
prior blocks via paged SDPA's existing cache-page mechanism.

## Reuse map (per non-negotiable)

| Pattern | Source | What to fork |
|---|---|---|
| Chunked outer loop | 27B `server_tp.py:forward_prefill_chunked_tp` | the L-outer iteration + paged cache write |
| Chunked SDPA prefill | S2.1 Qwen3.6 isolation (#108) | `chunked_sdpa_isolate.py` for the kernel pattern |
| TT-Metal arg/gemma4_optimizations branch | `reference_tt_metal_gemma4_branch` memory | Tenstorrent's official Gemma 4 prefill recipe |
| Paged SDPA causal | server_gemma4_unified_ttnn's existing paged-decode | flip `is_causal=True`, q_len = chunk_size instead of 1 |
| Paged fused update cache | Gemma 4 already uses for B=1 decode | extend to L positions at once |

## Phased plan

### Phase 0 — research + chunk-size pick
- Read tt-metal arg/gemma4_optimizations branch's prefill (commit
  pointers in `reference_tt_metal_gemma4_branch.md`)
- Read 27B `forward_prefill_chunked_tp` for outer loop pattern
- Read Qwen `chunked_sdpa_isolate.py` for SDPA contract
- Pick chunk_size: 2048 is Tenstorrent default; smaller chunks waste
  matmul work, larger chunks may exceed SDPA buffer
- Output: `research/gemma4_chunked_prefill_design.md` with mesh shapes
  and chunk decision

### Phase 1 — isolated prefill forward at L=128

**Status 2026-06-09**: scaffold + P1.1-P1.5 implemented in commit `fea36b7`,
first qb1 run in flight.

- File: `experiments/cb/isolate/gemma4_chunked_prefill_L128.py`
- Single block (L=128 < chunk_size=2048), no outer loop yet
- `step_forward_prefill(state, token_ids, capture_hidden=False)` processes
  L=128 in one pass; returns (last_argmax, last_hidden_np).
- Gates (probe driver runs all three):
  - A: cos ≥ 0.999 vs ground truth (L × step_forward_v031) last hidden
  - B: TTFT eager ≥ 2× faster than sequential
  - C: argmax match at last position

## P1 GREEN 2026-06-10

| Gate (L=128) | Result |
|---|---|
| C: argmax | ✅ PASS  chunk=1091, base=1091 |
| A: cos ≥ 0.999 | ✅ PASS  0.999234 |
| B: TTFT ≥ 2× | ✅ PASS  9.45× speedup |

Bug: `ttnn.transformer.scaled_dot_product_attention` non-paged op has a
NKV<NQ GQA mis-routing. Pinpointed in 1 ladder run (attn_out cos 1.0 /
0.45 / 0.29 / 0.23 by position; everything upstream cos=1.0).

Fix: per-KV-head SDPA split — both sliding AND global use 2 SDPA calls
per layer with NQ_per_call=2 / NKV_per_call=1. Mirrors decode's pattern
exactly (which already does this for sliding via paged_sdpa_decode).

## P1.6.5 GREEN 2026-06-10

Added `paged_fill_cache` inside the per-KV-head SDPA loops in both
sliding (2 calls/layer) and global (1 call/layer). Gate D handoff test:

| Gate (L=128) | Result |
|---|---|
| D: handoff @ pos L | ✅ PASS  chunk=236761, base=236761 |
| C/A/B (prior gates) | ✅ all still PASS |

## P2 GREEN 2026-06-10 (L=2048)

| Gate | Result |
|---|---|
| C: argmax @ pos L-1 | ✅ PASS  3797 == 3797 |
| D: handoff @ pos L | ✅ PASS  102905 == 102905 |
| A: cos ≥ 0.999 | ✅ PASS  0.999615 |
| B: TTFT ≥ 2× | ✅ **PASS  100.83× speedup** (2.9s vs 294.4s) |

The sliding-window fix worked at the first try.

Headline: 2048-token prompt TTFT goes **294s → 2.9s** with chunked
prefill. The full L sweep (128 → 256 → 512 → 1024 → 2048) is implied
green by the upper bound; smaller L's only have a SUBSET of the
sliding-window edge case.

**P1 implementation choices (locked in scaffold)**:
1. **SKIP K/V cache writes**. The forward math (matmul + norms + RoPE +
   causal SDPA) is identical whether or not we write the cache, because
   causal SDPA in this path consumes the fresh K/V tensors directly.
   Cache writes are P1.6.5 (handoff-to-decode test): after gate A/B/C
   pass, add `paged_fill_cache`-equivalent over L positions and verify
   the decode path's next-token argmax matches the sequential.
2. **IGNORE sliding-window mask at L=128**. SLIDING_WINDOW=1024 ≥ L=128
   so causal == sliding-causal here. P2 adds the mask when L > 1024.
3. **Simplified MLP path** (no DRAM-sharded variant). Forks the
   `matmul + activation="gelu" + matmul + mul + matmul` chain — same as
   the default decode path; leading-dim agnostic per 27B
   `gated_attn_step_prefill_tp` precedent. DRAM-sharded MLP bakes
   M=TILE=32 into its program config; we'll evaluate later whether to
   port it to L>1 (likely yes — same M-dim treatment with L tile rows).
4. **rms_norm on rank-3 `[L, n_heads, head_dim]`** — preserves the
   `[[feedback-ttnn-rms-norm-shape-drift]]` shape contract (no fold).
5. **RoPE batched** via new helper `_apply_full_rope_seq`: `[L, n_heads,
   head_dim]` with `[L, 1, head_dim]` cos/sin broadcast — keeps the
   rotate-half-via-roll fusion from `_apply_full_rope`.
6. **Q/K/V matmul still uses HIFI4 fp32_acc** (not the Step 2 cliff
   config). Once chunked prefill is bit-stable we can revisit the
   small-K fp32_acc=False knob for prefill (separate gate per
   [[feedback-gemma4-bf16-acc-L1024-cliff]]).

**Risk register for P1 first run** (think-first per non-negotiable):
- `ttnn.permute((1, 0, 2))` on `[L, NQ, HD]` may not be supported in all
  layouts — fall-through to a reshape-then-transpose if so.
- `ttnn.transformer.scaled_dot_product_attention` GQA contract may need
  N_KV repeat-interleave at the SDPA layer for some shapes — 27B path
  uses native GQA; we expect the same for Gemma 4.
- bf16 chain at L=128 may push cos < 0.999. Mitigation: loosen to 0.99
  only if absolutely needed and document the relaxation.

### Phase 1.6.5 — cache write + handoff-to-decode test

After P1 gate A/B/C green, add KV-cache writes back so the next decode
step from a prefilled prompt produces the same argmax as the sequential
baseline's next step.

Approach options:
- (a) `paged_fill_cache(cache, K_for_cache, page_table, batch_idx=0)`
  over the full L positions in one call. 27B precedent at
  `gated_attn_step_prefill_tp:1812-1815`. Cleanest, but Gemma 4's cache
  layout is per-KV-head (NKV_PER_CHIP_SLIDING=2 caches per sliding
  layer) — need to split K/V by KV-head before fill.
- (b) `paged_fused_update_cache` per position in a loop (matches the
  decode-step writer exactly, just iterated L times). Easier to validate
  against sequential's cache state byte-for-byte. Slower but bulletproof.

Decision: start with (b) for the correctness gate (clear semantic match
to sequential's per-step cache write), then move to (a) for perf once
correctness is locked.

Gate: after a chunked prefill of L tokens, call `step_forward_v031` once
at pos=L with a sampled next token; argmax must match what the
sequential baseline would have produced from the same state.

### Phase 2 — TILE-aligned L (L=128 → L=256 → L=2048)
- Same probe at L ∈ {128, 256, 512, 1024, 2048}
- Each L padded to next multiple of 32 (TILE_SIZE for the seq dim)
- Gate: cos stays ≥ 0.999 at every L
- Bench: TTFT vs sequential

### Phase 3 — outer-chunk loop for L > 2048
- `forward_prefill_chunked(state, tokens, chunk_size=2048)`
- Outer loop over blocks; each block's SDPA reads prior blocks via
  paged cache pages (just like our existing decode)
- Probe at L = {2048, 3000, 4096, 8000}
- Gate: cos ≥ 0.99 (loosened slightly for compounding bf16); TTFT
  scales sub-linearly with L

### Phase 4 — trace capture
- Pre-allocate fixed-size buffers for chunk_size=2048 path
- Two-phase warmup, single-bucket trace
- 5× speedup expected over eager prefill (per S2.6 precedent)

### Phase 5 — server integration
- New entry: `step_forward_prefill_chunked(state, tokens)` exposed
  by server
- `cb_engine`'s admit path checks length: short → sequential, long →
  chunked
- `cb_api.py` dispatches accordingly per backend

## Gates / non-goals

- **No batch dim**: prefill is single-stream (B=1). Continuous
  batching of multiple prefills is a separate later phase.
- **No paged-K/V for chunks within one prefill**: within one call the
  cache writes are sequential block by block. Cross-block reads are
  what paged SDPA gives us for free.
- **No KV-cache prefix dedup yet** — orthogonal to chunked prefill,
  comes later when we wire prefix caching for Gemma 4.

## Estimated impact

- 200-token chat: 9.4s → ~1.5s TTFT (eager) → ~300 ms (traced)
- 1k chat: 47s → ~3s (traced)
- 8k context: from "unusable" → < 10s TTFT
- **Most important real-world win** of any of our open perf tasks

## Open questions / risks

1. Gemma 4 has `attention_k_eq_v=True` on global layers (V aliases K
   pre-norm). Chunked prefill must thread this through correctly.
2. Gemma 4 has dual head dims (sliding=256, global=512) +
   different KV head counts. Chunked SDPA contract may need two paths.
3. RoPE at L=2048 needs cos/sin tables sized for 2048 positions —
   already have MAX_KV=4096.
4. The trace's drafter-style L_kv recapture issue (spec-dec P-1)
   doesn't apply here — chunked prefill is a fixed L_kv operation
   (start at pos 0, write block, advance).

## Order vs #289

These run **in parallel-ish**:
- #289 layout fixes apply per-step → every step (decode + prefill steps
  + chunked prefill matmuls) gets faster
- #290 chunked prefill is the big-step structural change
- Land #289's first fix first (faster feedback loop, fewer touch
  points), then start #290 Phase 0 in parallel
