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
- New file: `experiments/cb/isolate/gemma4_chunked_prefill_L128.py`
- Single block (L=128 < chunk_size=2048), no outer loop yet
- Bootstrap target, build `step_forward_prefill(state, token_ids,
  start_pos)` that processes the whole L=128 in one pass
- Gates:
  - cos ≥ 0.999 vs ground truth from L=128 × step_forward_v031
  - TTFT (eager): ≥ 2× faster than the sequential version
  - HF argmax at last position matches

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
