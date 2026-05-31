# S2 chunked prefill — milestone plan (2026-05-30)

Living plan; update as we go. Source: prior-art audit in
[`27b_chunked_prefill_prior_art.md`](27b_chunked_prefill_prior_art.md), original
scoping in [`27b_chunked_prefill_plan.md`](27b_chunked_prefill_plan.md).

## Design (locked from audit)

- **Scheduler**: alternate `PREFILL_ONLY` and `DECODE_ONLY` steps (TT vLLM pattern). No mixed batches; no token-budget mixer.
- **PREFILL step**: reuse existing S1a (chunked attention) + S1b (block-Neumann GDN) on a *temp full state*. Swap S1a's internal attention primitive to `ttnn.transformer.chunked_scaled_dot_product_attention` so the post-prefill KV is already paged.
- **State converter**: at end of PREFILL, write the temp state into slot `s` of `cb_kv[li]`, `cb_dn[li].ssm`, `cb_dn[li].conv_cols`. This is the only genuinely new code.
- **DECODE step**: existing CB engine unchanged.

## Milestones

Each gate is a permanent file under `experiments/cb/validate/` or `experiments/cb/isolate/`.

### S2.1 — Isolation: chunked SDPA on Qwen3.6 shapes ✅ DONE (commit `ef110c7`)
- **What**: standalone test calling `ttnn.transformer.chunked_scaled_dot_product_attention` at Qwen3.6-27B GQA shapes on (1,4): NQ=40, NKV=8, head_dim=128, page_size=32, single slot, C ∈ {8, 16, 32}.
- **Reference**: replay decode SDPA C times against the same paged KV (the slow loop we're replacing).
- **Gate**: per-position cos ≥ 0.99999; output bit-equivalent up to bf16 rounding.
- **File**: `experiments/cb/isolate/chunked_sdpa.py`
- **Why first**: this is the ONLY new primitive; everything else is plumbing. If the op doesn't accept our shapes / mesh layout, the whole path changes.

### S2.2 — Swap S1a's attention primitive (DEFERRED — perf-only)

S1a already populates production paged KV via `paged_fill_cache` (server_tp.py:1709-1712), so the transplant doesn't need the chunked-SDPA swap. The non-paged SDPA inside S1a is O(L²) in compute and O(L²) in memory, but at L≤8192 that's fine. Defer to a post-S2.4 perf check — pick back up only if attention SDPA shows up as the TTFT bottleneck.

### S2.3 — State converter ⏳ NOW
- **What**: in `server_tp.py:forward_prefill_chunked_tp`, replace the current attention call with `ttnn.transformer.chunked_scaled_dot_product_attention(... page_table_tensor=..., chunk_start_idx_tensor=...)`.
- **Gate**: existing `experiments/cb/validate/long_context.py` PASS (non-capture path + needle retrieval, single-seq). Cos ≥ 0.999 vs current S1a on a 200-tok needle prompt.
- **Touches**: only `forward_prefill_chunked_tp`. Decode path untouched.

- **What**: `cb_prefill_transplant(state, slot_s, L)` — writes the post-prefill temp state into slot `s`:
  - For each attention layer: post-prefill KV (whatever layout S1a leaves it in after the chunked-SDPA swap) → `cb_kv[li]['kc'/'vc']` slot-s blocks.
  - For each GDN layer: post-S1b SSM H/K → `cb_dn[li]['ssm']` slot-s row; post-S1b conv kdim state → `cb_dn[li]['conv_cols']` 3-col shiftacc layout for slot s.
- **Gate**: layer-by-layer, post-transplant slot-s state matches a reference (1-tok-per-iter CB prefill of the same prompt into the same slot) with cos ≥ 0.999.
- **File**: `experiments/cb/isolate/state_transplant.py` (isolation), code in `experiments/serve/server_tp_cb.py` (`cb_prefill_transplant`).
- **Risk**: shiftacc conv state requires the last 3 input columns; the kdim path may not expose them directly — needs design check during impl.

### S2.4 — Single-slot transplant end-to-end proof
- **What**: prefill prompt via swapped-S1a → transplant into slot 0 → 4 decode steps via existing CB engine. Compare against the reference (1-tok-per-iter CB prefill all the way through the same prompt).
- **Gate**: first 4 decode tokens IDENTICAL; logits cos ≥ 0.999 at each of the 4 positions.
- **File**: `experiments/cb/validate/prefill_transplant.py`.

### S2.5 — Alternating-step scheduler
- **What**: extend `cb_scheduler.Scheduler` to support two step modes. On admit, run `_step_prefill(rid)` (single-slot S2.4 path). When no admits pending, run normal `step()` (decode). Queue tracks `prefill_needed` separately from `slots[]`.
- **Gate**: serving demo (`experiments/cb/serving_demo.py`) — N clients with long prompts; all retrieve correctly; aggregate throughput ≥ existing decode-only path.
- **Touches**: `cb_scheduler.py`. No changes to `cb_engine.py` API.

### S2.6 — TTFT measurement + needle haystack
- **What**: rerun `experiments/cb/needle.py` with `--length` 200, 500, 1000 through the new prefill path. Measure TTFT explicitly.
- **Gate**: needle retrieval matches the existing CB needle test (Y verdict). TTFT at L=200 << current `200 × decode_step` (target: ≤ 5s vs current ~52s).
- **Touches**: maybe a `--prefill {tok-by-tok, chunked}` flag in `needle.py` to A/B.

### S2.7 — Production wire-up
- **What**: route `cb_api.py` prompt admission through alternating scheduler. Add `TT_CB_PREFILL_CHUNK_SIZE` env (default = whatever S2.1 lands at).
- **Gate**: `serve_cb.sh` daemon serves long prompts via real chat; TUI/openai-client smoke passes.

## Risks + open questions

1. **`chunk_start_idx` multiple of `q_chunk_size`**: easy to satisfy if we pick C=32 and align prefill positions to 32; otherwise need the tensor-idx (trace-safe) variant which is more flexible.
2. **Shiftacc conv state semantics**: the 3 columns are the last 3 INPUT tokens to the conv (per `server_tp_cb.py:259-273`). After S1b chunked prefill we have the full input sequence available, so this should be a simple `[-3:]` slice + tilize. Verify in S2.3.
3. **GDN H/K shape parity**: S1b leaves the SSM state as global tensors; `cb_dn[li]['ssm']` is per-slot row. Should be a slot-indexed copy.
4. **B=1 vs B>1 in PREFILL step**: TT vLLM prefills ONE request per PREFILL step. We follow the same — prefill is large enough that batching prefill itself isn't a near-term win.

## Order of execution

Strictly sequential; each gates the next.

1. S2.1 (today)
2. S2.2
3. S2.3 (~2 sub-steps: design check, then impl)
4. S2.4
5. S2.5
6. S2.6
7. S2.7

S2.1 starts now.
