# B17 Trace Capture — Handoff (2026-05-22)

## Status
Trace infra **validated** (5.72× speedup on DN block, matches 27B reference 5.23×).
Input-buffer refactor **shipped** (B17-A, commit `8517fac`).
MoE on-device gather **blocked** on L1 budget (B17-B reverted).
Fixed-size KV cache **not started** (B17-C).

## What works
- `trace_demo_dn_block.py`: captures one `dn_forward_ttnn` call, replays via
  `execute_trace`. **4.14 ms eager → 0.72 ms traced (5.72×)** on qb1.
- `step_forward_ttnn` no longer does per-step host writes for tok_id, cos, sin:
  - `state.tok_buf`, `state.rot_idxs_buf` allocated at bootstrap
  - `state.cos_table_tt`, `sin_table_tt` precomputed for all `[0, MAX_KV=4096)` positions
  - `update_input_buffers(state, tok_id, cur_pos)` writes via
    `ttnn.copy_host_to_device_tensor` (must be called OUTSIDE captured trace)
  - `step_forward_inner(state)` is the trace-friendly inner step
  - `step_forward_ttnn(state, tok_id, pos)` is the eager wrapper
- Correctness preserved: all 5 prompt argmax match HF, L39 cos ≥ 0.95.

## What's blocking full-step trace
Three issues, in order of how the trace capture currently fails:

### 1. MoE host readback (B17-B blocker — L1 OVERFLOW)
`moe_forward_ttnn` reads `top_idxs`/`weights` to host for the Python TOP_K=8
expert dispatch loop. Trace error: `Reads are not supported during trace capture`.

**Attempted fix (reverted):** upload `experts_gate_up_flat` and
`experts_down_flat` as `[256_E, HIDDEN*2*INTER_CHIP]` ROW_MAJOR, use
`ttnn.embedding(top_idxs, table)` to gather TOP_K experts' weights on-device,
then Python loop over `k_idx` with fixed-int slices.

**Why it failed:** per-slice `ttnn.to_layout(TILE)` of `[2048, 256] = 1MB`
bf16 tensor blows the 1.5MB L1 circular buffer budget (2.2MB required).

**Next attempts to try:**
- Per-chunk processing: split the matmul weight into 2-4 chunks, layout-convert
  each separately, matmul, sum. Adds ops but fits L1.
- Tile-aware indexed gather: check if ttnn has a gather op that outputs TILE
  directly (avoiding the ROW_MAJOR→TILE transition).
- Upload experts as TILE from the start; verify if `ttnn.embedding` works on
  TILE tables (may not — embeddings usually require ROW_MAJOR).
- Pre-bake all 256 experts into a single batched-matmul-friendly layout that
  doesn't need per-slice conversion (e.g. transpose so the matmul reduction
  dim is properly aligned).
- Accept MoE stays eager + trace only DN + final tail (~9% speedup of decode).

### 2. KV cache growth (B17-C blocker — NEEDED FOR ATTN TRACE)
Current naive `ttnn.concat([k_prev, k_n], dim=0)` grows cache by 1 per position.
Trace requires fixed shapes.

**Refactor needed:**
- Pre-allocate `state.kv_caches_tt[L]` as `[MAX_KV, NQ_PER_CHIP, HEAD_DIM]`
  fixed-size buffer (with BROADCAST_KV)
- In-place write current K, V at slot `cur_pos` via
  `ttnn.kv_cache.update_cache_for_token_` (7.2× faster than scatter per memory)
- Attention computes over full `[MAX_KV, ...]` with **position masking** for
  positions `> cur_pos` (set scores to -inf there)
- 27B production uses paged SDPA decode for this; we'd want the same.

### 3. Final argmax readback (B17-D — TRIVIAL)
`step_forward_inner` calls `ttnn.to_torch(argmax_tt)` to get next_id (8 bytes).
For trace: return the on-device argmax tensor and read OUTSIDE trace.

## Path to full-step trace
1. Fix B17-B (try per-chunk layout transition; if not, accept eager MoE)
2. Fix B17-C (fixed-size KV cache + position mask)
3. Fix B17-D (return on-device argmax, read outside)
4. Trace capture + benchmark

## Expected speedup at completion
27B reference: pre-trace ~200 ms/tok → post-trace ~87.5 ms/tok (2.3×).
35B current: 484 ms/tok (no RoPE) or 1178 ms/tok (with RoPE).
35B target (no RoPE + full trace): ~85-150 ms/tok = 7-12 tok/s.
35B target (with RoPE + full trace): ~200-400 ms/tok = 2.5-5 tok/s.

## Files
- `experiments/utils/trace_demo_dn_block.py` — DN block isolation trace demo
- `experiments/utils/trace_demo_full_step.py` — full-step trace attempt (currently fails on MoE readback)
- `experiments/serve/server_35b_ttnn.py` — main server with input buffers + step_forward_inner
- `experiments/utils/profile_35b_ttnn.py` — end-to-end perf benchmark
- `experiments/utils/profile_blocks_35b_ttnn.py` — per-block timing breakdown

## References
- `feedback_c761_tp_trace_wins_big.md` — 27B trace 5.23× speedup precedent
- `feedback_update_cache_replaces_scatter.md` — `ttnn.kv_cache.update_cache_for_token_` 7.2× faster than scatter
- `feedback_paged_sdpa_shipped_tp.md` — 27B paged SDPA with position masking
