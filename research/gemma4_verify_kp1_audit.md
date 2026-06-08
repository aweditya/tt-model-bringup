# Phase 2.B.1 audit — B=1 hardcoded surfaces in target server's decode path

Status: foreground audit 2026-06-07, supersedes the Phase 2.B agent's
"1.5-2 day" estimate. **Actual scope: ~320 LOC of mostly-mechanical fork
work** (target ~2-3 hours focused), since:
- `_lm_head_argmax` is already B-generic
- All kernels accept B=K+1 (validated by `c3124d2` gate)
- Most "B=1" hardcoded `1`s are deterministic shape constants that
  become `K+1` (or `K+1` in dim 1, depending on the call)
- No new architecture, just shape generalization through 4 functions

## Decision: fork, don't thread

Fork the 4 decode-path functions into `*_kp1` variants instead of
threading `B` as a parameter. Reasons:
1. **Zero risk** to existing B=1 path (shipped at 47 ms/tok traced, baseline of all bringup)
2. Hardcoded `1`s + `[1, 1]` shapes are intricately tied to B=1 logic; threading risks regression
3. Iteration on B=K+1 stays isolated to its own probe + smoke
4. Phase 3 calls the new path directly via `state.verify_trace_id` without touching B=1

## Surfaces audited (with line refs)

### Input buffers (state)
- `state.tok_buf` `[1, 1]` → need `state.tok_buf_kp1` `[K+1, 1]` for embed lookup
- `state.cur_pos_buf` `[1]` → need `state.cur_pos_buf_kp1` `[K+1]` (sdpa requires per-row pos)
- `state.rot_idxs_buf` `[1, 1]` → need `state.rot_idxs_buf_kp1` `[K+1, 1]` (rope-table lookup per row; for verify all rows = current_pos so same value broadcast)
- `state.page_table_tt` → need `state.page_table_kp1_tt` (alias via `build_verify_alias_page_table_host`)

### `forward_token_gm4_inner` (line 2119) — top-level forward
- `ttnn.embedding(state.tok_buf, state.embed_tt)` — B-generic, works at K+1 directly
- 48-layer loop — needs kp1 variant per below
- `_lm_head_argmax(state, final, capture_logits=False)` — **already B-generic** (lines 1124-1149 build slice indices from `gshape`, return `[B, 1]`). NO FORK NEEDED.

### `_layer_pos0_sliding_paged` (line 1425) — sliding attention layer
| Line | Hardcoded | Needs to become |
|---|---|---|
| 1456 | `[NQ_PER_CHIP, HEAD_DIM_SLIDING]` | `[K+1, NQ_PER_CHIP, HEAD_DIM_SLIDING]` |
| 1458/1460 | `[NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING]` | `[K+1, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING]` |
| 1507/1508 | `ttnn.slice(k_n, [kv_idx, 0], [kv_idx+1, HEAD_DIM])` | 3D slice across batch dim |
| 1519 | `paged_fused_update_cache(...)` | **SKIP** (read-only verify) |
| 1533 | `ttnn.reshape(q_half, [1, 1, Q_HALF, HEAD_DIM])` | `[1, K+1, Q_HALF, HEAD_DIM]` |
| 1535 | `cur_pos_tensor=state.cur_pos_buf` | `cur_pos_tensor=state.cur_pos_buf_kp1` |
| 1552 | `ttnn.reshape(attn_concat, [1, NQ_PER_CHIP * HEAD_DIM])` | `[K+1, NQ_PER_CHIP * HEAD_DIM]` |

### `_layer_pos0_global_paged` (line 1563) — full attention layer
Same pattern as sliding, plus:
- Line 1581/1583: 2D reshape drops batch
- Line 1622: `paged_fused_update_cache` SKIP
- Line 1632/1642: reshape with hardcoded `1`s

### `_layer_forward_pos0_paged` (line 1652) — per-layer dispatch
- Calls `_layer_pos0_*_paged` — fork to call `*_kp1` variants
- DRAM MLP path (line 1709): matmul is B-generic (uses `TILE=32`); needs B-aware reshape but tiles align (K+1 ≤ 32)
- Residual adds (`ttnn.add`) — B-generic

### `_apply_full_rope` — RoPE rotate-half
- Takes Q `[N_HEADS, HEAD_DIM]` (already 2D drop-batch). At B=K+1 input is 3D `[K+1, N_HEADS, HEAD_DIM]`. Need 3D-aware rotate or unfold-rotate-fold pattern. Worth verifying with a small probe before assuming.

### `_set_pos` (line 1804) — cur_pos_buf write
Need parallel `_set_pos_kp1(state, current_pos)` that writes K+1 copies of `current_pos` into `cur_pos_buf_kp1` and `rot_idxs_buf_kp1`.

### `update_input_buffers` (line 2096) — host writes
Need parallel `update_verify_inputs(state, current_pos, candidate_tokens)`:
- Writes `[candidate_tokens]` into `tok_buf_kp1` (K+1 token IDs)
- Calls `_set_pos_kp1(state, current_pos)`

## Forks needed (line estimates)

| Function | New LOC | Source line |
|---|---|---|
| `_layer_pos0_sliding_paged_kp1` | ~80 | fork 1425 |
| `_layer_pos0_global_paged_kp1` | ~60 | fork 1563 |
| `_layer_forward_pos0_paged_kp1` | ~50 | fork 1652 |
| `forward_token_gm4_inner_kp1` | ~40 | fork 2119 |
| `update_verify_inputs` + `_set_pos_kp1` | ~30 | new |
| `_capture_verify_trace_kp1` + buffer alloc | ~50 | new |
| **Total** | **~310 LOC** | |

## Outstanding risks

1. **`_apply_full_rope` at 3D input shape** — needs verification. If the rotate-half pattern doesn't broadcast cleanly across batch, fork it too.
2. **`_shard_for_paged_write` invariants at B=K+1** — skipped for read-only verify, so not a v0 concern. Will matter for write-then-rewind variant (Phase 3 follow-up).
3. **TileLayout alignment**: K+1=6 < TILE=32, so all matmuls will tile-pad. Check that the existing program configs (HIFI4, sdpa_compute_kernel_config) work at small B.
4. **`_compute_rope_for_forward`** returns cos/sin shape `[1, head_dim]` — broadcasts to any B fine since rope is per-position not per-row.

## Implementation order

1. **Buffer allocations** in `State` + bootstrap (~30 LOC)
2. **`_set_pos_kp1` + `update_verify_inputs`** (~30 LOC)
3. **`_layer_pos0_sliding_paged_kp1`** + isolate-probe to validate one sliding layer at B=K+1 (~80 LOC + probe)
4. **`_layer_pos0_global_paged_kp1`** + isolate-probe (~60 LOC + probe)
5. **`_layer_forward_pos0_paged_kp1`** (orchestrates 1+ layers) (~50 LOC)
6. **`forward_token_gm4_inner_kp1`** (full 48-layer) (~40 LOC)
7. **`_capture_verify_trace_kp1`** with two-phase warmup (~50 LOC)
8. **End-to-end smoke** validating K+1 trace replay vs K+1 independent B=1 forwards (cos ≥ 0.999 per row)

Each step ships as its own commit with the smoke gate that follows it.

## Why this is foreground work

- No 1000+ LOC reading required (just the 4 functions above)
- Mechanical shape updates, not architectural decisions
- Immediate visibility on each kernel call's behavior
- ~2-3 hours target, fits comfortably in one foreground session
- Iteration loop fast: dev harness `gm4` is alive (or can be launched in ~89s)
