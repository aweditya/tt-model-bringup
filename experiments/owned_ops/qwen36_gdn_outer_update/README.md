# qwen36_gdn_outer_update

Owned TTNN component-op source for the fourth GDN kernel bring-up stage:

```text
state_next = state_scaled + k_col * delta
```

This is not copied as correctness ground truth from `tt-qwen-36`. The reference
tree is used only for TTNN operation wiring and TT-Metal dataflow patterns.

Initial contract:

- `state_scaled`: rank-4 tiled FP32 or BF16 tensor
  `[1, slots, key_dim, value_dim]`
- `k_col`: rank-4 tiled tensor `[1, slots, key_dim, 32]`; each key scalar is
  repeated across the tile columns
- `delta`: rank-4 tiled tensor `[1, slots, 32, value_dim]`; the delta vector is
  repeated across tile rows
- output: rank-4 tiled tensor matching `state_scaled`

SPMD work unit:

```text
block = (slot, key_tile, value_tile)
slot = block / (key_tiles * value_tiles)
key_tile = (block % (key_tiles * value_tiles)) / value_tiles
value_tile = block % value_tiles
```

Each block reads one `state_scaled` tile, one `k_col` tile, and one `delta`
tile. Compute writes `k_col * delta` to a temporary L1 CB, then adds that
temporary tile to `state_scaled`.

Optimization debt:

- The temporary CB materialization is a bring-up simplification. In the fused
  full GDN kernel, keep `k_col * delta` in DST/register state and add directly
  to the resident state tile.
- `k_col` repeats K scalars across columns. This avoids a tile transpose during
  isolated bring-up, but the fused kernel should load K once per slot/key tile
  and map it directly to the row broadcast needed by the outer update.
- `delta` repeats the value vector across rows. The fused kernel should produce
  the shape consumed by the state update without round-tripping a full repeated
  tile through DRAM.

Bring-up status:

- local source and Python harness are written
- local Python syntax check passes
- qb2 TTNN build/device validation passed on 2026-05-16
- `ttnn.experimental.qwen36_gdn_outer_update` is registered in the qb2 Tracy
  TTNN source-package extension after rebuilding `ttnn`
- 32x32 BF16 debug-fill smoke passes exactly
- BF16 native ladder passes with `--max-abs-diff-threshold 0.0005`:
  - 32x32: PCC `0.9999989604301218`, max `0.000244140625`,
    mean `9.257346391677856e-06`
  - 32x128: PCC `0.9999992769448623`, max `0.00048828125`,
    mean `6.462098099291325e-06`
  - 128x32: PCC `0.9999992849920574`, max `0.00048828125`,
    mean `6.466783815994859e-06`
  - 128x128: PCC `0.9999991214594598`, max `0.00048828125`,
    mean `7.561116944998503e-06`

Artifacts:

- `.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_32x32_fill_bf16_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_32x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_32x128_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_128x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_128x128_bf16_native_tol_20260516.json`

This is only the isolated outer-update component. It is not validation of output
contraction, full GDN recurrence, or decode speed.
