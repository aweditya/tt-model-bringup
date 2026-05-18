# qwen36_gdn_output

Owned TTNN component-op source for the fifth GDN kernel bring-up stage:

```text
out = q @ state_next
```

This is not copied as correctness ground truth from `tt-qwen-36`. The reference
tree is used only for TTNN operation wiring and TT-Metal dataflow patterns.

Initial contract:

- `state_next`: rank-4 tiled FP32 or BF16 tensor
  `[1, slots, key_dim, value_dim]`
- `q`: rank-4 tiled tensor `[1, slots, 32, key_dim]`; for component bring-up,
  the logical query vector is repeated across the 32 tile rows
- output: rank-4 tiled tensor `[1, slots, 32, value_dim]`, same dtype as input;
  each output tile row contains the same output vector when `q` rows are
  repeated

SPMD work unit:

```text
block = (slot, value_tile)
slot = block / value_tiles
value_tile = block % value_tiles
```

Each block reads all key tiles for the selected state value tile, reads all `q`
tiles for the slot, computes the tile-reduced matmul across key tiles, and
writes one output tile.

Optimization debt:

- `q` is repeated across rows for bring-up. The fused kernel should keep the
  query vector resident and avoid writing or rereading a repeated-row tile.
- This isolated op writes the repeated-row output tile to DRAM. A full fused
  recurrence can feed only the rows/shape needed by the downstream decode path.
- The work block rereads `q` for every value tile. Later evaluate a slot-level
  work grouping that processes multiple value tiles while keeping `q` resident
  if L1 budget allows.

Bring-up status:

- local source and Python harness are written
- local Python syntax check passes
- qb2 TTNN build/device validation passed on 2026-05-16
- `ttnn.experimental.qwen36_gdn_output` is registered in the qb2 Tracy TTNN
  source-package extension after rebuilding `ttnn`
- 32x32 BF16 debug-fill smoke passes exactly
- BF16 native ladder passes with `--max-abs-diff-threshold 0.0005`:
  - 32x32: PCC `0.9999990198081428`, max `0.000030517578125`,
    mean `1.0132789611816406e-06`
  - 32x128: PCC `0.9999999762572402`, max `0.00000762939453125`,
    mean `2.6193447411060333e-07`
  - 128x32: PCC `0.9999999990836789`, max `0.0000019073486328125`,
    mean `5.960464477539063e-08`
  - 128x128: PCC `0.9999997676859007`, max `0.00006103515625`,
    mean `1.7285346984863281e-06`

Artifacts:

- `.cache/qb2_tp_deltanet/qwen36_gdn_output_32x32_fill_bf16_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_output_32x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_output_32x128_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_output_128x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_output_128x128_bf16_native_tol_20260516.json`

This is only the isolated output contraction component. It is not validation of
the full GDN recurrence or decode speed.
