# qwen36_gdn_delta

Owned TTNN component-op source for the third GDN kernel bring-up stage:

```text
delta = beta * (value - prediction)
```

This is not copied as correctness ground truth from `tt-qwen-36`. The reference
tree is used only for TTNN operation wiring and TT-Metal dataflow patterns.

Initial contract:

- `value`: rank-4 tiled FP32 or BF16 tensor `[1, slots, 32, value_dim]`
- `prediction`: rank-4 tiled tensor matching `value`
- `beta`: rank-4 tiled tensor `[1, slots, 32, 32]`; each tile is filled with
  the slot scalar
- output: rank-4 tiled tensor matching `value`

SPMD work unit:

```text
block = (slot, value_tile)
slot = block / value_tiles
value_tile = block % value_tiles
```

Each block reads one `value` tile, one `prediction` tile, and one slot `beta`
tile. Compute currently writes `(value - prediction)` to a temporary L1 CB,
then multiplies that temporary tile by `beta`.

Optimization debt:

- The temporary CB materialization is a bring-up simplification. In the fused
  full GDN kernel, keep the subtract result in DST/register state or combine the
  delta math with the outer update instead of writing an intermediate tile.
- `value` and `prediction` are represented as repeated-row tiles for component
  validation. The full kernel should only produce/consume the tile shape needed
  by the following outer-update dataflow.

Bring-up status:

- local source and Python harness are written
- local Python syntax check passes
- qb2 TTNN build/device validation passed on 2026-05-16
- `ttnn.experimental.qwen36_gdn_delta` is registered in the qb2 Tracy TTNN
  source-package extension after rebuilding `ttnn`
- 32x32 BF16 debug-fill smoke passes exactly
- BF16 native ladder passes with `--max-abs-diff-threshold 0.0005`:
  - 32x32: PCC `0.9999970432825998`, max `0.000244140625`,
    mean `2.47955322265625e-05`
  - 32x128: PCC `0.9999957909040874`, max `0.00006103515625`,
    mean `5.453824996948242e-06`
  - 128x32: PCC `0.9999971434654209`, max `0.000030517578125`,
    mean `5.364418029785156e-06`
  - 128x128: PCC `0.9999970846548589`, max `0.00048828125`,
    mean `2.5272369384765625e-05`

Artifacts:

- `.cache/qb2_tp_deltanet/qwen36_gdn_delta_32x32_fill_bf16_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_delta_32x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_delta_32x128_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_delta_128x32_bf16_native_tol_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_delta_128x128_bf16_native_tol_20260516.json`

This is only the isolated delta component. It is not validation of outer update,
output contraction, full GDN recurrence, or decode speed.
