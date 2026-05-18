# qwen36_gdn_prediction

Owned TTNN component-op source for the second GDN kernel bring-up stage:

```text
prediction = k @ state_scaled
```

This is not copied as correctness ground truth from `tt-qwen-36`. The reference
tree is used only for TTNN operation wiring and TT-Metal dataflow patterns.

Initial contract:

- `state_scaled`: rank-4 tiled FP32 or BF16 tensor
  `[1, slots, key_dim, value_dim]`, where `key_dim` and `value_dim` are
  whole-tile multiples up to `128`
- `k`: rank-4 tiled tensor with the same dtype as `state_scaled`,
  `[1, slots, 32, key_dim]`; for component bring-up the logical vector is
  repeated across the 32 tile rows
- output: rank-4 tiled tensor `[1, slots, 32, value_dim]`, same dtype as input;
  each output tile row should contain the same prediction vector when `k` rows
  are repeated

SPMD work unit:

```text
block = (slot, value_tile)
slot = block / value_tiles
value_tile = block % value_tiles
```

Each block reads all key tiles for the selected state value tile, reads all `k`
tiles for the slot, and writes one prediction tile.

Bring-up status:

- qb2 TTNN build integration compiles and registers
  `ttnn.experimental.qwen36_gdn_prediction`
- 32x32 BF16 debug-fill passes exactly
- the older matmul-reduce implementation passed the one-device BF16 native
  ladder `[32,32]`, `[32,128]`, `[128,32]`, `[128,128]` with
  `PCC >= 0.9999991` and `max_abs_diff <= 1.220703125e-4`
- a 2026-05-18 strict-reduce experiment first exposed a gross orientation
  failure with PCC `-0.2876` and max diff `11.5625` in
  `.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_nativek_nativeio_seeded_l0_20260518.json`.
- the live component source now keeps the strict path only behind `debug_mode`
  for isolated bring-up. Current resident-server results:
  mode 10 K materialization is exact; mode 11 product vs TTNN intermediate has
  PCC `0.9999995951668267`, max diff `0.00390625`; mode 12 one-tile reduce vs
  TTNN intermediate has PCC `0.9999996063019564`, max diff `0.0078125`; mode 2
  full strict prediction is wedge-prone/too slow in the resident path. Changing
  the expected TTNN path from pre-multiply first-tile slice to full multiply
  then slice/reduce produced the same mode 11/12 mismatch, so this is not just
  a comparison-order artifact.
  Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode10_kcol_nativeio_seeded_l0_20260518.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_ttnn_expected_nativeio_seeded_l0_20260518.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_ttnn_expected_nativeio_seeded_l0_20260518.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json`.
  Do not integrate this component into the full GDN kernel.
- a direct `ttnn.matmul(k4, state_scaled)` baseline was tested with the native
  shapes `[1,12,1,128] @ [1,12,128,128] -> [1,12,1,128]` in
  `.cache/qb2_tp_deltanet/results_owned_gdn_matmul_contract_nativeio_seeded_l0_20260518.json`.
  The op accepts the clean shape formulation, but it is not numerically
  equivalent to the current TTNN broadcast-reduce reference:
  `prediction_matmul_vs_broadcast` PCC `0.999993423459592`, max diff `0.0625`.
  The custom component is closer to broadcast-reduce than to matmul:
  `component_prediction` max diff `0.0078125` vs broadcast, while
  `component_prediction_vs_matmul` max diff is `0.0625`. Do not switch the
  default reference contract to matmul without a broader generated-token
  equivalence argument.

Deployment note for qb2 resident server:

- after rebuilding TTNN, copy `_ttnn.so` to both package extension names if
  needed: `.venv/lib/python3.10/site-packages/ttnn/_ttnn.so` and
  `.venv/lib/python3.10/site-packages/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so`
- refresh `.venv/lib/python3.10/site-packages/ttnn/_ttnncpp.so` and
  `.venv/lib/python3.10/site-packages/ttnn/build/lib/_ttnncpp.so`
- copy owned op source directories into
  `.venv/lib/python3.10/site-packages/ttnn/ttnn/cpp/ttnn/operations/experimental/transformer/`
  so runtime kernel-source lookup can find the reader/compute/writer files

This is still only the prediction component. It does not validate delta,
outer-update, full recurrence, or a decode speedup.
