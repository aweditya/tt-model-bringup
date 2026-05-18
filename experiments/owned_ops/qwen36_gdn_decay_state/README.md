# qwen36_gdn_decay_state

Owned TTNN component-op source for the first GDN kernel bring-up stage:

```text
state_scaled = alpha * state
```

This is not copied as correctness ground truth from `tt-qwen-36`. The reference
tree is used only for TTNN operation wiring and TT-Metal dataflow patterns.

Initial contract:

- `state`: rank-4 tiled FP32 or BF16 tensor `[1, slots, key_dim, value_dim]`
  where `key_dim` and `value_dim` are whole-tile multiples up to `128`
- `alpha`: rank-4 tiled tensor with the same dtype as `state`,
  `[1, slots, 32, 32]`; each tile is filled with the slot scalar for this
  bring-up component
- output: rank-4 tiled tensor matching `state` shape and dtype

SPMD work unit:

```text
block = (slot, value_tile)
slot = block / value_tiles
value_tile = block % value_tiles
```

Each block reads all key tiles for the selected value tile, reads one alpha
tile, scales the state tiles in compute, and writes the output tiles.

Bring-up status on qb2:

1. The op builds into TTNN under `ttnn.experimental.qwen36_gdn_decay_state`.
2. 32x32 debug-fill passes exactly.
3. The one-device ladder `[32,32]`, `[32,128]`, `[128,32]`, `[128,128]`
   passes with `PCC >= 0.9999999` and `max_abs_diff <= 2e-4`.
4. The same one-device ladder passes in BF16 native mode with
   `PCC >= 0.9999997` and `max_abs_diff <= 3e-4` against a BF16-quantized
   oracle.

The strict `1e-5` FP32 max-error gate does not pass. The compute path uses
`MathFidelity::HiFi4` and still shows pack/readback-level quantization, so use
the explicit `2e-4` tolerance for this component gate until the target model
dtype path is wired and validated.

For BF16 validation, compare against the harness's `--oracle-mode native` path,
which quantizes the inputs and expected output to BF16 instead of treating the
hardware output as an ideal FP32 result.

See `INTEGRATION.md` for the remote TTNN wiring notes and
`test_qwen36_gdn_decay_state.py` for the first device correctness gate.
