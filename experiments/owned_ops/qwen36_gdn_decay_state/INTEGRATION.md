# qwen36_gdn_decay_state integration notes

This directory is an owned source drop for the first GDN component kernel.  It
is intentionally kept outside the TTNN tree until we are ready for a controlled
remote build.

## Target install location

Copy this directory into the TTNN source tree as:

```text
ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decay_state
```

The device program factory uses TT-Metal kernel paths rooted at that location,
so the copied directory layout should remain unchanged.

The idempotent helper for this source install is:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_decay_state/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal
```

## TTNN wiring points

Wire the copied sources into the same experimental transformer build region as
the existing transformer experimental ops:

```text
qwen36_gdn_decay_state/qwen36_gdn_decay_state.cpp
qwen36_gdn_decay_state/device/qwen36_gdn_decay_state_device_operation.cpp
qwen36_gdn_decay_state/device/qwen36_gdn_decay_state_program_factory.cpp
qwen36_gdn_decay_state/qwen36_gdn_decay_state_nanobind.cpp
```

Register the nanobind function by including
`qwen36_gdn_decay_state/qwen36_gdn_decay_state_nanobind.hpp` from the
experimental nanobind module and calling:

```cpp
ttnn::operations::experimental::qwen36_gdn_decay_state::detail::bind_qwen36_gdn_decay_state(module);
```

The expected Python symbol is:

```python
ttnn.experimental.qwen36_gdn_decay_state
```

## Validation gate

Do not run this from the local laptop.  Run it on a TT host only after the op is
built and only when the persistent inference server is not holding the chips:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_decay_state/test_qwen36_gdn_decay_state.py \
  --device-id 0 \
  --max-abs-diff-threshold 0.0002 \
  --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_correctness.json
```

Passing this gate only validates the isolated component:

```text
state_scaled = alpha * state
```

It does not validate the full GDN recurrence and it must not be reported as a
decode speedup.

Run the one-device shape ladder from smallest to production tile shape:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_decay_state/test_qwen36_gdn_decay_state.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --max-abs-diff-threshold 0.0002 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_${shape}.json
done
```

The harness uploads `alpha` as a full `[1, slots, 32, 32]` tiled tensor with
the scalar repeated across the tile.  The device op validates this contract.
The current component gate is not FP32-exact: with `MathFidelity::HiFi4`, the
qb2 one-device ladder passes at `2e-4` max absolute error and PCC around
`0.9999999`.

The op also supports the native BF16 path.  Use the harness's native oracle
mode for this path so correctness is measured against BF16-quantized inputs and
output, not an idealized FP32 expression:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_decay_state/test_qwen36_gdn_decay_state.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --dtype bfloat16 \
    --oracle-mode native \
    --max-abs-diff-threshold 0.0003 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_${shape}_bf16.json
done
```

Validated qb2 BF16 native artifacts:

```text
32x32:   PCC 0.9999997731, max_abs_diff 0.000244140625
32x128:  PCC 1.0,          max_abs_diff 0.0
128x32:  PCC 0.9999999081, max_abs_diff 0.000244140625
128x128: PCC 1.0,          max_abs_diff 0.0
```
