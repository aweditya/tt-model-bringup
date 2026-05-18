# qwen36_gdn_delta integration notes

This directory is an owned source drop for the third GDN component kernel.

## Target install location

Copy this directory into the TTNN source tree as:

```text
ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_delta
```

The idempotent helper for this source install is:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_delta/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal \
  --source-dir ~/tt-xla/experiments/owned_ops/qwen36_gdn_delta
```

## Validation Gate

Do not run this from the local laptop. Run it on a TT host only after the op is
built and only when the persistent inference server is not holding the chips.

Smallest-shape debug-fill smoke:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_delta/test_qwen36_gdn_delta.py \
  --device-id 0 \
  --slots 1 \
  --key-dim 32 \
  --value-dim 32 \
  --dtype bfloat16 \
  --debug-fill \
  --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_delta_32x32_fill_bf16.json
```

BF16 native ladder:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_delta/test_qwen36_gdn_delta.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --dtype bfloat16 \
    --oracle-mode native \
    --max-abs-diff-threshold 0.0005 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_delta_${shape}_bf16.json
done
```

Validated qb2 results from 2026-05-16:

```text
32x32:   PCC 0.9999970432825998, max 0.000244140625,   mean 2.47955322265625e-05
32x128:  PCC 0.9999957909040874, max 0.00006103515625, mean 5.453824996948242e-06
128x32:  PCC 0.9999971434654209, max 0.000030517578125, mean 5.364418029785156e-06
128x128: PCC 0.9999970846548589, max 0.00048828125,   mean 2.5272369384765625e-05
```

Passing this gate only validates the isolated component:

```text
delta = beta * (value - prediction)
```

It does not validate outer update, output contraction, full GDN recurrence, or
any decode speedup.
