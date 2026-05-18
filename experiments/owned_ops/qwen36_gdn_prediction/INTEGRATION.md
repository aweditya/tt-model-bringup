# qwen36_gdn_prediction integration notes

This directory is an owned source drop for the second GDN component kernel.

## Target install location

Copy this directory into the TTNN source tree as:

```text
ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prediction
```

The idempotent helper for this source install is:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_prediction/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal \
  --source-dir ~/tt-xla/experiments/owned_ops/qwen36_gdn_prediction
```

## Validation Gate

Do not run this from the local laptop. Run it on a TT host only after the op is
built and only when the persistent inference server is not holding the chips.

Smallest-shape debug-fill smoke:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_prediction/test_qwen36_gdn_prediction.py \
  --device-id 0 \
  --slots 1 \
  --key-dim 32 \
  --value-dim 32 \
  --dtype bfloat16 \
  --debug-fill \
  --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_prediction_32x32_fill_bf16.json
```

BF16 native ladder:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_prediction/test_qwen36_gdn_prediction.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --dtype bfloat16 \
    --oracle-mode native \
    --max-abs-diff-threshold 0.0005 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_prediction_${shape}_bf16.json
done
```

Validated qb2 BF16 native artifacts:

```text
32x32:   PCC 0.9999995677, max_abs_diff 0.000030517578125
32x128:  PCC 0.9999994235, max_abs_diff 0.00006103515625
128x32:  PCC 0.9999999384, max_abs_diff 0.0000152587890625
128x128: PCC 0.9999991642, max_abs_diff 0.0001220703125
```

Passing this gate only validates the isolated component:

```text
prediction = k @ state_scaled
```

It does not validate delta, outer update, full GDN recurrence, or any decode
speedup.
