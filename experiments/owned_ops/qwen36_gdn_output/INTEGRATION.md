# qwen36_gdn_output integration notes

This directory is an owned source drop for the fifth GDN component kernel.

## Target install location

Copy this directory into the TTNN source tree as:

```text
ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_output
```

The idempotent helper for this source install is:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_output/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal \
  --source-dir ~/tt-xla/experiments/owned_ops/qwen36_gdn_output
```

When validating against a rebuilt source tree on qb2, set:

```bash
export PYTHONPATH=~/tenstorrent/tt-metal/ttnn
```

Without this, the local virtualenv wheel can shadow the rebuilt source-package
extension and the experimental symbol will not be visible.

## Validation Gate

Do not run this from the local laptop. Run it on a TT host only after the op is
built and only when the persistent inference server is not holding the chips.

Smallest-shape debug-fill smoke:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_output/test_qwen36_gdn_output.py \
  --device-id 0 \
  --slots 1 \
  --key-dim 32 \
  --value-dim 32 \
  --dtype bfloat16 \
  --debug-fill \
  --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_output_32x32_fill_bf16.json
```

BF16 native ladder:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_output/test_qwen36_gdn_output.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --dtype bfloat16 \
    --oracle-mode native \
    --max-abs-diff-threshold 0.0005 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_output_${shape}_bf16.json
done
```

Validated qb2 results from 2026-05-16:

```text
32x32:   PCC 0.9999990198081428, max 0.000030517578125,    mean 1.0132789611816406e-06
32x128:  PCC 0.9999999762572402, max 0.00000762939453125,  mean 2.6193447411060333e-07
128x32:  PCC 0.9999999990836789, max 0.0000019073486328125, mean 5.960464477539063e-08
128x128: PCC 0.9999997676859007, max 0.00006103515625,     mean 1.7285346984863281e-06
```

Passing this gate only validates the isolated component:

```text
out = q @ state_next
```

It does not validate full GDN recurrence or any decode speedup.
