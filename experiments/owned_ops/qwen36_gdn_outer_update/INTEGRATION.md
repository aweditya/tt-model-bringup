# qwen36_gdn_outer_update integration notes

This directory is an owned source drop for the fourth GDN component kernel.

## Target install location

Copy this directory into the TTNN source tree as:

```text
ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_outer_update
```

The idempotent helper for this source install is:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_outer_update/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal \
  --source-dir ~/tt-xla/experiments/owned_ops/qwen36_gdn_outer_update
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
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_outer_update/test_qwen36_gdn_outer_update.py \
  --device-id 0 \
  --slots 1 \
  --key-dim 32 \
  --value-dim 32 \
  --dtype bfloat16 \
  --debug-fill \
  --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_32x32_fill_bf16.json
```

BF16 native ladder:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_outer_update/test_qwen36_gdn_outer_update.py \
    --device-id 0 \
    --slots 1 \
    --key-dim "$key_dim" \
    --value-dim "$value_dim" \
    --dtype bfloat16 \
    --oracle-mode native \
    --max-abs-diff-threshold 0.0005 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_outer_update_${shape}_bf16.json
done
```

Validated qb2 results from 2026-05-16:

```text
32x32:   PCC 0.9999989604301218, max 0.000244140625, mean 9.257346391677856e-06
32x128:  PCC 0.9999992769448623, max 0.00048828125,  mean 6.462098099291325e-06
128x32:  PCC 0.9999992849920574, max 0.00048828125,  mean 6.466783815994859e-06
128x128: PCC 0.9999991214594598, max 0.00048828125,  mean 7.561116944998503e-06
```

Passing this gate only validates the isolated component:

```text
state_next = state_scaled + k_col * delta
```

It does not validate output contraction, full GDN recurrence, or any decode
speedup.
