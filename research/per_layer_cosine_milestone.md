# Per-Layer Cosine Milestone — Plan A Lands

**Date**: 2026-05-12
**Status**: Bug localized — DeltaNet has catastrophic position-dependent drift; full attention is correct.

## The data

```
 layer                 type      pos 0     pos 1     pos 2     pos 3     pos 4      worst
     0     linear_attention    0.99907   0.99743   0.99953   0.99832   0.99662    0.99662
     1     linear_attention    0.99410   0.98656   0.95195   0.98129   0.97866    0.95195
     2     linear_attention    0.99416   0.95082   0.50766   0.84510   0.87509    0.50766  ⚠️
     3       full_attention    1.00000   0.99999   0.99998   0.99999   0.99999    0.99998  ✓
     7       full_attention    0.99999   0.99998   0.99998   0.99997   0.99998    0.99997  ✓
    11       full_attention    1.00000   0.99999   0.99998   0.99998   0.99998    0.99998  ✓
    15       full_attention    1.00000   0.99999   0.99998   0.99998   0.99997    0.99997  ✓
    31       full_attention    1.00000   0.99999   0.99997   0.99994   0.99989    0.99989  ✓
    47       full_attention    0.99996   0.99998   0.99998   0.99995   0.99996    0.99995  ✓
    63       full_attention    0.98979   0.99103   0.99132   0.99433   0.98724    0.98724
```

## Findings

1. **Full attention layers: essentially perfect** (cosine ≥ 0.9999 at most positions). Our q_norm/k_norm fixes work end-to-end.
2. **DeltaNet has position-dependent drift that grows catastrophically.**
   - Layer 0: worst pos cosine 0.997
   - Layer 1: worst 0.95
   - Layer 2: worst 0.508 (essentially random)
3. **The drift correlates with input magnitude.** HF's hidden states grow in scale through deeper layers; DeltaNet handles small inputs OK and large inputs terribly.
4. **Layer 63 (last full_attention) shows 0.987** — possibly a different small issue, but small compared to DeltaNet's collapse.

## Hypotheses for DeltaNet bug

Ranked by suspicion:

1. **Unstable softplus formula**. We use `log(exp(x)+1)`, HF uses `F.softplus(x)`. For large `x`, `exp(x)` overflows. The decay `g = -A_log.exp() * softplus(a + dt_bias)` propagates the corruption.
2. **Conv1d state propagation error**. Possible off-by-one or wrong stride accumulating across positions.
3. **Recurrence math subtle bug**. The H update `H_new = H_decayed + outer(k, delta)` — maybe order of operations matters numerically.
4. **bf8 weight noise at large magnitudes**. Possible but unlikely given the cosine doesn't smoothly degrade with magnitude — it crashes.

## Why earlier DeltaNet validation passed

The A3 / B'2 / B'3 isolation tests:
- Compared ttnn to our own numpy reference — both had the same 5 bugs (since fixed)
- Used small-magnitude inputs (~embed scale) that don't trigger numerical instability
- Tested single tokens or small sequences (didn't exercise multi-position state accumulation)

Lesson: isolation tests must use **production-magnitude inputs** AND **multi-position state evolution** to catch bugs that depend on input scale or state corruption.

## The ttnn JIT bug, fixed along the way

To unblock 91r at all, we patched ttnn's LLK + operation SFPU headers locally:
182 call sites across 73 files, replacing `int 0` with `sfpi::RoundMode::NearestEven`
in calls to `int32_to_float`/`float_to_int16`/etc.

Backup at `~/tt-xla/.cache/ttnn_llk_backup/20260512-203732/`.
Restore command in `experiments/utils/patch_ttnn_llk_roundmode.py --restore <ts>`.

## Next step

Probe softplus stability empirically with the actual `a + dt_bias` values from
layer 2 on HF's hidden_2 prompt input.
