# B.2.2 Overnight Investigation Log — 2026-05-20

User went to bed; Claude continued debugging the v3 parallel_attn prefill
divergence autonomously. Read top-to-bottom for chronological progress.
Latest findings + morning summary at the bottom.

## Status when user signed off

- Layer 0 DN state (`dn['ssm']`) is BIT-IDENTICAL between v3 prefill and
  decode-loop at every position (Test 5 confirmed). owned_gdn kernel works
  perfectly under v3's call pattern at Layer 0.
- Layer 1 DN pos 0 partial differs (`[pre call=6]` ≠ `[dec call=2]`).
- Test 6 bootstrap in flight: probe Layer 1's state during Layer 0's
  processing. If Layer 1 ssm becomes non-zero during Layer 0 → buffer
  aliasing bug. If still zero → Layer 1's INPUT must differ despite
  chip_v0 matching, requiring drilling into x_seq value comparison.

## Hypothesis tree

```
                Bug in v3 layer-1+ DN
                       │
        ┌──────────────┴──────────────┐
        │                             │
  (I) State contamination       (II) Input divergence
        │                             │
  Layer 1 ssm non-zero          Layer 1 ssm zero
  at L0 boundaries              but x_pos differs
        │                             │
  Aliasing bug → ttnn.clone     Drill into x_seq.value
  or kernel bug → fix            comparison vs decode
```

## Test results log

(Filled in autonomously below as tests complete.)

### Test 6 — Layer 1 state contamination check

Status: bootstrapping (bmw47e4es polling)
Hypothesis: Layer 1 ssm becomes non-zero during Layer 0's 5 per-position DN calls.
Expected outcomes:
- Layer 1 ssm stays zero throughout Layer 0 → eliminates contamination, suggests input divergence
- Layer 1 ssm becomes non-zero at some position N → smoking gun; identify call

Result: [TBD]

---

