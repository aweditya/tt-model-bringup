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

Status: COMPLETE
Result: **Hypothesis (I) RULED OUT, but new finding emerged.**

Layer 1's ssm stays bit-zero during Layer 0's 5 calls in v3:
```
[pre L1 state L1view_during_L0_pos0..4] ssm(mean=0.0 v0=0.0 norm=0.0) conv_st(mean=0 norm=0)
```
→ No buffer aliasing or contamination.

But comparing Layer 1's state AFTER each Layer-1-DN call in both paths:

| pos | dec L1 ssm norm | v3 L1 ssm norm | ratio |
|---|---|---|---|
| 0 | 1.163 | 2.790 | 2.4× |
| 1 | 2.107 | 4.692 | 2.2× |
| 2 | 2.134 | 4.071 | 1.9× |
| 3 | 2.361 | 5.618 | 2.4× |
| 4 | 2.310 | 6.077 | 2.6× |

v3's Layer 1 state is consistently 2.2-2.6× larger. Since Layer 1 starts
at zeros in both paths and DN is deterministic on (input, state), the
**INPUT to Layer 1 DN must differ** in v3 vs decode. Bug is in Layer 0's
chain (DN reassembly via slice_write OR batched MLP) producing wrong x_seq.

### Test 7 — directly compare Layer 1 input between paths

Status: COMPLETE (after fixing wiring bug — initial commit set flag in wrong path).
Result: **SMOKING GUN. v3's x_seq[0] at L1 entry differs dramatically.**

```
                          DECODE (correct)    V3 PREFILL (broken)
chip_v0 (elem [0,0])      0.000122            0.019897           163× larger
chip_means                0.003242            0.000812           4× smaller
chip_norms                18.3482             5.7399             3.2× smaller
```

Not a uniform scale factor. Element [0,0] is huge while overall norm is
small → element-by-element divergence, not global scaling.

Bug confirmed: v3's Layer 0 chain (DN reassembly OR batched MLP) produces
wrong x_seq row 0. Test 8 narrows further by checking pre-MLP value.

### Test 8 — pre-MLP diag in both paths

Status: COMPLETE
Result: **DN reassembly is fine. Bug is in MLP.**

```
PRE-MLP (after Layer 0 DN, before MLP):
  decode: chip_v0=-0.019531  chip_means=0.002435  chip_norms=12.8629
  v3:     chip_v0=-0.019531  chip_means=0.002432  chip_norms=12.8622
  → MATCH within bf16 noise ✓
```

DN per-position loop + slice_write + to_layout + reshape + dealloc all work
correctly. The full v3 DN reassembly chain produces a `x_seq` whose row 0
is bit-for-bit (within bf16 noise) the same as decode's post-Layer-0-DN x.

Bug is somewhere in the MLP step on batched [5, 5120] input.

### Test 9 — dump _tp_all_reduce OUTPUT in both paths

Status: COMPLETE
Result: **The all_reduce OUTPUT matches between paths. Bug is the residual add.**

```
MLP all_reduce OUTPUT for layer 0:
  dec call=1 OUT  chip_v0=0.019653  chip_means=0.000813  chip_norms=5.7723
  pre call=5 OUT  chip_v0=0.019897  chip_means=0.000812  chip_norms=5.7399
  → MATCH within bf16 noise ✓
```

But comparing to post-MLP x_seq[0,:]:
- decode post-MLP chip_v0 = 0.000122 = (pre-MLP -0.019531) + (reduced 0.019653) ✓
- v3 post-MLP chip_v0 = 0.019897 = (reduced 0.019897) ONLY, no residual added ✗

**THE BUG: `ttnn.add(x_tt, reduced)` inside `mlp_step_tp` is dropping the
residual term entirely for multi-row [5, 5120] input.** Post-MLP = reduced,
NOT x_tt + reduced.

### Test 10 — diag inside mlp_step_tp around ttnn.add

Status: bootstrapping (b1jn83t6l polling)

Prints x_tt[0,0], reduced[0,0], and out[0,0] (with memory_config) around
the residual add inside mlp_step_tp. Fires once per probe path.

Three candidate causes for the missing residual:
1. ttnn.add bug on multi-row [5, 5120] — returns 2nd arg only
2. x_tt silently zeroed by some upstream MLP op
3. Mismatched memory_config between x_tt and reduced causing add to misbehave

Result: [TBD]

---

