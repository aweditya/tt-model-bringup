# Phase 5 Step 7 — Op Fusion + On-Device Broadcast Plan

Date: 2026-05-11
Author: PJRT track agent (continuing from Step 6 handoff)

## Goal

Make softmax / layer_norm / RMS-norm reach **trace replay** instead of falling
back to parse-cached eager. The Step 6 handoff identified two paths:

1. **Step 7a — On-device broadcast_in_dim.** If `_execute_broadcast_device`
   can stay on-device for the patterns JAX actually emits, then
   `broadcast_in_dim` drops out of `_HOST_TRANSFER_DEVICE_OPS` and
   softmax/LN/RMSNorm become traceable automatically. This is the higher-
   leverage win — it covers every composite pattern at once.
2. **Step 7b/c — Pattern-matched fused ops.** Detect the canonical
   `max → sub → exp → sum → div` (softmax) and `pow2 → mean → +eps → rsqrt
   → mul` (RMSNorm) chains in the parsed op list, and emit a single
   `ttnn.softmax` / `ttnn.rms_norm` call. This is the bigger speedup IF
   step 7a lands.

## What I actually saw in the IR (qb1, JAX 0.7.x)

Inspected `pjrt_plugin/tests/inspect_stablehlo.py` output. Softmax expands
to **13 ops**, not 7. Key features:

```
%0  reduce(max)             [2,64] -> [2]
%cst_0 constant(-inf)       []
%1  broadcast_in_dim %cst_0, dims=[]    [] -> [2]      <-- scalar broadcast
%2  maximum %1, %0          [2]
%3  broadcast_in_dim %2, dims=[0]       [2] -> [2,1]   <-- rank-up
%4  broadcast_in_dim %3, dims=[0,1]     [2,1] -> [2,64] <-- repeat dim 1
%5  subtract %arg0, %4      [2,64]
%6  exp %5                  [2,64]
%cst_1 constant(0)          []
%7  reduce(add)             [2,64] -> [2]
%8  broadcast_in_dim %7, dims=[0]       [2] -> [2,1]
%9  broadcast_in_dim %8, dims=[0,1]     [2,1] -> [2,64]
%10 divide %6, %9           [2,64]
```

JAX ALWAYS inserts a stability `maximum(-inf, reduce)` for softmax. RMS-norm
emits an explicit `divide %ms, %const_dim_size` instead of `mean` (so it
shows up as `reduce(add) → broadcast → divide`).

**Broadcast patterns we MUST handle on-device:**

- `[] → [N,M]` — scalar to full tensor (constants like eps, -inf, 0)
- `[N] → [N,1]` — rank-up after reduction (`dims=[0]`)
- `[N,1] → [N,M]` — broadcast across the reduced dim (`dims=[0,1]`)
- `[K] → [1,K]` — rank-up for per-channel weights (`dims=[1]`)
- `[1,K] → [N,K]` — broadcast weight across batch (`dims=[0,1]`)

All five patterns are already in the existing `_execute_broadcast_device`
code (it computes `inter_shape` and calls `ttnn.repeat`). What's not clear
is whether it stays robust under trace capture for ALL of them.

## Decisions

### Step 7a: enable broadcast_in_dim for trace capture

**Approach:** Remove `broadcast_in_dim` from `_HOST_TRANSFER_DEVICE_OPS`
unconditionally. The current `_execute_broadcast_device` already has the
on-device `ttnn.repeat` path. If it raises during capture, the trace
capture's outer try/except catches it and the program falls back to
parse-cached eager — exactly the current behavior. So this is **safe to
flip**.

**One concern from the cont. 3 reflection:** "behaviour under test was
opaque because the model-bringup process held the device lock." Plan: run
the full 50-test suite after flipping, on a freed-up qb1. If anything
regresses, document which broadcast pattern fails and skip that case.

**Subtle issue with constants:** the current `_capture_trace` SKIPS any op
in `_DATA_INDEPENDENT_OPS` during trace replay (it pins the warm-up value).
That means `constant` ops don't re-execute inside the trace — fine. But
broadcasts of constants (`broadcast_in_dim %cst_0, dims=[]`) currently
fall into the `_HOST_TRANSFER_DEVICE_OPS` skip-and-pin path. If we move
broadcast_in_dim out of that set, broadcasts of constants get re-executed
inside the trace. The constant's device tensor is pinned from warmup, so
that's safe — `ttnn.repeat` on a pinned tensor produces a fresh device
tensor every replay. (We should confirm `ttnn.repeat` doesn't allocate
into a freshly-malloc buffer that the trace can't recover; if it does we
keep the skip-and-pin path for *constant-input* broadcasts.)

### Step 7b: softmax pattern match

The 13-op pattern is rigid: JAX always emits the same shape. Detection:

1. Walk ops looking for `reduce(maximum)` whose result feeds a chain
   `→ broadcast_in_dim*(rank up to input rank)` ending in
   `subtract(input, broadcast_result)`.
2. The subtract result must feed `exp`.
3. The exp result must feed `reduce(add)`.
4. The sum result must feed another `broadcast` chain ending in
   `divide(exp_result, broadcast_sum_result)`.
5. The stability `maximum(-inf_broadcast, reduce_max_result)` is allowed
   between steps 1 and 2 — we just walk past it.

If detected: emit `ttnn.softmax(input_tensor, dim=reduction_axis)` and skip
ALL the ops in the pattern, mapping the final divide's SSA name to the
softmax output.

**Implementation choice — when to match:**
- Easiest: match at parse time, REPLACE the op sub-sequence with a
  synthetic `op:'softmax'` entry. Engine then runs ttnn.softmax directly.
- This means cache-by-bytecode-hash still works; the pattern match happens
  inside `parse_stablehlo` (or a post-pass).

Trade-off: editing parsed ops means our parse cache stores the rewritten
form. Fine.

**Expected savings:** 13 ops × ~80us each = 1040us → 1 ttnn.softmax (~150us
on the small shapes we test). That's ~7x on the softmax program ALONE.
Combined with trace capture (which softmax couldn't reach before): full
gain is **3-4x** on top of step6-validated (1492us → ~250-350us).

### Step 7c: RMS-norm pattern match

JAX emits 12 ops:

```
%0 multiply x, x                                   <-- pow2 as mul-self
%cst constant(0)
%1 reduce(add) %0                                  <-- sum
%2 broadcast %1 [0] → [N,1]
%cst_0 constant(64.0)                              <-- divisor (D)
%3 broadcast %cst_0 [] → [N,1]
%4 divide %2, %3                                   <-- mean(x*x)
%5 broadcast g [1] → [1,K]
%6 broadcast %5 [0,1] → [N,K]
%7 multiply %6, x                                  <-- g*x  (NOT g*(x/...) — order matters)
%cst_1 constant(1e-6)
%8 broadcast %cst_1 [] → [N,1]
%9 add %4, %8                                      <-- mean + eps
%10 sqrt %9                                        <-- NOT rsqrt!
%11 broadcast %10 [0,1] → [N,K]
%12 divide %7, %11                                 <-- (g*x) / sqrt(mean+eps)
```

Pattern detection is more delicate than softmax: it has two parallel chains
(the variance and the weight-times-input). For v0 we'll skip RMS-norm
fusion and rely on step 7a for the speedup. If step 7a alone gets RMS-norm
under 500us, that's good enough — we DON'T need the fused op. If not,
we revisit.

**Note ttnn.rms_norm signature (ttnn 0.69):** `ttnn.rms_norm(input, epsilon,
weight=None, bias=None)`. Takes the SAME input tensor that LayerNorm took,
fuses the variance + scale + epsilon. Shape constraint: input must be at
least 2D; weight must broadcast across the last dim.

### Step 7d: re-benchmark

Same bench_device.py. Add a `step7-fusion` row showing softmax/linear/full-
block latency. The acceptance criterion is "softmax under 500us." Linear
should also fall (it has `broadcast_in_dim` on the bias).

## Risks & honest scoping

1. **`ttnn.repeat` might fail on certain shape patterns during trace
   capture.** If so, the trace capture's exception handler drops the
   trace and we fall back to eager-with-parse-cache (current behavior).
   Net: no regression, just no speedup. Acceptable.

2. **Pattern matching is brittle.** If JAX changes its softmax lowering
   (e.g., new JAX version reorders the stability max), our matcher
   silently fails to fire. We'll log a one-line note when the matcher
   does/doesn't fire on each new bytecode. If the matcher misfires
   (matches WRONG pattern), correctness breaks. Mitigation: the
   correctness gate inside `_capture_trace` (the eager warmup must match
   the trace replay) catches misfires.

3. **Constants in broadcast.** If `ttnn.repeat` on a pinned-from-warmup
   constant tensor doesn't behave well inside trace capture, we keep
   `broadcast_in_dim` of CONSTANTS in the skip-and-pin set but allow it
   for runtime values. (We can detect at trace-build time by checking
   the operand's `_DATA_INDEPENDENT_OPS` status.)

4. **If we ship step 7a but skip 7b/c**, softmax becomes traceable but
   still runs all 13 ops in the trace. That's the eager-via-trace path:
   13 × ~80us dispatch in trace = ~150us replay (trace fuses dispatch).
   GOOD ENOUGH. The fused-op work is bonus.

## Execution order

1. Free up qb1, run 50/50 baseline tests in parse-cache mode to confirm
   green-baseline.
2. Flip `broadcast_in_dim` out of `_HOST_TRANSFER_DEVICE_OPS`. Re-run
   tests. If any fail, investigate. If all pass, re-run bench.
3. IF step 2 lands softmax at <500us, declare victory and commit.
4. IF softmax is still >500us in trace mode (i.e., on-device-broadcast
   path is slow, not just blocking), implement softmax pattern match.
5. RMS-norm fusion only if step 4 doesn't land it under 500us.
6. Document everything in pjrt_reflections.md.

## Files I'll touch

- `pjrt_plugin/jax_plugins/tt/engine.py`: 1-line change to
  `_HOST_TRANSFER_DEVICE_OPS` for 7a. If 7b lands, add a
  `_match_softmax_pattern(ops)` pass in `parse_stablehlo` (or a new
  function called after parse).
- `pjrt_plugin/tests/test_engine_device.py`: add a `TestTraceSoftmax` /
  `TestTraceLN` / `TestTraceRMSNorm` to assert that trace cache fires
  AND outputs match (correctness + trace-fired assertion).
- `pjrt_plugin/scripts/inspect_softmax_trace.py`: small permanent script
  that prints `trace_cache[key]['failed']` after running each composite
  test, to debug.
- `research/pjrt_phase5_benchmarks.md`: append `step7-fusion` row.
- `research/pjrt_reflections.md`: new dated entry.

## Done criteria

1. 50/50 tests still pass.
2. softmax (1x64) under 500us in `bench_device.py` traced surface,
   verified across two runs.
3. (Stretch) softmax pattern match cuts another 30%+ on top.
4. Reflection log entry describes what flipped, what didn't, what surprised.
