# PJRT Phase 5 Handoff — Steps 5+6 Complete

Date: 2026-05-11
Branch: main
Last commit: `087ba48 PJRT Phase 5: fix double-engine-instance crash + trace release at exit`

## TL;DR

Phase 5 is at **Step 6 complete** — trace capture lands 9-13x speedup on
pure-device programs through the full JAX → PJRT → engine → Blackhole
pipeline. All 50 device tests pass together in one process. Step 7
(op fusion) is the next obvious win but was not started.

## Completed

### Step 5 — Benchmarks
Permanent benchmark at `pjrt_plugin/tests/bench_device.py`. Six surfaces:
raw ttnn, engine eager, parse-only, eager e2e, traced e2e, jax.jit.
Results land in `research/pjrt_phase5_benchmarks.md` (committed and
append-only across runs). Plan in
`research/pjrt_phase5_step5_plan.md`.

Findings:
- Parse cost dominates eager mode (1.4-1.7ms per call).
- Engine dispatch overhead is negligible (~5us on top of ttnn).
- C++ PJRT shim is ~50us — basically free.
- Per-op dispatch 45-95us; matmul dispatch-bound at 64x64 AND 256x256.

### Step 6 — Trace capture
- `_parse_cache: bytecode_hash → parsed ops` (always applies).
- `_trace_cache: bytecode_hash → {trace_id, placeholders, outputs}`.
  Applies when no host-transfer op participates in data-dependent
  compute.
- Env opt-out `TT_PJRT_NO_TRACE=1`.

Numbers (qb1, warm cache, single device 0):

| program            | eager  | traced  | speedup |
|--------------------|-------:|--------:|--------:|
| x + 1              | 1999us |  156us  |  12.8x  |
| exp(x)             | 1707us |  155us  |  11.0x  |
| a @ b 64x64        | 1795us |  199us  |   9.0x  |
| linear (a@w+b)     | 2186us |  534us* |   4.1x  |
| softmax            | 2904us | 1012us* |   2.9x  |
| jit: x + 1         | 2086us |  254us  |   8.2x  |
| jit: a @ b 64x64   | 1829us |  313us  |   5.8x  |

`*` programs include `broadcast_in_dim` and fall back to the parse-cache
path (no actual trace). They still get a measurable win.

Plan and discussion in `research/pjrt_phase5_step6_plan.md`. Full
reflection in `research/pjrt_reflections.md` under the 2026-05-11
(cont. 3) entry.

## Test status (qb1, run together in one process)

- `test_engine_device.py`: 23/23
- `test_basic_ops.py`: 27/27
- Together: 50/50, 3.48s

A long-standing latent bug surfaced during Step 6: test_engine_device
imported the engine via `importlib.spec_from_file_location` — a
SECOND module instance with its own `_device` global. When run after
test_basic_ops (which uses the canonical `jax_plugins.tt.engine`), the
second instance crashed re-opening device 0. Fixed in the same commit
by switching to `from jax_plugins.tt import engine`.

## In flight / next

### Step 7 — Op fusion (not started)

The big unblocked win for the no-trace programs (linear, softmax) is
making `broadcast_in_dim` reliably on-device, so softmax becomes
traceable. I added an on-device `ttnn.repeat` path inside
`_execute_broadcast_device` but kept `broadcast_in_dim` in the
`_HOST_TRANSFER_DEVICE_OPS` set (so trace capture still skips it)
pending broader validation under contention. The fallback to CPU is
preserved, so correctness is unchanged.

Step 7 plan to write when picking this up:
1. Validate on-device broadcast_in_dim with the full test suite.
2. Drop `broadcast_in_dim` from `_HOST_TRANSFER_DEVICE_OPS` if step 1
   stays green.
3. Pattern-match (max → sub → exp → sum → div) → `ttnn.softmax` for
   layer-norm and RMS-norm too.
4. Re-benchmark. Expected: softmax/linear drop to ~200us like other
   traceable programs.

### Phase 6 (longer term)

The trace path's floor is currently ~75% host transfer (numpy → ttnn
input, ttnn → numpy output). Removing that requires changing the PJRT
ABI so C++ holds device pointers directly. That's a separate phase.

## Blocked / known issues

- qb1's device 0 is occasionally locked by the model-bringup track
  (PCIe mutex). Tests can hang when this happens. Not a code issue.
- bf16 precision: `test_basic_ops.py::TestMatmul::test_larger_matmul`
  needs `atol=1.0` for a 128-deep matmul. Already handled with
  mode-aware tolerances. Pre-existing.
- Cross-process trace cache: traces are per-process. Not a regression,
  but a potential Phase 6 improvement.

## Final benchmark numbers (committed in research/pjrt_phase5_benchmarks.md)

Three full runs are recorded in
`research/pjrt_phase5_benchmarks.md`: `baseline-eager`, `step6-trace`,
`step6-validated`. Comparing them shows trace capture is stable across
runs.

## One-liner

**Phase 5 is at Step 6 complete (trace capture, 9-13x). Next: Step 7 op
fusion to extend traceability to softmax/layer-norm/RMS-norm.**

---

## Step 7 result (appended 2026-05-11)

### What fused

Nothing. Step 7 turned out to need ONE LINE — dropping `broadcast_in_dim`
from `_HOST_TRANSFER_DEVICE_OPS` — because the existing on-device
`ttnn.repeat` path inside `_execute_broadcast_device` is already robust
for the five broadcast patterns JAX emits (scalar->tensor, rank-up after
reduction, broadcast across reduced dim, per-channel rank-up, batch
broadcast). All trace-capture-safe.

I did NOT implement softmax/RMSNorm pattern-match fusion. The on-device
broadcast change alone landed the target latency and the cost/risk of a
pattern matcher wasn't justified.

### New benchmark numbers vs step6-validated

|                          | step6-validated | step7-broadcast | speedup |
|--------------------------|----------------:|----------------:|--------:|
| traced: x + 1            |          156us  |          160us  | ~       |
| traced: exp(x)           |          157us  |          156us  | ~       |
| traced: a @ b 64x64      |          200us  |          201us  | ~       |
| traced: linear (a@w+b)   |          534us  |      **228us**  | **2.3x**|
| traced: softmax (1x64)   |         1012us  |      **198us**  | **5.1x**|

Inspected with `pjrt_plugin/scripts/inspect_trace_status.py`: softmax,
layer_norm, rms_norm, linear, attention ALL hit trace cache cleanly.

### What didn't fuse (and why we didn't try harder)

The trace replay path has a hard floor of ~150us per call dominated by
numpy<->ttnn host transfer. A 13-op softmax trace and a 1-op `x+1` trace
both land at ~155-200us. Per-op fusion (replace 13 ops with 1
`ttnn.softmax` call) would save ~30-50us in the replay body — not worth
the brittleness of a pattern matcher that breaks on any JAX lowering
change.

### Test status

- `test_engine_device.py`: 27/27 (added 4 TestTrace assertions for
  softmax, layer_norm, rms_norm, linear).
- `test_basic_ops.py`: 27/27 unchanged.
- `test_engine.py` + `test_buffer.py`: 50/50 CPU tests unchanged.
- Together: **54/54 device, 50/50 CPU, all green.**

### Files touched

- `pjrt_plugin/jax_plugins/tt/engine.py` — one-line change to
  `_HOST_TRANSFER_DEVICE_OPS`.
- `pjrt_plugin/tests/test_engine_device.py` — new `TestTrace` class.
- `pjrt_plugin/scripts/inspect_trace_status.py` — new debug helper.
- `research/pjrt_phase5_step7_plan.md` — design doc.
- `research/pjrt_phase5_benchmarks.md` — two `step7-broadcast-on-device`
  rows appended (run + replay).
- `research/pjrt_reflections.md` — 2026-05-11 (cont. 4) entry.

### One-liner

**Phase 5 is at Step 7 complete (on-device broadcast trace-safe).
Softmax/linear/LN/RMSNorm now trace. Trace replay floor is ~150us —
70-75% host transfer. Phase 6 lever is the PJRT ABI: keep device
tensors across calls.**

---

## Vanilla tt-nn comparison results (appended 2026-05-11)

### The question

"When the same computation is hand-written in native tt-nn, how does it
compare to running it through our PJRT-traced path?"

This is the answer the entire PJRT effort hinges on. If PJRT is slower
than vanilla, the plugin is a worse abstraction. If equal, the value
is convenience. If faster, op fusion / trace logic is paying off.

### Setup

Plan: `research/pjrt_vanilla_comparison_plan.md`. Bench script:
`pjrt_plugin/tests/bench_vanilla_vs_pjrt.py`. Six programs, three
implementations each. Median + p90 over 100 measurement iters, 5
warmup. Two consecutive runs for stability (run1 had cold-cache
anomaly on P1; run2 is steady-state and the headline).

Cosine-equivalence between all three implementations: 1.0000 for
5 of 6 programs, 0.9998 for softmax. All paths compute the same
program.

### Numbers (run2, steady-state)

| Program | Vanilla eager med/p90 (us) | Vanilla traced med/p90 (us) | PJRT traced med/p90 (us) | PJRT / vanilla-traced |
|---|---:|---:|---:|---:|
| P1 x + 1 (1x32)        | 229 / 241 | 149 / 157 | 153 / 163 | **1.03** |
| P2 exp(x) (1x32)       | 216 / 229 | 148 / 156 | 150 / 161 | **1.01** |
| P3 a @ b (64x64)       | 224 / 235 | 195 / 208 | 200 / 206 | **1.03** |
| P4 softmax (1x64)      | 181 / 192 | 150 / 158 | 312 / 327 | **2.08** |
| P5 linear (a@w+b)      | 378 / 479 | 217 / 226 | 227 / 236 | **1.05** |
| P6 attention (8x32)    | 440 / 452 | 269 / 278 | 328 / 336 | **1.22** |

Mean ratio: 1.23 (dominated by the P4 softmax outlier). Excluding P4,
mean ratio is 1.07.

### Honest prose answer

**PJRT-traced is at parity with vanilla tt-nn-traced for 4 of 6
programs.** Ratios 1.01-1.05 for P1, P2, P3, P5 — within 5% of
vanilla. The engine's parse-cache hit + trace replay add ~3-10us of
bookkeeping per call, which is essentially free.

**1.22x slower on attention (P6, +58us).** Multi-op programs pay a
small linear-in-op-count overhead from the engine's per-op dispatch
during trace replay. Real but bounded.

**2.08x slower on softmax (P4, +161us).** This is the only material
gap. It's NOT framework overhead — it's the algorithmic difference
between `ttnn.softmax(dim=-1)` (one fused kernel) and the JAX
lowering `max → broadcast → sub → exp → sum → broadcast → div`
(5-13 ops). An expert writing vanilla TT-NN would never write the
decomposition; JAX users get it because JAX lowers softmax that way.
Pattern-match fusion in the engine (~1 day of work) would close
this gap.

### Recommendation

**Don't invest Phase 6 (PJRT ABI / persistent device tensors) for
raw latency.** The plugin already matches vanilla on real programs
(P1, P2, P3, P5 at 1.01-1.05x). The remaining engine overhead is
small, bounded, and scales harmlessly. Phase 6's gain on top of
parity would be marginal.

**Invest in softmax/RMS-norm/LN pattern-match fusion.** This is the
one place PJRT loses materially (P4 at 2.08x). For any transformer
workload — Qwen3-Coder-Next, Gemma4, etc. — JAX will lower softmax
into the decomposed form and pay 2x vs the fused kernel a hand
TT-NN engineer would write. Estimated work: ~1 day per fused
kernel (softmax, then RMSNorm, then LayerNorm). This is the highest
ROI investment the PJRT track has remaining.

### Files

- `pjrt_plugin/tests/bench_vanilla_vs_pjrt.py` — the comparison bench.
- `research/pjrt_vanilla_comparison_plan.md` — plan + methodology.
- `research/pjrt_phase5_benchmarks.md` — full results with run1 + run2
  numbers and per-bucket prose analysis.

### Test status

50/50 device tests still pass (`test_engine_device.py` +
`test_basic_ops.py`). Engine unchanged; only added a bench script.

### One-liner

**PJRT-traced is at parity with vanilla tt-nn-traced on 4/6 programs
(within 5%). The only material slowdown is softmax (2.08x) — JAX
lowers `softmax` to a 5-13-op graph while vanilla uses
`ttnn.softmax`. Pattern-match fusion is the next highest-ROI
PJRT investment.**
