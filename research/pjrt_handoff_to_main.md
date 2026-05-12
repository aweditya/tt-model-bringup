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
