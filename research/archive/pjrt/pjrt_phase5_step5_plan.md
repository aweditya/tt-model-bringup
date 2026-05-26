# Phase 5 Step 5 — Performance Benchmarking Plan

Date: 2026-05-11

## Goal

Measure where time goes in the PJRT-on-Blackhole pipeline so we know what
Step 6 (trace capture) must beat. Numbers are the deliverable; the
benchmark code is permanent infrastructure for the rest of the project.

## What we're measuring

Three distinct surfaces — we time each one separately so we can attribute
overhead correctly.

### Surface 1: Raw ttnn (bare ops, no engine)

Direct ttnn calls on tensors already on device. This is the floor — the
fastest a single op can run from Python.

Measured ops (all bf16, TILE_LAYOUT):
- `ttnn.add(a, b)` on tile-aligned 1x32 and 32x32 tensors
- `ttnn.matmul(a, b)` on a 64x64 @ 64x64 (smallest realistic)
- `ttnn.matmul(a, b)` on a 256x256 @ 256x256 (bigger, dispatch-bound less)
- `ttnn.exp` on 1x32 (transcendental)

For each: time 1000 iterations after a warm-up, divide. Report mean +
p99 in microseconds.

### Surface 2: Engine eager (our `_execute_op_device`)

Same ops, dispatched through our op-table. This isolates engine overhead
(Python op-lookup, get_operands, dict updates) on top of ttnn dispatch.

We time:
- `engine._execute_op_device({op: 'add', ...}, values)` — same shapes as Surface 1
- `engine._execute_op_device({op: 'dot_general', ...}, values)` — 64x64 and 256x256

The input tensors are already on device (no `_to_device` cost included).
This isolates pure dispatch overhead introduced by our Python layer.

### Surface 3: Engine end-to-end (`engine.execute_stablehlo`)

Full path: numpy → `_to_device` for each input → execute every op →
`_from_device` for each output → numpy. This is what the C++ plugin
actually calls.

Programs we time:
- `lambda x: x + 1.0` (1 op, scalar broadcast + add) — minimal program
- `lambda x: jnp.exp(x)` (1 op, transcendental) — minimal transcendental
- `lambda x, w: x @ w` (1 op, matmul 64x64) — minimal matmul
- `lambda x, w, b: x @ w + b` (matmul + add) — linear layer
- A small softmax: `jnp.exp(x - x.max()) / jnp.exp(x - x.max()).sum()` (7-8 ops)

We measure StableHLO bytecode each time (caches in JAX),
then time `engine.execute_stablehlo(bc, inputs)` over 100 iterations.

### Surface 4 (informational): PJRT through `jax.jit`

`jax.jit(f)(*args)` — the path C++ + Python plugin both participate in.
The C++ layer adds overhead vs calling the engine directly.

Time the same programs as Surface 3, this time through `jax.jit` on the
`tt` device. Report deltas from Surface 3 to attribute C++ shim cost.

## Separating dispatch / transfer / compute

Three-way split per Surface 3 program:

1. **Compute** = Surface 2 (engine.execute_op_device, already on device).
2. **Transfer (per call)** = Surface 3 − Surface 2 × N_ops. The remainder
   is `_to_device` × N_inputs + `_from_device` × N_outputs, plus parse
   time. For single-op programs, this directly equals transfer + parse.
3. **Parse** = time `engine.bytecode_to_text(bc)` + `engine.parse_stablehlo(text)`
   in isolation, on the same bytecode, 100 iterations.
4. **Dispatch (per op)** = Surface 2 − Surface 1. The gap is what our
   Python op-table + dict bookkeeping costs.

## Comparison baseline

Three baselines:
- Numpy CPU (engine in numpy mode): floor for the parse/dispatch path,
  isolates the ttnn cost.
- Raw ttnn (Surface 1): floor for the hardware op cost.
- (Future) Engine traced (Step 6): the upper bound we hope to hit.

## Output

Permanent benchmark script: `pjrt_plugin/tests/bench_device.py`. Runs as
both a CLI (`python bench_device.py`) and via pytest collection (skipped
without `TT_PJRT_BENCH=1` so it doesn't slow down the regular test suite).

Numeric results land in `research/pjrt_phase5_benchmarks.md` as a markdown
table — committed and reviewable. Every run appends a section with the
date and git SHA so we can track regressions.

## Methodology guardrails

1. Warm-up: 10 iterations before timing. Kernel JIT is brutal on
   first call (~3s for cold cache); we measure WARM only.
2. Synchronization: call `ttnn.synchronize_device(_device)` AFTER each
   timed iteration in Surfaces 1+2. ttnn ops are async-queued, so without
   sync we'd just measure submission latency, not work-completion time.
   For Surface 3, the engine already syncs via `_from_device` at exit.
3. Run with `TT_PJRT_USE_DEVICE=1` for device measurements, unset for numpy.
4. Single device (device 0) — per CLAUDE.md non-negotiable.
5. No `/tmp`, no inline scripts.

## Tolerances

Correctness check inside each timed run — every benchmark validates the
output against numpy reference with the same atol/rtol as the device
tests (1e-2 atol, 5e-2 rtol). If correctness drifts during the bench, we
fail loudly — we don't time wrong answers.

## Risks

- bf16 matmul on 256x256: drift might exceed our 5% rtol on some
  elements. If so, widen for the bench-only runs (we're measuring perf,
  not correctness).
- Cold cache on first run: kernel JIT can take 3-7s. We mitigate with a
  10-iter warm-up. If we still see jitter we'll increase warm-up.
- Python timing precision: `time.perf_counter` gives sub-microsecond
  resolution; we batch 1000 iter so per-op resolution is fine.

## Done criteria

1. `pjrt_plugin/tests/bench_device.py` runs on qb1 and prints a clean table.
2. `research/pjrt_phase5_benchmarks.md` has the numbers committed.
3. A reflection-log entry summarizing what the numbers mean for Step 6.
