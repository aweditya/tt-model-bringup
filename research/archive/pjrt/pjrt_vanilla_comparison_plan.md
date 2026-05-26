# Plan — Vanilla tt-nn vs PJRT Comparison

Date: 2026-05-11
Author: PJRT track (autonomous)

## The question

The TL;DR everyone wants:

> When you hand-write the same computation in native tt-nn, how does it
> compare to running it through our PJRT-traced path?

Three possible outcomes:

1. **PJRT slower → plugin is worse abstraction.** Document why and where
   the overhead is. Decide whether to fix or accept.
2. **PJRT at parity (<10% delta) → plugin's value is convenience only.**
   Document and accept; performance work moves elsewhere.
3. **PJRT faster → op-fusion / trace logic is paying off.** Keep
   investing.

## Programs

Six programs, each implemented three ways:

| ID | Program           | Shape         | Op count | Why we picked it |
|----|-------------------|---------------|----------|------------------|
| P1 | `x + 1`           | (1, 32)       | 1        | Pure dispatch overhead — both paths reduce to 1 ttnn.add |
| P2 | `exp(x)`          | (1, 32)       | 1        | Pure unary — single ttnn.exp |
| P3 | `a @ b`           | (64, 64)      | 1        | Single matmul — measures matmul kernel time |
| P4 | `softmax(x)`      | (1, 64)       | 5-13     | Multi-op decomposition — PJRT collapses via trace |
| P5 | `linear(a, w, b)` | a:(2,64), w:(64,32), b:(32,) | 2 | matmul + broadcast add |
| P6 | `attention`       | x:(8,32), wq/wk/wv:(32,32), wo:(32,32) | ~10 | Real fusion test — many ops |

For P6, "attention" = `softmax(q @ k.T) @ v`, single head, no scale —
just enough sequential ops to be a real workload.

## Three implementations per program

### A. Vanilla tt-nn (eager, no trace)

Hand-written python. No JAX, no PJRT, no engine. Just `ttnn.add`,
`ttnn.matmul`, etc. directly. Tensors created with `_to_device` to share
the same bf16/TILE_LAYOUT path as the engine — but no other engine
infra. This is the "raw ttnn" baseline.

### B. Vanilla tt-nn + manual trace

Same hand-written python, but wrapped in
`ttnn.begin_trace_capture` / `ttnn.end_trace_capture` directly. Input
data goes in via `ttnn.copy_host_to_device_tensor` to placeholders, and
the output gets read out at the end. This is the user-writes-the-trace
baseline — the level a careful TT engineer would write by hand.

### C. PJRT-traced (engine.execute_stablehlo with warm trace cache)

Build the JAX function, lower to stablehlo, hand the bytecode to
`engine.execute_stablehlo`. Run once to warm trace cache, then time the
cache-hit replay loop. This is exactly Surface 5 in the existing
benchmarks.

## What's "vanilla"?

To prevent A/B drift, all three paths must:

- Use `engine._get_device()` to share the same Blackhole handle.
- Use `engine._to_device(np_arr)` to upload tensors — same dtype
  (bf16), same layout (TILE_LAYOUT), same padding. No
  WormholeComputeKernelConfig — let ttnn use its defaults consistently.
- Run **device 0 only**.

No per-path tuning: if you wouldn't write a `ttnn.matmul` with a
specific program config in vanilla, the PJRT path doesn't get one
either.

## Synchronize policy

Both paths under test should produce one numpy array at the end of each
iteration. That means:

- **Inputs:** numpy → device every iteration (host upload included in
  timing). PJRT pays this cost too, so this is fair.
- **Outputs:** device → numpy at the end of each iteration. Both pay
  this cost too.
- **No explicit `synchronize_device` between iterations.** Each path's
  output→numpy call already drains the queue (via `ttnn.to_torch`).

This isolates the **steady-state per-call cost** of each path with
matched I/O.

The existing `bench_device.py` uses `sync_each=False` for the e2e and
traced surfaces because `execute_stablehlo` already returns a numpy
array (so the output read is the implicit sync). We mirror that here.

## Timing protocol

- Warmup: 5 iters per (program, implementation).
- Measure: 100 iters with `time.perf_counter_ns()`.
- Report: median (us) and p90 (us). Median is robust to one-off
  dispatch hiccups; p90 catches tail latency.

(Note: the user asked for median+p90 specifically. Existing
`bench_device.py` reports mean+p99 — different. We're following the
user spec here.)

## Output

Append a `## Vanilla tt-nn vs PJRT comparison (YYYY-MM-DD)` section to
`research/pjrt_phase5_benchmarks.md` with:

| Program | Vanilla eager | Vanilla traced | PJRT traced | Ratio (PJRT / vanilla traced) |

Plus a short prose summary at the bottom answering the question
honestly.

## Fairness checks

Before reporting results, sanity-check:

1. All three implementations produce numerically close outputs
   (cosine > 0.99 on a sample input). Otherwise we're comparing
   different programs.
2. All three use the same input data (same `np.random.seed`).
3. The vanilla-traced path actually captured a trace (no fall-through).
   The PJRT path actually hit the trace cache (introspect
   `_trace_cache`).
4. Run the bench twice. If the second run is materially different from
   the first, document both.

## What I'm NOT doing

- Not changing `engine.py`.
- Not regressing existing tests (`test_engine_device.py` +
  `test_basic_ops.py` = 50/50).
- Not implementing new ops in the engine. If P6 (attention) can't be
  traced today, we report that honestly and use a simpler proxy.
- Not benchmarking on multi-device or batch > 1.

## File outputs

- `pjrt_plugin/tests/bench_vanilla_vs_pjrt.py` — the permanent bench
  script.
- Append to `research/pjrt_phase5_benchmarks.md`.
- Append "## Vanilla tt-nn comparison results" section to
  `research/pjrt_handoff_to_main.md`.

## Honest scoping

If PJRT is materially slower (>20%), document the suspected cause and
either fix it (small fixes only) or note for future work (big fixes).
No cherry-picking. The whole point is to know the truth.
