# Phase 5 Step 6 — Trace Capture Plan

Date: 2026-05-11

## Goal

Eliminate Python parse cost AND ttnn per-op dispatch cost from
`execute_stablehlo`. Replay the same program with new input data in a
single hardware-issued trace.

Step 5 told us:
- Parse: 1.4-1.7ms per call (50-70% of e2e time on small programs)
- Per-op dispatch: 45-95us (Surface 1+2 numbers)
- Transfers + bookkeeping: residual ~200-300us

Trace capture attacks all three:
- Parse: cache by bytecode hash → ~1us dict lookup
- Dispatch: collapse N ops into one `ttnn.execute_trace` call
- Transfers: still pay numpy→device for inputs and device→numpy for
  outputs (Phase 6 problem)

## Architecture

Add a `_trace_cache` dict in engine.py:
```
_trace_cache : { bytecode_hash : TraceEntry }

TraceEntry:
  parsed:        (func_args, ops, returns, private_fns)
  input_tensors: list[ttnn.Tensor]    # placeholders held across calls
  output_tensors:list[ttnn.Tensor]    # captured during begin/end
  output_shapes: list[tuple]          # logical shapes for _from_device
  output_kinds:  list['device'|'host']# some outputs are pure numpy (argmax)
  output_host:   list[np.ndarray|None]# for 'host' outputs, the warmup value
  trace_id:      int                  # ttnn trace handle
```

`execute_stablehlo` becomes:
1. Hash bytecode (use built-in hash of immutable bytes object — fast).
2. If hash in cache → `_replay_trace(key, inputs)`.
3. Else → `_eager_execute_and_capture(bytecode, inputs)` which:
   a. Parse + run once eagerly. Save the result for correctness checking.
   b. Build input placeholders (the SAME shapes as the actual inputs).
   c. `ttnn.begin_trace_capture(device, cq_id=0)` → re-execute with the
      placeholder tensors → save the OUTPUT tensors → `end_trace_capture`.
   d. Cache the entry.
   e. Return the eager result.

`_replay_trace(key, inputs)`:
1. For each input: `ttnn.copy_host_to_device_tensor(new, placeholder)`.
2. `ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)`.
3. `_from_device(output_tensor, output_shape)` for each output.

## Constraints — what trace capture forbids

From `experiments/tt_jax/trace.py` and ttnn docs:
- **No host-device transfers** during capture. Any `_to_device` /
  `_from_device` call inside captured ops will fail or produce a corrupt
  trace.
- **No allocation** during capture, in principle. ttnn re-uses pre-allocated
  output buffers; some compositions are OK but allocation patterns matter.

Our engine has several ops that ROUND-TRIP to host:
- `slice`, `gather`, `scatter`, `compare` (CPU fallback path),
  `iota`, `and`/`or`, `reduce_argmax`, `broadcast_in_dim` (CPU path)
- These all call `_to_device(np_result)` internally — host transfers.

### Strategy: skip-and-pin host-transfer ops in trace

Same pattern as `experiments/tt_jax/trace.py:TracedExecutor`:
1. Eager-execute the whole program ONCE, recording every intermediate
   value (the warm-up).
2. Identify ops whose device implementation does a host roundtrip.
3. During trace capture, SKIP those ops — re-use the value already
   computed during warm-up.
4. Capture only the pure-device ops.

The downside: the SKIPPED ops are not recompiled with new inputs. If we
replay the trace with new inputs, the skipped op's output is STALE
(still from warm-up). For shape-dependent / data-independent ops (iota,
constants), this is fine. For data-dependent ops (gather indices,
scatter), the trace is WRONG.

For Step 6 v0, we will:
- Only attempt trace capture when the program contains NO host-transfer
  ops. Detect at parse time by walking the op list.
- Programs with host-transfer ops fall back to eager (with parse cache).
- This is still a big win for math-heavy programs (softmax, linear,
  attention) which are 100% device-friendly.

### Detection

Host-transfer ops in our engine (from reading `_execute_op_device`):
- `slice` — `_execute_slice_device` calls `_operand_to_numpy` + `_to_device`
- `gather`, `scatter` — same
- `iota` — `execute_iota` (CPU) then `_to_device`
- `and`, `or` — same
- `reduce_argmax` — same
- `compare` — may fall back to CPU on failure
- `broadcast_in_dim` — currently always CPU roundtrip
- `constant` — generates on CPU then `_to_device`
- `concatenate` — may fall back

Of these, `constant` is special: it's host transfer ONLY at warm-up; the
captured tensor lives on device thereafter. So as long as constants are
**created once during the warm-up** and re-used in the trace by their
SSA name, they're safe. Same for `iota`. Same for broadcast_in_dim of
constants.

`broadcast_in_dim` is the biggest concern. Even broadcasting a runtime
value (a reduction result) goes through CPU in our current code. We'll
need to either (a) only trace programs without broadcast_in_dim, or
(b) implement broadcast_in_dim on-device.

For v0: define `HOST_TRANSFER_OPS = {'slice', 'gather', 'scatter', 'iota',
'and', 'or', 'reduce_argmax', 'broadcast_in_dim'}` and skip trace
capture if any of these appear in the op list. Constants are allowed
because they execute once at warm-up.

Actually, we can do better: SKIP the host-transfer op during trace
capture, but PIN its warmup value in the values dict. This works as long
as the input data doesn't change the skipped op's output. Constants and
iota satisfy this. broadcast_in_dim of a runtime value does NOT — its
broadcast result depends on the upstream value.

So v0: ONLY skip-and-pin ops whose result is genuinely data-independent
(`constant`, `iota`). For any other host-transfer op, refuse to trace.

## Cache key

Use `hash(bytecode)`. Python's `hash` on `bytes` is fast and stable
within a process. Across processes the hash changes (Python randomizes
hashes), so traces don't persist across runs. That's fine for v0 — JAX
keeps long-running processes.

Edge case: two different inputs produce the same bytecode but different
input SHAPES. JAX guarantees the StableHLO module encodes input
shapes/dtypes (no polymorphism in our path), so the same bytecode → same
shapes. Safe to key on bytecode alone.

## Input shapes during capture

When we begin_trace_capture, we need stable placeholder tensors with the
EXACT shapes the trace will replay with. We allocate them ONCE from the
first call's inputs (via `_to_device`), keep them, and `copy_host_to_device_tensor`
into them on subsequent calls.

If a future call passes inputs with different shapes → different bytecode →
different cache key. So we'll just build a fresh trace.

## Correctness gate

Each trace, on creation, immediately runs the eager path and the trace
path with the SAME inputs and compares outputs. If they don't agree
within bf16 tolerance, we drop the trace and fall back to eager forever
for that bytecode. (Future: log to a `trace_failures` list for debugging.)

## Disable switch

Env var `TT_PJRT_NO_TRACE=1` disables trace capture (fallback: eager).
Lets us A/B without rebuilding.

## File changes

1. `pjrt_plugin/jax_plugins/tt/engine.py`:
   - Add `_trace_cache` dict, `_TRACEABLE_OPS` set, `_DATA_INDEPENDENT_OPS` set.
   - New functions: `_classify_trace(bytecode)`, `_capture_trace`, `_replay_trace`.
   - Modify `execute_stablehlo` to consult cache.

2. `pjrt_plugin/tests/test_engine_device.py`:
   - Add `TestTrace` class: run a program twice, verify second run is
     much faster and results match.

3. `pjrt_plugin/tests/bench_device.py`:
   - Add Surface 5: `traced` — same e2e programs, expected 7-10x speedup.

4. `research/pjrt_phase5_benchmarks.md` — append numbers after Step 6.

5. `research/pjrt_reflections.md` — entry on what worked / didn't.

## Risks

1. **ttnn trace API edge cases.** Some ops may interact badly with trace
   capture (e.g., `ttnn.reshape` on tile-aligned tensors). We'll find
   out during implementation. Mitigation: catch exceptions during
   capture, drop the trace, log clearly.

2. **Cache memory.** Each trace pins input/output tensors on device. A
   long-running JAX program might create hundreds of distinct
   bytecodes (different shapes for different layers). Mitigation v0:
   cap cache at 64 entries with LRU. Probably enough; we'll measure.

3. **Output tensor lifetime.** ttnn output tensors captured during trace
   are owned by ttnn. `_from_device` must work after replay; we need to
   verify the captured tensors stay valid across multiple replays.

4. **Multi-output programs.** Decode step returns (hidden, k_cache, v_cache).
   Trace needs to capture all three. Our op interpreter handles this
   via `returns` list; we just save N output tensors and N shapes.

5. **`reduce_argmax` is multi-output AND host-roundtrip.** Skip-traceable
   only if argmax doesn't appear. Greedy decode programs will fall back
   to eager — acceptable trade-off; they're rare in latency-critical
   paths.

## Done criteria

1. Add trace cache to engine.py with `TT_PJRT_NO_TRACE` opt-out.
2. `test_engine_device.py` and `test_basic_ops.py` still pass.
3. Add a trace-specific test verifying second run matches first.
4. `bench_device.py` shows Surface 5 numbers, with measurable speedup
   vs Surface 3 on traceable programs.
5. Reflection log entry: what worked, what didn't, what the speedup is.
