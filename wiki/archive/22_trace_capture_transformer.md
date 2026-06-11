# Wiki 22: Trace Capture for Full Transformer — 2,564 fwd/sec

## Q: What is TT-NN trace capture and why does it matter?

**A:** Trace capture records a sequence of device operations into a replayable "trace" that executes without Python dispatch overhead. Normal execution has ~50μs of Python overhead per op × 56 ops = ~2.8ms of pure overhead. Trace capture eliminates this entirely.

The API:
```python
tid = ttnn.begin_trace_capture(device, cq_id=0)
# ... run all device operations ...
ttnn.end_trace_capture(device, tid, cq_id=0)

# Replay the exact same operations, no Python overhead
ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
```

**Critical constraint:** During trace capture, ALL host↔device transfers are forbidden — no reads (`ttnn.to_torch`) and no writes (`ttnn.from_torch`/`to_device`).

## Q: What was blocking trace capture for the transformer?

**A:** Two sources of host transfers during execution:

1. **CPU broadcast round-trips** — our original `broadcast_to_match()` read tensors back to host, did `np.broadcast_to`, then wrote back. Fixed by on-device `ttnn.repeat`.

2. **Literal materialization** — `eval_var()` for Jaxpr literals (like `-inf`, `64.0`, `1e-5`) called `to_device()` which writes to device. Fixed by pre-materializing all literals into a cache before trace capture.

3. **Scalar ops in _binary_with_broadcast** — the literal-handling paths called `interp.to_device(np.array(val))` instead of `interp.eval_var(var)` which checks the cache. Fixed by using `eval_var` consistently.

## Q: What's the trace capture strategy?

**A:** Three-phase approach:

1. **Pre-materialize**: Scan Jaxpr for all literal values, put them on device, store in `literal_cache` dict (keyed by float value)
2. **Pre-load inputs**: All 11 input tensors (weights, biases, input) loaded before trace starts
3. **Trace capture**: Create `Interpreter(device, literal_cache=cache)`, bind pre-loaded inputs to env, then execute all 56 equations inside `begin_trace_capture/end_trace_capture`

The transformer has exactly 3 literals: `-inf` (softmax clamp), `64.0` (attention scale √d), `1e-5` (layernorm epsilon).

## Q: What are the performance results?

**A:**

| Mode | Latency | Throughput | vs Original |
|------|---------|------------|-------------|
| Original (CPU broadcast) | 5.59 ms | 179 fwd/sec | 1.0x |
| On-device broadcast | 2.99 ms | 334 fwd/sec | 1.9x |
| Trace capture | 0.39 ms | 2,564 fwd/sec | **14.3x** |

The 7.7x speedup from trace capture confirms that Python dispatch overhead was the dominant cost at the interpreted level.

## Q: What does 0.39ms per forward pass mean?

**A:** This is the time for one forward pass of a single-layer transformer encoder (32×64 input, 64-dim attention, 256-dim FFN) on Blackhole. At 2,564 fwd/sec:

- The device is executing ~143,000 TT-NN operations per second (56 ops × 2,564)
- Average per-op latency: ~7μs (pure device time, no dispatch)
- This is close to the theoretical minimum for these tensor sizes

## Q: Does trace capture affect numerical accuracy?

**A:** No. Max error vs JAX reference is 0.029 in both traced and interpreted modes — identical. Trace capture replays the exact same operations.

## Q: What are the limitations of trace capture?

**A:**

1. **Static computation graph** — the trace replays the exact same operations. Any data-dependent control flow (if/else on tensor values) can't be captured.
2. **Fixed tensor shapes** — all shapes must be known at trace time. Dynamic shapes require re-tracing.
3. **No new host transfers** — you can't add new inputs during execution. To change inputs, you'd need to write to pre-allocated buffers (the `execute_trace` API supports this).
4. **Memory overhead** — the trace stores all intermediate tensors. For larger models, this could be significant.

## Q: How does this compare to other frameworks?

**A:** For reference, on the same model size:
- PyTorch eager on CPU: ~10-50ms (estimated)
- JAX on CPU: ~5-20ms (estimated)
- Our traced execution on Blackhole: 0.39ms

The comparison isn't entirely fair (different hardware, bfloat16 vs float32), but it demonstrates that Blackhole can achieve very low latency for transformer inference when dispatch overhead is eliminated.

## Q: What's the path to a production-ready system?

**A:** The trace capture experiment validates the core architecture:

1. **Jaxpr → TT-NN interpreter** handles the compilation step
2. **Pre-materialization** handles the setup step
3. **Trace capture** handles the execution step

This is essentially what a PJRT plugin's `Compile()` + `Execute()` would do. The next step is wrapping this in the PJRT C API so JAX can call it directly.
