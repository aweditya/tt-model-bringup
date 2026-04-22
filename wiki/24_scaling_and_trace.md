# Wiki 24: Scaling, Trace Capture, and Dynamic Inputs

## Q: How fast can Blackhole do raw matmul?

**A:** Traced matmul throughput on Blackhole (bfloat16):

| Size | Latency | TFLOPS |
|------|---------|--------|
| 32×32 | 0.028 ms | 0.002 |
| 128×128 | 0.032 ms | 0.129 |
| 512×512 | 0.044 ms | 6.1 |
| 1024×1024 | 0.071 ms | 30.4 |
| 2048×2048 | 0.181 ms | **95.0** |

At 2048×2048, Blackhole reaches **95 TFLOPS** — in the ballpark of its theoretical peak for bfloat16 computation. Below 256×256, the operation is dispatch/latency-bound, not compute-bound.

## Q: How does the transformer scale with model size?

**A:** Full transformer encoder (traced) at different configurations:

| Config (seq, d, ff) | Latency | Throughput | TFLOPS | Error |
|---------------------|---------|------------|--------|-------|
| 32, 64, 256 | 0.39 ms | 2,566/s | 0.009 | 0.026 |
| 32, 128, 512 | 0.39 ms | 2,543/s | 0.033 | 0.030 |
| 64, 128, 512 | 0.41 ms | 2,429/s | 0.066 | 0.035 |
| 64, 256, 1024 | 0.42 ms | 2,370/s | 0.249 | 0.039 |
| 128, 256, 1024 | 0.46 ms | 2,197/s | 0.479 | 0.039 |
| 128, 512, 2048 | 0.57 ms | 1,742/s | 1.461 | 0.042 |
| 256, 512, 2048 | 0.68 ms | 1,461/s | 2.550 | 0.042 |

Key observations:
- **Sub-linear scaling**: 8x more compute (tiny → XL) only increases latency 1.75x
- **Compute efficiency grows with size**: 0.009 → 2.55 TFLOPS as the model gets bigger
- **Error stays bounded**: max error < 0.05 across all configs (bfloat16 precision)

## Q: Why is the transformer only 2.5 TFLOPS when raw matmul hits 95?

**A:** Several factors:

1. **Small matrix sizes**: The transformer's matmuls are at most 256×512 or 512×2048 — far from the 2048×2048 where peak throughput is achieved. At 512×512, raw matmul is only 6 TFLOPS.

2. **Non-matmul overhead**: The 56-equation transformer includes reductions (mean, sum, max), elementwise ops (exp, sqrt, multiply), and broadcasts — these don't approach matmul throughput.

3. **Memory bandwidth**: At smaller sizes, operations are memory-bound (moving data to/from compute cores) rather than compute-bound.

4. **Op count**: 7 matmuls share time with 49 other operations (reductions, broadcasts, elementwise).

## Q: How does trace capture work for dynamic inputs?

**A:** Three-phase protocol:

```python
# Phase 1: Warmup (allocates all intermediate buffers)
interp = Interpreter(device, literal_cache=cache)
for eqn in jaxpr.eqns:
    interp._exec(eqn)

# Phase 2: Trace capture (reuses existing buffers)
tid = ttnn.begin_trace_capture(device, cq_id=0)
for eqn in jaxpr.eqns:
    interp._exec(eqn)  # Same ops, same buffer addresses
ttnn.end_trace_capture(device, tid, cq_id=0)

# Phase 3: Execute with new inputs
for new_x in batches:
    ttnn.copy(new_x_tt, x_tt)  # Overwrite input buffer
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    # Output is in interp.env[out_var] — same buffer, new values
```

The warmup is critical: trace capture forbids new buffer allocations. The warmup run creates all intermediate buffers, and the trace capture records operations on those exact memory addresses. `ttnn.copy` overwrites input buffer contents without changing addresses.

## Q: What's the overhead of ttnn.copy for input swapping?

**A:** Only **0.021 ms** — about 5% of the transformer's 0.41 ms total latency. This means a real backend can process different inputs at nearly the same speed as replaying the same input:

- **Trace only**: 0.390 ms (2,563 fwd/sec)
- **Copy + trace**: 0.411 ms (2,431 fwd/sec)

## Q: What does this validate for the PJRT plugin approach?

**A:** This proves the complete compile-execute pipeline:

| PJRT Function | Our Implementation |
|---------------|-------------------|
| `Compile(StableHLO)` | Jaxpr → interpreter warmup + trace capture |
| `Execute(inputs)` | `ttnn.copy(new_input, buffer)` + `execute_trace` |
| `BufferFromHostBuffer` | `tensors.to_device()` |
| `BufferToHost` | `tensors.from_device()` |

The performance characteristics (sub-ms latency, 95 TFLOPS peak matmul) are competitive with GPUs for inference at these model sizes.
