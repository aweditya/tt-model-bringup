# TT-NN Trace Capture: The Missing Piece

## Q: Can TT-NN eliminate dispatch overhead without a full compiler?

**A: Yes!** TT-NN's trace capture API (`begin_trace_capture` / `end_trace_capture` / `execute_trace`) records a sequence of ops and replays them as a single command. This is analogous to CUDA Graphs or XLA's compiled executables. Results:

| Batch | Eager | Traced | Speedup |
|-------|-------|--------|---------|
| 32 | 0.292 ms (110K/s) | **0.090 ms (354K/s)** | **3.23x** |
| 128 | 0.284 ms (450K/s) | **0.120 ms (1.07M/s)** | **2.37x** |
| 512 | 0.306 ms (1.68M/s) | **0.189 ms (2.72M/s)** | **1.62x** |

The smaller the batch (= more dispatch-dominated), the bigger the win. At batch 32, we're 3.23x faster — the dispatch overhead that was killing us is nearly eliminated.

## Q: How does trace capture work?

**A:** Three-step process:

```python
# 1. Capture: record ops into a trace
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = my_model(input_tensor)  # ops are recorded, not executed
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 2. Execute: replay the trace (fast!)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

# 3. Feed new data: overwrite input buffer, then replay
ttnn.copy_host_to_device_tensor(new_input, input_tensor)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
```

The trace records the command sequence on the device. Replay skips all Python-side dispatch — it just tells the device "run that same sequence again." You can update input data by writing directly into the input tensor's device memory before replaying.

## Q: Does the trace produce correct results with new data?

**A: Yes.** We verified:
- Same data through trace vs eager: **max error = 0.000000** (bit-exact)
- New data through trace: also **max error = 0.000000** vs eager with same new data
- Different inputs produce different outputs (magnitude 2.68), confirming the trace actually reads the updated buffer

## Q: What's the raw dispatch overhead saved?

For a 10-op chain on tiny (32x32) tensors:
- Eager: 0.246 ms (0.025 ms/op)
- Traced: 0.087 ms (0.009 ms/op)
- **Saved: 0.160 ms** (2.84x speedup)

Per-op cost drops from 25µs to 9µs. The remaining 9µs is actual device execution time; the 16µs savings per op is pure dispatch overhead eliminated.

## Q: What does this mean for tt-xla?

**This is a critical insight.** An XLA backend for Tenstorrent doesn't necessarily need to do sophisticated kernel fusion or MLIR lowering to be useful. The minimum viable approach is:

1. **Receive StableHLO graph from JAX**
2. **Map each op to TT-NN equivalent** (mostly 1:1 mapping)
3. **Wrap the sequence in a trace capture**
4. **On execution, copy inputs → replay trace → copy outputs**

This "trace-based" backend would give 2-3x speedup over eager TT-NN dispatch with minimal compiler complexity. It's not full XLA compilation (no fusion, no op scheduling), but it captures the single biggest performance win.

The compilation path could then be:
- **Level 0** (current): Eager dispatch — 11 Python→device round-trips per forward pass
- **Level 1** (trace-based): Capture + replay — 1 command to replay all 11 ops
- **Level 2** (MLIR-optimized): Fuse ops, optimize memory placement, generate custom kernels

Level 1 is achievable today with existing TT-NN APIs. Level 2 is what tt-mlir aims for.

## Experiment

`experiments/12_trace_capture.py` — run on Blackhole p150a device 0, 2026-04-21.
