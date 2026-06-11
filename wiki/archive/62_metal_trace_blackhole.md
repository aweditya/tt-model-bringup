# Wiki 62: Metal Trace on Blackhole — Capture, Replay, Limitations

## Q: What is Metal Trace and why does it matter?

**A:** Metal Trace records a device-side operation graph, then replays it without Python dispatch overhead. Each `ttnn` op call from Python costs ~30us of host-side dispatch latency. For a model with 1500+ ops per token, that adds up to ~45ms — often more than actual device compute. Trace capture eliminates this entirely by recording the ops once and replaying as a single device command.

## Q: How do you capture and replay a trace?

**A:** Three-step process:

```python
# 1. Capture: record the op graph (ops are recorded, not truly executed)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model_forward(input_buffer)  # all ttnn ops recorded
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# 2. Update inputs between replays
ttnn.copy(new_input_tensor, input_buffer)

# 3. Replay: execute the recorded graph
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
result = from_dev(output, shape)  # read from persistent output buffer
```

The trace records the GRAPH, not the VALUES. Device tensor contents can change between replays via `ttnn.copy()`.

## Q: What are the requirements for trace capture?

**A:** Five requirements, all discovered through experiments:

1. **Fixed tensor shapes** — every tensor must have identical shape at every replay. Single-token decode (1x1x1xhidden) qualifies naturally.

2. **No host-side control flow** — no Python if/else based on tensor values during the traced section. This is why MoE routing (which reads expert indices to the host) cannot be traced.

3. **Pre-allocated persistent buffers** — input and output tensors must exist before capture. Update contents with `ttnn.copy()`, never allocate new tensors.

4. **Program cache must be enabled** — `device.enable_program_cache()` before capture. Without it, kernels compile during trace, wasting time and potentially producing incorrect traces.

5. **Warmup first** — run one non-traced iteration to compile all kernels before capturing. This populates the program cache so the trace captures compiled programs.

## Q: What can be traced and what cannot?

**A:**

| Component | Traceable? | Why |
|-----------|-----------|-----|
| Matmul / linear | Yes | Static graph, fixed shapes |
| RMSNorm | Yes | Static |
| RoPE (rotary embedding) | Yes | cos/sin updated via `ttnn.copy` |
| SDPA decode | Yes | `cur_pos_tensor` is a device tensor |
| KV cache update | Yes | `update_idxs_tensor` is a device tensor |
| Embedding lookup | No | Host-side token-to-vector mapping |
| MoE routing | No | Requires host readback of expert indices |
| Dynamic control flow | No | Expert selection branches |
| Python scalar parameters | No | Baked into trace at capture time |

The key insight: any value that changes per step must be a **device tensor** updated via `ttnn.copy`, not a Python int/float. This was the breakthrough that made traced KV cache updates work (exp 53: `paged_update_cache` with `update_idxs_tensor`).

## Q: How does partial tracing work for MoE?

**A:** When a model has both traceable and non-traceable sections, capture separate traces for each traceable block:

```python
# Capture 24 attention traces (one per layer)
for i in range(24):
    x_in = allocate_persistent_buffer(shape)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    # ... attention ops for layer i ...
    x_out = ttnn.add(x_in, attn_output)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    traces.append(trace_id)

# During decode: interleave trace replay with eager MoE
for i in range(24):
    ttnn.copy(x, attn_inputs[i])
    ttnn.execute_trace(device, traces[i], blocking=False)
    x2 = attn_outputs[i]
    # ... eager MoE routing + expert dispatch ...
    x = ttnn.add(x2, moe_result)
```

Each layer needs its own persistent input/output buffers, but cos/sin/pos buffers can be shared across all 24 layers (updated once per decode step).

## Q: What speedups has trace capture achieved?

**A:**

| Model | Non-traced | Traced | Speedup |
|-------|-----------|--------|---------|
| Qwen2.5-0.5B (dense) | 21ms/tok | 7.4ms/tok | **2.83x** |
| MoE attention only | 11.4ms | ~3ms | **~3.8x** |
| MoE full model (partial trace) | 48ms/tok | 44ms/tok | **1.09x** |

For dense models, full tracing yields massive speedups because ALL ops are traced. For MoE, attention is only 24% of total time, so tracing it gives a smaller absolute improvement. The MoE expert dispatch (~32% of time) remains eager and dispatch-bound.

## Q: What are the gotchas?

**A:**

1. **Buffer allocation during active trace is unsafe.** The warning "Allocating device buffers is unsafe due to the existence of an active trace" means new allocations can corrupt trace memory. Pre-allocate everything before capture.

2. **Multiple active traces share device memory.** Each trace reserves buffer space. 24 attention traces are fine, but attempting hundreds could exhaust device memory.

3. **`ttnn.release_trace(device, trace_id)` is needed for cleanup.** Traces persist until explicitly released.

4. **`blocking=False` with `execute_trace` allows overlap** with subsequent eager ops on the same command queue, but you must sync before reading outputs.

---

*Experiments 37, 52-53, 95. Trace capture on Blackhole P150: 2.8x for dense models, partial trace for MoE. April 2026.*
