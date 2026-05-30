# Reducing Dispatch Overhead and Sync Latency on Blackhole

**Date:** April 2026
**Context:** MoE inference on Blackhole P150, ~840 ttnn ops per token, ~50ms/tok total, ~25ms from dispatch overhead alone (with program cache, without trace).

---

## 1. Program Cache Internals

### What it does
`device.enable_program_cache()` caches compiled kernel binaries so that the second and subsequent calls to the same op (with same shapes, dtypes, memory configs) skip recompilation. The first call compiles the kernel, subsequent calls look up the cached binary by a hash of the op parameters.

### What the ~30us per op buys you
With program cache enabled, each dispatched op still requires:
1. **Python overhead** -- ttnn Python binding -> C++ call, argument marshaling (~5-10us)
2. **Cache lookup** -- hash op parameters, find cached program (~2-5us)
3. **Command construction** -- build dispatch command with tensor addresses, kernel args (~5-10us)
4. **PCIe command submission** -- write command to device's command queue via PCIe MMIO (~5-10us)
5. **Device-side command processing** -- dedicated RISC-V dispatch core reads and processes command (~5us)

The 30us/op figure is realistic for eager dispatch with program cache. Without cache, ops can take 100-500us+ due to kernel compilation. With trace, per-op overhead drops to ~3-5us (just device-side replay from DRAM buffer, no host involvement).

### Key insight
Program cache eliminates compilation but NOT dispatch. For 840 ops at 30us = 25.2ms. This is consistent with our measurements.

---

## 2. Async Dispatch

### What exists
- **`ttnn.enable_asynchronous_slow_dispatch(device)`** -- Experimental. For slow dispatch mode only (not fast dispatch). Enables running multiple non-overlapping programs concurrently. Not relevant to our fast dispatch setup.
- **`blocking=False` on EnqueueReadBuffer/EnqueueWriteBuffer** -- At the C++ Metal level, read/write commands accept a `blocking` parameter. When `False`, the call returns immediately without waiting for the DMA to complete. This is how fast dispatch already works internally.
- **Fast dispatch is inherently async** -- Commands are queued into the command queue and the device's dedicated RISC-V dispatch core processes them independently. The host doesn't block on each op unless you explicitly call `synchronize_device()` or read back a tensor.

### The real problem
ttnn ops in Python are NOT fully async at the Python level. Each `ttnn.matmul(...)` call:
1. Computes output shape and allocates output buffer
2. Builds and submits the dispatch command
3. Returns a tensor handle (doesn't wait for completion)

Steps 1-2 are the ~30us overhead. The device execution IS overlapped with the next host dispatch, but at 840 ops the host-side dispatch pipeline itself is the bottleneck.

### Verdict
Async dispatch won't help us -- fast dispatch is already async. The bottleneck is the sheer number of host-side dispatch calls, not waiting for device completion.

---

## 3. Op Fusion

### Available fused operations

**Matmul + activation fusion (critical for MoE):**
```python
# Fuse activation directly into matmul/linear
output = ttnn.linear(x, weight, bias=bias, activation="silu")
output = ttnn.linear(x, weight, bias=bias, activation="gelu")
output = ttnn.linear(x, weight, bias=bias, activation="relu")
```
Supported activations include: relu, silu, gelu, sigmoid, tanh, and others via `ttnn.UnaryWithParam`. For sharded tensors, use `fused_activation` parameter in the `program_config` instead.

**SwiGLU / GeGLU (fused gate+activation+multiply):**
```python
# Fused: splits input, applies silu to second half, multiplies with first half
output = ttnn.swiglu(concatenated_gate_up)  # replaces 3 ops (split + silu + mul)
output = ttnn.geglu(concatenated_gate_up)   # replaces 3 ops (split + gelu + mul)
```
Requires input last dim divisible by 64. Supports bf16 and bf8_b, TILE and ROW_MAJOR layouts.

**Fused RMS norm (multi-device, may not apply to single device):**
```python
# Fuses pre-RMS, all-gather, post-RMS, residual add, gamma, resharding
ttnn.fused_rms_minimal(...)
```
Requires specific tensor shapes and is designed for multi-device (all-gather).

**SDPA (fused attention):**
```python
# Already fused: Q*K^T scaling, softmax, V multiplication
ttnn.transformer.scaled_dot_product_attention(q, k, v)
ttnn.transformer.scaled_dot_product_attention_decode(q, k, v)
```

**Batched matmul for MoE experts:**
```python
# Single dispatch for multiple weight matrices (e.g., all experts at once)
outputs = ttnn.matmul_batched_weights(input, [w1, w2, w3, ...], activation="silu")
```

### Impact estimate for MoE
Per MoE layer, we currently dispatch per-expert matmuls separately. With fusion:
- `matmul + silu` saves 1 op per expert per layer (was: matmul, then silu)
- `swiglu` saves 2 ops per gate_up projection (was: split, silu, multiply)
- `matmul_batched_weights` could batch all expert projections into 1 dispatch

For 24 layers with 8 experts each, even saving 3 ops per expert per layer = 576 fewer dispatches = ~17ms saved at 30us/op.

---

## 4. Command Batching

### No explicit API for batching
There is no user-facing API to batch multiple commands into a single PCIe submission. However:

1. **Fast dispatch already batches implicitly** -- The host writes commands to a ring buffer in host memory. The device's dispatch RISC-V core fetches commands in batches via DMA. So multiple commands submitted in quick succession ARE fetched together.

2. **The bottleneck is host-side command construction**, not PCIe round-trips. Each op still requires Python->C++ call, output allocation, and command construction. PCIe writes are posted (fire-and-forget for writes), so they don't require round-trips.

3. **Metal trace IS the ultimate command batch** -- It records ALL commands into a DRAM buffer and replays them with a single host command. This is the closest thing to "batching all commands."

### Verdict
No additional batching mechanism exists beyond trace. The path forward is either: (a) use trace, or (b) reduce the number of ops via fusion.

---

## 5. Partial Tracing

### Can you trace part of the model?

**Yes, but with constraints.** The key strategies from TT's documentation:

**Strategy 1: Multiple independent traces**
You can capture multiple traces and execute them sequentially:
```python
# Trace just attention
trace_attn = ttnn.begin_trace_capture(device, cq_id=ttnn.QueueId(0))
attn_out = attention_block(x_attn_input)
ttnn.end_trace_capture(device, trace_attn, cq_id=ttnn.QueueId(0))

# Trace just MLP/MoE
trace_mlp = ttnn.begin_trace_capture(device, cq_id=ttnn.QueueId(0))
mlp_out = mlp_block(x_mlp_input)
ttnn.end_trace_capture(device, trace_mlp, cq_id=ttnn.QueueId(0))

# Execute: alternate between eager ops and traced blocks
for layer in layers:
    # Eager: rms_norm (if it needs CPU readback)
    x = ttnn.rms_norm(x, weight)
    
    # Traced: attention (static graph, no readback)
    copy_into(x_attn_input, x)  # overwrite trace input buffer
    ttnn.execute_trace(device, trace_attn, cq_id=ttnn.QueueId(0), blocking=False)
    
    # Eager: residual add, rms_norm
    x = ttnn.add(x, attn_out)
    x = ttnn.rms_norm(x, weight2)
    
    # Traced: MLP
    copy_into(x_mlp_input, x)
    ttnn.execute_trace(device, trace_mlp, cq_id=ttnn.QueueId(0), blocking=False)
    x = ttnn.add(x, mlp_out)
```

**Strategy 2: Address pinning for trace reuse**
Traces bake in tensor addresses. To reuse a trace, you must ensure input/output tensors are allocated at the same addresses:
```python
# Allocate persistent buffers
x_input = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, mem_config)
addr = x_input.buffer_address()

# After trace capture, verify address hasn't changed
assert x_input.buffer_address() == addr, "Address shifted -- trace is invalid"
```

**Critical constraint: No event synchronization inside traces**
Issue #30762 confirms: `record_event` and `wait_for_event` cannot be used during trace capture. This means you cannot synchronize between command queues inside a trace. Multi-CQ + trace requires events OUTSIDE the trace.

**MoE challenge:**
The MoE routing decision (which experts to activate) is dynamic and requires CPU readback of the gating logits. This fundamentally breaks full-model tracing. Options:
1. Trace everything EXCEPT the routing decision. Execute routing eagerly between traced blocks.
2. Pre-compute all expert outputs (ignore sparsity), trace the full computation, select outputs after trace execution. Wastes compute but eliminates dispatch overhead.
3. Use a fixed routing pattern during trace (always activate same experts). Only valid for benchmarking, not production.

---

## 6. Zero-Overhead Sync / Lightweight Readback

### The problem
Reading a single scalar (e.g., top-k expert indices from gating) currently requires:
1. `ttnn.from_device(tensor)` which calls `EnqueueReadBuffer`
2. This drains/synchronizes the command queue
3. Then DMA transfers the data
4. Total: ~3-5ms overhead (measured in our benchmarks)

### Potential approaches

**a) Non-blocking read on a second command queue:**
Use CQ1 for the readback while CQ0 continues dispatching ops:
```python
# CQ0: dispatch gating matmul
gate_logits = ttnn.linear(x, gate_weight, cq_id=ttnn.QueueId(0))

# CQ0: record event after gate logits are computed
event = ttnn.record_event(device, cq_id=ttnn.QueueId(0))

# CQ1: wait for gate logits, then read them back
ttnn.wait_for_event(ttnn.QueueId(1), event)
# Read gate_logits via CQ1 (non-blocking on CQ0)
gate_values = ttnn.from_device(gate_logits, cq_id=ttnn.QueueId(1))

# CQ0: meanwhile, continue dispatching expert computations speculatively
```
This requires `num_command_queues=2` in `open_device()`.

**b) Direct L1/DRAM polling (low-level, not exposed in ttnn):**
At the Metalium level, the host can read device memory via PCIe BAR. In theory, you could poll an L1 address directly without going through the command queue. However:
- No public ttnn API exposes this
- Would require raw `tt_umd` or `tt_metal` C++ calls
- Risk of reading stale data (no cache coherence guarantee between PCIe reads and NoC writes)
- Blackhole has a small write-through L1 data cache (4x16B cachelines) that could cause stale reads

**c) Pre-compute all experts, avoid readback entirely:**
For batch=1 decode with 8 experts selecting top-2, computing all 8 experts wastes 4x compute but eliminates the readback entirely. If dispatch overhead (eliminated by tracing all experts) exceeds the extra compute cost, this wins. At 840 ops / 30us = 25ms dispatch overhead vs 8 expert matmuls at ~0.5ms each = 4ms extra compute, this could be a massive net win when combined with tracing.

### Verdict
Option (a) multi-CQ readback is the cleanest. Option (c) compute-all-experts is the most pragmatic for batch=1 MoE.

---

## 7. Multi-CQ: Overlapping Dispatch with Execution

### Hardware support
- Blackhole supports up to 2 command queues (set via `num_command_queues` parameter in `open_device()`)
- Each CQ gets a dedicated RISC-V dispatch core on the device
- CQs are independent: both can dispatch programs and do I/O transfers
- Synchronization via events: `record_event(device, cq_id)` and `wait_for_event(cq_id, event)`

### Pattern: Overlap I/O with compute
```python
device = ttnn.open_device(device_id=0, num_command_queues=2)

# CQ0: runs model ops
# CQ1: handles input write + output read

# Write next input on CQ1
ttnn.to_device(next_input, device, cq_id=ttnn.QueueId(1))
write_event = ttnn.record_event(device, cq_id=ttnn.QueueId(1))

# CQ0: wait for input, run model
ttnn.wait_for_event(ttnn.QueueId(0), write_event)
output = model_forward(input, cq_id=ttnn.QueueId(0))
compute_event = ttnn.record_event(device, cq_id=ttnn.QueueId(0))

# CQ1: wait for output, read it back
ttnn.wait_for_event(ttnn.QueueId(1), compute_event)
result = ttnn.from_device(output, cq_id=ttnn.QueueId(1))
```

### Pattern: Overlap readback with next dispatch
For MoE, the key win is overlapping the gate logit readback (CQ1) with speculative expert dispatch (CQ0):
```python
# CQ0: compute gate logits
gate = ttnn.linear(x, gate_weight)
gate_event = ttnn.record_event(device, cq_id=ttnn.QueueId(0))

# CQ1: read back gate logits (async, doesn't stall CQ0)
ttnn.wait_for_event(ttnn.QueueId(1), gate_event)
gate_host = ttnn.from_device(gate, cq_id=ttnn.QueueId(1))

# CQ0: speculatively compute ALL expert outputs while gate reads back
for expert in all_experts:
    expert_out[expert] = ttnn.linear(x, expert_weight[expert])

# Host: now gate_host is available, select which expert outputs to use
selected = select_experts(gate_host)
output = combine_expert_outputs(expert_out, selected)
```

### Constraint: Multi-CQ + Trace
Event synchronization is NOT supported during trace capture (issue #30762). So you cannot use multi-CQ inside a trace. You CAN use multi-CQ between trace executions:
```python
# Trace the static parts
ttnn.execute_trace(device, trace_id, cq_id=ttnn.QueueId(0))

# CQ1: overlap I/O with trace execution
ttnn.to_device(next_input, device, cq_id=ttnn.QueueId(1))
```

---

## 8. Recommended Strategy for MoE Dispatch Optimization

### Priority order (most impact first):

**1. Compute all experts + full trace (eliminates dispatch entirely)**
- For batch=1, compute all 8 expert MLPs regardless of routing
- This makes the entire decode step a static graph -- fully traceable
- Extra compute: 6 matmuls * 6 unused experts * 24 layers = 864 extra matmuls
- But at batch=1 these are tiny (1x14336 * 14336x3584) and bandwidth-bound anyway
- Estimated extra device time: ~2-4ms
- Dispatch savings: ~25ms (840 ops * 30us eliminated)
- **Net improvement: ~20ms/tok, from 50ms to ~30ms**
- Use `ttnn.swiglu()` inside MLP to further reduce op count

**2. Op fusion to reduce op count (if trace isn't viable)**
- `ttnn.linear(..., activation="silu")` -- fuse activation into matmul
- `ttnn.swiglu()` -- fuse gate projection
- `ttnn.matmul_batched_weights()` -- batch expert matmuls into single dispatch
- Target: reduce 840 ops to ~400 ops, saving ~13ms

**3. Multi-CQ for gate readback overlap (if sparse routing is needed)**
- Use `num_command_queues=2`
- CQ0 dispatches compute, CQ1 reads gate logits
- Eliminates the 3-5ms readback stall per layer

**4. Partial tracing of static blocks**
- Trace attention blocks (fully static at decode time)
- Trace expert MLPs individually
- Execute routing decisions eagerly between traces
- Complex to implement but could save ~15ms

---

## Sources
- [TT-Metal Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)
- [TT-Metal Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metal Blackhole Bring-Up Guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)
- [Event Sync in Trace Capture (Issue #30762)](https://github.com/tenstorrent/tt-metal/issues/30762)
- [ttnn.matmul documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.matmul.html)
- [ttnn.swiglu documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.swiglu.html)
- [EnqueueReadBuffer API](https://docs.tenstorrent.com/tt-metal/v0.58.0/tt-metalium/tt_metal/apis/host_apis/command_queue/EnqueueReadBuffer.html)
- [DeepWiki: Performance Optimization Techniques](https://deepwiki.com/tenstorrent/tt-metal/7.5-performance-optimization-techniques)
- [Corsix: Tenstorrent Wormhole Series](https://www.corsix.org/content/tt-wh-part1)
