# Wiki 61: MoE Dispatch Optimization — From 12.8 to 22.7 tok/s

## Q: What was the MoE performance bottleneck after first light?

**A:** Host-device data transfer, not compute. Experiment 90's eager decode ran at 12.8 tok/s (78ms/tok), but only 10% of that time was actual compute. The remaining 90% was bouncing data between device and CPU — 24 round-trips per token for router readback, expert output readback, and residual transfers.

## Q: How did on-device accumulation help? (Exp 91: 12.8 -> 20.2 tok/s)

**A:** Five changes eliminated most host-device transfers:

1. **Expert output accumulation on device** — `ttnn.multiply(d, prob)` + `ttnn.add(moe_acc, weighted)` instead of `from_dev` per expert
2. **Expert weighting on device** — `ttnn.multiply(tensor, prob_float)` instead of numpy multiply
3. **Shared expert gate on device** — `ttnn.matmul(h2, seg_w)` + CPU sigmoid scalar
4. **Residual connections on device** — no CPU round-trip between layers
5. **Program cache enabled** — `device.enable_program_cache()` for faster dispatch

Result: only 1 sync per layer remains (reading 60 router logits = 240 bytes). This cut latency from 78ms to 50ms — a **1.58x speedup**.

Remaining bottleneck: ~840 op dispatches per token at ~30us each = ~25ms of pure dispatch overhead.

## Q: What did device-side routing add? (Exp 92: ~20.5 tok/s)

**A:** Moved three more operations from CPU to device:

| Operation | Before (CPU) | After (device) |
|-----------|-------------|----------------|
| Softmax on router logits | `np.exp / np.sum` | `ttnn.softmax(rl, dim=-1)` |
| Top-4 selection | `np.argsort[-4:]` | `ttnn.topk(probs, 4)` |
| Shared expert gate | CPU sigmoid | `ttnn.sigmoid(seg_logit)` |

Sync payload dropped from 240 bytes (60 router logits) to 32 bytes (4 indices + 4 probs). The expert mask tensor (used in exp 91 to avoid dispatching unused experts) was eliminated — topk returns only the 4 needed indices directly.

Marginal speedup (~0.3 tok/s) because the readback was already small, but this is architecturally important: it moves toward a fully traceable forward pass.

## Q: What are fused ops and how much did they save? (Exp 93: 20.5 -> 22.7 tok/s)

**A:** Two TT-NN op fusion features eliminated dispatch overhead:

### 1. Fused SiLU activation
```python
# Before: 2 dispatches
g = ttnn.matmul(h2, gate_w)
g = ttnn.silu(g)

# After: 1 dispatch
g = ttnn.linear(h2, gate_w, activation="silu")
```
Savings: 1 dispatch x (4 routed experts + 1 shared expert) x 24 layers = **120 dispatches saved**

### 2. Fused bias addition
```python
# Before: 2 dispatches
q = ttnn.matmul(h, q_w)
q = ttnn.add(q, q_b)

# After: 1 dispatch
q = ttnn.linear(h, q_w, bias=q_b)
```
Savings: 3 projections (Q/K/V) x 24 layers = **72 dispatches saved**

Total: ~192 fewer dispatches x 30us = ~5.8ms saved. Measured improvement: 47ms -> 44ms = **22.7 tok/s** (steady-state after exp 95 partial trace).

## Q: What is partial tracing and why does it matter for MoE? (Exp 95)

**A:** MoE models cannot be fully traced because expert selection is data-dependent — different tokens route to different experts. But attention is a static computation graph with no branching.

Partial tracing captures 24 attention traces (one per layer) while leaving MoE routing and expert dispatch eager:

```python
# Capture once per layer
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# ... attention ops (rms_norm, Q/K/V proj, RoPE, SDPA, O proj, residual add) ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# During decode: replay trace, then run MoE eagerly
ttnn.copy(x, attn_x_ins[i])            # update input buffer
ttnn.execute_trace(device, trace_id)    # replay attention
x2 = attn_x_outs[i]                    # read output
# ... eager MoE routing + expert dispatch ...
```

Key requirements:
- Persistent input/output buffers per layer (`attn_x_ins[i]`, `attn_x_outs[i]`)
- Shared cos/sin/pos buffers updated via `ttnn.copy` before each decode step
- Program cache warmup before capture (one non-traced iteration per layer)

Profiling (exp 94) showed attention = 11.4ms of 48.3ms total. Tracing it eliminates ~360 dispatches, saving ~8ms. Result: **44ms/tok = 22.7 tok/s**.

## Q: Did multi-CQ pipelining help? (Exp 96)

**A:** No. The idea was to overlap routing readback with compute using two command queues:

```
CQ0: [attention] [router+softmax+topk] ──────────── [expert dispatch]
CQ1:                                    [read 32B]
                                         ↑ overlap with CQ0 compute
```

Event synchronization works on Blackhole:
```python
device = ttnn.open_device(device_id=0, num_command_queues=2)
ev = ttnn.record_event(device, cq_id=CQ0)
ttnn.wait_for_event(CQ1, ev)  # CQ1 waits for CQ0's event
```

**But `ttnn.synchronize_device()` drains BOTH queues.** Since we need to synchronize to read the routing indices back to the host, both queues stall anyway. There is no per-queue sync API. Net result: same performance as single-CQ, with added complexity.

**Lesson:** Multi-CQ only helps when you can avoid full device sync — e.g., using events for device-to-device dependencies without host readback.

## Q: What is the full MoE optimization timeline?

**A:**

```
exp 90:   78ms/tok   12.8 tok/s   Eager decode (CPU routing, per-expert readback)
exp 91:   50ms/tok   20.2 tok/s   On-device accumulation (1.58x)
exp 92:   49ms/tok   20.5 tok/s   Device-side routing (+marginal)
exp 93:   47ms/tok   21.1 tok/s   Fused ops (-192 dispatches)
exp 95:   44ms/tok   22.7 tok/s   Partial trace (attention traced, MoE eager)
exp 96:   44ms/tok   22.7 tok/s   Multi-CQ (no gain — sync drains both queues)
```

Total speedup: **1.77x** (78ms -> 44ms). The dominant remaining cost is MoE expert dispatch at ~15ms/tok (32% of total).

---

*Experiments 90-96. Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active) on Blackhole P150. April 2026.*
