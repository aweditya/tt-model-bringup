#!/usr/bin/env python3
"""
Experiment 81 — MoE regression diagnostic microbench.

The whole Qwen1.5-MoE decode regressed 22.7 -> 15.7 tok/s on ttnn 0.69 vs
0.68. That's +20 ms/tok over the prior ~44 ms steady-state. Roughly 600
ops per token (~25 ops × 24 layers), so each op would only need to add
+33 µs on average to produce 20 ms of total regression.

This bench isolates the dispatch cost of each MoE-decode-loop surface in
ttnn 0.69 so we can see where the time goes. We DON'T have ttnn 0.68
installed for A/B comparison; this is "what's expensive now" data and
we judge it against expectations (45-95 µs/op was our prior single-op
benchmark for simple ops).

Surfaces (all at decode shapes, single token):
  1. matmul (hidden -> intermediate, ~2048 -> 1408 or 1408 -> 2048)
  2. softmax + topk (router output, 64 logits → top-4)
  3. sync_device  (no-op? or genuine wait?)
  4. host readback (to_torch(top4_vals))
  5. silu + mul (gate * up)
  6. add (residual)
  7. one full "MoE-experts" group: 4× (g + u + silu + mul + d + multiply + add)
  8. shared expert: g + u + silu + mul + d + sigmoid + multiply + add

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/81_moe_regression_micro.py
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

# Qwen1.5-MoE-A2.7B-Chat arch
hidden = 2048
intermediate = 1408
n_experts = 60
top_k = 4

ITERS = 200
WARMUP = 10

# All-or-nothing per feedback_compute_kernel_config: HiFi4 on Blackhole
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def to_dev(arr, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def bench(label, fn, sync=True):
    """Time fn() over ITERS iters, return median µs/call."""
    for _ in range(WARMUP):
        fn()
    if sync:
        ttnn.synchronize_device(device)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        if sync:
            ttnn.synchronize_device(device)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)  # µs
    med = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[8]
    print(f"  {label:38s}  median={med:8.1f} µs   p90={p90:8.1f} µs")
    return med


def main():
    global device
    device = ttnn.open_device(device_id=0)
    print(f"Device: Blackhole P150, ttnn {ttnn.__version__ if hasattr(ttnn, '__version__') else 'unknown'}")
    print(f"Iters: warmup={WARMUP}, sample={ITERS}\n")

    # Build representative tensors
    np.random.seed(42)
    h_np = np.random.randn(1, 1, 1, hidden).astype(np.float32) * 0.1
    w_h2i_np = np.random.randn(hidden, intermediate).astype(np.float32) * 0.02
    w_i2h_np = np.random.randn(intermediate, hidden).astype(np.float32) * 0.02
    w_router_np = np.random.randn(hidden, n_experts).astype(np.float32) * 0.02

    # Reshape for ttnn — expect 2D for linear
    h = to_dev(h_np.reshape(1, hidden))             # [1, hidden]
    w_h2i = to_dev(w_h2i_np, dtype=ttnn.bfloat8_b)  # bfp8 experts (matches port)
    w_i2h = to_dev(w_i2h_np, dtype=ttnn.bfloat8_b)
    w_router = to_dev(w_router_np)                  # bf16 router (small)

    # Pre-allocated output buffers (avoid measuring alloc)
    g_buf = ttnn.matmul(h, w_h2i, compute_kernel_config=hifi4)  # warmup alloc
    rl_buf = ttnn.matmul(h, w_router, compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)

    print("=== Per-op dispatch cost (eager, includes sync) ===\n")

    # ── Big matmuls (expert size) ──
    bench("matmul h@w  [1,2048] @ [2048,1408]", lambda:
          ttnn.matmul(h, w_h2i, compute_kernel_config=hifi4))
    bench("matmul h@w  [1,1408] @ [1408,2048]", lambda: (
        # mid-size product
        lambda mid: ttnn.matmul(mid, w_i2h, compute_kernel_config=hifi4)
    )(ttnn.matmul(h, w_h2i, compute_kernel_config=hifi4)))

    # ── Router matmul ──
    bench("matmul router  [1,2048] @ [2048,60]", lambda:
          ttnn.matmul(h, w_router, compute_kernel_config=hifi4))

    # ── Activation + binary ──
    bench("silu(g)", lambda: ttnn.silu(g_buf))
    bench("mul(silu_g, u)", lambda: ttnn.mul(ttnn.silu(g_buf), g_buf))
    bench("add(h, h)", lambda: ttnn.add(h, h))
    bench("multiply(g, 0.5)", lambda: ttnn.multiply(g_buf, 0.5))

    # ── Routing ──
    bench("softmax(router_logits)", lambda: ttnn.softmax(rl_buf, dim=-1))
    bench("topk(probs, k=4)", lambda: ttnn.topk(rl_buf, top_k))

    # ── Synchronization itself ──
    bench("synchronize_device (noop after sync)", lambda: None, sync=True)

    # ── Host readback (the suspected hotspot) ──
    top_vals, top_idxs = ttnn.topk(rl_buf, top_k)
    ttnn.synchronize_device(device)
    bench("to_torch(top4_vals).float()", lambda: ttnn.to_torch(top_vals).float().numpy())
    bench("to_torch(top4_idxs).int()", lambda: ttnn.to_torch(top_idxs).int().numpy())

    # ── Composite: one expert (3 matmuls + silu + mul + multiply) ──
    def one_expert():
        g = ttnn.linear(h, w_h2i, activation="silu", compute_kernel_config=hifi4)
        u = ttnn.matmul(h, w_h2i, compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(g, u), w_i2h, compute_kernel_config=hifi4)
        return ttnn.multiply(d, 0.25)
    bench("one expert pass (3 matmul + silu+mul+scale)", one_expert)

    # ── Composite: 4-expert MoE group ──
    def moe_group():
        acc = None
        for _ in range(4):
            d = one_expert()
            acc = d if acc is None else ttnn.add(acc, d)
        return acc
    bench("4-expert MoE group (4× one_expert + 3 adds)", moe_group)

    # ── Composite: shared expert with sigmoid gate ──
    w_sg = w_router  # repurpose [hidden, 60] as [hidden, 60] — small linear
    def shared_expert():
        g = ttnn.linear(h, w_h2i, activation="silu", compute_kernel_config=hifi4)
        u = ttnn.matmul(h, w_h2i, compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(g, u), w_i2h, compute_kernel_config=hifi4)
        seg_logit = ttnn.matmul(h, w_router, compute_kernel_config=hifi4)
        seg_val = ttnn.sigmoid(seg_logit)
        return ttnn.mul(d, seg_val)
    bench("shared expert (3 matmul + sigmoid gate)", shared_expert)

    print("\n=== Putting numbers in context ===\n")
    print("Per-layer MoE-block ops at decode (16 ops): ~6 small + 12 big matmuls + sync+readback")
    print("Per-token (24 layers): expect 24 × per-layer time")
    print("Prior tok/s was 22.7 (44 ms/tok). Now 15.7 (64 ms/tok). Gap: +20 ms.")
    print("If matmul went from ~70 µs to ~150 µs: +80 µs × 288 matmuls = +23 ms. Plausible!")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
