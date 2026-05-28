#!/usr/bin/env python3
"""
Experiment 81b — probe ttnn fused SwiGLU at 2D decode shapes.

A0 found `mul(silu_g, u)` costs 137 µs vs 85 µs for a scalar multiply.
The user's instinct: try a fused ttnn op if one exists. Per prior
memory (feedback_compute_kernel_config / exp 97), ttnn.swiglu crashed
on 4D shapes on Blackhole — but our decode runs at 2D, so we should
re-test.

Three candidates we probe:
  (a) ttnn.experimental.swiglu (if present)
  (b) ttnn.swiglu (top-level, if present)
  (c) ttnn.mul(a, b) with an inline activation= kwarg (some elemwise
      ops accept activation= in 0.69+)

Whichever works, time it vs the unfused silu+mul pair.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/81b_swiglu_fusion_probe.py
"""
import os, sys, time, statistics, traceback
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

# Decode shapes from Qwen1.5-MoE
hidden = 2048
intermediate = 1408

ITERS = 200
WARMUP = 10


def to_dev(arr, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def bench(label, fn):
    """200-iter median µs/call, syncs after each."""
    try:
        for _ in range(WARMUP):
            fn()
        ttnn.synchronize_device(device)
    except Exception as e:
        return f"FAIL ({type(e).__name__}: {str(e)[:60]})"
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        try:
            fn()
            ttnn.synchronize_device(device)
        except Exception as e:
            return f"FAIL mid-bench: {e}"
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    med = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[8]
    return f"median={med:6.1f} µs   p90={p90:6.1f} µs"


def main():
    global device
    device = ttnn.open_device(device_id=0)
    print("Blackhole P150, probing fused SwiGLU candidates at 2D decode shapes\n")

    g = to_dev(np.random.randn(1, intermediate).astype(np.float32) * 0.1)
    u = to_dev(np.random.randn(1, intermediate).astype(np.float32) * 0.1)

    # Baseline: silu + binary mul
    print("Baseline (unfused):")
    print(f"  silu(g) then mul(_,u)               {bench('', lambda: ttnn.mul(ttnn.silu(g), u))}")

    # Candidate A: ttnn.experimental.swiglu (often the new home)
    print()
    print("Probing fused candidates:")
    cand_a = bench('', lambda: ttnn.experimental.swiglu(g, u)) if hasattr(ttnn.experimental, 'swiglu') else "NOT PRESENT"
    print(f"  ttnn.experimental.swiglu(g, u)      {cand_a}")

    cand_b = bench('', lambda: ttnn.swiglu(g, u)) if hasattr(ttnn, 'swiglu') else "NOT PRESENT"
    print(f"  ttnn.swiglu(g, u)                   {cand_b}")

    # Candidate C: mul with activation kwarg
    try:
        cand_c = bench('', lambda: ttnn.mul(g, u, activation="silu"))
    except TypeError as e:
        cand_c = f"NO activation kwarg: {str(e)[:60]}"
    print(f"  ttnn.mul(g, u, activation='silu')   {cand_c}")

    # Candidate D: linear with silu+mul epilogue (matmul+epilogue is a fused kernel,
    # but the question is whether elementwise+elementwise can fuse)
    # Skipping — would need pretend-weights to do a matmul before the elemwise.

    # Candidate E: any silu_mul or mul_silu name we might have missed
    for name in ['silu_mul', 'mul_silu', 'silu_and_mul', 'silu_mul_fused']:
        if hasattr(ttnn, name):
            fn = getattr(ttnn, name)
            try:
                cand = bench('', lambda fn=fn: fn(g, u))
            except Exception as e:
                cand = f"FAIL: {str(e)[:60]}"
            print(f"  ttnn.{name}(g, u)" + " " * max(0, 18 - len(name)) + f"   {cand}")

    # Sanity: list any ttnn op name containing 'silu' or 'swiglu'
    matching = [a for a in dir(ttnn) if 'silu' in a.lower() or 'swiglu' in a.lower()]
    matching_exp = [a for a in dir(ttnn.experimental) if 'silu' in a.lower() or 'swiglu' in a.lower()]
    print(f"\nttnn.* containing silu/swiglu: {matching}")
    print(f"ttnn.experimental.* containing silu/swiglu: {matching_exp}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
