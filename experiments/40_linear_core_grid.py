#!/usr/bin/env python3
"""
Experiment 40: Linear core grid optimization on Blackhole P150 (tt-metal #25503)

The issue reports that ttnn.linear defaults to 22-24 cores on P150 when 88 are
available. This experiment:
  1. Benchmarks default matmul vs explicit core_grid on GPT-2/Qwen shapes
  2. Measures actual core utilization
  3. Quantifies the speedup from using full grid

Reference: https://github.com/tenstorrent/tt-metal/issues/25503
"""

import numpy as np
import torch
import time
import ttnn

print("=" * 60)
print("Experiment 40: Linear Core Grid Optimization")
print("Benchmarking default vs full core grid on Blackhole P150")
print("=" * 60)

device = ttnn.open_device(device_id=0)

# Check device grid
compute_grid = device.compute_with_storage_grid_size()
print(f"\nDevice compute grid: {compute_grid.x} x {compute_grid.y} = {compute_grid.x * compute_grid.y} cores")

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor):
    return ttnn.to_torch(tensor).float().numpy()

# ── Benchmark shapes (from our actual models) ──────────────
shapes = [
    # (M, K, N) — matmul A[M,K] @ B[K,N]
    ("GPT-2 QKV", 32, 768, 768),
    ("GPT-2 QKV long", 128, 768, 768),
    ("GPT-2 MLP up", 32, 768, 3072),
    ("GPT-2 MLP down", 32, 3072, 768),
    ("Qwen QKV", 32, 896, 896),
    ("Qwen MLP gate", 32, 896, 4864),
    ("Qwen MLP down", 32, 4864, 896),
    ("Qwen QKV long", 128, 896, 896),
]

n_warmup = 3
n_bench = 20

print(f"\n{'Name':<20} {'Shape':<24} {'Default (ms)':<14} {'Full Grid (ms)':<16} {'Speedup'}")
print("-" * 90)

for name, M, K, N in shapes:
    A = np.random.randn(1, M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    a_tt = to_dev(A)
    b_tt = to_dev(B)

    # ── Default matmul ──
    for _ in range(n_warmup):
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.deallocate(c)

    times_default = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        times_default.append(dt)
        ttnn.deallocate(c)

    avg_default = np.mean(times_default) * 1000

    # ── Full grid matmul ──
    # Try specifying a larger core grid
    try:
        grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        )

        for _ in range(n_warmup):
            c = ttnn.matmul(a_tt, b_tt, core_grid=grid)
            ttnn.deallocate(c)

        times_grid = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            c = ttnn.matmul(a_tt, b_tt, core_grid=grid)
            ttnn.synchronize_device(device)
            dt = time.perf_counter() - t0
            times_grid.append(dt)
            ttnn.deallocate(c)

        avg_grid = np.mean(times_grid) * 1000
        speedup = avg_default / avg_grid
        print(f"{name:<20} ({M},{K},{N}){'':<{24-len(f'({M},{K},{N})')}} {avg_default:<14.2f} {avg_grid:<16.2f} {speedup:.2f}x")

    except Exception as e:
        # core_grid param might not work — try alternate approach
        print(f"{name:<20} ({M},{K},{N}){'':<{24-len(f'({M},{K},{N})')}} {avg_default:<14.2f} {'(error)':<16} --")
        print(f"    Error: {str(e)[:80]}")

    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)

# ── Also test ttnn.linear if available ──────────────────────
print("\n── Testing ttnn.linear (if available) ──")
try:
    # ttnn.linear wraps matmul + bias
    A = np.random.randn(1, 32, 768).astype(np.float32)
    W = np.random.randn(768, 768).astype(np.float32)
    bias = np.random.randn(1, 1, 768).astype(np.float32)

    a_tt = to_dev(A)
    w_tt = to_dev(W)
    b_tt = to_dev(bias)

    for _ in range(n_warmup):
        c = ttnn.linear(a_tt, w_tt, bias=b_tt)
        ttnn.deallocate(c)

    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        c = ttnn.linear(a_tt, w_tt, bias=b_tt)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        ttnn.deallocate(c)

    avg = np.mean(times) * 1000
    print(f"  ttnn.linear (32, 768, 768) default: {avg:.2f}ms")

    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)

except Exception as e:
    print(f"  ttnn.linear not available or failed: {str(e)[:80]}")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Device: Blackhole P150, {compute_grid.x}x{compute_grid.y} = {compute_grid.x * compute_grid.y} cores")
print(f"  Issue #25503 reports default linear uses only 22-24 cores")
print(f"  See results above for actual speedup from explicit core_grid")

ttnn.close_device(device)
print("\nDone!")
