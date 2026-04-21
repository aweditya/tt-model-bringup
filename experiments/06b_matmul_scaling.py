"""
Experiment 06b: Matmul Scaling on Blackhole
============================================
Hypothesis: Larger matmuls will achieve higher utilization because they
can keep all 110 Tensix cores busy. We sweep matrix sizes to find the
point where the hardware saturates.
"""

import ttnn
import torch
import time

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, grid {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print(f"Peak bf16: ~372 TFLOPS")
print()

sizes = [
    (32, 32, 32),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

print(f"{'M×K×N':>20} {'Time (ms)':>12} {'TFLOPS':>10} {'Util%':>8}")
print("-" * 55)

for M, K, N in sizes:
    try:
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT
        )
        w = ttnn.from_torch(
            torch.randn(1, 1, K, N), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT
        )

        # Warmup
        for _ in range(3):
            r = ttnn.matmul(x, w)
            r.deallocate()

        # Benchmark
        REPS = 20 if M >= 4096 else 50
        start = time.perf_counter()
        for _ in range(REPS):
            r = ttnn.matmul(x, w)
            r.deallocate()
        elapsed = (time.perf_counter() - start) / REPS

        flops = 2 * M * K * N
        tflops = flops / elapsed / 1e12
        util = tflops / 372 * 100

        print(f"{M}×{K}×{N:>5} {elapsed*1000:>12.3f} {tflops:>10.2f} {util:>7.1f}%")

        x.deallocate()
        w.deallocate()
    except Exception as e:
        print(f"{M}×{K}×{N:>5}  ERROR: {str(e)[:60]}")

ttnn.close_device(device)
