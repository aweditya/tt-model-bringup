"""
Experiment 06c: Matmul Scaling (Properly Synchronized)
=======================================================
Hypothesis: Our previous benchmark was measuring dispatch time, not
execution time. TT-NN dispatches async — we need to synchronize.
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

        # Pre-allocate output to avoid alloc overhead in loop
        r = ttnn.matmul(x, w)
        # Synchronize: read a value back to host to force completion
        _ = ttnn.to_torch(r)
        r.deallocate()

        # Warmup with sync
        for _ in range(3):
            r = ttnn.matmul(x, w)
            _ = ttnn.to_torch(r)
            r.deallocate()

        # Benchmark with FULL synchronization each iteration
        REPS = 10 if M >= 4096 else 20
        times = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.matmul(x, w)
            # Force synchronization by reading result back
            _ = ttnn.to_torch(r)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            r.deallocate()

        # Also measure just dispatch+sync (no readback)
        # Use ttnn.synchronize_device as the sync mechanism
        times_sync = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.matmul(x, w)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start
            times_sync.append(elapsed)
            r.deallocate()

        avg_with_readback = sum(times) / len(times)
        avg_sync_only = sum(times_sync) / len(times_sync)

        flops = 2 * M * K * N
        tflops_readback = flops / avg_with_readback / 1e12
        tflops_sync = flops / avg_sync_only / 1e12
        util_sync = tflops_sync / 372 * 100

        print(f"{M}×{K}×{N:>5}  sync:{avg_sync_only*1000:>7.3f}  "
              f"read:{avg_with_readback*1000:>7.3f}  "
              f"{tflops_sync:>7.2f} TF  {util_sync:>5.1f}%")

        x.deallocate()
        w.deallocate()
    except Exception as e:
        print(f"{M}×{K}×{N:>5}  ERROR: {str(e)[:60]}")

ttnn.close_device(device)
