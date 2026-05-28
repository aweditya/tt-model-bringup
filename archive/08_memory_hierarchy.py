"""
Experiment 08: Blackhole Memory Hierarchy Characterization
===========================================================
Hypothesis: L1 SRAM (1.5MB per core) is dramatically faster than GDDR6
DRAM for small tensors. The crossover point where DRAM becomes necessary
(because data doesn't fit in L1) is a key architectural parameter.

We also measure host↔device transfer overhead, which is critical for
understanding the end-to-end performance picture.
"""

import ttnn
import torch
import time

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print(f"L1 per core: 1.5 MB → total L1: {grid.x*grid.y*1.5:.0f} MB")
print(f"DRAM: 32 GB GDDR6")
print()

# ============================================================
# TEST 1: Host → Device transfer speed
# ============================================================
print("=" * 60)
print("TEST 1: Host ↔ Device transfer speed")
print("=" * 60)

transfer_sizes = [
    (32, 32),
    (128, 128),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]

print(f"\n{'Size':>12} {'H→D (ms)':>10} {'D→H (ms)':>10} {'H→D BW':>12} {'D→H BW':>12}")
print("-" * 60)

for M, N in transfer_sizes:
    data = torch.randn(1, 1, M, N)
    bytes_size = M * N * 2  # bf16 = 2 bytes

    # Host → Device
    REPS = 20
    times_h2d = []
    for _ in range(REPS):
        start = time.perf_counter()
        t = ttnn.from_torch(data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        times_h2d.append(time.perf_counter() - start)
        t.deallocate()

    # Device → Host
    t = ttnn.from_torch(data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    times_d2h = []
    for _ in range(REPS):
        start = time.perf_counter()
        _ = ttnn.to_torch(t)
        times_d2h.append(time.perf_counter() - start)
    t.deallocate()

    avg_h2d = sum(times_h2d) / len(times_h2d)
    avg_d2h = sum(times_d2h) / len(times_d2h)
    bw_h2d = bytes_size / avg_h2d / 1e9
    bw_d2h = bytes_size / avg_d2h / 1e9

    print(f"{M}x{N:>5}  {avg_h2d*1000:>8.3f}  {avg_d2h*1000:>8.3f}  "
          f"{bw_h2d:>8.2f} GB/s  {bw_d2h:>8.2f} GB/s")

# ============================================================
# TEST 2: Elementwise operation — L1 vs DRAM memory config
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Elementwise add — L1 vs DRAM")
print("=" * 60)

elem_sizes = [
    (32, 32),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
]

print(f"\n{'Size':>12} {'DRAM (ms)':>10} {'L1 (ms)':>10} {'Speedup':>10}")
print("-" * 50)

for M, N in elem_sizes:
    a_data = torch.randn(1, 1, M, N)
    b_data = torch.randn(1, 1, M, N)

    # DRAM config
    a_dram = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_dram = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Warmup
    for _ in range(3):
        r = ttnn.add(a_dram, b_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r.deallocate()

    REPS = 50
    times_dram = []
    for _ in range(REPS):
        start = time.perf_counter()
        r = ttnn.add(a_dram, b_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        times_dram.append(time.perf_counter() - start)
        r.deallocate()

    a_dram.deallocate()
    b_dram.deallocate()

    # L1 config
    try:
        a_l1 = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        b_l1 = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        for _ in range(3):
            r = ttnn.add(a_l1, b_l1, memory_config=ttnn.L1_MEMORY_CONFIG)
            r.deallocate()

        times_l1 = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.add(a_l1, b_l1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            times_l1.append(time.perf_counter() - start)
            r.deallocate()

        a_l1.deallocate()
        b_l1.deallocate()

        avg_dram = sum(times_dram) / len(times_dram)
        avg_l1 = sum(times_l1) / len(times_l1)
        speedup = avg_dram / avg_l1

        print(f"{M}x{N:>5}  {avg_dram*1000:>8.3f}  {avg_l1*1000:>8.3f}  {speedup:>8.2f}x")
    except Exception as e:
        avg_dram = sum(times_dram) / len(times_dram)
        print(f"{M}x{N:>5}  {avg_dram*1000:>8.3f}  {'OOM':>8}  {'N/A':>8}")

# ============================================================
# TEST 3: Matmul — L1 vs DRAM for output
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Matmul output — L1 vs DRAM")
print("=" * 60)

mm_sizes = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

print(f"\n{'M×K×N':>16} {'DRAM (ms)':>10} {'L1 (ms)':>10} {'Speedup':>10}")
print("-" * 50)

for M, K, N in mm_sizes:
    x_data = torch.randn(1, 1, M, K)
    w_data = torch.randn(1, 1, K, N)

    x_tt = ttnn.from_torch(x_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(w_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # DRAM output
    for _ in range(3):
        r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r.deallocate()

    REPS = 20
    times_dram = []
    for _ in range(REPS):
        start = time.perf_counter()
        r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        times_dram.append(time.perf_counter() - start)
        r.deallocate()

    # L1 output
    try:
        for _ in range(3):
            r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            r.deallocate()

        times_l1 = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            times_l1.append(time.perf_counter() - start)
            r.deallocate()

        avg_dram = sum(times_dram) / len(times_dram)
        avg_l1 = sum(times_l1) / len(times_l1)
        speedup = avg_dram / avg_l1
        print(f"{M}×{K}×{N:>5}  {avg_dram*1000:>8.3f}  {avg_l1*1000:>8.3f}  {speedup:>8.2f}x")
    except Exception as e:
        avg_dram = sum(times_dram) / len(times_dram)
        print(f"{M}×{K}×{N:>5}  {avg_dram*1000:>8.3f}  {'OOM/ERR':>8}  N/A ({str(e)[:40]})")

    x_tt.deallocate()
    w_tt.deallocate()

# ============================================================
# TEST 4: Multi-op chain — L1 intermediate vs DRAM intermediate
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: matmul→relu chain — intermediate in L1 vs DRAM")
print("=" * 60)
print("  (This simulates what XLA fusion would optimize)")

chain_sizes = [(512, 512, 512), (1024, 1024, 1024)]

for M, K, N in chain_sizes:
    x_data = torch.randn(1, 1, M, K)
    w_data = torch.randn(1, 1, K, N)

    x_tt = ttnn.from_torch(x_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(w_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Path A: intermediate goes to DRAM (default, like unfused)
    for _ in range(3):
        h = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.relu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate()
        h.deallocate()

    REPS = 30
    times_dram_chain = []
    for _ in range(REPS):
        start = time.perf_counter()
        h = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.relu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        times_dram_chain.append(elapsed)
        out.deallocate()
        h.deallocate()

    # Path B: intermediate stays in L1 (like fused)
    try:
        for _ in range(3):
            h = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            out = ttnn.relu(h, memory_config=ttnn.L1_MEMORY_CONFIG)
            out.deallocate()
            h.deallocate()

        times_l1_chain = []
        for _ in range(REPS):
            start = time.perf_counter()
            h = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            out = ttnn.relu(h, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start
            times_l1_chain.append(elapsed)
            out.deallocate()
            h.deallocate()

        avg_dram = sum(times_dram_chain) / len(times_dram_chain)
        avg_l1 = sum(times_l1_chain) / len(times_l1_chain)

        print(f"\n  {M}×{K}×{N}:")
        print(f"    DRAM intermediate: {avg_dram*1000:.3f} ms")
        print(f"    L1 intermediate:   {avg_l1*1000:.3f} ms")
        print(f"    Speedup:           {avg_dram/avg_l1:.2f}x")
    except Exception as e:
        avg_dram = sum(times_dram_chain) / len(times_dram_chain)
        print(f"\n  {M}×{K}×{N}:")
        print(f"    DRAM intermediate: {avg_dram*1000:.3f} ms")
        print(f"    L1 intermediate:   ERROR ({str(e)[:50]})")

    x_tt.deallocate()
    w_tt.deallocate()

ttnn.close_device(device)
print("\nDone!")
