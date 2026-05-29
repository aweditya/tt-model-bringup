"""
Experiment 09: Sharded Memory & Matmul Program Configs
======================================================
Hypothesis: Experiment 08 showed interleaved L1 is SLOWER than DRAM for
matmul. The real speedup comes from sharded memory configs where each
core's data is local to its L1. TT-NN's matmul program configs control
this — we test whether proper sharding unlocks L1's potential.

We also explore what program configs exist and how they affect performance.
"""

import ttnn
import torch
import time

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print()

# ============================================================
# TEST 1: Explore available matmul program configs
# ============================================================
print("=" * 60)
print("TEST 1: Matmul with different core grids")
print("=" * 60)
print("  Default matmul uses the full grid. What if we vary it?")
print()

# We'll test matmul with explicit compute_with_storage_grid_size
# to see if TT-NN auto-selects different strategies
sizes = [(1024, 1024, 1024), (2048, 2048, 2048)]

for M, K, N in sizes:
    x_data = torch.randn(1, 1, M, K)
    w_data = torch.randn(1, 1, K, N)

    x_tt = ttnn.from_torch(x_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(w_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Default matmul
    for _ in range(5):
        r = ttnn.matmul(x_tt, w_tt)
        r.deallocate()

    REPS = 20
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        r = ttnn.matmul(x_tt, w_tt)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - start)
        r.deallocate()

    avg = sum(times) / len(times)
    flops = 2 * M * K * N
    tflops = flops / avg / 1e12

    print(f"  {M}x{K}x{N}: {avg*1000:.3f} ms = {tflops:.1f} TFLOPS ({tflops/372*100:.1f}%)")

    x_tt.deallocate()
    w_tt.deallocate()

# ============================================================
# TEST 2: Block-sharded matmul (if supported)
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Sharded memory for elementwise (L1 block shard)")
print("=" * 60)
print("  For elementwise ops, block-sharding should let each core")
print("  work on its local L1 data with zero NoC traffic.")
print()

shard_sizes = [(1024, 1024), (2048, 2048)]

for M, N in shard_sizes:
    a_data = torch.randn(1, 1, M, N)
    b_data = torch.randn(1, 1, M, N)
    bytes_per_elem = 2  # bf16

    # Interleaved DRAM (baseline)
    a_dram = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_dram = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for _ in range(5):
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

    # Try block-sharded L1
    try:
        # Calculate shard shape: divide tensor across cores
        num_cores_y = grid.y  # 10
        num_cores_x = grid.x  # 11
        total_cores = num_cores_x * num_cores_y  # 110

        # For block sharding: divide rows across cores_y, cols across cores_x
        # Must be tile-aligned (multiples of 32)
        shard_h = (M + num_cores_y - 1) // num_cores_y
        shard_h = ((shard_h + 31) // 32) * 32  # round up to tile
        shard_w = (N + num_cores_x - 1) // num_cores_x
        shard_w = ((shard_w + 31) // 32) * 32

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
            )}),
            (shard_h, shard_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        a_shard = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                                   layout=ttnn.TILE_LAYOUT, memory_config=sharded_mem_config)
        b_shard = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                                   layout=ttnn.TILE_LAYOUT, memory_config=sharded_mem_config)

        for _ in range(5):
            r = ttnn.add(a_shard, b_shard, memory_config=sharded_mem_config)
            r.deallocate()

        times_shard = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.add(a_shard, b_shard, memory_config=sharded_mem_config)
            ttnn.synchronize_device(device)
            times_shard.append(time.perf_counter() - start)
            r.deallocate()

        a_shard.deallocate()
        b_shard.deallocate()

        avg_dram = sum(times_dram) / len(times_dram)
        avg_shard = sum(times_shard) / len(times_shard)
        total_bytes = M * N * bytes_per_elem

        print(f"  {M}x{N}:")
        print(f"    DRAM interleaved:  {avg_dram*1000:.3f} ms  ({total_bytes/avg_dram/1e9:.1f} GB/s)")
        print(f"    L1 block-sharded:  {avg_shard*1000:.3f} ms  ({total_bytes/avg_shard/1e9:.1f} GB/s)")
        print(f"    Speedup:           {avg_dram/avg_shard:.2f}x")

    except Exception as e:
        avg_dram = sum(times_dram) / len(times_dram)
        print(f"  {M}x{N}:")
        print(f"    DRAM interleaved:  {avg_dram*1000:.3f} ms")
        print(f"    L1 block-sharded:  ERROR ({str(e)[:80]})")

# ============================================================
# TEST 3: Multi-op chain with sharded intermediates
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: add→relu→add chain — sharded vs interleaved")
print("=" * 60)
print("  Three consecutive elementwise ops. Sharded intermediates")
print("  should eliminate DRAM round-trips between ops.")
print()

for M, N in [(1024, 1024), (2048, 2048)]:
    a_data = torch.randn(1, 1, M, N)
    b_data = torch.randn(1, 1, M, N)
    c_data = torch.randn(1, 1, M, N)

    # Path A: DRAM interleaved (like unfused eager)
    a_dram = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_dram = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c_dram = ttnn.from_torch(c_data, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for _ in range(5):
        h1 = ttnn.add(a_dram, b_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h2 = ttnn.relu(h1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(h2, c_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(); h2.deallocate(); h1.deallocate()

    REPS = 50
    times_dram = []
    for _ in range(REPS):
        start = time.perf_counter()
        h1 = ttnn.add(a_dram, b_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h2 = ttnn.relu(h1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(h2, c_dram, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        times_dram.append(time.perf_counter() - start)
        out.deallocate(); h2.deallocate(); h1.deallocate()

    a_dram.deallocate(); b_dram.deallocate(); c_dram.deallocate()

    # Path B: L1 interleaved intermediates (naive "fusion")
    a_dram2 = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_dram2 = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c_dram2 = ttnn.from_torch(c_data, dtype=ttnn.bfloat16, device=device,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for _ in range(5):
        h1 = ttnn.add(a_dram2, b_dram2, memory_config=ttnn.L1_MEMORY_CONFIG)
        h2 = ttnn.relu(h1, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.add(h2, c_dram2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(); h2.deallocate(); h1.deallocate()

    times_l1 = []
    for _ in range(REPS):
        start = time.perf_counter()
        h1 = ttnn.add(a_dram2, b_dram2, memory_config=ttnn.L1_MEMORY_CONFIG)
        h2 = ttnn.relu(h1, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.add(h2, c_dram2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        times_l1.append(time.perf_counter() - start)
        out.deallocate(); h2.deallocate(); h1.deallocate()

    a_dram2.deallocate(); b_dram2.deallocate(); c_dram2.deallocate()

    # Path C: Block-sharded intermediates (proper fusion)
    try:
        num_cores_y = grid.y
        num_cores_x = grid.x
        shard_h = ((M + num_cores_y - 1) // num_cores_y + 31) // 32 * 32
        shard_w = ((N + num_cores_x - 1) // num_cores_x + 31) // 32 * 32

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
            )}),
            (shard_h, shard_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        a_sh = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
        b_sh = ttnn.from_torch(b_data, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
        c_sh = ttnn.from_torch(c_data, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)

        for _ in range(5):
            h1 = ttnn.add(a_sh, b_sh, memory_config=sharded_cfg)
            h2 = ttnn.relu(h1, memory_config=sharded_cfg)
            out = ttnn.add(h2, c_sh, memory_config=sharded_cfg)
            out.deallocate(); h2.deallocate(); h1.deallocate()

        times_shard = []
        for _ in range(REPS):
            start = time.perf_counter()
            h1 = ttnn.add(a_sh, b_sh, memory_config=sharded_cfg)
            h2 = ttnn.relu(h1, memory_config=sharded_cfg)
            out = ttnn.add(h2, c_sh, memory_config=sharded_cfg)
            ttnn.synchronize_device(device)
            times_shard.append(time.perf_counter() - start)
            out.deallocate(); h2.deallocate(); h1.deallocate()

        a_sh.deallocate(); b_sh.deallocate(); c_sh.deallocate()

        avg_dram = sum(times_dram) / len(times_dram)
        avg_l1 = sum(times_l1) / len(times_l1)
        avg_shard = sum(times_shard) / len(times_shard)

        print(f"  {M}x{N} (add→relu→add chain):")
        print(f"    DRAM interleaved:     {avg_dram*1000:.3f} ms")
        print(f"    L1 interleaved:       {avg_l1*1000:.3f} ms  ({avg_dram/avg_l1:.2f}x vs DRAM)")
        print(f"    L1 block-sharded:     {avg_shard*1000:.3f} ms  ({avg_dram/avg_shard:.2f}x vs DRAM)")

    except Exception as e:
        avg_dram = sum(times_dram) / len(times_dram)
        avg_l1 = sum(times_l1) / len(times_l1)
        print(f"  {M}x{N} (add→relu→add chain):")
        print(f"    DRAM interleaved:     {avg_dram*1000:.3f} ms")
        print(f"    L1 interleaved:       {avg_l1*1000:.3f} ms  ({avg_dram/avg_l1:.2f}x vs DRAM)")
        print(f"    L1 block-sharded:     ERROR ({str(e)[:80]})")

# ============================================================
# TEST 4: Dispatch overhead — how much does each op call cost?
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Per-op dispatch overhead")
print("=" * 60)
print("  Measure the host-side cost of dispatching each TT-NN op.")
print("  This is the 'interpretation tax' that XLA compilation avoids.")
print()

# Tiny tensor — compute is negligible, we're measuring dispatch
a_data = torch.randn(1, 1, 32, 32)
a_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
b_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Single op dispatch (no sync)
REPS = 200
for _ in range(10):
    r = ttnn.add(a_tt, b_tt)
    r.deallocate()

times_nosync = []
for _ in range(REPS):
    start = time.perf_counter()
    r = ttnn.add(a_tt, b_tt)
    times_nosync.append(time.perf_counter() - start)
    r.deallocate()

# Single op dispatch (with sync)
times_sync = []
for _ in range(REPS):
    start = time.perf_counter()
    r = ttnn.add(a_tt, b_tt)
    ttnn.synchronize_device(device)
    times_sync.append(time.perf_counter() - start)
    r.deallocate()

# Chain of 10 ops (no sync until end)
times_chain = []
for _ in range(REPS):
    start = time.perf_counter()
    r = a_tt
    for _ in range(10):
        r = ttnn.add(r, b_tt)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start
    times_chain.append(elapsed)
    # deallocate intermediates (they're already freed by overwrite)

avg_nosync = sum(times_nosync) / len(times_nosync) * 1000
avg_sync = sum(times_sync) / len(times_sync) * 1000
avg_chain = sum(times_chain) / len(times_chain) * 1000
per_op_chain = avg_chain / 10

print(f"  Single add (no sync):      {avg_nosync:.3f} ms  (dispatch only)")
print(f"  Single add (with sync):    {avg_sync:.3f} ms  (dispatch + execute + sync)")
print(f"  10-op chain (sync at end): {avg_chain:.3f} ms total, {per_op_chain:.3f} ms/op")
print(f"  Dispatch overhead:         ~{avg_nosync:.3f} ms per op")
print(f"  → A 100-op graph dispatched eagerly: ~{avg_nosync*100:.1f} ms in dispatch alone")
print(f"  → This is the tax that XLA compilation would eliminate")

a_tt.deallocate()
b_tt.deallocate()

ttnn.close_device(device)
print("\nDone!")
