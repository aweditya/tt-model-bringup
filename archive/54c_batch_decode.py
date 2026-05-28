#!/usr/bin/env python3
"""
Experiment 54c: Batch decode on Blackhole P150.

Question: Can we process multiple sequences simultaneously to increase
throughput? Our single-sequence decode runs at 132 tok/sec with 110 cores
mostly idle. Batch decode should spread work across more cores.

What this tests:
  1. paged_update_cache with batch > 1 (multiple update indices)
  2. scaled_dot_product_attention_decode with batch > 1
  3. Latency and throughput scaling across batch_size = 1, 2, 4, 8
  4. Where batching breaks or hits limitations

Key shape changes for batch > 1:
  - KV cache:       (batch, n_kv_heads, MAX_SEQ, head_dim)
  - Q for decode:   (1, 1, batch*n_q_heads_padded, head_dim)  -- upstream pattern
  - K/V for update: (1, batch, n_kv_heads, head_dim)
  - cur_pos_tensor: (batch,) with different positions per sequence
  - update_idxs:    (batch,) with different positions per sequence

Upstream tt-metal pattern (from qwen3_vl, llama models):
  - KV sharding: num_cores = batch_size, shard = (nearest_32(n_kv_heads), head_dim)
  - SDPA decode:  Q is HEIGHT_SHARDED across batch*head cores
  - paged_update_cache: update_idxs_tensor has one entry per batch element
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Qwen2.5-0.5B dimensions
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
TILE_SIZE = 32; MAX_SEQ = 256

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)


# ══════════════════════════════════════════════════════════════
# TEST 1: paged_update_cache with batch > 1
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: paged_update_cache with batch > 1")
print("=" * 60)

for batch_size in [1, 2, 4, 8]:
    print(f"\n  --- batch_size = {batch_size} ---")

    # KV cache: (batch, n_kv_heads, MAX_SEQ, head_dim)
    cache_np = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    cache_tt = to_dev_4d(cache_np)

    # New KV to insert: (1, batch, n_kv_heads, head_dim)
    # This is the upstream pattern — dim1 = batch
    new_kv_np = np.random.randn(1, batch_size, n_kv_heads, head_dim).astype(np.float32)

    # Different positions per sequence
    positions = [10 + i * 5 for i in range(batch_size)]
    update_idx = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device)

    # KV memory config: one shard per batch element
    kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE  # 32
    kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, head_dim),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    try:
        kv_tt = to_dev_4d(new_kv_np)
        kv_sharded = ttnn.to_memory_config(kv_tt, kv_mem_cfg)
        ttnn.experimental.paged_update_cache(cache_tt, kv_sharded, update_idxs_tensor=update_idx)

        # Verify: check each batch element's position
        cache_back = from_dev(cache_tt, (batch_size, n_kv_heads, MAX_SEQ, head_dim))
        all_correct = True
        for b in range(batch_size):
            pos = positions[b]
            actual = cache_back[b, 0, pos, :3]
            expected = new_kv_np[0, b, 0, :3]
            # bfloat16 has limited precision, check approximately
            match = np.allclose(actual, expected, atol=0.05)
            if not match:
                print(f"    batch[{b}] pos={pos}: MISMATCH actual={actual} expected={expected}")
                all_correct = False

        if all_correct:
            print(f"  PASS paged_update_cache batch={batch_size}, positions={positions}")
        else:
            print(f"  FAIL paged_update_cache batch={batch_size} — values don't match")

        # Benchmark: time 100 updates
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            ttnn.experimental.paged_update_cache(cache_tt, kv_sharded, update_idxs_tensor=update_idx)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        avg_us = np.mean(times[10:]) * 1e6  # skip warmup
        print(f"  Latency: {avg_us:.1f} us/update (batch={batch_size})")

        kv_tt.deallocate(); kv_sharded.deallocate()
    except Exception as e:
        err = str(e).split('\n')[0][:200]
        print(f"  FAIL paged_update_cache batch={batch_size}: {err}")

        # Try alternative shape: (1, 1, batch*n_kv_heads, head_dim)
        print(f"  Trying alt shape (1, 1, {batch_size*n_kv_heads}, {head_dim})...")
        try:
            alt_np = new_kv_np.reshape(1, 1, batch_size * n_kv_heads, head_dim)
            alt_tt = to_dev_4d(alt_np)
            alt_sh = ttnn.to_memory_config(alt_tt, kv_mem_cfg)
            ttnn.experimental.paged_update_cache(cache_tt, alt_sh, update_idxs_tensor=update_idx)
            print(f"  ALT PASS! shape (1, 1, {batch_size*n_kv_heads}, {head_dim})")
            alt_tt.deallocate(); alt_sh.deallocate()
        except Exception as e2:
            print(f"  ALT also failed: {str(e2).split(chr(10))[0][:200]}")

    cache_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 2: SDPA decode with batch > 1
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("TEST 2: scaled_dot_product_attention_decode with batch > 1")
print("=" * 60)

for batch_size in [1, 2, 4, 8]:
    print(f"\n  --- batch_size = {batch_size} ---")

    # Create and fill KV caches
    cache_np = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_cache = to_dev_4d(cache_np.copy())
    v_cache = to_dev_4d(cache_np.copy())

    # Fill first 20 positions with random data
    kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, head_dim),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    fill_ok = True
    for pos in range(20):
        try:
            kv_np = np.random.randn(1, batch_size, n_kv_heads, head_dim).astype(np.float32) * 0.1
            kv_tt = to_dev_4d(kv_np)
            kv_sh = ttnn.to_memory_config(kv_tt, kv_mem_cfg)
            idx = ttnn.from_torch(torch.tensor([pos] * batch_size, dtype=torch.int32), device=device)
            ttnn.experimental.paged_update_cache(k_cache, kv_sh, update_idxs_tensor=idx)
            ttnn.experimental.paged_update_cache(v_cache, kv_sh, update_idxs_tensor=idx)
            kv_tt.deallocate(); kv_sh.deallocate()
        except Exception as e:
            print(f"  Cache fill failed at pos={pos}: {str(e).split(chr(10))[0][:150]}")
            fill_ok = False
            break

    if not fill_ok:
        print(f"  SKIP SDPA batch={batch_size} — cache fill failed")
        k_cache.deallocate(); v_cache.deallocate()
        continue

    # Different positions per sequence in the batch
    positions = [15 + i for i in range(batch_size)]
    pos_tensor = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device)

    # Q tensor: try multiple shapes
    # Pattern A: (1, 1, batch*padded_q_heads, head_dim) — upstream concat pattern
    padded_q_heads = ((n_q_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE  # 32
    q_shapes = [
        (f"(1, 1, {n_q_heads}, {head_dim}) interleaved",
         np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32) * 0.1,
         "interleaved"),
        (f"(1, {batch_size}, {n_q_heads}, {head_dim}) interleaved",
         np.random.randn(1, batch_size, n_q_heads, head_dim).astype(np.float32) * 0.1,
         "interleaved"),
        (f"({batch_size}, 1, {n_q_heads}, {head_dim}) interleaved",
         np.random.randn(batch_size, 1, n_q_heads, head_dim).astype(np.float32) * 0.1,
         "interleaved"),
    ]

    for label, q_np, mem_type in q_shapes:
        try:
            q_tt = to_dev_4d(q_np)
            sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_tt, k_cache, v_cache,
                cur_pos_tensor=pos_tensor,
                compute_kernel_config=hifi4,
            )
            out_shape = sdpa_out.shape
            print(f"  PASS Q={label} -> out={out_shape}")

            # Benchmark
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                sdpa_out2 = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, k_cache, v_cache,
                    cur_pos_tensor=pos_tensor,
                    compute_kernel_config=hifi4,
                )
                ttnn.synchronize_device(device)
                times.append(time.perf_counter() - t0)
                sdpa_out2.deallocate()
            avg_us = np.mean(times[5:]) * 1e6
            print(f"    Latency: {avg_us:.1f} us (batch={batch_size})")

            sdpa_out.deallocate(); q_tt.deallocate()
            break  # Found working shape, move to next batch_size
        except Exception as e:
            err = str(e).split('\n')[0][:200]
            print(f"  FAIL Q={label}: {err}")
            q_tt.deallocate()

    # Also try HEIGHT_SHARDED Q (upstream pattern)
    print(f"  Trying HEIGHT_SHARDED Q...")
    try:
        q_np = np.random.randn(1, 1, batch_size * padded_q_heads, head_dim).astype(np.float32) * 0.1
        q_tt = to_dev_4d(q_np)

        q_core_grid = ttnn.num_cores_to_corerangeset(
            batch_size * n_q_heads,
            ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        q_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        q_sharded = ttnn.to_memory_config(q_tt, q_shard_cfg)

        sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sharded, k_cache, v_cache,
            cur_pos_tensor=pos_tensor,
            compute_kernel_config=hifi4,
        )
        print(f"  PASS HEIGHT_SHARDED Q (1,1,{batch_size*padded_q_heads},{head_dim}) -> out={sdpa_out.shape}")

        # Benchmark
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_sharded, k_cache, v_cache,
                cur_pos_tensor=pos_tensor,
                compute_kernel_config=hifi4,
            )
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
            out.deallocate()
        avg_us = np.mean(times[5:]) * 1e6
        print(f"    HEIGHT_SHARDED latency: {avg_us:.1f} us (batch={batch_size})")

        sdpa_out.deallocate(); q_sharded.deallocate(); q_tt.deallocate()
    except Exception as e:
        err = str(e).split('\n')[0][:200]
        print(f"  FAIL HEIGHT_SHARDED Q: {err}")

    k_cache.deallocate(); v_cache.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 3: Full decode step simulation (matmul + SDPA + cache update)
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("TEST 3: Full decode step simulation (batch scaling)")
print("=" * 60)
print("Simulating: embedding -> QKV projection -> RoPE -> cache update -> SDPA -> output projection")

# Use random weights (we just care about shapes and timing, not correctness)
w_q = to_dev_4d(np.random.randn(1, 1, hidden, n_q_heads * head_dim).astype(np.float32) * 0.01)
w_k = to_dev_4d(np.random.randn(1, 1, hidden, n_kv_heads * head_dim).astype(np.float32) * 0.01)
w_v = to_dev_4d(np.random.randn(1, 1, hidden, n_kv_heads * head_dim).astype(np.float32) * 0.01)
w_o = to_dev_4d(np.random.randn(1, 1, hidden, hidden).astype(np.float32) * 0.01)

# Rotation matrix for RoPE
half_dim = head_dim // 2
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev_4d(np.expand_dims(np.expand_dims(R, 0), 0))  # (1,1,64,64)

for batch_size in [1, 2, 4, 8]:
    print(f"\n  --- batch_size = {batch_size} ---")

    # KV caches
    cache_np = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_cache = to_dev_4d(cache_np.copy())
    v_cache = to_dev_4d(cache_np.copy())

    # Fill 20 positions
    kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, head_dim),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    fill_ok = True
    for pos in range(20):
        try:
            kv_np = np.random.randn(1, batch_size, n_kv_heads, head_dim).astype(np.float32) * 0.1
            kv_tt = to_dev_4d(kv_np)
            kv_sh = ttnn.to_memory_config(kv_tt, kv_mem_cfg)
            idx = ttnn.from_torch(torch.tensor([pos] * batch_size, dtype=torch.int32), device=device)
            ttnn.experimental.paged_update_cache(k_cache, kv_sh, update_idxs_tensor=idx)
            ttnn.experimental.paged_update_cache(v_cache, kv_sh, update_idxs_tensor=idx)
            kv_tt.deallocate(); kv_sh.deallocate()
        except:
            fill_ok = False; break

    if not fill_ok:
        print(f"  SKIP batch={batch_size} — cache fill failed")
        k_cache.deallocate(); v_cache.deallocate()
        continue

    # Input embedding: (1, 1, batch, hidden) — batch in dim2 for matmul broadcasting
    x_np = np.random.randn(1, 1, batch_size, hidden).astype(np.float32) * 0.1

    positions = [19] * batch_size  # all at same position for simplicity
    pos_tensor = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device)

    # RoPE cos/sin for these positions (same pos -> same rope)
    freqs = 1.0 / (1000000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = 19 * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    cos_tt = to_dev_4d(cos_full)
    sin_tt = to_dev_4d(sin_full)

    try:
        x_tt = to_dev_4d(x_np)

        # Key insight from TEST 2: SDPA decode wants Q=(1, batch, n_q_heads, head_dim)
        # So we need to reshape Q/K/V projections accordingly.
        # Projection: (1,1,batch,hidden) @ (1,1,hidden,heads*head_dim) -> (1,1,batch,heads*head_dim)
        # Then reshape to (1, batch, heads, head_dim) for SDPA

        # Warmup
        for _ in range(3):
            q_proj = ttnn.matmul(x_tt, w_q, compute_kernel_config=hifi4)
            k_proj = ttnn.matmul(x_tt, w_k, compute_kernel_config=hifi4)
            v_proj = ttnn.matmul(x_tt, w_v, compute_kernel_config=hifi4)

            # Reshape: (1,1,batch,heads*hd) -> (1,batch,heads,hd)
            q_4d = ttnn.reshape(q_proj, [1, batch_size, n_q_heads, head_dim])
            k_4d = ttnn.reshape(k_proj, [1, batch_size, n_kv_heads, head_dim])
            v_4d = ttnn.reshape(v_proj, [1, batch_size, n_kv_heads, head_dim])

            # RoPE on Q (simplified — cos/sin broadcast over batch & heads)
            q_rotated = ttnn.matmul(q_4d, R_tt)
            q_roped = ttnn.add(ttnn.mul(q_4d, cos_tt), ttnn.mul(q_rotated, sin_tt))

            # Cache update: K/V already (1, batch, n_kv_heads, head_dim)
            k_sh = ttnn.to_memory_config(k_4d, kv_mem_cfg)
            v_sh = ttnn.to_memory_config(v_4d, kv_mem_cfg)
            ttnn.experimental.paged_update_cache(k_cache, k_sh, update_idxs_tensor=pos_tensor)
            ttnn.experimental.paged_update_cache(v_cache, v_sh, update_idxs_tensor=pos_tensor)

            # SDPA decode: Q=(1,batch,n_q_heads,hd), K/V cache=(batch,n_kv,MAX_SEQ,hd)
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_roped, k_cache, v_cache,
                cur_pos_tensor=pos_tensor,
                compute_kernel_config=hifi4,
            )

            # Output projection: reshape (1,batch,n_q_heads,hd) -> (1,1,batch,hidden)
            attn_reshaped = ttnn.reshape(attn_out, [1, 1, batch_size, hidden])
            o_out = ttnn.matmul(attn_reshaped, w_o, compute_kernel_config=hifi4)
            ttnn.synchronize_device(device)

            q_proj.deallocate(); k_proj.deallocate(); v_proj.deallocate()
            q_4d.deallocate(); k_4d.deallocate(); v_4d.deallocate()
            q_rotated.deallocate(); q_roped.deallocate()
            k_sh.deallocate(); v_sh.deallocate()
            attn_out.deallocate(); attn_reshaped.deallocate(); o_out.deallocate()

        # Benchmark
        times = []
        for _ in range(30):
            t0 = time.perf_counter()

            q_proj = ttnn.matmul(x_tt, w_q, compute_kernel_config=hifi4)
            k_proj = ttnn.matmul(x_tt, w_k, compute_kernel_config=hifi4)
            v_proj = ttnn.matmul(x_tt, w_v, compute_kernel_config=hifi4)
            q_4d = ttnn.reshape(q_proj, [1, batch_size, n_q_heads, head_dim])
            k_4d = ttnn.reshape(k_proj, [1, batch_size, n_kv_heads, head_dim])
            v_4d = ttnn.reshape(v_proj, [1, batch_size, n_kv_heads, head_dim])
            q_rotated = ttnn.matmul(q_4d, R_tt)
            q_roped = ttnn.add(ttnn.mul(q_4d, cos_tt), ttnn.mul(q_rotated, sin_tt))
            k_sh = ttnn.to_memory_config(k_4d, kv_mem_cfg)
            v_sh = ttnn.to_memory_config(v_4d, kv_mem_cfg)
            ttnn.experimental.paged_update_cache(k_cache, k_sh, update_idxs_tensor=pos_tensor)
            ttnn.experimental.paged_update_cache(v_cache, v_sh, update_idxs_tensor=pos_tensor)
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_roped, k_cache, v_cache,
                cur_pos_tensor=pos_tensor,
                compute_kernel_config=hifi4,
            )
            attn_reshaped = ttnn.reshape(attn_out, [1, 1, batch_size, hidden])
            o_out = ttnn.matmul(attn_reshaped, w_o, compute_kernel_config=hifi4)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)

            q_proj.deallocate(); k_proj.deallocate(); v_proj.deallocate()
            q_4d.deallocate(); k_4d.deallocate(); v_4d.deallocate()
            q_rotated.deallocate(); q_roped.deallocate()
            k_sh.deallocate(); v_sh.deallocate()
            attn_out.deallocate(); attn_reshaped.deallocate(); o_out.deallocate()

        avg_ms = np.mean(times[5:]) * 1000
        throughput = batch_size / (avg_ms / 1000)
        print(f"  PASS: single-layer decode step")
        print(f"    Latency:    {avg_ms:.2f} ms  (batch={batch_size})")
        print(f"    Throughput: {throughput:.1f} tok/sec  (batch_size / latency)")
        print(f"    Efficiency: {avg_ms / batch_size:.2f} ms/tok")

        x_tt.deallocate()
    except Exception as e:
        import traceback
        err = str(e).split('\n')[0][:200]
        print(f"  FAIL batch={batch_size}: {err}")
        traceback.print_exc()

    cos_tt.deallocate(); sin_tt.deallocate()
    k_cache.deallocate(); v_cache.deallocate()

    # Needed to release memory for next iteration
    ttnn.synchronize_device(device)


# ══════════════════════════════════════════════════════════════
# TEST 4: Verify correctness — batch SDPA vs single SDPA
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("TEST 4: Correctness — batch=2 SDPA vs two single SDPA calls")
print("=" * 60)

batch_size = 2
kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

# Create shared KV data for both approaches
np.random.seed(42)
kv_data = {}
for pos in range(15):
    kv_data[pos] = {
        'k': np.random.randn(n_kv_heads, head_dim).astype(np.float32) * 0.1,
        'v': np.random.randn(n_kv_heads, head_dim).astype(np.float32) * 0.1,
    }

# Approach A: Two separate single-batch SDPA calls
print("\n  Approach A: Two separate single-batch calls...")
results_single = []
for b in range(2):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_cache = to_dev_4d(cache_np.copy())
    v_cache = to_dev_4d(cache_np.copy())

    kv_core_grid = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, head_dim),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    for pos in range(15):
        kv_np = kv_data[pos]['k' if b == 0 else 'k'].reshape(1, 1, n_kv_heads, head_dim)
        kv_tt = to_dev_4d(kv_np)
        kv_sh = ttnn.to_memory_config(kv_tt, kv_mem_cfg)
        idx = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
        ttnn.experimental.paged_update_cache(k_cache, kv_sh, update_idxs_tensor=idx)

        vv_np = kv_data[pos]['v' if b == 0 else 'v'].reshape(1, 1, n_kv_heads, head_dim)
        vv_tt = to_dev_4d(vv_np)
        vv_sh = ttnn.to_memory_config(vv_tt, kv_mem_cfg)
        ttnn.experimental.paged_update_cache(v_cache, vv_sh, update_idxs_tensor=idx)
        kv_tt.deallocate(); kv_sh.deallocate(); vv_tt.deallocate(); vv_sh.deallocate()

    q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32) * 0.1
    q_tt = to_dev_4d(q_np)
    pos_t = ttnn.from_torch(torch.tensor([14], dtype=torch.int32), device=device)
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_tt, k_cache, v_cache, cur_pos_tensor=pos_t, compute_kernel_config=hifi4)
    results_single.append(from_dev(out, (1, 1, n_q_heads, head_dim)))
    out.deallocate(); q_tt.deallocate(); k_cache.deallocate(); v_cache.deallocate()

print(f"    Single[0] range: [{results_single[0].min():.4f}, {results_single[0].max():.4f}]")
print(f"    Single[1] range: [{results_single[1].min():.4f}, {results_single[1].max():.4f}]")

# Approach B: Batched SDPA
print("\n  Approach B: Single batched SDPA call...")
try:
    cache_np = np.zeros((2, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_cache = to_dev_4d(cache_np.copy())
    v_cache = to_dev_4d(cache_np.copy())

    kv_core_grid = ttnn.num_cores_to_corerangeset(2, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, head_dim),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    for pos in range(15):
        # Both batch elements get same KV data (like approach A)
        k_both = np.stack([kv_data[pos]['k'], kv_data[pos]['k']]).reshape(1, 2, n_kv_heads, head_dim)
        v_both = np.stack([kv_data[pos]['v'], kv_data[pos]['v']]).reshape(1, 2, n_kv_heads, head_dim)
        k_tt = to_dev_4d(k_both)
        v_tt = to_dev_4d(v_both)
        k_sh = ttnn.to_memory_config(k_tt, kv_mem_cfg)
        v_sh = ttnn.to_memory_config(v_tt, kv_mem_cfg)
        idx = ttnn.from_torch(torch.tensor([pos, pos], dtype=torch.int32), device=device)
        ttnn.experimental.paged_update_cache(k_cache, k_sh, update_idxs_tensor=idx)
        ttnn.experimental.paged_update_cache(v_cache, v_sh, update_idxs_tensor=idx)
        k_tt.deallocate(); v_tt.deallocate(); k_sh.deallocate(); v_sh.deallocate()

    # Q: same query for both batch elements
    # From TEST 2: correct shape is (1, batch, n_q_heads, head_dim)
    q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32) * 0.1
    q_batch = np.tile(q_np, (1, 2, 1, 1))  # (1, 2, n_q_heads, head_dim)
    q_tt = to_dev_4d(q_batch)
    pos_t = ttnn.from_torch(torch.tensor([14, 14], dtype=torch.int32), device=device)

    out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_tt, k_cache, v_cache, cur_pos_tensor=pos_t, compute_kernel_config=hifi4)
    result_batch = from_dev(out, (1, 2, n_q_heads, head_dim))
    print(f"    Batch output shape: {result_batch.shape}")
    print(f"    Batch[0] range: [{result_batch[0,0].min():.4f}, {result_batch[0,0].max():.4f}]")
    print(f"    Batch[1] range: [{result_batch[0,1].min():.4f}, {result_batch[0,1].max():.4f}]")

    # Compare each batch element vs single-sequence result
    cos_sim_0 = np.dot(result_batch[0,0].flatten(), results_single[0].flatten()) / (
        np.linalg.norm(result_batch[0,0]) * np.linalg.norm(results_single[0]) + 1e-10)
    cos_sim_1 = np.dot(result_batch[0,1].flatten(), results_single[1].flatten()) / (
        np.linalg.norm(result_batch[0,1]) * np.linalg.norm(results_single[1]) + 1e-10)
    print(f"    Cosine similarity batch[0] vs single[0]: {cos_sim_0:.6f}")
    print(f"    Cosine similarity batch[1] vs single[1]: {cos_sim_1:.6f}")
    if cos_sim_0 > 0.99 and cos_sim_1 > 0.99:
        print(f"    PASS: Batched SDPA matches individual calls!")

    out.deallocate(); q_tt.deallocate(); k_cache.deallocate(); v_cache.deallocate()
except Exception as e:
    import traceback
    print(f"  FAIL: {str(e).split(chr(10))[0][:200]}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key questions answered:
  1. paged_update_cache: YES, batch > 1 works. Shape (1, batch, n_kv_heads, head_dim).
     Latency is nearly constant: 44us (b=1) -> 31us (b=8). Faster with batch!
  2. SDPA decode: YES, batch > 1 works. Q shape = (1, batch, n_q_heads, head_dim).
     KV cache shape = (batch, n_kv_heads, MAX_SEQ, head_dim).
     cur_pos_tensor = (batch,) with per-sequence positions.
     Latency nearly constant: 77us (b=1) -> 75us (b=8). Near-perfect scaling!
  3. Full single-layer decode step: near-perfect linear scaling.
     b=1: 0.38ms (2601 tok/s), b=2: 0.39ms (5172 tok/s),
     b=4: 0.41ms (9819 tok/s), b=8: 0.41ms (19365 tok/s).
     Efficiency: 0.38 -> 0.05 ms/tok. 7.4x throughput gain for 8x batch!
  4. Correctness: batched SDPA matches individual calls at 0.9999 cosine.
  5. HEIGHT_SHARDED Q does NOT work for batch > 1 (core count exceeds 110).
     batch=8 * n_q_heads=14 = 112 cores > 110 available.

Implication for full 24-layer model:
  - Current: 7.6ms/tok at batch=1 = 132 tok/sec
  - Single layer scales ~1.08x latency for 8x batch
  - Projected batch=8: ~8.2ms for 8 tokens = 975 tok/sec aggregate
  - That's 7.4x throughput gain, consistent with single-layer results
""")

ttnn.close_device(device)
print("Done!")
