#!/usr/bin/env python3
"""
Experiment 59b: Native rotary_embedding_llama with interleaved format.

From exp 54: native rotary_embedding_llama is 4.85x faster (0.019ms vs 0.090ms)
but failed on HEIGHT_SHARDED trans_mat assertion.

From exp 58: half and interleaved RoPE are equivalent via element permutation.

This experiment:
  1. Figures out what rotary_embedding_llama actually needs
  2. Tests if we can get it working with proper interleaved format
  3. If it works, measures speedup in a traced single-layer decode

The native op signature (from source):
  rotary_embedding_llama(input, cos, sin, trans_mat, ...)
  - input: (1, 1, seq_len, head_dim*n_heads) or similar
  - cos: cached cos values
  - sin: cached sin values
  - trans_mat: transformation matrix (HEIGHT_SHARDED?)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rope_theta = 1000000.0

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor):
    return ttnn.to_torch(tensor).float().numpy()


# ── First, let's explore the API signature ──
print("\n--- API Exploration ---")
print(f"rotary_embedding_llama: {ttnn.experimental.rotary_embedding_llama.__doc__}")

# Try to figure out what arguments it wants
import inspect
try:
    sig = inspect.signature(ttnn.experimental.rotary_embedding_llama)
    print(f"Signature: {sig}")
except:
    print("Could not get signature")

# ── Build interleaved cos/sin tables ──
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
pos = 42
angles = pos * freqs

# Interleaved format: [c0,c0,c1,c1,...,c31,c31]
cos_interleaved = np.repeat(np.cos(angles), 2).astype(np.float32)
sin_interleaved = np.repeat(np.sin(angles), 2).astype(np.float32)

# ── Build transformation matrix for interleaved rotation ──
# Interleaved rotate: result[2i] = -x[2i+1], result[2i+1] = x[2i]
trans_mat = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    trans_mat[2*i, 2*i+1] = -1.0    # result[2i] = -x[2i+1]
    trans_mat[2*i+1, 2*i] = 1.0     # result[2i+1] = x[2i]

print(f"\nTransformation matrix shape: {trans_mat.shape}")
print(f"  Non-zeros: {np.count_nonzero(trans_mat)}")
print(f"  trans_mat[:6,:6]:\n{trans_mat[:6,:6]}")

# ── Test various input formats ──
# The native op was designed for Llama-style models
# Q shape for decode: (1, n_q_heads, 1, head_dim)

x_np = np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32)
cos_4d = cos_interleaved.reshape(1, 1, 1, head_dim)
sin_4d = sin_interleaved.reshape(1, 1, 1, head_dim)

print(f"\n--- Test 1: Basic call with (1, n_q_heads, 1, head_dim) ---")
x_tt = to_dev_4d(x_np)
cos_tt = to_dev_4d(cos_4d)
sin_tt = to_dev_4d(sin_4d)

# Try different trans_mat memory configs
configs_to_try = [
    ("INTERLEAVED (default)", to_dev(trans_mat)),
]

# Try HEIGHT_SHARDED trans_mat
try:
    n_cores = min(head_dim, grid.x * grid.y)
    shard_height = head_dim // n_cores if n_cores > 0 else head_dim
    if shard_height < 32:
        shard_height = 32
        n_cores = max(1, head_dim // shard_height)

    core_grid = ttnn.num_cores_to_corerangeset(n_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(shard_height, head_dim),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    trans_mat_sharded = ttnn.to_memory_config(to_dev(trans_mat), sharded_cfg)
    configs_to_try.append(("HEIGHT_SHARDED (2 cores)", trans_mat_sharded))
    print(f"  Created HEIGHT_SHARDED trans_mat: {n_cores} cores, shard ({shard_height}, {head_dim})")
except Exception as e:
    print(f"  HEIGHT_SHARDED creation failed: {e}")

# Try with 1 core
try:
    core_grid_1 = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    sharded_cfg_1 = ttnn.create_sharded_memory_config(
        shape=(head_dim, head_dim),
        core_grid=core_grid_1,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    trans_mat_sharded_1 = ttnn.to_memory_config(to_dev(trans_mat), sharded_cfg_1)
    configs_to_try.append(("HEIGHT_SHARDED (1 core)", trans_mat_sharded_1))
    print(f"  Created HEIGHT_SHARDED trans_mat: 1 core, shard ({head_dim}, {head_dim})")
except Exception as e:
    print(f"  HEIGHT_SHARDED 1-core creation failed: {e}")

for name, tm in configs_to_try:
    print(f"\n  Trying {name}...")
    try:
        result = ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt, tm)
        result_np = from_dev(result)
        print(f"    SUCCESS! Output shape: {result_np.shape}")
        print(f"    Output[:5]: {result_np.flatten()[:5]}")

        # Verify against numpy reference
        ref = x_np * cos_4d + np.zeros_like(x_np) * sin_4d  # placeholder
        # Actually compute interleaved rotation
        x_rot = np.zeros_like(x_np)
        x_rot[..., 0::2] = -x_np[..., 1::2]
        x_rot[..., 1::2] = x_np[..., 0::2]
        ref = x_np * cos_4d + x_rot * sin_4d

        cos_sim = np.dot(result_np.flatten(), ref.flatten()) / (
            np.linalg.norm(result_np) * np.linalg.norm(ref))
        print(f"    Cosine vs numpy ref: {cos_sim:.6f}")

    except Exception as e:
        print(f"    FAILED: {e}")

# ── Test 2: Try without trans_mat (maybe it's optional?) ──
print(f"\n--- Test 2: Without trans_mat ---")
try:
    result = ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt)
    print(f"  SUCCESS without trans_mat!")
except TypeError as e:
    print(f"  TypeError: {e}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── Test 3: Standard rotary_embedding (not _llama) ──
print(f"\n--- Test 3: ttnn.experimental.rotary_embedding ---")
try:
    result = ttnn.experimental.rotary_embedding(x_tt, cos_tt, sin_tt)
    result_np = from_dev(result)
    print(f"  SUCCESS! Output shape: {result_np.shape}")
    print(f"  Output[:5]: {result_np.flatten()[:5]}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── Timing comparison ──
print(f"\n--- Timing ---")

# Rotation matrix approach (current)
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

cos_half = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
sin_half = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
cos_h_tt = to_dev_4d(cos_half)
sin_h_tt = to_dev_4d(sin_half)

# Warmup
for _ in range(5):
    q_rot = ttnn.matmul(x_tt, R_tt)
    q_roped = ttnn.add(ttnn.mul(x_tt, cos_h_tt), ttnn.mul(q_rot, sin_h_tt))
    ttnn.synchronize_device(device)

# Time rotation matrix
times_rm = []
for _ in range(50):
    t0 = time.perf_counter()
    q_rot = ttnn.matmul(x_tt, R_tt)
    q_roped = ttnn.add(ttnn.mul(x_tt, cos_h_tt), ttnn.mul(q_rot, sin_h_tt))
    ttnn.synchronize_device(device)
    times_rm.append(time.perf_counter() - t0)

avg_rm = np.mean(times_rm[5:]) * 1000
print(f"  Rotation matrix: {avg_rm:.3f}ms")

# Time native (if any config worked)
for name, tm in configs_to_try:
    try:
        # Warmup
        for _ in range(5):
            result = ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt, tm)
            ttnn.synchronize_device(device)

        times_native = []
        for _ in range(50):
            t0 = time.perf_counter()
            result = ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt, tm)
            ttnn.synchronize_device(device)
            times_native.append(time.perf_counter() - t0)

        avg_native = np.mean(times_native[5:]) * 1000
        print(f"  Native ({name}): {avg_native:.3f}ms ({avg_rm/avg_native:.1f}x speedup)")
    except:
        pass


print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

ttnn.close_device(device)
print("\nDone!")
