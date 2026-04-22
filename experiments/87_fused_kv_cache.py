#!/usr/bin/env python3
"""
Experiment 87: Fused KV cache update — paged_fused_update_cache

Currently we do 4 separate paged_update_cache calls per layer:
  paged_update_cache(k_lo, kr_lo, ...)
  paged_update_cache(v_lo, v_lo, ...)
  paged_update_cache(k_hi, kr_hi, ...)
  paged_update_cache(v_hi, v_hi, ...)

paged_fused_update_cache updates K and V in parallel:
  paged_fused_update_cache(k_lo, v_lo, kr_lo, v_lo, ...)
  paged_fused_update_cache(k_hi, v_hi, kr_hi, v_hi, ...)

This halves the cache update dispatch count: 4 → 2 per layer, 128 → 64 across 32 layers.

Test on Qwen2.5-0.5B first for fast iteration.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device grid: {grid.x}x{grid.y}")

TILE = 32
batch_size = 1
head_dim = 64
n_kv_heads = 2
MAX_SEQ = 512

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

kv_sh = ((n_kv_heads + TILE - 1) // TILE) * TILE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# Create test caches and update tensors
k_cache = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))
v_cache = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))

kr = to_dev_4d(np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32))
vr = to_dev_4d(np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32))
kr_s = ttnn.to_memory_config(kr, kv_cfg)
vr_s = ttnn.to_memory_config(vr, kv_cfg)
pos_buf = ttnn.from_torch(torch.tensor([42], dtype=torch.int32), device=device)

print("\n" + "="*60)
print("TEST 1: Separate paged_update_cache (current approach)")
print("="*60)

try:
    t0 = time.perf_counter()
    ttnn.experimental.paged_update_cache(k_cache, kr_s, update_idxs_tensor=pos_buf)
    ttnn.experimental.paged_update_cache(v_cache, vr_s, update_idxs_tensor=pos_buf)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"  2x separate: {(t1-t0)*1000:.2f}ms")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "="*60)
print("TEST 2: paged_fused_update_cache")
print("="*60)

# Reset caches
k_cache2 = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))
v_cache2 = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))

try:
    t0 = time.perf_counter()
    ttnn.experimental.paged_fused_update_cache(
        k_cache2, v_cache2, kr_s, vr_s, update_idxs_tensor=pos_buf)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"  Fused: {(t1-t0)*1000:.2f}ms")

    # Verify correctness: fused result should match separate result
    k_np = ttnn.to_torch(k_cache).float().numpy()
    k2_np = ttnn.to_torch(k_cache2).float().numpy()
    v_np = ttnn.to_torch(v_cache).float().numpy()
    v2_np = ttnn.to_torch(v_cache2).float().numpy()

    k_match = np.allclose(k_np, k2_np, atol=1e-5)
    v_match = np.allclose(v_np, v2_np, atol=1e-5)
    print(f"  K cache match: {k_match}")
    print(f"  V cache match: {v_match}")

    if not k_match:
        diff = np.abs(k_np - k2_np).max()
        print(f"  K max diff: {diff}")
    if not v_match:
        diff = np.abs(v_np - v2_np).max()
        print(f"  V max diff: {diff}")

except Exception as e:
    err = str(e)[:300]
    print(f"  FAILED: {err}")

    # Try without sharded inputs
    print("\n  Trying with non-sharded inputs...")
    k_cache3 = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))
    v_cache3 = to_dev_4d(np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32))
    kr_ns = to_dev_4d(np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32))
    vr_ns = to_dev_4d(np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32))
    try:
        ttnn.experimental.paged_fused_update_cache(
            k_cache3, v_cache3, kr_ns, vr_ns, update_idxs_tensor=pos_buf)
        ttnn.synchronize_device(device)
        print("  Non-sharded: OK!")
    except Exception as e2:
        print(f"  Non-sharded also FAILED: {str(e2)[:200]}")


# Test 3: Benchmark in a loop (if fused works)
print("\n" + "="*60)
print("TEST 3: Benchmark separate vs fused (100 iterations)")
print("="*60)

# Benchmark separate
times_sep = []
for pos in range(100):
    pos_t = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
    t0 = time.perf_counter()
    ttnn.experimental.paged_update_cache(k_cache, kr_s, update_idxs_tensor=pos_t)
    ttnn.experimental.paged_update_cache(v_cache, vr_s, update_idxs_tensor=pos_t)
    ttnn.synchronize_device(device)
    times_sep.append(time.perf_counter() - t0)

avg_sep = np.mean(times_sep[10:]) * 1000  # skip warmup
print(f"  Separate (2 calls): avg {avg_sep:.3f}ms")
print(f"  Per-layer overhead (with split SDPA = 4 calls): {avg_sep*2:.3f}ms")
print(f"  Total across 32 layers: {avg_sep*2*32:.1f}ms")

ttnn.close_device(device)
print("\nDone!")
