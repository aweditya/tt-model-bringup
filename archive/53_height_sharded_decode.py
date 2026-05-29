#!/usr/bin/env python3
"""
Experiment 53: HEIGHT_SHARDED decode — the unified optimization.

Hypothesis: HEIGHT_SHARDED memory layout enables three things at once:
  1. paged_update_cache with update_idxs_tensor (tensor-based KV position)
  2. Native ttnn.experimental.rotary_embedding (no rotation matrix overhead)
  3. L1 SRAM residency (no DRAM round-trips between ops)

This experiment probes each in isolation on Blackhole P150, starting with
the simplest possible HEIGHT_SHARDED tensor and working up to full decode.

Plan:
  Test A: Create HEIGHT_SHARDED tensors with our model dimensions
  Test B: Elementwise ops on HEIGHT_SHARDED tensors
  Test C: Matmul with HEIGHT_SHARDED input
  Test D: paged_update_cache with update_idxs_tensor
  Test E: Native rotary_embedding_llama on HEIGHT_SHARDED
  Test F: Full HEIGHT_SHARDED decode step (if all above pass)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Model dimensions (Qwen2.5-0.5B)
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)


# ══════════════════════════════════════════════════════════════
# TEST A: Create HEIGHT_SHARDED tensors
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST A: HEIGHT_SHARDED tensor creation")
print("=" * 60)

# For decode, we process single tokens: shape (1, 1, 1, hidden) = (1, 1, 1, 896)
# In TILE_LAYOUT, this becomes (1, 1, 32, 896) due to tile padding.
# HEIGHT_SHARDED distributes rows across cores.
# With 1 batch, we need 1 core for height sharding.

# Method 1: Convenience constant (let op infer sharding)
print("\n  Method 1: L1_HEIGHT_SHARDED_MEMORY_CONFIG (convenience constant)")
try:
    test_data = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    test_tt = ttnn.from_torch(test_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    test_sharded = ttnn.to_memory_config(test_tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    print(f"    ✓ Created: {test_sharded.shape}")
    print(f"    Memory config: {test_sharded.memory_config()}")
    test_sharded.deallocate()
    test_tt.deallocate()
except Exception as e:
    print(f"    ✗ Failed: {e}")

# Method 2: Manual ShardSpec
print("\n  Method 2: Manual ShardSpec (1 core)")
try:
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        (32, 64),  # shard shape: full tile
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    test_data = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    test_tt = ttnn.from_torch(test_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    test_sharded = ttnn.to_memory_config(test_tt, mem_cfg)
    print(f"    ✓ Created: {test_sharded.shape}")
    print(f"    Memory config: {test_sharded.memory_config()}")
    test_sharded.deallocate()
    test_tt.deallocate()
except Exception as e:
    print(f"    ✗ Failed: {e}")

# Method 3: create_sharded_memory_config helper
print("\n  Method 3: create_sharded_memory_config helper")
try:
    batch_grid = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem_cfg_helper = ttnn.create_sharded_memory_config(
        shape=(32, 64),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    test_data = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    test_tt = ttnn.from_torch(test_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    test_sharded = ttnn.to_memory_config(test_tt, mem_cfg_helper)
    print(f"    ✓ Created: {test_sharded.shape}")
    print(f"    Memory config: {test_sharded.memory_config()}")
    test_sharded.deallocate()
    test_tt.deallocate()
except Exception as e:
    print(f"    ✗ Failed: {e}")

# Method 4: Actual model shapes
print("\n  Testing with actual model tensor shapes:")
shapes_to_test = {
    "decode_hidden (1,1,1,896)": (1, 1, 32, 896),  # tile-padded
    "q_heads (1,14,1,64)": (1, 14, 32, 64),        # tile-padded
    "k_heads (1,2,1,64)": (1, 2, 32, 64),           # tile-padded
    "kv_cache (1,2,256,64)": (1, 2, 256, 64),
}

for name, shape in shapes_to_test.items():
    try:
        data = torch.randn(*shape, dtype=torch.float32)
        tt = ttnn.from_torch(data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        sharded = ttnn.to_memory_config(tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        print(f"    ✓ {name}: {sharded.memory_config()}")
        sharded.deallocate()
        tt.deallocate()
    except Exception as e:
        print(f"    ✗ {name}: {e}")


# ══════════════════════════════════════════════════════════════
# TEST B: Elementwise ops on HEIGHT_SHARDED
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST B: Elementwise ops on HEIGHT_SHARDED tensors")
print("=" * 60)

test_np = np.random.randn(1, 1, 32, 64).astype(np.float32)
a_tt = to_dev(test_np.reshape(32, 64))

for sharded_input in [False, True]:
    label = "HEIGHT_SHARDED" if sharded_input else "INTERLEAVED (baseline)"
    try:
        if sharded_input:
            a = ttnn.to_memory_config(a_tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        else:
            a = a_tt

        # Try basic ops
        r1 = ttnn.neg(a)
        r2 = ttnn.relu(a)
        r3 = ttnn.silu(a)

        print(f"\n  {label}:")
        print(f"    neg:  ✓ shape {r1.shape}")
        print(f"    relu: ✓ shape {r2.shape}")
        print(f"    silu: ✓ shape {r3.shape}")

        # Check output memory config
        print(f"    neg output memory: {r1.memory_config()}")

        r1.deallocate(); r2.deallocate(); r3.deallocate()
        if sharded_input and a is not a_tt:
            a.deallocate()
    except Exception as e:
        print(f"\n  {label}: ✗ {e}")


# ══════════════════════════════════════════════════════════════
# TEST C: Matmul with HEIGHT_SHARDED input
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST C: Matmul with HEIGHT_SHARDED input")
print("=" * 60)

# Simulate Q projection: (1, 1, 896) @ (896, 896) -> (1, 1, 896)
x_np = np.random.randn(1, 1, hidden).astype(np.float32)
w_np = np.random.randn(hidden, hidden).astype(np.float32)
x_tt = to_dev(x_np.reshape(1, hidden))
w_tt = to_dev(w_np)

# Baseline: INTERLEAVED matmul
try:
    r_baseline = ttnn.matmul(x_tt, w_tt, compute_kernel_config=hifi4)
    baseline_np = from_dev(r_baseline, (1, hidden))
    print(f"\n  INTERLEAVED matmul: ✓ shape {r_baseline.shape}")
    r_baseline.deallocate()
except Exception as e:
    print(f"\n  INTERLEAVED matmul: ✗ {e}")
    baseline_np = None

# HEIGHT_SHARDED input matmul
try:
    x_sharded = ttnn.to_memory_config(x_tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    r_sharded = ttnn.matmul(x_sharded, w_tt, compute_kernel_config=hifi4)
    sharded_np = from_dev(r_sharded, (1, hidden))
    print(f"  HEIGHT_SHARDED matmul: ✓ shape {r_sharded.shape}")
    print(f"  Output memory: {r_sharded.memory_config()}")

    if baseline_np is not None:
        cos_sim = np.dot(baseline_np.flatten(), sharded_np.flatten()) / (
            np.linalg.norm(baseline_np) * np.linalg.norm(sharded_np) + 1e-8)
        print(f"  Cosine vs INTERLEAVED: {cos_sim:.6f}")
    r_sharded.deallocate()
    x_sharded.deallocate()
except Exception as e:
    print(f"  HEIGHT_SHARDED matmul: ✗ {e}")

x_tt.deallocate()
w_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST D: paged_update_cache with update_idxs_tensor
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST D: paged_update_cache with update_idxs_tensor")
print("=" * 60)

# Create a KV cache: (1, n_kv_heads, MAX_SEQ, head_dim)
MAX_SEQ = 256
cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
cache_tt = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# New KV data: (1, n_kv_heads, 1, head_dim) — single token
new_kv_np = np.random.randn(1, n_kv_heads, 1, head_dim).astype(np.float32)
new_kv_tt = ttnn.from_torch(torch.from_numpy(new_kv_np.copy()),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Position tensor
pos_val = 5
update_idx = ttnn.from_torch(torch.tensor([pos_val], dtype=torch.int32), device=device)

# Test 1: Standard update_cache_for_token_ (Python int) — our known working path
print("\n  Test 1: update_cache_for_token_ (int) — baseline")
try:
    ttnn.kv_cache.update_cache_for_token_(cache_tt, new_kv_tt, update_index=pos_val, batch_offset=0)
    cache_readback = from_dev(cache_tt, (1, n_kv_heads, MAX_SEQ, head_dim))
    check_pos = cache_readback[0, 0, pos_val, :5]
    check_zero = cache_readback[0, 0, pos_val + 1, :5]
    print(f"    ✓ Cache[{pos_val}] = {check_pos} (non-zero)")
    print(f"    ✓ Cache[{pos_val+1}] = {check_zero} (should be zero)")
except Exception as e:
    print(f"    ✗ {e}")

# Test 2: paged_update_cache with update_idxs_tensor (device tensor)
print("\n  Test 2: paged_update_cache with update_idxs_tensor")

# Reset cache
cache_tt2 = ttnn.from_torch(torch.zeros(1, n_kv_heads, MAX_SEQ, head_dim),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# paged_update_cache may need HEIGHT_SHARDED input
# Try with INTERLEAVED first, then HEIGHT_SHARDED
for sharded in [False, True]:
    label = "HEIGHT_SHARDED" if sharded else "INTERLEAVED"
    try:
        if sharded:
            # HEIGHT_SHARD the new KV data on 1 core (batch_size=1)
            batch_grid = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
            kv_shard_cfg = ttnn.create_sharded_memory_config(
                shape=(32, head_dim),  # tile-padded height for 1 kv head
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            new_kv_sharded = ttnn.to_memory_config(new_kv_tt, kv_shard_cfg)
        else:
            new_kv_sharded = new_kv_tt

        pos_val2 = 10
        update_idx2 = ttnn.from_torch(torch.tensor([pos_val2], dtype=torch.int32), device=device)

        ttnn.experimental.paged_update_cache(cache_tt2, new_kv_sharded,
                                             update_idxs_tensor=update_idx2)

        cache_readback2 = from_dev(cache_tt2, (1, n_kv_heads, MAX_SEQ, head_dim))
        check_pos2 = cache_readback2[0, 0, pos_val2, :5]
        print(f"\n    {label}: ✓ paged_update_cache worked!")
        print(f"    Cache[{pos_val2}] = {check_pos2}")
        break  # Success, no need to try the other
    except Exception as e:
        print(f"\n    {label}: ✗ {e}")

cache_tt.deallocate()
cache_tt2.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST E: Native rotary_embedding_llama
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST E: Native rotary_embedding_llama")
print("=" * 60)

# rotary_embedding_llama expects:
#   input: (seq_len, 1, batch, head_dim) or HEIGHT_SHARDED
#   cos_cache, sin_cache: (1, 1, max_seq, head_dim)
rope_theta = 1000000.0
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

# Build full cos/sin caches
angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)
cos_cache = np.cos(angles).astype(np.float32)  # (MAX_SEQ, half_dim)
sin_cache = np.sin(angles).astype(np.float32)

cos_cache_tt = ttnn.from_torch(
    torch.from_numpy(cos_cache.reshape(1, 1, MAX_SEQ, half_dim)),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
sin_cache_tt = ttnn.from_torch(
    torch.from_numpy(sin_cache.reshape(1, 1, MAX_SEQ, half_dim)),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Test input: Q heads (1, 1, n_q_heads, head_dim) for decode
# rotary_embedding expects: (seq_len, 1, batch, head_dim) or similar
q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)
q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

test_pos = 5

# Test rotary_embedding (standard)
print("\n  Test 1: ttnn.experimental.rotary_embedding (standard)")
try:
    q_roped = ttnn.experimental.rotary_embedding(q_tt, cos_cache_tt, sin_cache_tt, token_idx=test_pos)
    print(f"    ✓ Output shape: {q_roped.shape}")
    q_roped.deallocate()
except Exception as e:
    print(f"    ✗ {e}")

# Test rotary_embedding_llama (half-format, what Qwen needs)
print("\n  Test 2: ttnn.experimental.rotary_embedding_llama")
try:
    q_roped_llama = ttnn.experimental.rotary_embedding_llama(q_tt, cos_cache_tt, sin_cache_tt, None)
    print(f"    ✓ Output shape: {q_roped_llama.shape}")
    q_roped_llama.deallocate()
except Exception as e:
    print(f"    ✗ {e}")

# Test with explicit token_idx (if supported)
print("\n  Test 3: rotary_embedding_llama with transformation matrix")
try:
    # Build transformation matrix
    trans_mat = torch.zeros(1, 1, head_dim, head_dim)
    for i in range(half_dim):
        trans_mat[0, 0, i, i + half_dim] = 1.0     # result[:32] = x[32:]
        trans_mat[0, 0, i + half_dim, i] = -1.0     # result[32:] = -x[:32]
    trans_mat_tt = ttnn.from_torch(trans_mat, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    q_roped_llama2 = ttnn.experimental.rotary_embedding_llama(q_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt)
    print(f"    ✓ Output shape: {q_roped_llama2.shape}")

    # Verify against our numpy reference
    q_np_4d = q_np.copy()
    angles_pos = test_pos * freqs
    cos_pos = np.cos(angles_pos)
    sin_pos = np.sin(angles_pos)
    # Half-format: rotate_half(x) * sin + x * cos
    q_rot = np.concatenate([-q_np_4d[..., half_dim:], q_np_4d[..., :half_dim]], axis=-1)
    cos_full = np.concatenate([cos_pos, cos_pos])
    sin_full = np.concatenate([sin_pos, sin_pos])
    q_ref = q_np_4d * cos_full + q_rot * sin_full

    q_out = from_dev(q_roped_llama2, (1, 1, n_q_heads, head_dim))
    cos_sim = np.dot(q_ref.flatten(), q_out.flatten()) / (
        np.linalg.norm(q_ref) * np.linalg.norm(q_out) + 1e-8)
    print(f"    Cosine vs numpy reference: {cos_sim:.6f}")
    q_roped_llama2.deallocate()
except Exception as e:
    print(f"    ✗ {e}")

# Test with HEIGHT_SHARDED input
print("\n  Test 4: rotary_embedding_llama with HEIGHT_SHARDED input")
try:
    q_sharded = ttnn.to_memory_config(q_tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    q_roped_sharded = ttnn.experimental.rotary_embedding_llama(q_sharded, cos_cache_tt, sin_cache_tt, trans_mat_tt)
    print(f"    ✓ Output shape: {q_roped_sharded.shape}")
    q_roped_sharded.deallocate()
    q_sharded.deallocate()
except Exception as e:
    print(f"    ✗ {e}")


# ══════════════════════════════════════════════════════════════
# TEST F: Speed comparison — INTERLEAVED vs HEIGHT_SHARDED decode ops
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST F: Speed comparison")
print("=" * 60)

# Time a chain of ops that simulates one layer's attention
x_np = np.random.randn(1, 1, hidden).astype(np.float32)
w_np = np.random.randn(hidden, n_q_heads * head_dim).astype(np.float32)

x_tt = to_dev(x_np.reshape(1, hidden))
w_tt = to_dev(w_np)

# Warmup
for _ in range(5):
    r = ttnn.matmul(x_tt, w_tt, compute_kernel_config=hifi4)
    r.deallocate()

# INTERLEAVED timing
REPS = 50
times_interleaved = []
for _ in range(REPS):
    t0 = time.perf_counter()
    r = ttnn.matmul(x_tt, w_tt, compute_kernel_config=hifi4)
    r2 = ttnn.silu(r)
    ttnn.synchronize_device(device)
    times_interleaved.append(time.perf_counter() - t0)
    r.deallocate(); r2.deallocate()

# HEIGHT_SHARDED timing (if sharding worked above)
times_sharded = []
try:
    x_sharded = ttnn.to_memory_config(x_tt, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    # Warmup
    for _ in range(5):
        r = ttnn.matmul(x_sharded, w_tt, compute_kernel_config=hifi4)
        r.deallocate()

    for _ in range(REPS):
        t0 = time.perf_counter()
        r = ttnn.matmul(x_sharded, w_tt, compute_kernel_config=hifi4)
        r2 = ttnn.silu(r)
        ttnn.synchronize_device(device)
        times_sharded.append(time.perf_counter() - t0)
        r.deallocate(); r2.deallocate()
    x_sharded.deallocate()
except Exception as e:
    print(f"  HEIGHT_SHARDED matmul+silu: ✗ {e}")

avg_i = np.mean(times_interleaved) * 1e6  # microseconds
print(f"\n  INTERLEAVED (matmul+silu): {avg_i:.0f} µs")
if times_sharded:
    avg_s = np.mean(times_sharded) * 1e6
    speedup = avg_i / avg_s
    print(f"  HEIGHT_SHARDED (matmul+silu): {avg_s:.0f} µs ({speedup:.2f}x)")
else:
    print(f"  HEIGHT_SHARDED: skipped (creation failed)")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Test A: HEIGHT_SHARDED tensor creation
Test B: Elementwise ops on HEIGHT_SHARDED
Test C: Matmul with HEIGHT_SHARDED input
Test D: paged_update_cache with update_idxs_tensor
Test E: Native rotary_embedding_llama
Test F: Speed comparison

If D+E pass → we can build a fully HEIGHT_SHARDED decode that:
  1. Uses paged_update_cache with tensor positions (traceable!)
  2. Uses native RoPE (no rotation matrix overhead)
  3. Runs entirely in L1 SRAM (no DRAM round-trips)

Combined with trace capture → correct 100+ tok/sec target.
""")

ttnn.close_device(device)
print("Done!")
