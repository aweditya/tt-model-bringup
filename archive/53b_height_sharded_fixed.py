#!/usr/bin/env python3
"""
Experiment 53b: HEIGHT_SHARDED decode — fixed API shapes.

Fixes from 53:
  - Use explicit ShardSpec (convenience constant has no shard_spec)
  - cos/sin caches: (1, 1, MAX_SEQ, head_dim) not half_dim
  - paged_update_cache: input shape (1, batch, 32, head_dim) with batch=1
  - rotary_embedding: token_index= not token_idx=
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; MAX_SEQ = 256; rope_theta = 1000000.0

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

def make_height_shard_config(num_cores, shard_h, shard_w):
    """Create HEIGHT_SHARDED memory config with explicit ShardSpec."""
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# ══════════════════════════════════════════════════════════════
# TEST A: HEIGHT_SHARDED with actual model shapes (explicit ShardSpec)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST A: HEIGHT_SHARDED with model tensor shapes")
print("=" * 60)

# All our decode tensors are tile-padded to height 32
shapes_to_test = [
    # (name, torch_shape, num_cores, shard_h, shard_w)
    ("hidden (1,1,32,896)", (1, 1, 32, 896), 1, 32, 896),
    ("q_proj (1,1,32,896)", (1, 1, 32, n_q_heads * head_dim), 1, 32, n_q_heads * head_dim),
    ("q_heads (1,14,32,64)", (1, n_q_heads, 32, head_dim), n_q_heads, 32, head_dim),
    ("k_heads (1,2,32,64)", (1, n_kv_heads, 32, head_dim), n_kv_heads, 32, head_dim),
    ("single_head (1,1,32,64)", (1, 1, 32, head_dim), 1, 32, head_dim),
]

for name, shape, ncores, sh, sw in shapes_to_test:
    try:
        mem_cfg = make_height_shard_config(ncores, sh, sw)
        data = torch.randn(*shape, dtype=torch.float32)
        tt = ttnn.from_torch(data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        sharded = ttnn.to_memory_config(tt, mem_cfg)
        print(f"  ✓ {name}: sharded on {ncores} core(s)")
        sharded.deallocate(); tt.deallocate()
    except Exception as e:
        print(f"  ✗ {name}: {e}")


# ══════════════════════════════════════════════════════════════
# TEST B: Elementwise + matmul on HEIGHT_SHARDED (explicit shard spec)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST B: Ops on HEIGHT_SHARDED tensors")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, 896).astype(np.float32)
x_tt = to_dev_4d(x_np)
shard_cfg = make_height_shard_config(1, 32, 896)

try:
    x_sharded = ttnn.to_memory_config(x_tt, shard_cfg)
    print(f"  Created HEIGHT_SHARDED (1,1,32,896)")

    # Elementwise
    for op_name, op_fn in [("neg", ttnn.neg), ("relu", ttnn.relu), ("silu", ttnn.silu)]:
        try:
            r = op_fn(x_sharded)
            print(f"  ✓ {op_name}: output memory = {r.memory_config().memory_layout}")
            r.deallocate()
        except Exception as e:
            print(f"  ✗ {op_name}: {e}")

    # Matmul: need to unshard first? or does matmul accept sharded input?
    w_np = np.random.randn(896, 896).astype(np.float32)
    w_tt = to_dev(w_np)

    # First try matmul on sharded input directly
    try:
        r = ttnn.matmul(x_sharded, w_tt, compute_kernel_config=hifi4)
        print(f"  ✓ matmul(sharded, interleaved): output memory = {r.memory_config().memory_layout}")
        r.deallocate()
    except Exception as e:
        print(f"  ✗ matmul(sharded, interleaved): {e}")
        # Try unsharding first
        try:
            x_unshard = ttnn.to_memory_config(x_sharded, ttnn.DRAM_MEMORY_CONFIG)
            r = ttnn.matmul(x_unshard, w_tt, compute_kernel_config=hifi4)
            print(f"  ✓ matmul(unshard→interleaved): works after unsharding")
            r.deallocate(); x_unshard.deallocate()
        except Exception as e2:
            print(f"  ✗ matmul(unshard→interleaved): {e2}")

    w_tt.deallocate()
    x_sharded.deallocate()
except Exception as e:
    print(f"  ✗ HEIGHT_SHARDED creation failed: {e}")
x_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST C: paged_update_cache with correct shapes
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST C: paged_update_cache with update_idxs_tensor")
print("=" * 60)

# Cache: (batch, n_kv_heads, MAX_SEQ, head_dim) = (1, 2, 256, 64)
cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
cache_tt = to_dev_4d(cache_np)

# paged_update_cache expects input: (1, batch, padded_heads, head_dim)
# where batch matches cache batch (1), padded to tile (32)
# Actually, looking at the error: "input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[0]"
# cache_tensor.padded_shape()[0] = 1 (batch)
# So input_tensor.padded_shape()[1] must = 1
# Input shape should be: (n_kv_heads, 1, 32, head_dim) or (1, 1, 32, head_dim)?

# Let's try multiple input shapes
new_kv_np = np.random.randn(1, n_kv_heads, 1, head_dim).astype(np.float32)
pos_val = 10
update_idx = ttnn.from_torch(torch.tensor([pos_val], dtype=torch.int32), device=device)

input_shapes = [
    ("(1, 1, 32, 64) — single head", np.random.randn(1, 1, 1, head_dim).astype(np.float32)),
    ("(n_kv, 1, 32, 64) — per head", np.random.randn(n_kv_heads, 1, 1, head_dim).astype(np.float32)),
    ("(1, n_kv, 32, 64) — batch×heads", np.random.randn(1, n_kv_heads, 1, head_dim).astype(np.float32)),
]

for label, inp_np in input_shapes:
    try:
        inp_tt = to_dev_4d(inp_np)
        ttnn.experimental.paged_update_cache(cache_tt, inp_tt, update_idxs_tensor=update_idx)
        cache_back = from_dev(cache_tt, (1, n_kv_heads, MAX_SEQ, head_dim))
        val = cache_back[0, 0, pos_val, :3]
        print(f"  ✓ Input {label}: cache[{pos_val}] = {val}")
        inp_tt.deallocate()
        break
    except Exception as e:
        short_err = str(e).split('\n')[0][:100]
        print(f"  ✗ Input {label}: {short_err}")
        inp_tt.deallocate()

# Try with HEIGHT_SHARDED input
print("\n  HEIGHT_SHARDED input variants:")
for ncores in [1, 2, n_kv_heads]:
    for inp_shape, inp_np in [
        (f"({n_kv_heads},1,32,{head_dim})", np.random.randn(n_kv_heads, 1, 1, head_dim).astype(np.float32)),
        (f"(1,{n_kv_heads},32,{head_dim})", np.random.randn(1, n_kv_heads, 1, head_dim).astype(np.float32)),
    ]:
        try:
            inp_tt = to_dev_4d(inp_np)
            shard_h = max(32, inp_tt.shape[-2])  # tile-padded
            shard_w = head_dim
            scfg = make_height_shard_config(ncores, shard_h, shard_w)
            inp_sharded = ttnn.to_memory_config(inp_tt, scfg)
            ttnn.experimental.paged_update_cache(cache_tt, inp_sharded,
                                                 update_idxs_tensor=update_idx)
            cache_back = from_dev(cache_tt, (1, n_kv_heads, MAX_SEQ, head_dim))
            val = cache_back[0, 0, pos_val, :3]
            print(f"    ✓ {inp_shape} on {ncores} core(s): cache[{pos_val}] = {val}")
            inp_sharded.deallocate(); inp_tt.deallocate()
        except Exception as e:
            short_err = str(e).split('\n')[0][:80]
            print(f"    ✗ {inp_shape} on {ncores} core(s): {short_err}")
            inp_tt.deallocate()

cache_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST D: Native rotary_embedding with fixed cos/sin shapes
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST D: rotary_embedding / rotary_embedding_llama")
print("=" * 60)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)

# rotary_embedding_llama expects cos/sin with last dim = head_dim
# Half-format: cos/sin are duplicated [cos, cos] to match full head_dim
cos_cache_full = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (MAX_SEQ, head_dim)
sin_cache_full = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

cos_cache_tt = to_dev_4d(cos_cache_full.reshape(1, 1, MAX_SEQ, head_dim))
sin_cache_tt = to_dev_4d(sin_cache_full.reshape(1, 1, MAX_SEQ, head_dim))

# Also try half_dim caches for standard rotary_embedding
cos_cache_half = np.cos(angles)  # (MAX_SEQ, half_dim)
sin_cache_half = np.sin(angles)
cos_half_tt = to_dev_4d(cos_cache_half.reshape(1, 1, MAX_SEQ, half_dim))
sin_half_tt = to_dev_4d(sin_cache_half.reshape(1, 1, MAX_SEQ, half_dim))

# Build transformation matrix for rotary_embedding_llama
# rotate_half: result[:32] = -x[32:], result[32:] = x[:32]
trans_mat = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    trans_mat[0, 0, i, i + half_dim] = -1.0     # result[:32] = -x[32:]
    trans_mat[0, 0, i + half_dim, i] = 1.0      # result[32:] = x[:32]
trans_mat_tt = to_dev_4d(trans_mat)

test_pos = 5

# Q tensor for decode: (1, 1, n_q_heads, head_dim)
q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)
q_tt = to_dev_4d(q_np)

# Numpy reference (half-format RoPE)
def apply_rope_np(x, pos):
    a = pos * freqs
    cos_f = np.concatenate([np.cos(a), np.cos(a)])
    sin_f = np.concatenate([np.sin(a), np.sin(a)])
    rotated = np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
    return x * cos_f + rotated * sin_f

q_ref = apply_rope_np(q_np, test_pos)

# Test 1: rotary_embedding (standard, interleaved format)
print("\n  Test 1: rotary_embedding (standard, half_dim cos/sin)")
try:
    r = ttnn.experimental.rotary_embedding(q_tt, cos_half_tt, sin_half_tt, token_index=test_pos)
    r_np = from_dev(r, q_np.shape)
    cos_sim = np.dot(r_np.flatten(), q_ref.flatten()) / (np.linalg.norm(r_np) * np.linalg.norm(q_ref) + 1e-8)
    print(f"    ✓ Output shape: {r.shape}")
    print(f"    Cosine vs numpy ref: {cos_sim:.6f}")
    r.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"    ✗ {short_err}")

# Test 2: rotary_embedding (full head_dim cos/sin)
print("\n  Test 2: rotary_embedding (standard, full head_dim cos/sin)")
try:
    r = ttnn.experimental.rotary_embedding(q_tt, cos_cache_tt, sin_cache_tt, token_index=test_pos)
    r_np = from_dev(r, q_np.shape)
    cos_sim = np.dot(r_np.flatten(), q_ref.flatten()) / (np.linalg.norm(r_np) * np.linalg.norm(q_ref) + 1e-8)
    print(f"    ✓ Output shape: {r.shape}")
    print(f"    Cosine vs numpy ref: {cos_sim:.6f}")
    r.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"    ✗ {short_err}")

# Test 3: rotary_embedding_llama (half-format, full head_dim cos/sin)
print("\n  Test 3: rotary_embedding_llama (full head_dim cos/sin, trans_mat)")
try:
    r = ttnn.experimental.rotary_embedding_llama(
        q_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
        is_decode_mode=True)
    r_np = from_dev(r, q_np.shape)
    cos_sim = np.dot(r_np.flatten(), q_ref.flatten()) / (np.linalg.norm(r_np) * np.linalg.norm(q_ref) + 1e-8)
    print(f"    ✓ Output shape: {r.shape}")
    print(f"    Cosine vs numpy ref: {cos_sim:.6f}")
    r.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"    ✗ {short_err}")

# Test 4: rotary_embedding_llama without decode mode
print("\n  Test 4: rotary_embedding_llama (is_decode_mode=False)")
try:
    r = ttnn.experimental.rotary_embedding_llama(
        q_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
        is_decode_mode=False)
    r_np = from_dev(r, q_np.shape)
    cos_sim = np.dot(r_np.flatten(), q_ref.flatten()) / (np.linalg.norm(r_np) * np.linalg.norm(q_ref) + 1e-8)
    print(f"    ✓ Output shape: {r.shape}")
    print(f"    Cosine vs numpy ref: {cos_sim:.6f}")
    r.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"    ✗ {short_err}")

# Test 5: fused QK rotation
print("\n  Test 5: rotary_embedding_llama_fused_qk")
k_np = np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32)
k_tt = to_dev_4d(k_np)
try:
    q_r, k_r = ttnn.experimental.rotary_embedding_llama_fused_qk(
        q_tt, k_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt)
    q_r_np = from_dev(q_r, q_np.shape)
    k_r_np = from_dev(k_r, k_np.shape)
    q_cos = np.dot(q_r_np.flatten(), q_ref.flatten()) / (np.linalg.norm(q_r_np) * np.linalg.norm(q_ref) + 1e-8)
    k_ref = apply_rope_np(k_np, test_pos)
    k_cos = np.dot(k_r_np.flatten(), k_ref.flatten()) / (np.linalg.norm(k_r_np) * np.linalg.norm(k_ref) + 1e-8)
    print(f"    ✓ Q cosine: {q_cos:.6f}, K cosine: {k_cos:.6f}")
    q_r.deallocate(); k_r.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"    ✗ {short_err}")
k_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST E: nlp_create_qkv_heads_decode
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST E: nlp_create_qkv_heads_decode")
print("=" * 60)

# This op splits a combined QKV tensor into separate Q, K, V with HEIGHT_SHARDED output
# Input: (1, 1, 32, q_dim + 2*kv_dim) = (1, 1, 32, 896 + 128 + 128) = (1, 1, 32, 1152)
qkv_dim = n_q_heads * head_dim + 2 * n_kv_heads * head_dim  # 896 + 128 + 128 = 1152
qkv_np = np.random.randn(1, 1, 1, qkv_dim).astype(np.float32)
qkv_tt = to_dev_4d(qkv_np)

try:
    q_out, k_out, v_out = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt,
        num_heads=n_q_heads,
        num_kv_heads=n_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    print(f"  ✓ Q: {q_out.shape}, memory: {q_out.memory_config().memory_layout}")
    print(f"  ✓ K: {k_out.shape}, memory: {k_out.memory_config().memory_layout}")
    print(f"  ✓ V: {v_out.shape}, memory: {v_out.memory_config().memory_layout}")
    q_out.deallocate(); k_out.deallocate(); v_out.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"  ✗ {short_err}")

# Try without HEIGHT_SHARDED output
try:
    q_out, k_out, v_out = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt,
        num_heads=n_q_heads,
        num_kv_heads=n_kv_heads,
    )
    print(f"  ✓ (default mem) Q: {q_out.shape}, K: {k_out.shape}, V: {v_out.shape}")
    q_out.deallocate(); k_out.deallocate(); v_out.deallocate()
except Exception as e:
    short_err = str(e).split('\n')[0][:100]
    print(f"  ✗ (default mem): {short_err}")

qkv_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key findings:
  A: Explicit ShardSpec needed (convenience constant fails)
  B: Elementwise / matmul compatibility with sharded tensors
  C: paged_update_cache input shape requirements
  D: Native RoPE API requirements and correctness
  E: nlp_create_qkv_heads_decode (produces HEIGHT_SHARDED output)
""")

ttnn.close_device(device)
print("Done!")
