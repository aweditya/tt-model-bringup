#!/usr/bin/env python3
"""
Experiment 53c: HEIGHT_SHARDED decode using upstream tt-metal patterns.

Key discoveries from reading upstream gpt_oss and qwen3_vl:
  1. trans_mat is (1,1,TILE_SIZE,TILE_SIZE) = (1,1,32,32) with adjacent-pair swaps
  2. nlp_create_qkv_heads_decode → HEIGHT_SHARDED Q/K/V
  3. K/V moved to kv_mem_cfg (HEIGHT_SHARDED, batch cores, kv_heads*32 height)
  4. paged_update_cache takes HEIGHT_SHARDED K/V
  5. SDPA decode outputs to HEIGHT_SHARDED
  6. nlp_concat_heads_decode merges heads back

This tests each component in isolation with Qwen2.5-0.5B dimensions.
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
TILE_SIZE = 32; batch_size = 1

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


# ══════════════════════════════════════════════════════════════
# Build transformation matrix (upstream pattern)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Building transformation matrix (upstream pattern)")
print("=" * 60)

# From get_rot_transformation_mat: adjacent-pair swaps at TILE_SIZE=32
trans_mat = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
trans_mat[..., torch.arange(0, TILE_SIZE, 2), torch.arange(1, TILE_SIZE, 2)] = 1
trans_mat[..., torch.arange(1, TILE_SIZE, 2), torch.arange(0, TILE_SIZE, 2)] = -1
trans_mat_tt = ttnn.from_torch(trans_mat, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
print(f"  trans_mat shape: (1, 1, {TILE_SIZE}, {TILE_SIZE})")
print(f"  Pattern: adjacent pairs (0↔1, 2↔3, ...) with sign flip")

# Build cos/sin cache: (1, 1, MAX_SEQ, head_dim) — full head_dim
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)  # (MAX_SEQ, half_dim=32)

# Interleave cos/sin: [c0, c0, c1, c1, ...] to match adjacent-pair rotation
cos_interleaved = np.repeat(np.cos(angles), 2, axis=-1)  # (MAX_SEQ, head_dim)
sin_interleaved = np.repeat(np.sin(angles), 2, axis=-1)  # (MAX_SEQ, head_dim)

cos_cache_tt = to_dev_4d(cos_interleaved.reshape(1, 1, MAX_SEQ, head_dim))
sin_cache_tt = to_dev_4d(sin_interleaved.reshape(1, 1, MAX_SEQ, head_dim))
print(f"  cos/sin cache shape: (1, 1, {MAX_SEQ}, {head_dim})")
print(f"  Format: interleaved [c0, c0, c1, c1, ...]")


# ══════════════════════════════════════════════════════════════
# TEST 1: nlp_create_qkv_heads_decode
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: nlp_create_qkv_heads_decode")
print("=" * 60)

# Create QKV tensor: (1, 1, batch, q_dim + 2*kv_dim)
# For batch=1, this is (1, 1, 1, 1152)
qkv_dim = n_q_heads * head_dim + 2 * n_kv_heads * head_dim  # 896 + 128 + 128 = 1152
qkv_np = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)
qkv_tt = to_dev_4d(qkv_np)

try:
    q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt,
        num_heads=n_q_heads,
        num_kv_heads=n_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    print(f"  ✓ Q: {q_tt.shape}, memory: {q_tt.memory_config().memory_layout}")
    print(f"  ✓ K: {k_tt.shape}, memory: {k_tt.memory_config().memory_layout}")
    print(f"  ✓ V: {v_tt.shape}, memory: {v_tt.memory_config().memory_layout}")
    qkv_split_ok = True
except Exception as e:
    print(f"  ✗ {str(e)[:100]}")
    qkv_split_ok = False
qkv_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 2: rotary_embedding_llama on HEIGHT_SHARDED heads
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: rotary_embedding_llama on HEIGHT_SHARDED Q/K")
print("=" * 60)

if qkv_split_ok:
    # Q is HEIGHT_SHARDED from test 1
    try:
        q_roped = ttnn.experimental.rotary_embedding_llama(
            q_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
            is_decode_mode=True)
        print(f"  ✓ Q RoPE: {q_roped.shape}")
        q_roped.deallocate()
    except Exception as e:
        print(f"  ✗ Q RoPE: {str(e)[:150]}")

    try:
        k_roped = ttnn.experimental.rotary_embedding_llama(
            k_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
            is_decode_mode=True)
        print(f"  ✓ K RoPE: {k_roped.shape}")
        k_roped.deallocate()
    except Exception as e:
        print(f"  ✗ K RoPE: {str(e)[:150]}")

    # Try fused QK
    try:
        q_r, k_r = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_tt, k_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
            is_decode_mode=True)
        print(f"  ✓ Fused QK RoPE: Q={q_r.shape}, K={k_r.shape}")
        q_r.deallocate(); k_r.deallocate()
    except Exception as e:
        print(f"  ✗ Fused QK RoPE: {str(e)[:150]}")
else:
    # Create HEIGHT_SHARDED Q/K manually
    print("  (creating HEIGHT_SHARDED Q/K manually)")
    grid_size = ttnn.CoreCoord(grid.x, grid.y)

    q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)
    q_tt_interleaved = to_dev_4d(q_np)
    q_shard = ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, head_dim),
        core_grid=ttnn.num_cores_to_corerangeset(n_q_heads, grid_size, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_tt = ttnn.to_memory_config(q_tt_interleaved, q_shard)

    try:
        q_roped = ttnn.experimental.rotary_embedding_llama(
            q_tt, cos_cache_tt, sin_cache_tt, trans_mat_tt,
            is_decode_mode=True)
        print(f"  ✓ Q RoPE (manual shard): {q_roped.shape}")
        q_roped.deallocate()
    except Exception as e:
        print(f"  ✗ Q RoPE (manual shard): {str(e)[:150]}")
    q_tt.deallocate(); q_tt_interleaved.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 3: Build KV memory config (upstream pattern)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: KV memory config + paged_update_cache")
print("=" * 60)

# From upstream get_kv_memory_config:
# KV shape for decode: (1, batch, n_kv_heads, head_dim)
# shard_height = nearest_32(n_kv_heads) = 32 (for n_kv_heads=2)
# shard_width = head_dim = 64
# num_cores = batch_size = 1
kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE  # nearest_32(2) = 32
kv_num_cores = batch_size  # 1

kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)
print(f"  KV mem config: shard ({kv_shard_height}, {head_dim}) on {kv_num_cores} core(s)")

# Create KV cache: (batch, n_kv_heads, MAX_SEQ, head_dim)
cache_np = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
k_cache = to_dev_4d(cache_np.copy())
v_cache = to_dev_4d(cache_np.copy())

# Create K/V tensors in the right format for paged_update_cache
# From upstream: K/V from nlp_create_qkv_heads_decode are HEIGHT_SHARDED
# Then moved to kv_mem_cfg via to_memory_config
# Shape: (1, 1, n_kv_heads, head_dim) HEIGHT_SHARDED on batch cores

# If we got HEIGHT_SHARDED K from test 1, use that pattern
# Otherwise, create manually
k_np = np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32)
k_interleaved = to_dev_4d(k_np)

try:
    k_sharded = ttnn.to_memory_config(k_interleaved, kv_mem_cfg)
    print(f"  ✓ K to kv_mem_cfg: {k_sharded.shape}, {k_sharded.memory_config().memory_layout}")

    # paged_update_cache
    pos_val = 5
    update_idx = ttnn.from_torch(torch.tensor([pos_val], dtype=torch.int32), device=device)

    ttnn.experimental.paged_update_cache(k_cache, k_sharded, update_idxs_tensor=update_idx)
    cache_back = from_dev(k_cache, (batch_size, n_kv_heads, MAX_SEQ, head_dim))
    val = cache_back[0, 0, pos_val, :5]
    print(f"  ✓ paged_update_cache: cache[{pos_val}] = {val}")
    print(f"  ✓ TENSOR-BASED KV POSITION UPDATE WORKS!")
    k_sharded.deallocate()
except Exception as e:
    print(f"  ✗ paged_update_cache: {str(e)[:150]}")

k_interleaved.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 4: SDPA decode with HEIGHT_SHARDED output
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: SDPA decode with HEIGHT_SHARDED")
print("=" * 60)

# Fill some data into caches first
for pos in range(10):
    kv_data = np.random.randn(1, 1, n_kv_heads, head_dim).astype(np.float32) * 0.1
    kv_tt = to_dev_4d(kv_data)
    kv_sh = ttnn.to_memory_config(kv_tt, kv_mem_cfg)
    idx = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
    ttnn.experimental.paged_update_cache(k_cache, kv_sh, update_idxs_tensor=idx)
    ttnn.experimental.paged_update_cache(v_cache, kv_sh, update_idxs_tensor=idx)
    kv_tt.deallocate(); kv_sh.deallocate()

# Q tensor: HEIGHT_SHARDED (1, 1, n_q_heads, head_dim)
padded_heads = ((n_q_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE  # 32
q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)
q_interleaved = to_dev_4d(q_np)

batch_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
sdpa_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(padded_heads, head_dim),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

# First shard Q
q_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, head_dim),
    core_grid=ttnn.num_cores_to_corerangeset(n_q_heads, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
q_sharded = ttnn.to_memory_config(q_interleaved, q_shard_cfg)

pos_tensor = ttnn.from_torch(torch.tensor([9], dtype=torch.int32), device=device)

try:
    sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_sharded, k_cache, v_cache,
        cur_pos_tensor=pos_tensor,
        compute_kernel_config=hifi4,
        memory_config=sdpa_mem_cfg,
    )
    print(f"  ✓ SDPA decode: {sdpa_out.shape}")
    print(f"  Output memory: {sdpa_out.memory_config().memory_layout}")
    sdpa_out.deallocate()
except Exception as e:
    print(f"  ✗ SDPA decode with mem_cfg: {str(e)[:150]}")

    # Try without specifying memory_config
    try:
        sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sharded, k_cache, v_cache,
            cur_pos_tensor=pos_tensor,
            compute_kernel_config=hifi4,
        )
        print(f"  ✓ SDPA decode (default mem): {sdpa_out.shape}")
        sdpa_out.deallocate()
    except Exception as e2:
        print(f"  ✗ SDPA decode (default mem): {str(e2)[:150]}")

    # Try with INTERLEAVED Q
    try:
        sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_interleaved, k_cache, v_cache,
            cur_pos_tensor=pos_tensor,
            compute_kernel_config=hifi4,
        )
        print(f"  ✓ SDPA decode (interleaved Q): {sdpa_out.shape}")
        sdpa_out.deallocate()
    except Exception as e3:
        print(f"  ✗ SDPA decode (interleaved Q): {str(e3)[:150]}")

q_sharded.deallocate()
q_interleaved.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 5: nlp_concat_heads_decode
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: nlp_concat_heads_decode")
print("=" * 60)

# Create a HEIGHT_SHARDED tensor like SDPA output: (1, batch, n_q_heads, head_dim)
attn_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)
attn_tt = to_dev_4d(attn_np)

# Shard it
attn_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(padded_heads, head_dim),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

try:
    attn_sharded = ttnn.to_memory_config(attn_tt, attn_shard_cfg)
    merged = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=n_q_heads)
    print(f"  ✓ nlp_concat_heads_decode: {merged.shape}")
    print(f"  Output memory: {merged.memory_config().memory_layout}")
    merged.deallocate()
    attn_sharded.deallocate()
except Exception as e:
    print(f"  ✗ nlp_concat_heads_decode: {str(e)[:150]}")
attn_tt.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 6: Correctness check — rotary_embedding_llama vs numpy
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: RoPE correctness (interleaved vs half format)")
print("=" * 60)

test_pos = 5
q_np = np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32)

# Numpy reference: INTERLEAVED format (adjacent pairs)
def apply_rope_interleaved_np(x, pos):
    """RoPE with interleaved rotation: pairs (x0,x1), (x2,x3), ..."""
    a = pos * freqs  # (half_dim,)
    cos_vals = np.cos(a)  # (half_dim,)
    sin_vals = np.sin(a)

    # Interleave: [c0, c0, c1, c1, ...] and [s0, s0, s1, s1, ...]
    cos_full = np.repeat(cos_vals, 2)  # (head_dim,)
    sin_full = np.repeat(sin_vals, 2)

    # Rotation: swap adjacent pairs with sign
    x_rotated = np.empty_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2]   # even positions get negated odd
    x_rotated[..., 1::2] = x[..., 0::2]    # odd positions get original even
    return x * cos_full + x_rotated * sin_full

# Numpy reference: HALF format (split at midpoint)
def apply_rope_half_np(x, pos):
    """RoPE with half rotation: split at midpoint, negate, swap."""
    a = pos * freqs
    cos_full = np.concatenate([np.cos(a), np.cos(a)])
    sin_full = np.concatenate([np.sin(a), np.sin(a)])
    rotated = np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
    return x * cos_full + rotated * sin_full

ref_interleaved = apply_rope_interleaved_np(q_np, test_pos)
ref_half = apply_rope_half_np(q_np, test_pos)

# Cross-check the two formats
cos_formats = np.dot(ref_interleaved.flatten(), ref_half.flatten()) / (
    np.linalg.norm(ref_interleaved) * np.linalg.norm(ref_half) + 1e-8)
print(f"  Cosine between interleaved and half format: {cos_formats:.6f}")
print(f"  (This should be ~0.5 — they are DIFFERENT rotations)")

# Now test ttnn rotary_embedding_llama if it worked
if qkv_split_ok:
    # Recreate QKV and split
    qkv_np2 = np.zeros((1, 1, batch_size, qkv_dim), dtype=np.float32)
    # Pack Q into the QKV tensor
    qkv_np2[0, 0, 0, :n_q_heads * head_dim] = q_np.flatten()
    qkv_tt2 = to_dev_4d(qkv_np2)
    q2, k2, v2 = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt2, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    qkv_tt2.deallocate()

    try:
        q2_roped = ttnn.experimental.rotary_embedding_llama(
            q2, cos_cache_tt, sin_cache_tt, trans_mat_tt, is_decode_mode=True)
        q2_np = from_dev(q2_roped, (1, 1, n_q_heads, head_dim))

        cos_vs_interleaved = np.dot(q2_np.flatten(), ref_interleaved.flatten()) / (
            np.linalg.norm(q2_np) * np.linalg.norm(ref_interleaved) + 1e-8)
        cos_vs_half = np.dot(q2_np.flatten(), ref_half.flatten()) / (
            np.linalg.norm(q2_np) * np.linalg.norm(ref_half) + 1e-8)

        print(f"\n  ttnn rotary_embedding_llama output:")
        print(f"    Cosine vs interleaved numpy: {cos_vs_interleaved:.6f}")
        print(f"    Cosine vs half numpy:        {cos_vs_half:.6f}")

        if cos_vs_interleaved > 0.99:
            print(f"    → rotary_embedding_llama uses INTERLEAVED format")
        elif cos_vs_half > 0.99:
            print(f"    → rotary_embedding_llama uses HALF format")
        else:
            print(f"    → Neither format matches! Investigate further.")

        q2_roped.deallocate()
    except Exception as e:
        print(f"  ✗ RoPE test failed: {str(e)[:150]}")

    q2.deallocate(); k2.deallocate(); v2.deallocate()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
If Tests 1-5 pass, we have the full HEIGHT_SHARDED decode pipeline:
  1. nlp_create_qkv_heads_decode (fused QKV split → HEIGHT_SHARDED)
  2. rotary_embedding_llama (native on-device RoPE)
  3. paged_update_cache + update_idxs_tensor (tensor KV positions!)
  4. SDPA decode with cur_pos_tensor
  5. nlp_concat_heads_decode (merge heads)

This is the path to CORRECT traced decode at 100+ tok/sec.
""")

k_cache.deallocate(); v_cache.deallocate()
ttnn.close_device(device)
print("Done!")
