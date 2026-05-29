#!/usr/bin/env python3
"""
Experiment 88: Native RoPE for Llama (interleaved format)

Currently our Llama decode uses 6 ops for RoPE per Q and K:
  qr = add(mul(q, cos), mul(matmul(q, R_tt), sin))  # 4 ops for Q
  kr = add(mul(k, cos), mul(matmul(k, R_tt), sin))  # 4 ops for K
  Total: 8 ops per layer (matmul, mul, mul, add) x2

ttnn.experimental.rotary_embedding_llama should do this in 1 op for Llama's
interleaved RoPE format. If it works: 8 ops → 2 per layer, 256 → 64 across 32 layers.

ttnn.experimental.rotary_embedding_llama_fused_qk could do BOTH Q and K in 1 op:
8 ops → 1 per layer, 256 → 32 across 32 layers.

Test on Qwen2.5-0.5B first with half-format rotary_embedding to verify API,
then try Llama-specific variant.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device grid: {grid.x}x{grid.y}")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

TILE = 32
head_dim = 128  # Llama-3.1-8B
half_dim = head_dim // 2
n_q_heads = 32
n_kv_heads = 8
rope_theta = 500000.0

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))


# ══════════════════════════════════════════════════════════════
# TEST 1: ttnn.experimental.rotary_embedding_llama on single tensor
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 1: rotary_embedding_llama (single tensor)")
print("="*60)

# Create test Q tensor: [1, n_q_heads, 1, head_dim]
q_np = np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32)
q_tt = to_dev_4d(q_np)

# Precompute cos/sin cache for position 42
pos = 42
angles = pos * freqs  # [half_dim]

# rotary_embedding_llama expects cos/sin of shape [1, 1, MAX_SEQ, head_dim]
# with interleaved format: cos[i] = cos(freq[i//2])
MAX_SEQ = 512
all_angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)  # [MAX_SEQ, half_dim]
cos_cache = np.repeat(np.cos(all_angles), 2, axis=-1)  # [MAX_SEQ, head_dim]
sin_cache = np.repeat(np.sin(all_angles), 2, axis=-1)  # [MAX_SEQ, head_dim]

# Try different shapes for cos/sin cache
for cache_shape_name, cos_np, sin_np in [
    ("4D [1,1,SEQ,HD]", cos_cache.reshape(1, 1, MAX_SEQ, head_dim), sin_cache.reshape(1, 1, MAX_SEQ, head_dim)),
    ("2D [SEQ,HD]", cos_cache, sin_cache),
]:
    cos_tt = to_dev_4d(cos_np) if cos_np.ndim == 4 else to_bf16(cos_np)
    sin_tt = to_dev_4d(sin_np) if sin_np.ndim == 4 else to_bf16(sin_np)

    # Create trans_mat for rotary_embedding_llama
    trans_mat = np.zeros((1, 1, TILE, TILE), dtype=np.float32)
    for i in range(head_dim // 2):
        if i < TILE and 2*i+1 < TILE:
            trans_mat[0, 0, 2*i, 2*i+1] = -1.0
            trans_mat[0, 0, 2*i+1, 2*i] = 1.0
    trans_tt = to_dev_4d(trans_mat)

    try:
        t0 = time.perf_counter()
        result = ttnn.experimental.rotary_embedding_llama(
            q_tt, cos_tt, sin_tt, trans_tt)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()

        out = from_dev(result, (1, n_q_heads, 1, head_dim))
        print(f"  {cache_shape_name}: OK! {(t1-t0)*1000:.1f}ms, shape={result.shape}")

        # Verify against numpy reference
        cos_pos = np.repeat(np.cos(angles), 2).reshape(1, 1, 1, head_dim)
        sin_pos = np.repeat(np.sin(angles), 2).reshape(1, 1, 1, head_dim)
        rot = np.zeros_like(q_np)
        rot[..., 0::2] = -q_np[..., 1::2]
        rot[..., 1::2] = q_np[..., 0::2]
        expected = q_np * cos_pos + rot * sin_pos
        cos_sim = np.dot(out.flatten(), expected.flatten()) / (np.linalg.norm(out) * np.linalg.norm(expected) + 1e-10)
        print(f"           Cosine vs numpy: {cos_sim:.6f}")

    except Exception as e:
        err = str(e)[:200]
        print(f"  {cache_shape_name}: FAILED — {err}")


# ══════════════════════════════════════════════════════════════
# TEST 2: rotary_embedding_llama_fused_qk (both Q and K at once)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 2: rotary_embedding_llama_fused_qk")
print("="*60)

q_tt = to_dev_4d(np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32))
k_tt = to_dev_4d(np.random.randn(1, n_kv_heads, 1, head_dim).astype(np.float32))

cos_tt = to_dev_4d(cos_cache.reshape(1, 1, MAX_SEQ, head_dim))
sin_tt = to_dev_4d(sin_cache.reshape(1, 1, MAX_SEQ, head_dim))
trans_tt = to_dev_4d(trans_mat)

try:
    t0 = time.perf_counter()
    q_rot, k_rot = ttnn.experimental.rotary_embedding_llama_fused_qk(
        q_tt, k_tt, cos_tt, sin_tt, trans_tt)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"  Fused QK: OK! {(t1-t0)*1000:.1f}ms")
    print(f"  Q_rot shape: {q_rot.shape}")
    print(f"  K_rot shape: {k_rot.shape}")
except Exception as e:
    err = str(e)[:300]
    print(f"  FAILED: {err}")


# ══════════════════════════════════════════════════════════════
# TEST 3: Compare against our current approach (rotation matrix)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 3: Benchmark rotation matrix vs native RoPE (if available)")
print("="*60)

# Our current approach: rotation matrix
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

cos_buf = to_dev_4d(np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32))
sin_buf = to_dev_4d(np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32))

q_tt = to_dev_4d(np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32))

# Benchmark rotation matrix approach
times_rot = []
for _ in range(20):
    t0 = time.perf_counter()
    qr = ttnn.add(ttnn.mul(q_tt, cos_buf), ttnn.mul(ttnn.matmul(q_tt, R_tt), sin_buf))
    ttnn.synchronize_device(device)
    times_rot.append(time.perf_counter() - t0)
avg_rot = np.mean(times_rot[2:]) * 1000
print(f"  Rotation matrix (4 ops): avg {avg_rot:.3f}ms")
print(f"  Per Q+K (8 ops): {avg_rot*2:.3f}ms")
print(f"  Across 32 layers: {avg_rot*2*32:.1f}ms")

ttnn.close_device(device)
print("\nDone!")
