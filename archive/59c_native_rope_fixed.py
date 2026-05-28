#!/usr/bin/env python3
"""
Experiment 59c: Fix native rotary_embedding_llama — trans_mat must be 4D.

From 59b: rotary_embedding_llama asserts trans_mat.logical_shape()[0] == 1 && [1] == 1.
Our 2D (head_dim, head_dim) gets padded to (1, head_dim) — fails.
Need (1, 1, head_dim, head_dim) 4D tensor.

Also discovered: ttnn.experimental.rotary_embedding (non-llama) works!
This test tries both with proper shapes and measures timing.
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

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor):
    return ttnn.to_torch(tensor).float().numpy()


# ── Test data ──
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
pos = 42
angles = pos * freqs

x_np = np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32)
x_tt = to_dev_4d(x_np)

# ── Half-format cos/sin (what Qwen uses) ──
cos_half = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
sin_half = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)

# ── Interleaved cos/sin ──
cos_interleaved = np.repeat(np.cos(angles), 2).reshape(1, 1, 1, head_dim).astype(np.float32)
sin_interleaved = np.repeat(np.sin(angles), 2).reshape(1, 1, 1, head_dim).astype(np.float32)

# ── Trans mat for half-format: rotate_half ──
# rotate_half: [-x[32:], x[:32]] -> matrix form
trans_half = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    trans_half[0, 0, i + half_dim, i] = -1.0  # result[i] = -x[i+32]... wait
    trans_half[0, 0, i, i + half_dim] = 1.0

# Actually rotate_half: result[i] = -x[i+half_dim] for i < half_dim
#                       result[i+half_dim] = x[i] for i < half_dim
# As matrix: result = x @ R where R[j,i] gives contribution of x[j] to result[i]
# result[i] = sum_j x[j] * R[j,i]
# For i < half_dim: result[i] = -x[i+half_dim] so R[i+half_dim, i] = -1
# For i >= half_dim: result[i] = x[i-half_dim] so R[i-half_dim, i] = 1
trans_half = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    trans_half[0, 0, i + half_dim, i] = -1.0    # result[i] = -x[i+32]
    trans_half[0, 0, i, i + half_dim] = 1.0     # result[i+32] = x[i]

# ── Trans mat for interleaved: rotate_interleaved ──
trans_interleaved = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    trans_interleaved[0, 0, 2*i+1, 2*i] = -1.0    # result[2i] = -x[2i+1]
    trans_interleaved[0, 0, 2*i, 2*i+1] = 1.0      # result[2i+1] = x[2i]


# ── Test 1: rotary_embedding_llama with 4D trans_mat ──
print(f"\n{'='*60}")
print("Test 1: rotary_embedding_llama with 4D trans_mat")
print(f"{'='*60}")

for name, cos_np, sin_np, trans_np in [
    ("half-format", cos_half, sin_half, trans_half),
    ("interleaved", cos_interleaved, sin_interleaved, trans_interleaved),
]:
    cos_tt = to_dev_4d(cos_np)
    sin_tt = to_dev_4d(sin_np)
    trans_tt = to_dev_4d(trans_np)

    print(f"\n  {name}:")
    print(f"    x shape: {x_np.shape}")
    print(f"    cos shape: {cos_np.shape}")
    print(f"    trans_mat shape: {trans_np.shape}")

    try:
        result = ttnn.experimental.rotary_embedding_llama(
            x_tt, cos_tt, sin_tt, trans_tt, is_decode_mode=True)
        result_np = from_dev(result)
        print(f"    SUCCESS! Output shape: {result_np.shape}")
        print(f"    Output[:5]: {result_np.flatten()[:5]}")

        # Verify: compute numpy reference
        if name == "half-format":
            x_rot = np.zeros_like(x_np)
            x_rot[..., :half_dim] = -x_np[..., half_dim:]
            x_rot[..., half_dim:] = x_np[..., :half_dim]
        else:
            x_rot = np.zeros_like(x_np)
            x_rot[..., 0::2] = -x_np[..., 1::2]
            x_rot[..., 1::2] = x_np[..., 0::2]

        ref = x_np * cos_np + x_rot * sin_np
        cos_sim = np.dot(result_np.flatten()[:head_dim*n_q_heads], ref.flatten()) / (
            np.linalg.norm(result_np.flatten()[:head_dim*n_q_heads]) * np.linalg.norm(ref.flatten()))
        print(f"    Cosine vs numpy: {cos_sim:.6f}")

    except Exception as e:
        err_msg = str(e).split('\n')[0]
        print(f"    FAILED: {err_msg}")


# ── Test 2: rotary_embedding (non-llama) ──
print(f"\n{'='*60}")
print("Test 2: rotary_embedding (non-llama)")
print(f"{'='*60}")

# The non-llama version — what cos/sin shape does it want?
# From 59b it returned (1, 14, 32, 64) from (1, 14, 1, 64) input
# The seq_len=1 got padded to 32 (tile size)
cos_tt = to_dev_4d(cos_half)
sin_tt = to_dev_4d(sin_half)

try:
    result = ttnn.experimental.rotary_embedding(x_tt, cos_tt, sin_tt)
    result_np = from_dev(result)
    print(f"  Output shape: {result_np.shape}")

    # Extract just the first seq position
    if result_np.ndim == 4 and result_np.shape[2] > 1:
        result_np = result_np[:, :, 0:1, :]

    x_rot_half = np.zeros_like(x_np)
    x_rot_half[..., :half_dim] = -x_np[..., half_dim:]
    x_rot_half[..., half_dim:] = x_np[..., :half_dim]
    ref = x_np * cos_half + x_rot_half * sin_half
    cos_sim = np.dot(result_np.flatten(), ref.flatten()) / (
        np.linalg.norm(result_np) * np.linalg.norm(ref))
    print(f"  Cosine vs numpy half-format ref: {cos_sim:.6f}")
    print(f"  Output[:5]: {result_np.flatten()[:5]}")
    print(f"  Ref[:5]:    {ref.flatten()[:5]}")
except Exception as e:
    print(f"  FAILED: {e}")


# ── Timing comparison ──
print(f"\n{'='*60}")
print("Timing comparison")
print(f"{'='*60}")

# Current rotation matrix approach
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)
cos_h_tt = to_dev_4d(cos_half)
sin_h_tt = to_dev_4d(sin_half)

# Warmup
for _ in range(10):
    q_rot = ttnn.matmul(x_tt, R_tt)
    q_roped = ttnn.add(ttnn.mul(x_tt, cos_h_tt), ttnn.mul(q_rot, sin_h_tt))
    ttnn.synchronize_device(device)

times_rm = []
for _ in range(100):
    t0 = time.perf_counter()
    q_rot = ttnn.matmul(x_tt, R_tt)
    q_roped = ttnn.add(ttnn.mul(x_tt, cos_h_tt), ttnn.mul(q_rot, sin_h_tt))
    ttnn.synchronize_device(device)
    times_rm.append(time.perf_counter() - t0)

avg_rm = np.mean(times_rm[10:]) * 1000
print(f"  Rotation matrix:  {avg_rm:.3f}ms/call")

# Try native ops
for op_name, op_fn, extra_args in [
    ("rotary_embedding (non-llama)",
     lambda: ttnn.experimental.rotary_embedding(x_tt, cos_h_tt, sin_h_tt), []),
]:
    try:
        # Warmup
        for _ in range(10):
            _ = op_fn()
            ttnn.synchronize_device(device)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = op_fn()
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)

        avg = np.mean(times[10:]) * 1000
        speedup = avg_rm / avg
        print(f"  {op_name}: {avg:.3f}ms/call ({speedup:.1f}x {'faster' if speedup > 1 else 'slower'})")
    except Exception as e:
        print(f"  {op_name}: FAILED — {str(e)[:80]}")

# Try rotary_embedding_llama with properly shaped trans_mat
for name, trans_np, cos_np, sin_np in [
    ("llama half-format", trans_half, cos_half, sin_half),
    ("llama interleaved", trans_interleaved, cos_interleaved, sin_interleaved),
]:
    trans_tt = to_dev_4d(trans_np)
    cos_t = to_dev_4d(cos_np)
    sin_t = to_dev_4d(sin_np)
    try:
        # Warmup
        for _ in range(10):
            _ = ttnn.experimental.rotary_embedding_llama(x_tt, cos_t, sin_t, trans_tt, is_decode_mode=True)
            ttnn.synchronize_device(device)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = ttnn.experimental.rotary_embedding_llama(x_tt, cos_t, sin_t, trans_tt, is_decode_mode=True)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)

        avg = np.mean(times[10:]) * 1000
        speedup = avg_rm / avg
        print(f"  {name}: {avg:.3f}ms/call ({speedup:.1f}x {'faster' if speedup > 1 else 'slower'})")
    except Exception as e:
        print(f"  {name}: FAILED — {str(e).split(chr(10))[0][:80]}")


print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  If native RoPE works and is faster, we can eliminate the rotation")
print(f"  matrix matmul (1 matmul + 2 mul + 1 add) per Q and K projection.")

ttnn.close_device(device)
print("\nDone!")
