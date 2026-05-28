#!/usr/bin/env python3
"""
Experiment 48: Test native on-device RoPE APIs.

Experiment 45 discovered these APIs:
  - ttnn.experimental.rotary_embedding
  - ttnn.experimental.rotary_embedding_llama
  - ttnn.experimental.rotary_embedding_llama_fused_qk

Test each on Qwen-shaped tensors (B=1, T=5, n_q_heads=14, n_kv_heads=2, head_dim=64)
and compare against our numpy RoPE reference.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn
import inspect

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)

# Config
B, T = 1, 5
n_q_heads, n_kv_heads, head_dim = 14, 2, 64
rope_theta = 1000000.0

print("=" * 60)
print("Experiment 48: Native On-Device RoPE APIs")
print("=" * 60)

# ── Numpy RoPE reference ─────────────────────────────────────
np.random.seed(42)
q_np = np.random.randn(B, n_q_heads, T, head_dim).astype(np.float32) * 0.1
k_np = np.random.randn(B, n_kv_heads, T, head_dim).astype(np.float32) * 0.1

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(T, dtype=np.float32), freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope_np(x_4d):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_table[None, None, :, :] - x_4d[..., 1::2] * sin_table[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_table[None, None, :, :] + x_4d[..., 1::2] * cos_table[None, None, :, :]
    return out

q_ref = apply_rope_np(q_np)
k_ref = apply_rope_np(k_np)
print(f"Reference Q RoPE norm: {np.linalg.norm(q_ref):.4f}")
print(f"Reference K RoPE norm: {np.linalg.norm(k_ref):.4f}")

# ── Device ────────────────────────────────────────────────────
device = ttnn.open_device(device_id=0)

def to_dev(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(t, shape):
    return ttnn.to_torch(t).float().reshape(shape).numpy()

# ── Explore APIs ──────────────────────────────────────────────
print("\n── Available rotary embedding APIs ──")
apis = {}
for name in dir(ttnn.experimental):
    if 'rotar' in name.lower() or 'rope' in name.lower():
        func = getattr(ttnn.experimental, name)
        apis[name] = func
        print(f"  ttnn.experimental.{name}")
        try:
            sig = inspect.signature(func)
            print(f"    Signature: {sig}")
        except (ValueError, TypeError):
            print(f"    (signature not available)")

# Also check ttnn.transformer
for name in dir(ttnn.transformer):
    if 'rotar' in name.lower() or 'rope' in name.lower():
        func = getattr(ttnn.transformer, name)
        apis[f"transformer.{name}"] = func
        print(f"  ttnn.transformer.{name}")
        try:
            sig = inspect.signature(func)
            print(f"    Signature: {sig}")
        except (ValueError, TypeError):
            print(f"    (signature not available)")

# ── Precompute cos/sin in various formats ─────────────────────
# Format 1: (1, 1, T, head_dim) with interleaved cos/sin
cos_interleaved = np.zeros((1, 1, T, head_dim), dtype=np.float32)
sin_interleaved = np.zeros((1, 1, T, head_dim), dtype=np.float32)
cos_interleaved[..., 0::2] = cos_table[None, None, :, :]
cos_interleaved[..., 1::2] = cos_table[None, None, :, :]
sin_interleaved[..., 0::2] = sin_table[None, None, :, :]
sin_interleaved[..., 1::2] = sin_table[None, None, :, :]

# Format 2: (1, T, 1, head_dim) — some APIs might expect this
cos_f2 = cos_interleaved.transpose(0, 2, 1, 3)  # (1, T, 1, head_dim)
sin_f2 = sin_interleaved.transpose(0, 2, 1, 3)

# Format 3: (T, head_dim//2) — raw angles
cos_raw = cos_table  # (T, head_dim//2)
sin_raw = sin_table

# ── Test each API ─────────────────────────────────────────────

# Test 1: ttnn.experimental.rotary_embedding
print("\n── Test 1: ttnn.experimental.rotary_embedding ──")
if hasattr(ttnn.experimental, 'rotary_embedding'):
    func = ttnn.experimental.rotary_embedding
    # Try different argument patterns
    q_tt = to_dev(q_np)

    # Pattern A: (input, cos, sin)
    for fmt_name, cos_arr, sin_arr in [
        ("interleaved 4d", cos_interleaved, sin_interleaved),
        ("transposed 4d", cos_f2, sin_f2),
    ]:
        try:
            cos_tt = to_dev(cos_arr)
            sin_tt = to_dev(sin_arr)
            q_tt = to_dev(q_np)
            out = func(q_tt, cos_tt, sin_tt)
            out_np = from_dev(out, q_ref.shape)
            cos_val = cosine(out_np, q_ref)
            print(f"  Pattern (input, cos_{fmt_name}, sin): cosine={cos_val:.6f}")
        except Exception as e:
            print(f"  Pattern (input, cos_{fmt_name}, sin): FAILED — {str(e)[:120]}")

    # Pattern B: (input, cos, sin, token_idx) — for decode mode
    for token_idx in [0, T-1]:
        try:
            cos_tt = to_dev(cos_interleaved)
            sin_tt = to_dev(sin_interleaved)
            q_tt = to_dev(q_np)
            out = func(q_tt, cos_tt, sin_tt, token_idx)
            out_np = from_dev(out, q_ref.shape)
            cos_val = cosine(out_np, q_ref)
            print(f"  Pattern (input, cos, sin, token_idx={token_idx}): cosine={cos_val:.6f}")
        except Exception as e:
            print(f"  Pattern (input, cos, sin, token_idx={token_idx}): FAILED — {str(e)[:120]}")

# Test 2: ttnn.experimental.rotary_embedding_llama
print("\n── Test 2: ttnn.experimental.rotary_embedding_llama ──")
if hasattr(ttnn.experimental, 'rotary_embedding_llama'):
    func = ttnn.experimental.rotary_embedding_llama

    # Llama-style RoPE uses split-half, not interleaved
    # x1 = x[..., :head_dim//2], x2 = x[..., head_dim//2:]
    # rotated = cat(-x2, x1) * sin + x * cos
    # Try with cos/sin in (1, 1, T, head_dim) format
    for fmt_name, cos_arr, sin_arr in [
        ("interleaved 4d", cos_interleaved, sin_interleaved),
    ]:
        try:
            cos_tt = to_dev(cos_arr)
            sin_tt = to_dev(sin_arr)
            q_tt = to_dev(q_np)
            out = func(q_tt, cos_tt, sin_tt)
            out_np = from_dev(out, q_ref.shape)
            cos_val = cosine(out_np, q_ref)
            print(f"  Pattern (input, cos_{fmt_name}, sin): cosine={cos_val:.6f}")
        except Exception as e:
            print(f"  Pattern (input, cos_{fmt_name}, sin): FAILED — {str(e)[:120]}")

    # Try with transformation matrix (mentioned in research)
    try:
        # Some APIs need a trans_mat for the rotation
        trans_mat = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
        for j in range(head_dim // 2):
            trans_mat[0, 0, j, j + head_dim//2] = 1.0
            trans_mat[0, 0, j + head_dim//2, j] = -1.0
        trans_tt = to_dev(trans_mat)
        cos_tt = to_dev(cos_interleaved)
        sin_tt = to_dev(sin_interleaved)
        q_tt = to_dev(q_np)
        out = func(q_tt, cos_tt, sin_tt, trans_tt)
        out_np = from_dev(out, q_ref.shape)
        cos_val = cosine(out_np, q_ref)
        print(f"  Pattern (input, cos, sin, trans_mat): cosine={cos_val:.6f}")
    except Exception as e:
        print(f"  Pattern (input, cos, sin, trans_mat): FAILED — {str(e)[:120]}")

# Test 3: ttnn.experimental.rotary_embedding_llama_fused_qk
print("\n── Test 3: ttnn.experimental.rotary_embedding_llama_fused_qk ──")
if hasattr(ttnn.experimental, 'rotary_embedding_llama_fused_qk'):
    func = ttnn.experimental.rotary_embedding_llama_fused_qk

    # This should take both Q and K and apply RoPE to both at once
    try:
        cos_tt = to_dev(cos_interleaved)
        sin_tt = to_dev(sin_interleaved)
        q_tt = to_dev(q_np)
        k_tt = to_dev(k_np)
        out = func(q_tt, k_tt, cos_tt, sin_tt)
        if isinstance(out, tuple) and len(out) == 2:
            q_out_np = from_dev(out[0], q_ref.shape)
            k_out_np = from_dev(out[1], k_ref.shape)
            q_cos = cosine(q_out_np, q_ref)
            k_cos = cosine(k_out_np, k_ref)
            print(f"  Q cosine: {q_cos:.6f}, K cosine: {k_cos:.6f}")
        else:
            print(f"  Output type: {type(out)}, not a 2-tuple")
    except Exception as e:
        print(f"  Pattern (q, k, cos, sin): FAILED — {str(e)[:120]}")

    # Try with trans_mat
    try:
        trans_mat = np.zeros((1, 1, head_dim, head_dim), dtype=np.float32)
        for j in range(head_dim // 2):
            trans_mat[0, 0, j, j + head_dim//2] = 1.0
            trans_mat[0, 0, j + head_dim//2, j] = -1.0
        trans_tt = to_dev(trans_mat)
        cos_tt = to_dev(cos_interleaved)
        sin_tt = to_dev(sin_interleaved)
        q_tt = to_dev(q_np)
        k_tt = to_dev(k_np)
        out = func(q_tt, k_tt, cos_tt, sin_tt, trans_tt)
        if isinstance(out, tuple) and len(out) == 2:
            q_out_np = from_dev(out[0], q_ref.shape)
            k_out_np = from_dev(out[1], k_ref.shape)
            q_cos = cosine(q_out_np, q_ref)
            k_cos = cosine(k_out_np, k_ref)
            print(f"  Q cosine: {q_cos:.6f}, K cosine: {k_cos:.6f}")
        else:
            print(f"  Output type: {type(out)}, not a 2-tuple")
    except Exception as e:
        print(f"  Pattern (q, k, cos, sin, trans_mat): FAILED — {str(e)[:120]}")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Goal: Find a native RoPE API that eliminates the Q/K CPU round-trip")
print("If any test above shows cosine > 0.99, we can use it in Qwen!")

ttnn.close_device(device)
print("\nDone!")
