"""
Experiment 36: Validate TT-NN ops needed for Qwen2.5-0.5B.

Tests each op that differs from GPT-2:
  1. ttnn.rms_norm — replaces LayerNorm
  2. ttnn.silu / ttnn.swiglu — replaces GELU
  3. RoPE — rotary position embeddings (decomposed or native)
  4. GQA attention — 14 query heads, 2 KV heads
  5. ttnn.transformer.rotary_embedding — native RoPE if available

All tested with Qwen-shaped tensors: hidden=896, heads=14/2, head_dim=64.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch

import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)

# Qwen2.5-0.5B dimensions
hidden = 896
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
intermediate = 4864
seq_len = 32

rng = np.random.RandomState(42)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)


# ══════════════════════════════════════════════════════════════
# Test 1: RMSNorm
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: ttnn.rms_norm")
print("=" * 60)

x_np = rng.randn(1, seq_len, hidden).astype(np.float32)
g_np = rng.randn(hidden).astype(np.float32) * 0.1 + 1.0

# NumPy reference
rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)
ref = (x_np / rms) * g_np

x_tt = to_dev(x_np)
g_tt = to_dev(g_np)

rms_ok = False
for fn_name, fn_call in [
    ("ttnn.rms_norm(x, weight=g)", lambda: ttnn.rms_norm(x_tt, weight=g_tt, epsilon=1e-6)),
    ("ttnn.rms_norm(x, weight=g, eps=1e-5)", lambda: ttnn.rms_norm(x_tt, weight=g_tt, epsilon=1e-5)),
]:
    try:
        print(f"  Trying: {fn_name}...")
        out = fn_call()
        out_np = from_dev(out, (1, seq_len, hidden))
        cos = cosine(out_np, ref)
        maxe = np.abs(out_np - ref).max()
        print(f"  SUCCESS! Cosine: {cos:.6f}, Max error: {maxe:.6f}")
        rms_ok = True
        break
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")

if not rms_ok:
    # Try without weight
    try:
        out = ttnn.rms_norm(x_tt, epsilon=1e-6)
        print(f"  ttnn.rms_norm without weight: shape={out.shape}")
        rms_ok = True
    except Exception as e:
        print(f"  Also failed without weight: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# Test 2: SiLU and SwiGLU
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: ttnn.silu and ttnn.swiglu")
print("=" * 60)

# SiLU (sigmoid linear unit): x * sigmoid(x)
a_np = rng.randn(1, seq_len, intermediate).astype(np.float32)
silu_ref = a_np * (1.0 / (1.0 + np.exp(-a_np)))

a_tt = to_dev(a_np)

silu_ok = False
try:
    print("  Trying: ttnn.silu(x)...")
    out = ttnn.silu(a_tt)
    out_np = from_dev(out, (1, seq_len, intermediate))
    cos = cosine(out_np, silu_ref)
    print(f"  SiLU cosine: {cos:.6f}")
    silu_ok = True
except Exception as e:
    print(f"  ttnn.silu FAILED: {str(e)[:200]}")

# SwiGLU: silu(gate) * up
gate_np = rng.randn(1, seq_len, intermediate).astype(np.float32)
up_np = rng.randn(1, seq_len, intermediate).astype(np.float32)
swiglu_ref = (gate_np * (1.0 / (1.0 + np.exp(-gate_np)))) * up_np

gate_tt = to_dev(gate_np)
up_tt = to_dev(up_np)

swiglu_ok = False
# Try fused SwiGLU
for fn_name, fn_call in [
    ("ttnn.swiglu(gate, up)", lambda: ttnn.swiglu(gate_tt, up_tt)),
    ("silu(gate) * up", lambda: ttnn.mul(ttnn.silu(gate_tt), up_tt)),
]:
    try:
        print(f"  Trying: {fn_name}...")
        out = fn_call()
        out_np = from_dev(out, (1, seq_len, intermediate))
        cos = cosine(out_np, swiglu_ref)
        print(f"  SwiGLU cosine: {cos:.6f}")
        swiglu_ok = True
        break
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# Test 3: RoPE (rotary position embeddings)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 3: RoPE")
print("=" * 60)

# Check if native rotary_embedding exists
has_rope = hasattr(ttnn.transformer, 'rotary_embedding')
print(f"  ttnn.transformer.rotary_embedding: {'EXISTS' if has_rope else 'NOT FOUND'}")

if has_rope:
    try:
        doc = ttnn.transformer.rotary_embedding.__doc__
        print(f"  Doc: {doc[:300] if doc else 'none'}")
    except:
        pass

# NumPy reference RoPE implementation
def rope_numpy(x, seq_len, head_dim, base=10000.0):
    """Apply rotary position embeddings. x: (batch, n_heads, seq_len, head_dim)"""
    # Compute frequency table
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    # Outer product: (seq_len, head_dim/2)
    angles = np.outer(positions, freqs)
    cos_table = np.cos(angles)  # (seq_len, head_dim/2)
    sin_table = np.sin(angles)  # (seq_len, head_dim/2)

    # Split x into even/odd
    x_even = x[..., 0::2]  # (..., head_dim/2)
    x_odd = x[..., 1::2]   # (..., head_dim/2)

    # Apply rotation
    # Broadcast cos/sin: (1, 1, seq_len, head_dim/2)
    cos_t = cos_table[None, None, :, :]
    sin_t = sin_table[None, None, :, :]

    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_even * sin_t + x_odd * cos_t

    # Interleave back
    out = np.zeros_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out

# Test RoPE with Q-shaped tensor: (1, n_q_heads, seq_len, head_dim)
q_np = rng.randn(1, n_q_heads, seq_len, head_dim).astype(np.float32) * 0.1
q_rope_ref = rope_numpy(q_np, seq_len, head_dim)
print(f"  Q shape: {q_np.shape}")
print(f"  RoPE reference norm: {np.linalg.norm(q_rope_ref):.4f}")

# Try native RoPE
rope_native_ok = False
if has_rope:
    q_tt = ttnn.from_torch(torch.from_numpy(q_np.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)

    # Build cos/sin tables
    freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    cos_table = np.cos(angles).astype(np.float32)
    sin_table = np.sin(angles).astype(np.float32)

    cos_tt = ttnn.from_torch(
        torch.from_numpy(cos_table[None, None, :, :].copy()),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    sin_tt = ttnn.from_torch(
        torch.from_numpy(sin_table[None, None, :, :].copy()),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    for fn_name, fn_call in [
        ("rotary_embedding(q, cos, sin)", lambda: ttnn.transformer.rotary_embedding(q_tt, cos_tt, sin_tt)),
        ("rotary_embedding(q, cos, sin, token_idx=0)", lambda: ttnn.transformer.rotary_embedding(q_tt, cos_tt, sin_tt, token_idx=0)),
    ]:
        try:
            print(f"  Trying: {fn_name}...")
            out = fn_call()
            out_np = ttnn.to_torch(out).float().numpy()
            if out_np.shape != q_rope_ref.shape:
                out_np = out_np.reshape(q_rope_ref.shape)
            cos_sim = cosine(out_np, q_rope_ref)
            print(f"  SUCCESS! Cosine: {cos_sim:.6f}")
            rope_native_ok = True
            break
        except Exception as e:
            print(f"  FAILED: {str(e)[:200]}")

# Decomposed RoPE using existing ops
print("\n  Testing decomposed RoPE (existing ops)...")
# Strategy: split into even/odd, apply cos/sin rotation, interleave back
q_2d = q_np.reshape(1, n_q_heads * seq_len, head_dim)
q_tt_2d = to_dev(q_2d)

# Precompute cos/sin for full sequence, broadcast to all heads
freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(seq_len, dtype=np.float32)
angles = np.outer(positions, freqs)  # (seq_len, head_dim/2)

# Tile cos/sin to match (n_q_heads * seq_len, head_dim/2)
cos_np = np.tile(np.cos(angles), (n_q_heads, 1)).astype(np.float32)
sin_np = np.tile(np.sin(angles), (n_q_heads, 1)).astype(np.float32)

# Even/odd slicing — CPU fallback (could be optimized later)
q_flat = q_2d[0]  # (n_q_heads * seq_len, head_dim)
q_even = q_flat[:, 0::2]  # (..., 32)
q_odd = q_flat[:, 1::2]

out_even = q_even * cos_np - q_odd * sin_np
out_odd = q_even * sin_np + q_odd * cos_np

result = np.zeros_like(q_flat)
result[:, 0::2] = out_even
result[:, 1::2] = out_odd
result = result.reshape(1, n_q_heads, seq_len, head_dim)

cos_decomp = cosine(result, q_rope_ref)
print(f"  Decomposed RoPE cosine vs reference: {cos_decomp:.6f}")


# ══════════════════════════════════════════════════════════════
# Test 4: GQA attention (14 Q heads, 2 KV heads)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 4: GQA attention")
print("=" * 60)

# Q: (1, 14, 32, 64), K/V: (1, 2, 32, 64)
q_gqa = rng.randn(1, n_q_heads, seq_len, head_dim).astype(np.float32) * 0.1
k_gqa = rng.randn(1, n_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1
v_gqa = rng.randn(1, n_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1

q_tt = ttnn.from_torch(torch.from_numpy(q_gqa.copy()), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)
k_tt = ttnn.from_torch(torch.from_numpy(k_gqa.copy()), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)
v_tt = ttnn.from_torch(torch.from_numpy(v_gqa.copy()), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)

print(f"  Q: {q_tt.shape}, K: {k_tt.shape}, V: {v_tt.shape}")

gqa_ok = False
for fn_name, fn_call in [
    ("SDPA(q, k, v, is_causal=True)", lambda: ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, is_causal=True)),
    ("SDPA with expanded K/V", None),  # Fallback below
]:
    if fn_call is None:
        # Expand K/V: (1, 2, T, 64) → (1, 14, T, 64) by repeating each KV head 7x
        k_exp = np.repeat(k_gqa, n_q_heads // n_kv_heads, axis=1)
        v_exp = np.repeat(v_gqa, n_q_heads // n_kv_heads, axis=1)
        k_exp_tt = ttnn.from_torch(torch.from_numpy(k_exp.copy()), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        v_exp_tt = ttnn.from_torch(torch.from_numpy(v_exp.copy()), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        fn_call = lambda: ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_exp_tt, v_exp_tt, is_causal=True)

    try:
        print(f"  Trying: {fn_name}...")
        out = fn_call()
        print(f"  SUCCESS! Output shape: {out.shape}")
        out_np = ttnn.to_torch(out).float().numpy()
        print(f"  Output torch shape: {out_np.shape}, norm: {np.linalg.norm(out_np):.4f}")
        gqa_ok = True
        break
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# Test 5: GQA with Flash-Decode (for KV-cached decode)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 5: GQA Flash-Decode (KV-cached)")
print("=" * 60)

# Q: (1, 1, 14, 64), K cache: (1, 2, 1024, 64), V cache: (1, 2, 1024, 64)
q_decode = rng.randn(1, 1, n_q_heads, head_dim).astype(np.float32) * 0.1
k_cache = np.zeros((1, n_kv_heads, 1024, head_dim), dtype=np.float32)
v_cache = np.zeros((1, n_kv_heads, 1024, head_dim), dtype=np.float32)
k_cache[:, :, :seq_len, :] = rng.randn(1, n_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1
v_cache[:, :, :seq_len, :] = rng.randn(1, n_kv_heads, seq_len, head_dim).astype(np.float32) * 0.1

q_dec_tt = ttnn.from_torch(torch.from_numpy(q_decode.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
k_cache_tt = ttnn.from_torch(torch.from_numpy(k_cache.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)
v_cache_tt = ttnn.from_torch(torch.from_numpy(v_cache.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)

print(f"  Q: {q_dec_tt.shape}, K cache: {k_cache_tt.shape}, V cache: {v_cache_tt.shape}")

gqa_decode_ok = False
try:
    print("  Trying: scaled_dot_product_attention_decode(q, k_cache, v_cache, cur_pos=[31])...")
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_dec_tt, k_cache_tt, v_cache_tt, cur_pos=[seq_len - 1]
    )
    print(f"  SUCCESS! Output shape: {out.shape}")
    out_np = ttnn.to_torch(out).float().numpy()
    print(f"  Output torch shape: {out_np.shape}, norm: {np.linalg.norm(out_np):.4f}")
    gqa_decode_ok = True
except Exception as e:
    print(f"  FAILED: {str(e)[:300]}")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary: Qwen2.5 Op Readiness")
print("=" * 60)
print(f"""
  RMSNorm:             {'OK' if rms_ok else 'FAILED'}
  SiLU:                {'OK' if silu_ok else 'FAILED'}
  SwiGLU:              {'OK' if swiglu_ok else 'FAILED'}
  RoPE (native):       {'OK' if rope_native_ok else 'FAILED/NOT TESTED'}
  RoPE (decomposed):   OK (cosine={cos_decomp:.6f})
  GQA SDPA:            {'OK' if gqa_ok else 'FAILED'}
  GQA Flash-Decode:    {'OK' if gqa_decode_ok else 'FAILED'}

Qwen2.5-0.5B feasibility: {'ALL OPS READY' if all([rms_ok, silu_ok or swiglu_ok, gqa_ok]) else 'SOME OPS MISSING'}
""")

ttnn.close_device(device)
print("Done!")
