#!/usr/bin/env python3
"""
Experiment 51: On-device Qwen decode — eliminate CPU round-trips.

Current bottleneck: 192+ CPU↔device transfers per decode step for RoPE
and tensor reshaping. Each transfer ~0.1ms → ~19ms wasted per step.

Approach:
  1. Switch from interleaved to half-format RoPE (split at head_dim//2)
     Half-format only needs split/neg/concat — all tile-aligned at 32
  2. Pre-upload cos/sin tables for all positions to device
  3. Keep residual stream on device between layers (no CPU round-trip)
  4. Reshape attention output on device

Phase 1: Test critical ttnn ops (split, neg, concat on tiled 4D tensors)
Phase 2: Numpy reference with half-format RoPE (correctness baseline)
Phase 3: Full on-device decode
Phase 4: Speedup measurement
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=20)
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2  # 32
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936
MAX_SEQ = 256

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# ── Load weights ─────────────────────────────────────────────
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights["lm_head.weight"].T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ── Device + helpers ─────────────────────────────────────────
device = ttnn.open_device(device_id=0)

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
# PHASE 1: Test critical ttnn ops
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1: Testing ttnn ops for on-device RoPE")

# Test tensor: (1, 14, 1, 64) — Q shape during decode
test_data = np.random.randn(1, 14, 1, 64).astype(np.float32)
test_tt = to_dev_4d(test_data)
print(f"  Test tensor shape: {test_tt.shape}")

# Test 1: Split along last dim at midpoint
ops_ok = {}
try:
    # Try ttnn.split
    halves = ttnn.split(test_tt, 2, dim=-1)
    first_half = halves[0]
    second_half = halves[1]
    f_np = from_dev(first_half, (1, 14, 1, 32))
    s_np = from_dev(second_half, (1, 14, 1, 32))
    assert np.allclose(f_np, test_data[:, :, :, :32], atol=0.05), "split data mismatch"
    ops_ok['split'] = True
    print(f"  ✓ ttnn.split works: {first_half.shape} + {second_half.shape}")
except Exception as e:
    ops_ok['split'] = False
    print(f"  ✗ ttnn.split failed: {e}")

# Test 2: Negate
try:
    neg_tt = ttnn.neg(test_tt)
    neg_np = from_dev(neg_tt, (1, 14, 1, 64))
    assert np.allclose(neg_np, -test_data, atol=0.05), "neg data mismatch"
    ops_ok['neg'] = True
    print(f"  ✓ ttnn.neg works")
except Exception as e:
    try:
        neg_tt = ttnn.multiply(test_tt, -1.0)
        neg_np = from_dev(neg_tt, (1, 14, 1, 64))
        assert np.allclose(neg_np, -test_data, atol=0.05)
        ops_ok['neg'] = 'multiply'
        print(f"  ✓ ttnn.multiply(-1) works as neg")
    except Exception as e2:
        ops_ok['neg'] = False
        print(f"  ✗ neg failed: {e}, multiply: {e2}")

# Test 3: Concat along last dim
try:
    if ops_ok['split']:
        concat_tt = ttnn.concat([first_half, second_half], dim=-1)
        concat_np = from_dev(concat_tt, (1, 14, 1, 64))
        assert np.allclose(concat_np, test_data, atol=0.05), "concat data mismatch"
        ops_ok['concat'] = True
        print(f"  ✓ ttnn.concat works: {concat_tt.shape}")
except Exception as e:
    ops_ok['concat'] = False
    print(f"  ✗ ttnn.concat failed: {e}")

# Test 4: Reshape for Q: (1, 1, 896) → (1, 14, 1, 64) and back
try:
    flat = to_dev(np.random.randn(1, 1, 896).astype(np.float32))
    reshaped = ttnn.reshape(flat, [1, 14, 1, 64])
    back = ttnn.reshape(reshaped, [1, 1, 1, 896])
    ops_ok['reshape_q'] = True
    print(f"  ✓ reshape (1,1,896)→(1,14,1,64)→(1,1,896)")
except Exception as e:
    ops_ok['reshape_q'] = False
    print(f"  ✗ reshape failed: {e}")

# Test 5: Reshape for K: (1, 1, 128) → (1, 2, 1, 64) and back
try:
    flat_k = to_dev(np.random.randn(1, 1, 128).astype(np.float32))
    reshaped_k = ttnn.reshape(flat_k, [1, 2, 1, 64])
    ops_ok['reshape_k'] = True
    print(f"  ✓ reshape (1,1,128)→(1,2,1,64)")
except Exception as e:
    ops_ok['reshape_k'] = False
    print(f"  ✗ reshape K failed: {e}")

# Test 6: Broadcast multiply (1,1,1,64) * (1,14,1,64)
try:
    cos_test = to_dev_4d(np.random.randn(1, 1, 1, 64).astype(np.float32))
    q_test = to_dev_4d(np.random.randn(1, 14, 1, 64).astype(np.float32))
    result = ttnn.mul(q_test, cos_test)
    ops_ok['broadcast_mul'] = True
    print(f"  ✓ broadcast mul (1,1,1,64)*(1,14,1,64)={result.shape}")
except Exception as e:
    ops_ok['broadcast_mul'] = False
    print(f"  ✗ broadcast mul failed: {e}")

# Test 7: Full rotate_half + RoPE on device
can_do_ondevice_rope = all(ops_ok.get(k) for k in ['split', 'neg', 'concat', 'broadcast_mul'])
print(f"\n  On-device RoPE feasible: {can_do_ondevice_rope}")
print(f"  Op status: {ops_ok}")

if not can_do_ondevice_rope:
    print("\n⚠ Cannot do full on-device RoPE. Will test partial optimization.")

# ══════════════════════════════════════════════════════════════
# PHASE 2: Numpy reference with HALF-format RoPE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Numpy reference (half-format RoPE)")

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def get_rope_tables_half(T):
    """Half-format: cos/sin are (T, head_dim) with duplicated halves."""
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)  # (T, half_dim)
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (T, head_dim)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos_full, sin_full

def rotate_half_np(x):
    """x: (..., head_dim). Split in half, negate second, swap."""
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope_half_np(x_4d, cos_t, sin_t):
    """Half-format RoPE. x: (B, heads, T, head_dim), cos/sin: (T, head_dim)."""
    return x_4d * cos_t[None, None, :, :] + rotate_half_np(x_4d) * sin_t[None, None, :, :]

# Also keep interleaved for comparison
def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return np.cos(angles), np.sin(angles)

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_t[None, None] - x_4d[..., 1::2] * sin_t[None, None]
    out[..., 1::2] = x_4d[..., 0::2] * sin_t[None, None] + x_4d[..., 1::2] * cos_t[None, None]
    return out

# Quick test: are both formats equivalent?
test_q = np.random.randn(1, 14, 5, 64).astype(np.float32)
cos_h, sin_h = get_rope_tables_half(5)
cos_i, sin_i = get_rope_tables_interleaved(5)
roped_half = apply_rope_half_np(test_q, cos_h, sin_h)
roped_inter = apply_rope_interleaved_np(test_q, cos_i, sin_i)
cos_sim = np.dot(roped_half.flatten(), roped_inter.flatten()) / (
    np.linalg.norm(roped_half.flatten()) * np.linalg.norm(roped_inter.flatten()))
print(f"  Half vs interleaved RoPE cosine: {cos_sim:.6f}")
print(f"  (1.0 = identical, <1.0 = different rotation)")

# Full numpy forward with half-format RoPE
def rms_norm_np(x, g, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms * g

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

prompt = args.prompt
tokens = tokenizer.encode(prompt)
T = len(tokens)

print(f"\n  Running numpy reference for '{prompt}' ({T} tokens)...")
ref_x = embed_w[np.array(tokens)].reshape(1, T, hidden)
cos_t, sin_t = get_rope_tables_half(T)

for i in range(n_layers):
    lw = layer_weights[i]
    h = rms_norm_np(ref_x, lw["input_layernorm.weight"], rms_eps)
    q = (h @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]).reshape(1, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k = (h @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]).reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (h @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]).reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q = apply_rope_half_np(q, cos_t, sin_t)
    k = apply_rope_half_np(k, cos_t, sin_t)
    # GQA: repeat K/V for each Q head group
    k_exp = np.repeat(k, n_q_heads // n_kv_heads, axis=1)
    v_exp = np.repeat(v, n_q_heads // n_kv_heads, axis=1)
    sc = (q @ k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
    sc += np.triu(np.ones((T,T))*-1e9, k=1)[None,None]
    e = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
    attn_w = e / np.sum(e, axis=-1, keepdims=True)
    attn_out = (attn_w @ v_exp).transpose(0,2,1,3).reshape(1, T, hidden)
    ref_x = ref_x + attn_out @ lw["self_attn.o_proj.weight"].T
    h2 = rms_norm_np(ref_x, lw["post_attention_layernorm.weight"], rms_eps)
    gate = silu_np(h2 @ lw["mlp.gate_proj.weight"].T)
    up = h2 @ lw["mlp.up_proj.weight"].T
    ref_x = ref_x + (gate * up) @ lw["mlp.down_proj.weight"].T

ref_x = rms_norm_np(ref_x, final_norm_g, rms_eps)
ref_logits = (ref_x @ embed_w.T).reshape(1, T, vocab_size)
ref_top5 = np.argsort(ref_logits[0, -1])[-5:][::-1]
print(f"  Reference top-5: {[tokenizer.decode([int(t)]) for t in ref_top5]}")

# Also run interleaved reference for comparison
ref_x_i = embed_w[np.array(tokens)].reshape(1, T, hidden)
cos_ti, sin_ti = get_rope_tables_interleaved(T)
for i in range(n_layers):
    lw = layer_weights[i]
    h = rms_norm_np(ref_x_i, lw["input_layernorm.weight"], rms_eps)
    q = (h @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]).reshape(1, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k = (h @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]).reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (h @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]).reshape(1, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q = apply_rope_interleaved_np(q, cos_ti, sin_ti)
    k = apply_rope_interleaved_np(k, cos_ti, sin_ti)
    k_exp = np.repeat(k, n_q_heads // n_kv_heads, axis=1)
    v_exp = np.repeat(v, n_q_heads // n_kv_heads, axis=1)
    sc = (q @ k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
    sc += np.triu(np.ones((T,T))*-1e9, k=1)[None,None]
    e = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
    attn_w = e / np.sum(e, axis=-1, keepdims=True)
    attn_out = (attn_w @ v_exp).transpose(0,2,1,3).reshape(1, T, hidden)
    ref_x_i = ref_x_i + attn_out @ lw["self_attn.o_proj.weight"].T
    h2 = rms_norm_np(ref_x_i, lw["post_attention_layernorm.weight"], rms_eps)
    gate = silu_np(h2 @ lw["mlp.gate_proj.weight"].T)
    up = h2 @ lw["mlp.up_proj.weight"].T
    ref_x_i = ref_x_i + (gate * up) @ lw["mlp.down_proj.weight"].T

ref_x_i = rms_norm_np(ref_x_i, final_norm_g, rms_eps)
ref_logits_i = (ref_x_i @ embed_w.T).reshape(1, T, vocab_size)
ref_top5_i = np.argsort(ref_logits_i[0, -1])[-5:][::-1]

cos_half_vs_inter = np.dot(ref_logits[0,-1].flatten(), ref_logits_i[0,-1].flatten()) / (
    np.linalg.norm(ref_logits[0,-1]) * np.linalg.norm(ref_logits_i[0,-1]))
print(f"  Interleaved top-5: {[tokenizer.decode([int(t)]) for t in ref_top5_i]}")
print(f"  Half vs interleaved logit cosine: {cos_half_vs_inter:.6f}")
print(f"  Top-1 match: {ref_top5[0] == ref_top5_i[0]}")

# ══════════════════════════════════════════════════════════════
# PHASE 3: Upload weights + pre-compute RoPE tables on device
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3: Upload weights + RoPE tables")

t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights[i]
    dev_layers.append({
        "ln1_g": to_dev(lw["input_layernorm.weight"]),
        "q_w": to_dev(lw["self_attn.q_proj.weight"].T),
        "q_b": to_dev(lw["self_attn.q_proj.bias"]),
        "k_w": to_dev(lw["self_attn.k_proj.weight"].T),
        "k_b": to_dev(lw["self_attn.k_proj.bias"]),
        "v_w": to_dev(lw["self_attn.v_proj.weight"].T),
        "v_b": to_dev(lw["self_attn.v_proj.bias"]),
        "o_w": to_dev(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_dev(lw["post_attention_layernorm.weight"]),
        "gate_w": to_dev(lw["mlp.gate_proj.weight"].T),
        "up_w": to_dev(lw["mlp.up_proj.weight"].T),
        "down_w": to_dev(lw["mlp.down_proj.weight"].T),
    })
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# Pre-compute cos/sin for all positions, upload to device
# Shape: (1, 1, 1, head_dim) for each position — broadcasts over heads
print("  Uploading RoPE cos/sin tables...")
rope_cos_tt = []
rope_sin_tt = []
for pos in range(MAX_SEQ):
    angles = pos * freqs  # (half_dim,)
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    rope_cos_tt.append(to_dev_4d(cos_full.reshape(1, 1, 1, head_dim)))
    rope_sin_tt.append(to_dev_4d(sin_full.reshape(1, 1, 1, head_dim)))
print(f"  {MAX_SEQ} position tables uploaded")

# ── KV Caches ────────────────────────────────────────────────
print("  Allocating KV caches...")
k_caches = []
v_caches = []
for i in range(n_layers):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))

# ══════════════════════════════════════════════════════════════
# PHASE 3b: On-device RoPE helper
# ══════════════════════════════════════════════════════════════

def rotate_half_tt(x_tt):
    """On-device rotate_half: split, negate, concat."""
    halves = ttnn.split(x_tt, 2, dim=-1)
    neg_second = ttnn.neg(halves[1])
    return ttnn.concat([neg_second, halves[0]], dim=-1)

def apply_rope_ondevice(x_tt, pos):
    """Apply half-format RoPE on device. x: (1, heads, 1, head_dim)."""
    cos_tt = rope_cos_tt[pos]
    sin_tt = rope_sin_tt[pos]
    # x * cos + rotate_half(x) * sin
    return ttnn.add(
        ttnn.mul(x_tt, cos_tt),
        ttnn.mul(rotate_half_tt(x_tt), sin_tt)
    )

# Quick test: on-device RoPE matches numpy?
if can_do_ondevice_rope:
    test_q_np = np.random.randn(1, 14, 1, 64).astype(np.float32)
    test_q_tt = to_dev_4d(test_q_np)

    # Numpy half-format at position 3
    pos_test = 3
    angles = pos_test * freqs
    cos_f = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, 64)
    sin_f = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, 64)
    np_result = test_q_np * cos_f + rotate_half_np(test_q_np) * sin_f

    # Device
    tt_result = apply_rope_ondevice(test_q_tt, pos_test)
    tt_result_np = from_dev(tt_result, (1, 14, 1, 64))

    rope_cos = np.dot(np_result.flatten(), tt_result_np.flatten()) / (
        np.linalg.norm(np_result.flatten()) * np.linalg.norm(tt_result_np.flatten()))
    print(f"\n  On-device RoPE test: cosine = {rope_cos:.6f} (should be >0.999)")

# ══════════════════════════════════════════════════════════════
# PHASE 4: On-device prefill (half-format RoPE, still with CPU RoPE
# since prefill has variable T and runs once)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4: Prefill + On-device Decode")

def prefill(token_ids):
    """Process full prompt, fill KV caches. Uses CPU RoPE (runs once)."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        # CPU RoPE for prefill (variable T, runs once)
        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_half_np(
            q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_half_np(
            k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        k_4d_tt = to_dev_4d(k_4d)
        v_4d_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), k_4d_tt, v_4d_tt,
            is_causal=True, compute_kernel_config=hifi4)

        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)

        x_np = from_dev(out_tt, (B, T, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, T, vocab_size))[0, -1]


def decode_step_ondevice(token_id, pos):
    """Single-token decode with on-device RoPE — zero CPU round-trips for RoPE."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        # Reshape Q/K/V to 4D on device
        q_4d = ttnn.reshape(q_tt, [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, n_kv_heads, 1, head_dim])

        # On-device RoPE!
        q_roped = apply_rope_ondevice(q_4d, pos)
        k_roped = apply_rope_ondevice(k_4d, pos)

        # Update KV caches
        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                               update_index=pos, batch_offset=0)

        # Flash-Decode: need Q as (1, 1, n_q_heads, head_dim)
        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos],
            compute_kernel_config=hifi4,
        )

        # Reshape attention output on device: (1, 1, n_q_heads, head_dim) → (1, 1, hidden)
        merged = ttnn.reshape(attn, [1, 1, 1, hidden])

        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)

        x_np = from_dev(out_tt, (B, 1, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    logits = from_dev(logits_tt, (B, 1, vocab_size))
    return logits[0, 0]


# Also keep the CPU-RoPE decode for comparison
def decode_step_cpu_rope(token_id, pos):
    """Original decode with CPU round-trips for RoPE (baseline)."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, 1, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, 1, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, 1, n_kv_heads * head_dim))

        angles = pos * freqs
        cos_f = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
        sin_f = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)

        q_4d = q_np.reshape(B, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k_4d = k_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_4d = v_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q_4d = q_4d * cos_f + rotate_half_np(q_4d) * sin_f
        k_4d = k_4d * cos_f + rotate_half_np(k_4d) * sin_f

        ttnn.kv_cache.update_cache_for_token_(k_caches[i], to_dev_4d(k_4d),
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], to_dev_4d(v_4d),
                                               update_index=pos, batch_offset=0)

        q_decode = to_dev_4d(q_4d.transpose(0, 2, 1, 3))
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos], compute_kernel_config=hifi4)

        attn_np = ttnn.to_torch(attn).float().numpy()
        merged = to_dev(attn_np.reshape(B, 1, hidden))

        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)

        x_np = from_dev(out_tt, (B, 1, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    logits = from_dev(logits_tt, (B, 1, vocab_size))
    return logits[0, 0]


# ═══════════════════════════════════════════════════════════════
# RUN: Prefill + decode with on-device RoPE
# ═══════════════════════════════════════════════════════════════
tokens_list = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens...\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
t_prefill = time.perf_counter() - t0

# Check prefill correctness
cos_prefill = np.dot(logits.flatten(), ref_logits[0,-1].flatten()) / (
    np.linalg.norm(logits.flatten()) * np.linalg.norm(ref_logits[0,-1].flatten()))
print(f"  Prefill cosine vs half-format ref: {cos_prefill:.6f}")
print(f"  Prefill time: {t_prefill*1000:.0f}ms")

next_id = int(np.argmax(logits))
tokens_list.append(next_id)
sys.stdout.write(args.prompt + tokenizer.decode([next_id]))
sys.stdout.flush()

# Decode with ON-DEVICE RoPE
print(f"\n  [Trying on-device decode...]")
decode_times_ondev = []
generated_ondev = [next_id]

try:
    for step in range(max_gen - 1):
        pos = len(tokens_list) - 1
        t0 = time.perf_counter()
        logits = decode_step_ondevice(next_id, pos)
        dt = time.perf_counter() - t0
        decode_times_ondev.append(dt)

        next_id = int(np.argmax(logits))
        tokens_list.append(next_id)
        generated_ondev.append(next_id)
        sys.stdout.write(tokenizer.decode([next_id]))
        sys.stdout.flush()

        if next_id == tokenizer.eos_token_id:
            break

    ondevice_success = True
except Exception as e:
    print(f"\n  ⚠ On-device decode failed at step {len(decode_times_ondev)}: {e}")
    ondevice_success = False

# ═══════════════════════════════════════════════════════════════
# ALSO: Run CPU-RoPE decode for speed comparison (reset caches first)
# ═══════════════════════════════════════════════════════════════
print("\n\n  [Running CPU-RoPE baseline for comparison...]")

# Reset caches
for i in range(n_layers):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

tokens_cpu = tokenizer.encode(args.prompt)
logits = prefill(np.array(tokens_cpu))
next_id_cpu = int(np.argmax(logits))
tokens_cpu.append(next_id_cpu)

decode_times_cpu = []
for step in range(min(5, max_gen - 1)):  # Just 5 steps for baseline
    pos = len(tokens_cpu) - 1
    t0 = time.perf_counter()
    logits = decode_step_cpu_rope(next_id_cpu, pos)
    dt = time.perf_counter() - t0
    decode_times_cpu.append(dt)
    next_id_cpu = int(np.argmax(logits))
    tokens_cpu.append(next_id_cpu)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if decode_times_cpu:
    avg_cpu = np.mean(decode_times_cpu[1:]) * 1000 if len(decode_times_cpu) > 1 else decode_times_cpu[0] * 1000
    print(f"\nCPU-RoPE baseline (half-format):")
    print(f"  First decode:  {decode_times_cpu[0]*1000:.0f}ms")
    if len(decode_times_cpu) > 1:
        print(f"  Sustained:     {avg_cpu:.0f}ms/tok ({1000/avg_cpu:.1f} tok/sec)")

if ondevice_success and decode_times_ondev:
    avg_ondev = np.mean(decode_times_ondev[1:]) * 1000 if len(decode_times_ondev) > 1 else decode_times_ondev[0] * 1000
    print(f"\nOn-device RoPE:")
    print(f"  First decode:  {decode_times_ondev[0]*1000:.0f}ms")
    if len(decode_times_ondev) > 1:
        print(f"  Sustained:     {avg_ondev:.0f}ms/tok ({1000/avg_ondev:.1f} tok/sec)")
        if decode_times_cpu and len(decode_times_cpu) > 1:
            speedup = avg_cpu / avg_ondev
            print(f"  Speedup:       {speedup:.2f}x vs CPU-RoPE")

    print(f"\n  Transfers per decode step:")
    print(f"    CPU-RoPE:    ~8 per layer × 24 = ~192")
    print(f"    On-device:   ~2 per layer × 24 = ~48 (embed in + logits out + residual)")
    print(f"    Saved:       ~144 transfers per step")

print(f"\n  Baseline (exp 49): 35ms/tok, 29.3 tok/sec")
print(f"  RoPE format used: half (rotate_half)")

ttnn.close_device(device)
print("\nDone!")
