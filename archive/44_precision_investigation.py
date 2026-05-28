#!/usr/bin/env python3
"""
Experiment 44: Precision investigation — isolate bfloat16 error source.

From experiment 43, we know each layer adds ~0.008 cosine error.
This experiment isolates whether the error comes from:
  A) SDPA (softmax in bfloat16)
  B) Matmul (weight projections in bfloat16)
  C) RMSNorm (normalization in bfloat16)
  D) Residual connection accumulation

Method: Run single-layer forward with:
  1. All on device (baseline)
  2. Replace SDPA with numpy reference — does cosine improve?
  3. Replace matmuls with numpy reference — does cosine improve?
  4. Test with different TT-NN dtype options (if available)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)

# Config
hidden = 896
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0

print("=" * 60)
print("Experiment 44: Precision Investigation")
print("Isolating bfloat16 error in Qwen2.5-0.5B")
print("=" * 60)

# Load first layer weights
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        if key.startswith("model.layers.0.") or key in [
            "model.embed_tokens.weight", "model.norm.weight"]:
            all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
lw = {k.replace("model.layers.0.", ""): v for k, v in all_weights.items()
      if k.startswith("model.layers.0.")}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
input_ids = tokenizer.encode("The capital of France is", return_tensors="np")[0]
T = len(input_ids)
B = 1

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(T, dtype=np.float32)
angles = np.outer(positions, freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope(x_4d, n_heads):
    t = x_4d.shape[2]
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_table[None, None, :t, :] - x_4d[..., 1::2] * sin_table[None, None, :t, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_table[None, None, :t, :] + x_4d[..., 1::2] * cos_table[None, None, :t, :]
    return out

def rms_norm_np(x, weight, eps=1e-6):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# ── Numpy reference (float32) ────────────────────────────────
x_np = embed_w[input_ids].reshape(B, T, hidden).copy()

# RMSNorm
h_ref = rms_norm_np(x_np, lw["input_layernorm.weight"])

# Q/K/V
q_ref = h_ref @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]
k_ref = h_ref @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]
v_ref = h_ref @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]

q_4d_ref = apply_rope(q_ref.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
k_4d_ref = apply_rope(k_ref.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
v_4d_ref = v_ref.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

# GQA expand
kv_repeat = n_q_heads // n_kv_heads
k_4d_ref_exp = np.repeat(k_4d_ref, kv_repeat, axis=1)
v_4d_ref_exp = np.repeat(v_4d_ref, kv_repeat, axis=1)

# Attention (float32)
scale = 1.0 / np.sqrt(head_dim)
scores_ref = (q_4d_ref @ k_4d_ref_exp.transpose(0, 1, 3, 2)) * scale
mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
scores_ref += mask[None, None]
attn_w_ref = softmax_np(scores_ref)
attn_out_ref = (attn_w_ref @ v_4d_ref_exp).transpose(0, 2, 1, 3).reshape(B, T, hidden)

# Output proj + residual
o_ref = attn_out_ref @ lw["self_attn.o_proj.weight"].T
x_after_attn_ref = x_np + o_ref

# MLP
h2_ref = rms_norm_np(x_after_attn_ref, lw["post_attention_layernorm.weight"])
gate_ref = h2_ref @ lw["mlp.gate_proj.weight"].T
up_ref = h2_ref @ lw["mlp.up_proj.weight"].T
mlp_out_ref = silu_np(gate_ref) * up_ref
mlp_out_ref = mlp_out_ref @ lw["mlp.down_proj.weight"].T
x_out_ref = x_after_attn_ref + mlp_out_ref

print(f"\nReference layer output norm: {np.linalg.norm(x_out_ref):.4f}")

# ── TT-NN device tests ───────────────────────────────────────
device = ttnn.open_device(device_id=0)

def to_dev(arr, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# Upload layer 0 weights
dl = {
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
}

print("\n" + "=" * 60)
print("Test 1: Full TT-NN layer (baseline)")
print("=" * 60)

x_tt = to_dev(x_np)
h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"]), dl["q_b"])
k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"]), dl["k_b"])
v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"]), dl["v_b"])

q_np_tt = from_dev(q_tt, (B, T, n_q_heads * head_dim))
k_np_tt = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
v_np_tt = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

print(f"  Q projection cosine vs ref: {cosine(q_np_tt, q_ref):.6f}")
print(f"  K projection cosine vs ref: {cosine(k_np_tt, k_ref):.6f}")
print(f"  V projection cosine vs ref: {cosine(v_np_tt, v_ref):.6f}")

# RoPE on CPU (same as exp 41)
q_4d = apply_rope(q_np_tt.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
k_4d = apply_rope(k_np_tt.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
v_4d = v_np_tt.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

print(f"  Q+RoPE cosine vs ref: {cosine(q_4d, q_4d_ref):.6f}")
print(f"  K+RoPE cosine vs ref: {cosine(k_4d, k_4d_ref):.6f}")

# SDPA on device
attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
    to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d), is_causal=True)
attn_out_np_tt = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
attn_merged_tt = attn_out_np_tt.transpose(0, 2, 1, 3).reshape(B, T, hidden)

print(f"  SDPA output cosine vs ref: {cosine(attn_merged_tt, attn_out_ref):.6f}")

# Output proj + residual
o_tt_dev = ttnn.matmul(to_dev(attn_merged_tt), dl["o_w"])
x_tt2 = ttnn.add(x_tt, o_tt_dev)
x_after_attn_tt = from_dev(x_tt2, (B, T, hidden))
print(f"  Post-attention residual cosine vs ref: {cosine(x_after_attn_tt, x_after_attn_ref):.6f}")

# MLP
h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
up_tt = ttnn.matmul(h2_tt, dl["up_w"])
swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])
out_tt = ttnn.add(x_tt2, down_tt)
x_out_tt = from_dev(out_tt, (B, T, hidden))

print(f"  Full layer output cosine vs ref: {cosine(x_out_tt, x_out_ref):.6f}")

print("\n" + "=" * 60)
print("Test 2: TT-NN matmuls + numpy SDPA (isolate SDPA)")
print("=" * 60)

# Use TT-NN for projections but numpy for attention
q_4d_mixed = apply_rope(q_np_tt.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
k_4d_mixed = apply_rope(k_np_tt.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
v_4d_mixed = v_np_tt.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

# GQA expand for numpy SDPA
k_expanded = np.repeat(k_4d_mixed, kv_repeat, axis=1)
v_expanded = np.repeat(v_4d_mixed, kv_repeat, axis=1)

scores = (q_4d_mixed @ k_expanded.transpose(0, 1, 3, 2)) * scale
scores += mask[None, None]
attn_w = softmax_np(scores)
attn_out_numpy = (attn_w @ v_expanded).transpose(0, 2, 1, 3).reshape(B, T, hidden)

print(f"  SDPA (numpy) on TT-NN Q/K/V cosine vs ref: {cosine(attn_out_numpy, attn_out_ref):.6f}")

# Continue with TT-NN for output proj
o_mixed = ttnn.matmul(to_dev(attn_out_numpy), dl["o_w"])
x_mixed = ttnn.add(x_tt, o_mixed)
x_after_attn_mixed = from_dev(x_mixed, (B, T, hidden))
print(f"  Post-attention (numpy SDPA) cosine vs ref: {cosine(x_after_attn_mixed, x_after_attn_ref):.6f}")

# MLP with TT-NN
h2_m = ttnn.rms_norm(x_mixed, weight=dl["ln2_g"], epsilon=rms_eps)
gate_m = ttnn.matmul(h2_m, dl["gate_w"])
up_m = ttnn.matmul(h2_m, dl["up_w"])
swiglu_m = ttnn.mul(ttnn.silu(gate_m), up_m)
down_m = ttnn.matmul(swiglu_m, dl["down_w"])
out_m = ttnn.add(x_mixed, down_m)
x_out_mixed = from_dev(out_m, (B, T, hidden))
print(f"  Full layer (numpy SDPA) cosine vs ref: {cosine(x_out_mixed, x_out_ref):.6f}")

print(f"\n  SDPA contribution to error:")
print(f"    Full TT-NN layer cosine:      {cosine(x_out_tt, x_out_ref):.6f}")
print(f"    TT-NN + numpy SDPA cosine:    {cosine(x_out_mixed, x_out_ref):.6f}")
print(f"    Delta (SDPA error):            {cosine(x_out_mixed, x_out_ref) - cosine(x_out_tt, x_out_ref):.6f}")

print("\n" + "=" * 60)
print("Test 3: Numpy matmuls + TT-NN SDPA (isolate matmul)")
print("=" * 60)

# Use float32 numpy for projections, TT-NN for SDPA
attn_out_tt3 = ttnn.transformer.scaled_dot_product_attention(
    to_dev_4d(q_4d_ref), to_dev_4d(k_4d_ref), to_dev_4d(v_4d_ref), is_causal=True)
attn_merged_tt3 = from_dev(attn_out_tt3, (B, n_q_heads, T, head_dim))
attn_merged_tt3 = attn_merged_tt3.transpose(0, 2, 1, 3).reshape(B, T, hidden)

print(f"  SDPA (TT-NN) on ref Q/K/V cosine vs ref: {cosine(attn_merged_tt3, attn_out_ref):.6f}")

# Continue in numpy
o_3 = attn_merged_tt3 @ lw["self_attn.o_proj.weight"].T
x_after_attn_3 = x_np + o_3
print(f"  Post-attention (ref matmuls + TT-NN SDPA) cosine vs ref: {cosine(x_after_attn_3, x_after_attn_ref):.6f}")

# Full layer in numpy after
h2_3 = rms_norm_np(x_after_attn_3, lw["post_attention_layernorm.weight"])
gate_3 = h2_3 @ lw["mlp.gate_proj.weight"].T
up_3 = h2_3 @ lw["mlp.up_proj.weight"].T
mlp_3 = silu_np(gate_3) * up_3 @ lw["mlp.down_proj.weight"].T
x_out_3 = x_after_attn_3 + mlp_3
print(f"  Full layer (ref matmuls + TT-NN SDPA) cosine vs ref: {cosine(x_out_3, x_out_ref):.6f}")

print(f"\n  Matmul contribution to error:")
print(f"    Full TT-NN layer cosine:      {cosine(x_out_tt, x_out_ref):.6f}")
print(f"    Ref matmuls + TT-NN SDPA:     {cosine(x_out_3, x_out_ref):.6f}")
print(f"    Delta (matmul error):          {cosine(x_out_3, x_out_ref) - cosine(x_out_tt, x_out_ref):.6f}")

# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRECISION SUMMARY")
print("=" * 60)
print(f"  Error breakdown for single Qwen layer 0:")
print(f"    Q projection:    {cosine(q_np_tt, q_ref):.6f}")
print(f"    K projection:    {cosine(k_np_tt, k_ref):.6f}")
print(f"    V projection:    {cosine(v_np_tt, v_ref):.6f}")
print(f"    SDPA output:     {cosine(attn_merged_tt, attn_out_ref):.6f}")
print(f"    Full layer:      {cosine(x_out_tt, x_out_ref):.6f}")
print(f"")
print(f"  Ablation (what helps):")
full_cos = cosine(x_out_tt, x_out_ref)
mixed_cos = cosine(x_out_mixed, x_out_ref)
ref3_cos = cosine(x_out_3, x_out_ref)
print(f"    Replace SDPA with numpy:  {full_cos:.6f} -> {mixed_cos:.6f} (delta {mixed_cos-full_cos:+.6f})")
print(f"    Replace matmuls with ref: {full_cos:.6f} -> {ref3_cos:.6f} (delta {ref3_cos-full_cos:+.6f})")

if mixed_cos - full_cos > ref3_cos - full_cos:
    print(f"\n  >> SDPA (bfloat16 softmax) is the LARGER error source")
else:
    print(f"\n  >> Matmul (bfloat16 weight projection) is the LARGER error source")

ttnn.close_device(device)
print("\nDone!")
