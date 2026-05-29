#!/usr/bin/env python3
"""
Experiment 45: Test SDPA with HiFi4 / fp32 accumulation.

Research found that ttnn.transformer.scaled_dot_product_attention accepts
a compute_kernel_config parameter. Test whether fp32_dest_acc_en improves
the 0.985 cosine we see with default bfloat16 SDPA.

Key parameters to try:
  - fp32_dest_acc_en=True (32-bit accumulation in Dst registers)
  - math_approx_mode=False (no math approximations)
  - math_fidelity=ttnn.MathFidelity.HiFi4 (highest fidelity)
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
print("Experiment 45: SDPA with HiFi4 / FP32 Accumulation")
print("=" * 60)

# Load first layer weights
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        if key.startswith("model.layers.0.") or key == "model.embed_tokens.weight":
            all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
lw = {k.replace("model.layers.0.", ""): v for k, v in all_weights.items()
      if k.startswith("model.layers.0.")}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
input_ids = tokenizer.encode("The capital of France is", return_tensors="np")[0]
T, B = len(input_ids), 1

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(T, dtype=np.float32), freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope(x_4d, n_heads):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_table[None, None, :, :] - x_4d[..., 1::2] * sin_table[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_table[None, None, :, :] + x_4d[..., 1::2] * cos_table[None, None, :, :]
    return out

def rms_norm_np(x, weight, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * weight

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# Compute numpy reference
x_np = embed_w[input_ids].reshape(B, T, hidden)
h_ref = rms_norm_np(x_np, lw["input_layernorm.weight"])
q_ref = h_ref @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]
k_ref = h_ref @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]
v_ref = h_ref @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]

q_4d = apply_rope(q_ref.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
k_4d = apply_rope(k_ref.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
v_4d = v_ref.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

# Numpy SDPA reference (float32)
kv_repeat = n_q_heads // n_kv_heads
k_exp = np.repeat(k_4d, kv_repeat, axis=1)
v_exp = np.repeat(v_4d, kv_repeat, axis=1)
scale = 1.0 / np.sqrt(head_dim)
scores = (q_4d @ k_exp.transpose(0, 1, 3, 2)) * scale
scores += np.triu(np.ones((T, T)) * -1e9, k=1)[None, None]
attn_out_ref = (softmax_np(scores) @ v_exp).transpose(0, 2, 1, 3).reshape(B, T, hidden)

print(f"Reference attention output norm: {np.linalg.norm(attn_out_ref):.4f}")

# ── Device tests ──────────────────────────────────────────
device = ttnn.open_device(device_id=0)

def to_dev_4d(arr):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except: return t.squeeze().numpy().reshape(shape)

# Upload Q/K/V
q_tt = to_dev_4d(q_4d)
k_tt = to_dev_4d(k_4d)
v_tt = to_dev_4d(v_4d)

# ── Test 1: Default SDPA (baseline) ──────────────────────
print("\n── Test 1: Default SDPA (bfloat16) ──")
out1 = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=True)
out1_np = from_dev(out1, (B, n_q_heads, T, head_dim))
out1_merged = out1_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)
cos1 = cosine(out1_merged, attn_out_ref)
print(f"  Cosine vs float32 ref: {cos1:.6f}")

# ── Test 2: SDPA with compute_kernel_config ───────────────
configs_to_try = []

# Try MathFidelity.HiFi4
try:
    configs_to_try.append(("HiFi4", ttnn.DeviceComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )))
except Exception as e:
    print(f"\n  DeviceComputeKernelConfig with HiFi4 failed: {e}")

# Try just fp32 acc
try:
    configs_to_try.append(("fp32_acc_only", ttnn.DeviceComputeKernelConfig(
        fp32_dest_acc_en=True,
    )))
except Exception as e:
    print(f"\n  DeviceComputeKernelConfig with fp32_acc failed: {e}")

# Try BlackholeComputeKernelConfig if it exists
try:
    configs_to_try.append(("BlackholeCompute", ttnn.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )))
except Exception as e:
    print(f"\n  BlackholeComputeKernelConfig failed: {e}")

# Try WormholeComputeKernelConfig (might work for Blackhole too)
try:
    configs_to_try.append(("WormholeCompute", ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )))
except Exception as e:
    print(f"\n  WormholeComputeKernelConfig failed: {e}")

# Try listing available attributes
print("\n── Exploring compute config options ──")
for attr in dir(ttnn):
    if 'compute' in attr.lower() or 'kernel' in attr.lower() or 'fidelity' in attr.lower():
        print(f"  ttnn.{attr}")

for attr in dir(ttnn):
    if 'MathFidelity' in attr or 'HiFi' in attr:
        print(f"  ttnn.{attr}")

if hasattr(ttnn, 'MathFidelity'):
    print(f"\n  MathFidelity values: {list(ttnn.MathFidelity)}")

for name, config in configs_to_try:
    print(f"\n── Test: SDPA with {name} ──")
    try:
        q_tt2 = to_dev_4d(q_4d)
        k_tt2 = to_dev_4d(k_4d)
        v_tt2 = to_dev_4d(v_4d)
        out = ttnn.transformer.scaled_dot_product_attention(
            q_tt2, k_tt2, v_tt2, is_causal=True,
            compute_kernel_config=config)
        out_np = from_dev(out, (B, n_q_heads, T, head_dim))
        out_merged = out_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)
        cos_val = cosine(out_merged, attn_out_ref)
        print(f"  Cosine vs float32 ref: {cos_val:.6f}")
        print(f"  Improvement over baseline: {cos_val - cos1:+.6f}")
    except Exception as e:
        print(f"  FAILED: {str(e)[:100]}")

# ── Also test rotary_embedding if available ────────────────
print("\n── Checking for rotary_embedding API ──")
for attr in dir(ttnn.transformer):
    if 'rotar' in attr.lower() or 'rope' in attr.lower():
        print(f"  ttnn.transformer.{attr}")
if hasattr(ttnn, 'experimental'):
    for attr in dir(ttnn.experimental):
        if 'rotar' in attr.lower() or 'rope' in attr.lower():
            print(f"  ttnn.experimental.{attr}")

# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Default SDPA cosine: {cos1:.6f}")
for name, _ in configs_to_try:
    print(f"  {name}: see results above")

ttnn.close_device(device)
print("\nDone!")
