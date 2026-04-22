#!/usr/bin/env python3
"""
Experiment 46b: Quick SDPA config sweep — which configs help/hurt?

Exp 46 showed HiFi4 makes layers 0-2 better but layer 3+ catastrophically worse.
Test: HiFi2, HiFi3, fp32_acc only, and default — all on a 5-layer forward pass.
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

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 6; vocab_size = 151936

print("=" * 60)
print("Experiment 46b: SDPA Config Sweep (first 6 layers)")
print("=" * 60)

# Load weights
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        if key.startswith("model.layers.") and int(key.split(".")[2]) >= n_layers:
            continue
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
input_ids = tokenizer.encode("The capital of France is", return_tensors="np")[0]
T, B = len(input_ids), 1

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(T, dtype=np.float32), freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope_np(x_4d, n_heads):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_table[None, None, :, :] - x_4d[..., 1::2] * sin_table[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_table[None, None, :, :] + x_4d[..., 1::2] * cos_table[None, None, :, :]
    return out

def rms_norm_np(x, w, eps=1e-6):
    return (x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * w

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# Numpy reference
ref_x = embed_w[input_ids].reshape(B, T, hidden)
ref_outputs = [ref_x.copy()]
for i in range(n_layers):
    lw = layer_weights[i]
    h = rms_norm_np(ref_x, lw["input_layernorm.weight"])
    q = h @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]
    k = h @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]
    v = h @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]
    q_4d = apply_rope_np(q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
    k_4d = apply_rope_np(k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
    v_4d = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    kv_r = n_q_heads // n_kv_heads
    k_exp = np.repeat(k_4d, kv_r, axis=1); v_exp = np.repeat(v_4d, kv_r, axis=1)
    sc = (q_4d @ k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
    sc += np.triu(np.ones((T,T))*-1e9, k=1)[None,None]
    attn_out = (softmax_np(sc) @ v_exp).transpose(0,2,1,3).reshape(B,T,hidden)
    ref_x = ref_x + attn_out @ lw["self_attn.o_proj.weight"].T
    h2 = rms_norm_np(ref_x, lw["post_attention_layernorm.weight"])
    mlp = silu_np(h2 @ lw["mlp.gate_proj.weight"].T) * (h2 @ lw["mlp.up_proj.weight"].T)
    ref_x = ref_x + mlp @ lw["mlp.down_proj.weight"].T
    ref_outputs.append(ref_x.copy())
print(f"Reference computed ({n_layers} layers)")

# Device
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
    except: return t.squeeze().numpy().reshape(shape)

# Upload weights once
print("Uploading weights...")
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

# Configs to test
configs = {
    "default": None,
    "HiFi2": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True),
    "HiFi4": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False),
    "fp32_acc": ttnn.WormholeComputeKernelConfig(
        fp32_dest_acc_en=True),
    "HiFi4+fp32": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
    "HiFi2+fp32": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False),
}

def run_forward(sdpa_config):
    """Run n_layers forward, return per-layer cosines."""
    tt_x = embed_w[input_ids].reshape(B, T, hidden)
    cosines = []
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(tt_x)
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"]), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"]), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"]), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
        k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        kwargs = {"is_causal": True}
        if sdpa_config is not None:
            kwargs["compute_kernel_config"] = sdpa_config
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d), **kwargs)

        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_np = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_np), dl["o_w"])
        x_tt2 = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
        up_tt = ttnn.matmul(h2_tt, dl["up_w"])
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])
        out_tt = ttnn.add(x_tt2, down_tt)
        tt_x = from_dev(out_tt, (B, T, hidden))

        cosines.append(cosine(tt_x[0], ref_outputs[i + 1][0]))
    return cosines

# Run sweep
print(f"\n{'Config':<16}", end="")
for i in range(n_layers):
    print(f"{'L'+str(i):<10}", end="")
print()
print("-" * (16 + 10 * n_layers))

for name, cfg in configs.items():
    try:
        cosines = run_forward(cfg)
        print(f"{name:<16}", end="")
        for c in cosines:
            status = "" if c > 0.99 else "*" if c > 0.95 else "!"
            print(f"{c:.4f}{status:<5}", end="")
        print()
    except Exception as e:
        print(f"{name:<16} FAILED: {str(e)[:60]}")

ttnn.close_device(device)
print("\nDone!")
