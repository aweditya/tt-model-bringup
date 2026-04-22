#!/usr/bin/env python3
"""
Experiment 46e: Full 24-layer Qwen with HiFi4+fp32 on ALL ops.

Experiment 46d showed that applying HiFi4+fp32 to ALL ops (not just SDPA)
gives essentially perfect results (0.9995-1.0000 per layer for 6 layers).

The key insight: there's a kernel config state leak on Blackhole when
different ops use different compute_kernel_configs. Applying HiFi4+fp32
uniformly avoids this.

This experiment runs the full 24-layer correctness validation.
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

hidden = 896; intermediate = 4864; n_q_heads = 14; n_kv_heads = 2
head_dim = 64; rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936

print("=" * 60)
print("Experiment 46e: Full Qwen — HiFi4+fp32 ALL OPS")
print("=" * 60)

# Load weights
print("\nLoading weights...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)

ref_layer_weights = [lw.copy() for lw in layer_weights]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="np")[0]
T, B = len(input_ids), 1
print(f"Prompt: '{prompt}' ({T} tokens)")

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

# ═══════════════════════════════════════════════════════════
# NUMPY REFERENCE
# ═══════════════════════════════════════════════════════════
print("\nPhase 1: Numpy float32 reference...")
ref_x = embed_w[input_ids].reshape(B, T, hidden)
ref_outputs = [ref_x.copy()]

for i in range(n_layers):
    lw = ref_layer_weights[i]
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

ref_x_normed = rms_norm_np(ref_x, final_norm_g)
ref_logits = (ref_x_normed @ lm_head_w).reshape(B, T, vocab_size)
ref_top5 = np.argsort(ref_logits[0, -1])[-5:][::-1]
print("Reference top-5:", [tokenizer.decode([idx]) for idx in ref_top5])

del ref_layer_weights

# ═══════════════════════════════════════════════════════════
# TT-NN FORWARD — HiFi4+fp32 ALL OPS
# ═══════════════════════════════════════════════════════════
print("\nPhase 2: TT-NN with HiFi4+fp32 on ALL ops...")
device = ttnn.open_device(device_id=0)

# THE KEY CONFIG — applied to EVERYTHING
cfg = ttnn.WormholeComputeKernelConfig(
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
    except: return t.squeeze().numpy().reshape(shape)

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
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights

print(f"\n{'Layer':<8} {'Hidden cos':<14} {'Attn cos':<14} {'MLP cos':<14} {'Status'}")
print("-" * 60)

tt_x = embed_w[input_ids].reshape(B, T, hidden)
emb_cos = cosine(tt_x, ref_outputs[0])
print(f"{'embed':<8} {emb_cos:<14.6f} {'--':<14} {'--':<14} PASS")

layer_cosines = []
t_start = time.perf_counter()

for i in range(n_layers):
    dl = dev_layers[i]
    x_tt = to_dev(tt_x)

    h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
    q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=cfg), dl["q_b"])
    k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=cfg), dl["k_b"])
    v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=cfg), dl["v_b"])

    q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
    k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
    v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

    q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
    k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
    v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
        is_causal=True, compute_kernel_config=cfg)

    attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
    attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

    o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=cfg)
    x_tt2 = ttnn.add(x_tt, o_tt)

    h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
    gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=cfg)
    up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=cfg)
    swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
    down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=cfg)
    out_tt = ttnn.add(x_tt2, down_tt)
    tt_x = from_dev(out_tt, (B, T, hidden))

    hidden_cos = cosine(tt_x[0], ref_outputs[i + 1][0])
    layer_cosines.append(hidden_cos)

    status = "PASS" if hidden_cos > 0.99 else "LOW" if hidden_cos > 0.95 else "FAIL"
    print(f"{i:<8} {hidden_cos:<14.6f} {'--':<14} {'--':<14} {status}")

t_fwd = time.perf_counter() - t_start

# Final logits
x_tt = to_dev(tt_x)
x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=cfg)
tt_logits = from_dev(logits_tt, (B, T, vocab_size))

# Comparison
last_cos = cosine(tt_logits[0, -1], ref_logits[0, -1])
tt_top5 = np.argsort(tt_logits[0, -1])[-5:][::-1]
top1_match = tt_top5[0] == ref_top5[0]
overlap = len(set(tt_top5.tolist()) & set(ref_top5.tolist()))

print("\n" + "=" * 60)
print("RESULTS — HiFi4+fp32 ALL OPS vs Baseline (exp 43)")
print("=" * 60)
print(f"  Forward time:        {t_fwd*1000:.0f}ms ({t_fwd*1000/n_layers:.1f}ms/layer)")
print(f"  Last-token cosine:   {last_cos:.6f} (baseline: 0.956183)")
print(f"  Top-1 match:         {'YES' if top1_match else 'NO'}")
print(f"  Top-5 overlap:       {overlap}/5")
print()

print(f"  TT-NN top-5:")
for r, idx in enumerate(tt_top5):
    print(f"    {r+1}. '{tokenizer.decode([idx])}' (logit={tt_logits[0,-1,idx]:.2f})")
print(f"  Reference top-5:")
for r, idx in enumerate(ref_top5):
    print(f"    {r+1}. '{tokenizer.decode([idx])}' (logit={ref_logits[0,-1,idx]:.2f})")

print(f"\n  Per-layer cosine stats:")
print(f"    Min:  {min(layer_cosines):.6f} (layer {layer_cosines.index(min(layer_cosines))})")
print(f"    Max:  {max(layer_cosines):.6f} (layer {layer_cosines.index(max(layer_cosines))})")
print(f"    Mean: {np.mean(layer_cosines):.6f}")
print(f"    All > 0.99: {all(c > 0.99 for c in layer_cosines)}")
print(f"    All > 0.999: {all(c > 0.999 for c in layer_cosines)}")

ttnn.close_device(device)
print("\nDone!")
