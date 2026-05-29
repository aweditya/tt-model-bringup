#!/usr/bin/env python3
"""
Experiment 46: Qwen2.5-0.5B correctness with HiFi4 SDPA.

Re-runs experiment 43's per-layer correctness validation but with
WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True) for all SDPA calls.

Experiment 45 showed this improves single-layer SDPA cosine from 0.980 → 0.996.
This experiment verifies the improvement compounds across 24 layers to push
final logit cosine above 0.99 (was 0.956 without HiFi4).
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

# ── Config ───────────────────────────────────────────────────
hidden = 896
intermediate = 4864
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0
n_layers = 24
vocab_size = 151936

print("=" * 60)
print("Experiment 46: Qwen2.5-0.5B Correctness with HiFi4 SDPA")
print("=" * 60)

# ── Load weights ─────────────────────────────────────────────
print("\nLoading weights...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
has_lm_head = "lm_head.weight" in all_weights
lm_head_w = all_weights["lm_head.weight"].T if has_lm_head else embed_w.T.copy()

layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)

ref_layer_weights = layer_weights.copy()
ref_embed_w = embed_w.copy()
ref_final_norm_g = final_norm_g.copy()
ref_lm_head_w = lm_head_w.copy()

# ── Tokenize ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="np")[0]
T = len(input_ids)
B = 1
print(f"Prompt: '{prompt}' ({T} tokens: {input_ids.tolist()})")

# ── RoPE helpers ─────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(T, dtype=np.float32)
angles = np.outer(positions, freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope_np(x_4d, n_heads):
    t = x_4d.shape[2]
    x_even = x_4d[..., 0::2]
    x_odd = x_4d[..., 1::2]
    cos_t = cos_table[None, None, :t, :]
    sin_t = sin_table[None, None, :t, :]
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out

def rms_norm_np(x, weight, eps=1e-6):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# ═══════════════════════════════════════════════════════════
# NUMPY REFERENCE FORWARD (float32)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Pure numpy float32 reference")
print("=" * 60)

ref_x = ref_embed_w[input_ids].reshape(B, T, hidden)
ref_layer_outputs = [ref_x.copy()]
ref_attn_outputs = []
ref_mlp_outputs = []

for i in range(n_layers):
    lw = ref_layer_weights[i]
    h = rms_norm_np(ref_x, lw["input_layernorm.weight"])
    q = h @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]
    k = h @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]
    v = h @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]

    q_4d = q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k_4d = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_4d = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    q_4d = apply_rope_np(q_4d, n_q_heads)
    k_4d = apply_rope_np(k_4d, n_kv_heads)

    kv_repeat = n_q_heads // n_kv_heads
    k_4d = np.repeat(k_4d, kv_repeat, axis=1)
    v_4d = np.repeat(v_4d, kv_repeat, axis=1)

    scale = 1.0 / np.sqrt(head_dim)
    scores = (q_4d @ k_4d.transpose(0, 1, 3, 2)) * scale
    mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
    scores = scores + mask[None, None, :, :]
    attn_weights = softmax_np(scores)
    attn_out = attn_weights @ v_4d
    attn_merged = attn_out.transpose(0, 2, 1, 3).reshape(B, T, hidden)
    ref_attn_outputs.append(attn_merged.copy())

    o = attn_merged @ lw["self_attn.o_proj.weight"].T
    ref_x = ref_x + o

    h2 = rms_norm_np(ref_x, lw["post_attention_layernorm.weight"])
    gate = h2 @ lw["mlp.gate_proj.weight"].T
    up = h2 @ lw["mlp.up_proj.weight"].T
    mlp_out = silu_np(gate) * up
    mlp_out = mlp_out @ lw["mlp.down_proj.weight"].T
    ref_mlp_outputs.append(mlp_out.copy())
    ref_x = ref_x + mlp_out
    ref_layer_outputs.append(ref_x.copy())

    if (i + 1) % 6 == 0 or i == 0:
        print(f"  Layer {i:2d} done | x norm: {np.linalg.norm(ref_x):.4f}")

ref_x_normed = rms_norm_np(ref_x, ref_final_norm_g)
ref_logits = (ref_x_normed @ ref_lm_head_w).reshape(B, T, vocab_size)

ref_top5 = np.argsort(ref_logits[0, -1])[-5:][::-1]
print(f"\nNumpy reference top-5:")
for r, idx in enumerate(ref_top5):
    print(f"  {r+1}. '{tokenizer.decode([idx])}' (id={idx}, logit={ref_logits[0,-1,idx]:.4f})")

del ref_layer_weights

# ═══════════════════════════════════════════════════════════
# TT-NN FORWARD WITH HiFi4 SDPA
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: TT-NN forward with HiFi4 SDPA — per-layer comparison")
print("=" * 60)

device = ttnn.open_device(device_id=0)

# THE KEY CONFIG: HiFi4 + fp32 accumulation for SDPA
sdpa_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)
print(f"SDPA config: HiFi4, fp32_dest_acc=True, math_approx=False")

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# Upload weights
print("\nUploading weights...")
dev_layers = []
for i in range(n_layers):
    lw = layer_weights[i]
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
    dev_layers.append(dl)
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights

# TT-NN forward with per-layer tracking
print(f"\n{'Layer':<8} {'Hidden cos':<14} {'Attn cos':<14} {'MLP cos':<14} {'Status'}")
print("-" * 60)

tt_x = embed_w[input_ids].reshape(B, T, hidden)

emb_cos = cosine(tt_x, ref_layer_outputs[0])
print(f"{'embed':<8} {emb_cos:<14.6f} {'--':<14} {'--':<14} {'PASS' if emb_cos > 0.999 else 'CHECK'}")

layer_cosines = []
for i in range(n_layers):
    dl = dev_layers[i]
    x_tt = to_dev(tt_x)

    # Self-attention
    h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
    q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"]), dl["q_b"])
    k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"]), dl["k_b"])
    v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"]), dl["v_b"])

    q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
    k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
    v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

    q_4d = q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k_4d = k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    q_4d = apply_rope_np(q_4d, n_q_heads)
    k_4d = apply_rope_np(k_4d, n_kv_heads)

    # SDPA with HiFi4 config!
    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
        is_causal=True,
        compute_kernel_config=sdpa_config,
    )

    attn_out_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
    attn_out_np = attn_out_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)
    attn_cos = cosine(attn_out_np[0], ref_attn_outputs[i][0])

    o_tt = ttnn.matmul(to_dev(attn_out_np), dl["o_w"])
    x_tt2 = ttnn.add(x_tt, o_tt)

    # MLP
    h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
    gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
    up_tt = ttnn.matmul(h2_tt, dl["up_w"])
    swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
    down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])

    mlp_out_np = from_dev(down_tt, (B, T, hidden))
    mlp_cos = cosine(mlp_out_np[0], ref_mlp_outputs[i][0])

    out_tt = ttnn.add(x_tt2, down_tt)
    tt_x = from_dev(out_tt, (B, T, hidden))

    hidden_cos = cosine(tt_x[0], ref_layer_outputs[i + 1][0])
    layer_cosines.append(hidden_cos)

    status = "PASS" if hidden_cos > 0.99 else "LOW" if hidden_cos > 0.95 else "FAIL"
    print(f"{i:<8} {hidden_cos:<14.6f} {attn_cos:<14.6f} {mlp_cos:<14.6f} {status}")

# Final logits
x_tt = to_dev(tt_x)
x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
logits_tt = ttnn.matmul(x_tt, lm_head_w_tt)
tt_logits = from_dev(logits_tt, (B, T, vocab_size))

# ── Logit comparison ──────────────────────────────────────
print("\n" + "=" * 60)
print("Final Logit Comparison")
print("=" * 60)

last_cos = cosine(tt_logits[0, -1], ref_logits[0, -1])
max_err = np.abs(tt_logits[0, -1] - ref_logits[0, -1]).max()
mean_err = np.abs(tt_logits[0, -1] - ref_logits[0, -1]).mean()

print(f"  Last-token cosine:  {last_cos:.6f}")
print(f"  Max absolute error: {max_err:.4f}")
print(f"  Mean absolute error: {mean_err:.4f}")

# Per-token cosine
print(f"\n  Per-token cosine:")
for t in range(T):
    tok_cos = cosine(tt_logits[0, t], ref_logits[0, t])
    tok_str = tokenizer.decode([input_ids[t]])
    print(f"    Token {t}: '{tok_str:>12s}' -> {tok_cos:.6f}")

# Top-5
tt_top5 = np.argsort(tt_logits[0, -1])[-5:][::-1]
print(f"\n  {'Rank':<6} {'TT-NN':<30} {'Numpy Ref':<30} {'Match'}")
print("  " + "-" * 70)
for r in range(5):
    tt_tok = tokenizer.decode([tt_top5[r]])
    ref_tok = tokenizer.decode([ref_top5[r]])
    match = "Y" if tt_top5[r] == ref_top5[r] else "N"
    print(f"  {r+1:<6} '{tt_tok}' (logit={tt_logits[0,-1,tt_top5[r]]:.2f}){'':<10}"
          f"'{ref_tok}' (logit={ref_logits[0,-1,ref_top5[r]]:.2f}){'':<10} {match}")

overlap = len(set(tt_top5.tolist()) & set(ref_top5.tolist()))
top1_match = tt_top5[0] == ref_top5[0]

# ── Comparison with baseline (exp 43) ─────────────────────
print("\n" + "=" * 60)
print("CORRECTNESS SUMMARY — HiFi4 vs Baseline")
print("=" * 60)
print(f"  Last-token cosine:  {last_cos:.6f} ({'PASS' if last_cos > 0.99 else 'INVESTIGATE'})")
print(f"  Top-1 match:        {'YES' if top1_match else 'NO'}")
print(f"  Top-5 overlap:      {overlap}/5")
print(f"  Max logit error:    {max_err:.4f}")
print()
print(f"  Baseline (exp 43):  0.956183 cosine, top-1 mismatch")
print(f"  HiFi4 (this exp):   {last_cos:.6f} cosine, top-1 {'match' if top1_match else 'mismatch'}")
print(f"  Improvement:        {last_cos - 0.956183:+.6f}")
print()
print(f"  Per-layer cosine stats:")
print(f"    Min:  {min(layer_cosines):.6f} (layer {layer_cosines.index(min(layer_cosines))})")
print(f"    Max:  {max(layer_cosines):.6f} (layer {layer_cosines.index(max(layer_cosines))})")
print(f"    Mean: {np.mean(layer_cosines):.6f}")
print(f"    Layers > 0.99: {sum(1 for c in layer_cosines if c > 0.99)}/{n_layers}")

ttnn.close_device(device)
print("\nDone!")
