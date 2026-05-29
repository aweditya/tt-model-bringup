#!/usr/bin/env python3
"""
Experiment 61: Per-layer quantization sensitivity ablation.

Quantize ONLY layer i to bf8 (all weights: Q/K/V/O/gate/up/down), keep rest bf16.
Run 24 experiments to build a sensitivity curve.

Hypothesis: Layers 0, 21-23 are precision-sensitive (embedding transform,
error tipping point, final decision layer). Layers 1-20 are safe for bf8.

This tells us exactly which layers to keep in bf16 for optimal quality/speed tradeoff.

Measurement: Final logit cosine vs all-bf16 baseline, generated text comparison.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print("Device: Blackhole P150")

# Load model
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights["lm_head.weight"].T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bf8(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# RoPE (numpy for full recompute)
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))
def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


def upload_with_bf8_layer(bf8_layer_idx):
    """Upload all weights bf16 except layer bf8_layer_idx which is all-bf8.
    If bf8_layer_idx < 0, all bf16 (baseline)."""
    dev_layers = []
    for i in range(n_layers):
        lw = layer_weights_np[i]
        w_fn = to_bf8 if i == bf8_layer_idx else to_bf16
        dev_layers.append({
            "ln1_g": to_bf16(lw["input_layernorm.weight"]),
            "q_w": w_fn(lw["self_attn.q_proj.weight"].T),
            "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
            "k_w": w_fn(lw["self_attn.k_proj.weight"].T),
            "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
            "v_w": w_fn(lw["self_attn.v_proj.weight"].T),
            "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
            "o_w": w_fn(lw["self_attn.o_proj.weight"].T),
            "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
            "gate_w": w_fn(lw["mlp.gate_proj.weight"].T),
            "up_w": w_fn(lw["mlp.up_proj.weight"].T),
            "down_w": w_fn(lw["mlp.down_proj.weight"].T),
        })
    return dev_layers, to_bf16(final_norm_g), to_bf16(lm_head_w)


def forward_full_recompute(dev_layers, final_g, lm_h, token_ids):
    """Full recompute forward — returns final logits."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = apply_rope_half_np(from_dev(q,(B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_half_np(from_dev(k,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn,(B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)

        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))


# ══════════════════════════════════════════════════════════════
# Run ablation
# ══════════════════════════════════════════════════════════════
tokens = np.array(tokenizer.encode("The capital of France is"))
print(f'\nPrompt: "The capital of France is" ({len(tokens)} tokens)')
print(f"Running 24-layer ablation (+ baseline)...\n")

# Baseline: all bf16
print("--- Baseline (all bf16) ---")
dev_layers_base, final_g, lm_h = upload_with_bf8_layer(-1)
logits_base = forward_full_recompute(dev_layers_base, final_g, lm_h, tokens)
top1_base = int(np.argmax(logits_base[-1]))
for dl in dev_layers_base:
    for v in dl.values(): v.deallocate()
final_g.deallocate(); lm_h.deallocate()
print(f"  Top-1: {top1_base} ({tokenizer.decode([top1_base])})")

# Per-layer ablation
results = []
for layer_idx in range(n_layers):
    dev_layers, final_g, lm_h = upload_with_bf8_layer(layer_idx)
    logits = forward_full_recompute(dev_layers, final_g, lm_h, tokens)

    # Compare to baseline
    last_logits = logits[-1]
    base_last = logits_base[-1]
    cos_sim = np.dot(last_logits, base_last) / (np.linalg.norm(last_logits) * np.linalg.norm(base_last))
    top1 = int(np.argmax(last_logits))
    top1_match = top1 == top1_base

    results.append({
        "layer": layer_idx,
        "cosine": cos_sim,
        "top1_match": top1_match,
        "top1_token": tokenizer.decode([top1]),
    })

    # Deallocate
    for dl in dev_layers:
        for v in dl.values(): v.deallocate()
    final_g.deallocate(); lm_h.deallocate()

    bar = "#" * int(cos_sim * 50 - 49.5) if cos_sim > 0.99 else "!" * 5
    print(f"  Layer {layer_idx:2d}: cosine={cos_sim:.6f} top1={'MATCH' if top1_match else 'DIFF':5s} {bar}")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SENSITIVITY CURVE (per-layer bf8 ablation)")
print(f"{'='*60}")

# Sort by sensitivity (lowest cosine = most sensitive)
sorted_results = sorted(results, key=lambda x: x["cosine"])

print(f"\n  Most sensitive layers (lowest cosine):")
for r in sorted_results[:5]:
    print(f"    Layer {r['layer']:2d}: cosine={r['cosine']:.6f} top1={r['top1_token']} ({'MATCH' if r['top1_match'] else 'DIFF'})")

print(f"\n  Least sensitive layers (highest cosine):")
for r in sorted_results[-5:]:
    print(f"    Layer {r['layer']:2d}: cosine={r['cosine']:.6f} top1={r['top1_token']} ({'MATCH' if r['top1_match'] else 'DIFF'})")

n_match = sum(1 for r in results if r["top1_match"])
avg_cosine = np.mean([r["cosine"] for r in results])
print(f"\n  Summary:")
print(f"    Average cosine:    {avg_cosine:.6f}")
print(f"    Top-1 matches:     {n_match}/{n_layers}")
print(f"    Min cosine:        {min(r['cosine'] for r in results):.6f} (layer {sorted_results[0]['layer']})")
print(f"    Max cosine:        {max(r['cosine'] for r in results):.6f} (layer {sorted_results[-1]['layer']})")

# Recommendation
sensitive = [r["layer"] for r in results if r["cosine"] < 0.999 or not r["top1_match"]]
safe = [r["layer"] for r in results if r["cosine"] >= 0.999 and r["top1_match"]]
print(f"\n  Recommendation:")
print(f"    Keep bf16: layers {sensitive}")
print(f"    Safe for bf8: layers {safe}")
print(f"    bf8 coverage: {len(safe)}/{n_layers} = {100*len(safe)/n_layers:.0f}% of layers")

ttnn.close_device(device)
print("\nDone!")
