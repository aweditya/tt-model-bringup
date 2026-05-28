#!/usr/bin/env python3
"""
Experiment 63: HiFi2 math fidelity for MLP matmuls.

We use HiFi4+fp32_dest_acc everywhere (required due to kernel config state leak).
But HiFi2 halves compute cost per matmul. If MLP matmuls can tolerate HiFi2
(they're less precision-sensitive than attention), we could save significant compute.

WARNING: The kernel config state leak (exp 46) means ALL ops must use the same config.
If mixing HiFi2 MLP + HiFi4 attention corrupts results, this won't work.

BUT: exp 46 showed the leak was between SDPA and matmul specifically.
Maybe two different matmul configs (both matmul, different fidelity) don't leak?

Tests:
  A: HiFi4 everywhere (baseline, 7.1ms at b=1)
  B: HiFi2 everywhere (if no quality loss, max speedup)
  C: HiFi2 MLP + HiFi4 attention (if config mixing works)
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
batch_size = 1

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Define configs
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)
hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)
lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en=False,
    math_approx_mode=True,
)

# Load
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

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))
def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# Upload weights (bf16)
print("Uploading weights...")
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    })
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np

tokens = np.array(tokenizer.encode("The capital of France is"))
print(f'Prompt: "The capital of France is" ({len(tokens)} tokens)')


def generate_with_config(attn_cfg, mlp_cfg, sdpa_cfg, label, n_gen=30):
    """Full recompute generation with specified math configs."""
    B = 1
    gen = list(tokens)

    for step in range(n_gen):
        T = len(gen)
        x_np = embed_w[gen].reshape(B, T, hidden)
        cos_t, sin_t = get_rope_tables_half(T)

        for i in range(n_layers):
            dl = dev_layers[i]
            x_tt = to_bf16(x_np.reshape(B*T, hidden))
            h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
            q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=attn_cfg), dl["q_b"])
            k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=attn_cfg), dl["k_b"])
            v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=attn_cfg), dl["v_b"])

            q_np = apply_rope_half_np(from_dev(q,(B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
            k_np = apply_rope_half_np(from_dev(k,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
            v_np = from_dev(v,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
                is_causal=True, compute_kernel_config=sdpa_cfg)
            a_np = from_dev(attn_out,(B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)

            o = ttnn.matmul(to_bf16(a_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=attn_cfg)
            x2 = ttnn.add(x_tt, o)
            h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
            g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=mlp_cfg)
            u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=mlp_cfg)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=mlp_cfg)
            x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)

        x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
        logits = from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=mlp_cfg), (B*T, vocab_size))[-1]
        gen.append(int(np.argmax(logits)))

    return gen


# Test configs
configs = [
    ("A: HiFi4 everywhere",     hifi4, hifi4, hifi4),
    ("B: HiFi2 everywhere",     hifi2, hifi2, hifi2),
    ("C: HiFi2-MLP HiFi4-attn", hifi4, hifi2, hifi4),
    ("D: LoFi everywhere",      lofi,  lofi,  lofi),
    ("E: LoFi-MLP HiFi4-attn",  hifi4, lofi,  hifi4),
]

print(f"\nGenerating 30 tokens with each config...\n")

results = {}
for label, attn_cfg, mlp_cfg, sdpa_cfg in configs:
    print(f"--- {label} ---")
    t0 = time.perf_counter()
    gen = generate_with_config(attn_cfg, mlp_cfg, sdpa_cfg, label, n_gen=30)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(gen)
    results[label] = {"text": text, "time": dt, "tokens": gen}
    print(f"  {dt*1000:.0f}ms total")
    print(f"  {text[:150]}")
    print()


# Compare
print("=" * 60)
print("SUMMARY")
print("=" * 60)

baseline_tokens = results["A: HiFi4 everywhere"]["tokens"]
for label, r in results.items():
    match = r["tokens"] == baseline_tokens
    quality = "MATCH" if match else "DIFF"
    # Check if text starts coherently
    has_paris = "Paris" in r["text"]
    print(f"  {label:30s}: [{quality}] {has_paris and 'Paris' or '???':5s} {r['time']*1000:6.0f}ms | {r['text'][:80]}...")

ttnn.close_device(device)
print("\nDone!")
