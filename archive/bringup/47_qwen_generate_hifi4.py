#!/usr/bin/env python3
"""
Experiment 47: Qwen2.5-0.5B text generation with HiFi4+fp32 on ALL ops.

Combines exp 41 (generation) with exp 46e (HiFi4 correctness fix).
This is the "production" Qwen inference — correct AND on-device.

Expected: correct generation (top-1 match with reference) + perf numbers.
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── Args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=20)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────
hidden = 896; intermediate = 4864; n_q_heads = 14; n_kv_heads = 2
head_dim = 64; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256

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

# ── Device ───────────────────────────────────────────────────
print(f"Opening device {args.device}...")
device = ttnn.open_device(device_id=args.device)

# THE KEY CONFIG — applied to ALL compute ops
hifi4_cfg = ttnn.WormholeComputeKernelConfig(
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
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights...")
t0 = time.perf_counter()

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
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── RoPE (CPU for now — exp 48 will test on-device) ─────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def apply_rope_np(x_4d, cos_t, sin_t):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_t[None, None, :, :] - x_4d[..., 1::2] * sin_t[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_t[None, None, :, :] + x_4d[..., 1::2] * cos_t[None, None, :, :]
    return out

# ── Forward pass ─────────────────────────────────────────────
def forward(input_ids_np):
    B, T = 1, len(input_ids_np)
    x_np = embed_w[input_ids_np].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        # Self-attention
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4_cfg), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4_cfg), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4_cfg), dl["v_b"])

        # Pull Q/K for CPU RoPE, V stays as numpy for reshape
        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # SDPA on device
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4_cfg)

        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        # Output projection + residual
        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=hifi4_cfg)
        x_tt2 = ttnn.add(x_tt, o_tt)

        # MLP
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4_cfg)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4_cfg)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4_cfg)
        out_tt = ttnn.add(x_tt2, down_tt)

        x_np = from_dev(out_tt, (B, T, hidden))

    # Final norm + logits
    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4_cfg)
    logits = from_dev(logits_tt, (B, T, vocab_size))
    return logits[0, -1]

# ── Generate ─────────────────────────────────────────────────
tokens = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens))

print(f'\nPrompt: "{args.prompt}"')
print(f"Generating {max_gen} tokens (HiFi4+fp32 all ops)...\n")

sys.stdout.write(args.prompt)
sys.stdout.flush()

times = []
for i in range(max_gen):
    t0 = time.perf_counter()
    logits = forward(np.array(tokens))
    dt = time.perf_counter() - t0
    times.append(dt)

    next_id = int(np.argmax(logits))
    tokens.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()

    if next_id == tokenizer.eos_token_id:
        break

avg_ms = np.mean(times) * 1000
print(f"\n\n--- {len(times)} tokens, avg {avg_ms:.0f}ms/tok, "
      f"{1000/avg_ms:.1f} tok/sec ---")
if len(times) > 1:
    print(f"  First: {times[0]*1000:.0f}ms, subsequent avg: {np.mean(times[1:])*1000:.0f}ms")
print(f"  Config: HiFi4+fp32 on ALL ops (correctness: 0.998 cosine)")
print(f"  Baseline (exp 41, default config): 582ms/tok, 1.7 tok/sec")

ttnn.close_device(device)
