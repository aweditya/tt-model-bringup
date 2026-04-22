#!/usr/bin/env python3
"""
Experiment 41: Qwen2.5-0.5B text generation on Blackhole.

Builds on experiment 38's working forward pass to do autoregressive generation.
Full-recompute per token (no KV cache yet) — establishes baseline correctness.

Usage:
  python3 experiments/41_qwen_generate.py
  python3 experiments/41_qwen_generate.py "Your prompt here" --tokens 20
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
parser = argparse.ArgumentParser(description="Qwen2.5-0.5B on Blackhole")
parser.add_argument("prompt", nargs="?", default="The capital of France is",
                    help="Text prompt to complete")
parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
parser.add_argument("--device", type=int, default=0, help="Device ID")
args = parser.parse_args()

# ── Model config ─────────────────────────────────────────────
hidden = 896
intermediate = 4864
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0
n_layers = 24
vocab_size = 151936
MAX_SEQ = 256  # Safety cap for full-recompute mode

# ── Load weights ─────────────────────────────────────────────
print("Loading Qwen2.5-0.5B (490M params)...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")

all_weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key)

embed_w = all_weights["model.embed_tokens.weight"].astype(np.float32)
final_norm_g = all_weights["model.norm.weight"].astype(np.float32)
has_lm_head = "lm_head.weight" in all_weights
lm_head_w = (all_weights["lm_head.weight"].astype(np.float32).T if has_lm_head
             else embed_w.T.copy())

# Per-layer weights (keep on CPU, upload to device below)
layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v.astype(np.float32)
          for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)
del all_weights

# ── Tokenizer ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ── Open device ──────────────────────────────────────────────
print(f"Opening Blackhole device {args.device}...")
device = ttnn.open_device(device_id=args.device)

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
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)

# ── Upload weights to device ─────────────────────────────────
print("Uploading 490M parameters to device...")
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
print(f"  Done in {(time.perf_counter()-t0)*1000:.0f}ms")

del layer_weights

# ── RoPE tables ──────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(MAX_SEQ, dtype=np.float32)
angles = np.outer(positions, freqs)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope(x_4d, n_heads):
    T = x_4d.shape[2]
    x_even = x_4d[..., 0::2]
    x_odd = x_4d[..., 1::2]
    cos_t = cos_table[None, None, :T, :]
    sin_t = sin_table[None, None, :T, :]
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out

# ── Forward pass ─────────────────────────────────────────────
def forward(input_ids_np):
    """Full Qwen forward. Returns logits for the last token only."""
    B, T = 1, len(input_ids_np)
    x_np = embed_w[input_ids_np].reshape(B, T, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

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

        q_4d = apply_rope(q_4d, n_q_heads)
        k_4d = apply_rope(k_4d, n_kv_heads)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d), is_causal=True)

        attn_out_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_out_np = attn_out_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_out_np), dl["o_w"])
        x_tt2 = ttnn.add(x_tt, o_tt)

        # MLP
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
        up_tt = ttnn.matmul(h2_tt, dl["up_w"])
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])
        out_tt = ttnn.add(x_tt2, down_tt)

        x_np = from_dev(out_tt, (B, T, hidden))

    # Final norm + logits (last token only)
    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt)
    logits = from_dev(logits_tt, (B, T, vocab_size))
    return logits[0, -1]  # (vocab,)

# ── Generate! ────────────────────────────────────────────────
tokens = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens))
if max_gen < args.tokens:
    print(f"Note: capping at {max_gen} tokens (full-recompute, max context={MAX_SEQ})")

print(f'\nPrompt: "{args.prompt}"')
print(f"Prompt tokens: {len(tokens)}")
print(f"Generating {max_gen} tokens on Blackhole...\n")

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

    # EOS
    if next_id == tokenizer.eos_token_id:
        break

avg_ms = np.mean(times) * 1000
print(f"\n\n--- {len(times)} tokens, avg {avg_ms:.0f}ms/tok, "
      f"{1000/avg_ms:.1f} tok/sec ---")
print(f"  First token: {times[0]*1000:.0f}ms (cold)")
if len(times) > 1:
    print(f"  Subsequent avg: {np.mean(times[1:])*1000:.0f}ms")

ttnn.close_device(device)
