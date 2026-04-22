#!/usr/bin/env python3
"""
tt-xla demo: GPT-2 text generation on Tenstorrent Blackhole.

This script loads GPT-2 small (124M params) from HuggingFace and
generates text token-by-token on a Tenstorrent Blackhole accelerator.

Requirements (on the Tenstorrent host):
  pip install ttnn safetensors huggingface_hub jax jaxlib torch

Usage:
  python3 demo.py
  python3 demo.py "Your custom prompt here"
  python3 demo.py "Your prompt" --tokens 50
"""

import sys, os, time, argparse
import numpy as np
import torch

# ── Args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GPT-2 on Tenstorrent Blackhole")
parser.add_argument("prompt", nargs="?", default="The meaning of life is",
                    help="Text prompt to complete")
parser.add_argument("--tokens", type=int, default=30, help="Tokens to generate")
parser.add_argument("--device", type=int, default=0, help="Device ID")
args = parser.parse_args()

# ── Load GPT-2 ───────────────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 small (124M params)...")
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    cfg = json.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

id_to_token = {v: k for k, v in vocab.items()}
n_heads, d_model, n_layers = cfg['n_head'], cfg['n_embd'], cfg['n_layer']
head_dim = d_model // n_heads
wte, wpe = weights["wte.weight"], weights["wpe.weight"]

# ── Tokenizer (greedy byte-level) ────────────────────────────
def encode(text):
    tokens, i = [], 0
    text_bytes = text.encode('utf-8')
    while i < len(text_bytes):
        best_len = 0
        for length in range(min(20, len(text_bytes) - i), 0, -1):
            candidate = text_bytes[i:i+length].decode('utf-8', errors='ignore')
            if i > 0 and text_bytes[i] == ord(' '):
                cand = '\u0120' + candidate[1:] if len(candidate) > 1 else '\u0120'
                if cand in vocab:
                    tokens.append(vocab[cand]); best_len = length; break
            if candidate in vocab:
                tokens.append(vocab[candidate]); best_len = length; break
        if best_len == 0:
            tokens.append(vocab.get(chr(text_bytes[i]), 0)); best_len = 1
        i += best_len
    return tokens

def decode(ids):
    return ''.join(id_to_token.get(int(i), '?').replace('\u0120', ' ') for i in ids)

# ── Open device ──────────────────────────────────────────────
import ttnn

print(f"Opening Blackhole device {args.device}...")
device = ttnn.open_device(device_id=args.device)

def to_dev(arr):
    """Upload numpy array to Blackhole."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    """Download tensor from Blackhole."""
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── Upload weights ───────────────────────────────────────────
print("Uploading 124M parameters to device...")
t0 = time.perf_counter()

layer_w = []
for i in range(n_layers):
    p = f"h.{i}"
    layer_w.append({
        'ln1_g': to_dev(weights[f"{p}.ln_1.weight"]),
        'ln1_b': to_dev(weights[f"{p}.ln_1.bias"]),
        'w_attn': to_dev(weights[f"{p}.attn.c_attn.weight"]),
        'b_attn': to_dev(weights[f"{p}.attn.c_attn.bias"]),
        'w_proj': to_dev(weights[f"{p}.attn.c_proj.weight"]),
        'b_proj': to_dev(weights[f"{p}.attn.c_proj.bias"]),
        'ln2_g': to_dev(weights[f"{p}.ln_2.weight"]),
        'ln2_b': to_dev(weights[f"{p}.ln_2.bias"]),
        'w_fc': to_dev(weights[f"{p}.mlp.c_fc.weight"]),
        'b_fc': to_dev(weights[f"{p}.mlp.c_fc.bias"]),
        'w_mlp': to_dev(weights[f"{p}.mlp.c_proj.weight"]),
        'b_mlp': to_dev(weights[f"{p}.mlp.c_proj.bias"]),
    })
ln_f_g = to_dev(weights["ln_f.weight"])
ln_f_b = to_dev(weights["ln_f.bias"])
print(f"  Done in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── GPT-2 forward pass ──────────────────────────────────────
def gpt2_layer(x, w, seq_len):
    """One transformer layer. Device ops except QKV split/head concat."""
    # LayerNorm → QKV projection
    h = ttnn.layer_norm(x, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)
    qkv = ttnn.add(ttnn.matmul(h, w['w_attn']), w['b_attn'])

    # QKV split + reshape to 4D (CPU round-trip)
    qkv_np = from_dev(qkv, (1, seq_len, 3 * d_model))
    q = qkv_np[:,:,:d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0,2,1,3)
    k = qkv_np[:,:,d_model:2*d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0,2,1,3)
    v = qkv_np[:,:,2*d_model:].reshape(1, seq_len, n_heads, head_dim).transpose(0,2,1,3)
    q_t = ttnn.from_torch(torch.from_numpy(q.copy()), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    k_t = ttnn.from_torch(torch.from_numpy(k.copy()), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    v_t = ttnn.from_torch(torch.from_numpy(v.copy()), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # FlashAttention-2 on device
    attn = ttnn.transformer.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)

    # Merge heads (CPU round-trip)
    a_np = ttnn.to_torch(attn).float().numpy()
    merged = to_dev(a_np.transpose(0,2,1,3).reshape(1, seq_len, d_model))

    # Output proj + residual
    x = ttnn.add(x, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))

    # LayerNorm → MLP
    h2 = ttnn.layer_norm(x, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))

def forward(token_ids):
    """Full GPT-2 forward. Returns logits for next token."""
    seq_len = len(token_ids)
    pad_len = ((seq_len + 31) // 32) * 32
    ids = list(token_ids) + [50256] * (pad_len - seq_len)

    x = to_dev((wte[ids] + wpe[:pad_len])[None, :, :])
    for i in range(n_layers):
        x = gpt2_layer(x, layer_w[i], pad_len)
    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)

    out = from_dev(x, (1, pad_len, d_model))
    return out[0, seq_len - 1, :] @ wte.T

# ── Generate! ────────────────────────────────────────────────
tokens = encode(args.prompt)
print(f'\nPrompt: "{args.prompt}"')
print(f"Generating {args.tokens} tokens on Blackhole...\n")

sys.stdout.write(args.prompt)
sys.stdout.flush()

times = []
for i in range(args.tokens):
    t0 = time.perf_counter()
    logits = forward(tokens)
    dt = time.perf_counter() - t0
    times.append(dt)

    next_id = int(np.argmax(logits))
    tokens.append(next_id)
    sys.stdout.write(decode([next_id]))
    sys.stdout.flush()

    if next_id == 50256:
        break

avg_ms = np.mean(times) * 1000
print(f"\n\n--- {len(times)} tokens, avg {avg_ms:.0f}ms/tok, "
      f"{1000/avg_ms:.1f} tok/sec ---")

ttnn.close_device(device)
