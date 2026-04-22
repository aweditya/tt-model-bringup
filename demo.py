#!/usr/bin/env python3
"""
tt-xla demo: GPT-2 text generation on Tenstorrent Blackhole.

This script loads GPT-2 small (124M params) from HuggingFace and
generates text token-by-token on a Tenstorrent Blackhole accelerator.

Zero CPU round-trips in the forward pass: QKV weights are pre-split
at upload time, all reshapes happen on-device, and the full 12-layer
forward is captured as a TT-NN trace for replay.

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

# ── Upload weights (pre-split QKV) ──────────────────────────
print("Uploading 124M parameters to device (pre-splitting QKV)...")
t0 = time.perf_counter()

layer_w = []
for i in range(n_layers):
    p = f"h.{i}"
    w_attn = weights[f"{p}.attn.c_attn.weight"]  # (768, 2304)
    b_attn = weights[f"{p}.attn.c_attn.bias"]     # (2304,)
    layer_w.append({
        'ln1_g': to_dev(weights[f"{p}.ln_1.weight"]),
        'ln1_b': to_dev(weights[f"{p}.ln_1.bias"]),
        # Pre-split QKV: 3 separate (768,768) weight matrices
        'w_q': to_dev(w_attn[:, :d_model]),
        'w_k': to_dev(w_attn[:, d_model:2*d_model]),
        'w_v': to_dev(w_attn[:, 2*d_model:]),
        'b_q': to_dev(b_attn[:d_model]),
        'b_k': to_dev(b_attn[d_model:2*d_model]),
        'b_v': to_dev(b_attn[2*d_model:]),
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

# ── GPT-2 forward pass (zero CPU round-trips) ───────────────
def gpt2_layer(x, w, seq_len):
    """One transformer layer. All ops on device — zero CPU round-trips."""
    # LayerNorm 1
    h = ttnn.layer_norm(x, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)

    # Separate Q, K, V matmuls (weights pre-split at upload time)
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])

    # Reshape + transpose on device: (1,T,768) → (1,12,T,64)
    q = ttnn.transpose(ttnn.reshape(q, [1, seq_len, n_heads, head_dim]), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, [1, seq_len, n_heads, head_dim]), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, [1, seq_len, n_heads, head_dim]), 1, 2)

    # FlashAttention-2 on device
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Merge heads on device
    merged = ttnn.transformer.concatenate_heads(attn)

    # Output proj + residual
    x = ttnn.add(x, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))

    # LayerNorm 2 → MLP
    h2 = ttnn.layer_norm(x, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))

def forward_body(x, pad_len):
    """12-layer transformer + final LN. Fully on-device, traceable."""
    for i in range(n_layers):
        x = gpt2_layer(x, layer_w[i], pad_len)
    return ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)

# ── Trace capture ────────────────────────────────────────────
# Capture traces for multiple pad lengths to cover typical generation.
# All 5 buckets fit simultaneously in device memory (tested in exp 37b).
TRACE_PAD_LENS = [32, 64, 128, 256, 512]
traces = {}  # pad_len -> (trace_id, input_buf, output_buf)

for pad_len in TRACE_PAD_LENS:
    print(f"Capturing trace for seq_len={pad_len}...")
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)

    # Warmup (establishes buffer sizes)
    x_warm = to_dev(dummy)
    x_warm = forward_body(x_warm, pad_len)

    # Capture
    x_in = to_dev(dummy)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    x_out = forward_body(x_in, pad_len)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Warmup replay
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    traces[pad_len] = (tid, x_in, x_out)

print(f"  {len(traces)} traces captured!")

def forward(token_ids):
    """Full GPT-2 forward. Returns logits for next token."""
    seq_len = len(token_ids)
    pad_len = ((seq_len + 31) // 32) * 32
    ids = list(token_ids) + [50256] * (pad_len - seq_len)

    emb = (wte[ids] + wpe[:pad_len])[None, :, :]

    if pad_len in traces:
        # Fast path: write embeddings into trace input buffer, replay
        tid, x_in, x_out = traces[pad_len]
        emb_t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
        while emb_t.dim() < 2:
            emb_t = emb_t.unsqueeze(0)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(emb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            x_in
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        out = from_dev(x_out, (1, pad_len, d_model))
    else:
        # Fallback: untraced for uncommon lengths
        x = to_dev(emb)
        x = forward_body(x, pad_len)
        out = from_dev(x, (1, pad_len, d_model))

    return out[0, seq_len - 1, :] @ wte.T

# ── Generate! ────────────────────────────────────────────────
MAX_TRACED = max(TRACE_PAD_LENS)
tokens = encode(args.prompt)
max_gen = min(args.tokens, MAX_TRACED - len(tokens))
if max_gen < args.tokens:
    print(f"Note: capping generation at {max_gen} tokens (max context = {MAX_TRACED})")
    print(f"  Prompt uses {len(tokens)} tokens, {MAX_TRACED - len(tokens)} slots remaining.")

print(f'\nPrompt: "{args.prompt}"')
print(f"Generating {max_gen} tokens on Blackhole...\n")

sys.stdout.write(args.prompt)
sys.stdout.flush()

times = []
for i in range(max_gen):
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

for tid, _, _ in traces.values():
    ttnn.release_trace(device, tid)
ttnn.close_device(device)
