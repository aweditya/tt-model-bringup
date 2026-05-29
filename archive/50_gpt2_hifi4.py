#!/usr/bin/env python3
"""
Experiment 50: GPT-2 with HiFi4+fp32 — does it improve precision?

Apply the Qwen HiFi4 fix to GPT-2. Compare default vs HiFi4 all-ops
on the full 12-layer model.

GPT-2 has 12 layers (vs Qwen's 24), so precision was never as severe.
But let's quantify the improvement.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json
import ttnn

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)

# ── Load GPT-2 ──────────────────────────────────────────────
print("Loading GPT-2...")
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    config = json.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

id_to_token = {v: k for k, v in vocab.items()}
n_heads = config['n_head']       # 12
d_model = config['n_embd']       # 768
head_dim = d_model // n_heads    # 64
n_layers = config['n_layer']     # 12
vocab_size = 50257

wte = weights["wte.weight"]
wpe = weights["wpe.weight"]

def encode_simple(text):
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

def decode_tokens(ids):
    return ''.join(id_to_token.get(int(i), '?').replace('\u0120', ' ') for i in ids)

# ── Numpy reference ──────────────────────────────────────────
def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm_np(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

prompt = "The meaning of life is"
token_ids = encode_simple(prompt)
T = len(token_ids)
print(f"Prompt: '{prompt}' ({T} tokens)")

# Numpy forward
ref_x = (wte[token_ids] + wpe[:T])[None, :, :]  # (1, T, 768)
for i in range(n_layers):
    p = f"h.{i}"
    h = layer_norm_np(ref_x, weights[f"{p}.ln_1.weight"], weights[f"{p}.ln_1.bias"])
    w_attn = weights[f"{p}.attn.c_attn.weight"]
    b_attn = weights[f"{p}.attn.c_attn.bias"]
    qkv = h @ w_attn + b_attn
    q = qkv[:, :, :d_model].reshape(1, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = qkv[:, :, d_model:2*d_model].reshape(1, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = qkv[:, :, 2*d_model:].reshape(1, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    sc = (q @ k.transpose(0,1,3,2)) / np.sqrt(head_dim)
    sc += np.triu(np.ones((T,T))*-1e9, k=1)[None,None]
    attn = (softmax_np(sc) @ v).transpose(0,2,1,3).reshape(1, T, d_model)
    ref_x = ref_x + attn @ weights[f"{p}.attn.c_proj.weight"] + weights[f"{p}.attn.c_proj.bias"]
    h2 = layer_norm_np(ref_x, weights[f"{p}.ln_2.weight"], weights[f"{p}.ln_2.bias"])
    ff = gelu_np(h2 @ weights[f"{p}.mlp.c_fc.weight"] + weights[f"{p}.mlp.c_fc.bias"])
    ref_x = ref_x + ff @ weights[f"{p}.mlp.c_proj.weight"] + weights[f"{p}.mlp.c_proj.bias"]

ref_x = layer_norm_np(ref_x, weights["ln_f.weight"], weights["ln_f.bias"])
ref_logits = (ref_x @ wte.T).reshape(1, T, vocab_size)
ref_top5 = np.argsort(ref_logits[0, -1])[-5:][::-1]
print(f"Reference top-5: {[decode_tokens([t]) for t in ref_top5]}")

# ── Device ───────────────────────────────────────────────────
device = ttnn.open_device(device_id=0)

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# Upload weights
print("Uploading weights...")
layer_w = []
for i in range(n_layers):
    p = f"h.{i}"
    w_attn = weights[f"{p}.attn.c_attn.weight"]
    b_attn = weights[f"{p}.attn.c_attn.bias"]
    layer_w.append({
        'ln1_g': to_dev(weights[f"{p}.ln_1.weight"]),
        'ln1_b': to_dev(weights[f"{p}.ln_1.bias"]),
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

def gpt2_forward(token_ids, use_hifi4=False):
    """Run GPT-2 forward, optionally with HiFi4+fp32."""
    T = len(token_ids)
    emb = (wte[token_ids] + wpe[:T])[None, :, :]
    x = to_dev(emb)
    cfg = {"compute_kernel_config": hifi4} if use_hifi4 else {}

    for i in range(n_layers):
        w = layer_w[i]
        h = ttnn.layer_norm(x, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)
        q = ttnn.add(ttnn.matmul(h, w['w_q'], **cfg), w['b_q'])
        k = ttnn.add(ttnn.matmul(h, w['w_k'], **cfg), w['b_k'])
        v = ttnn.add(ttnn.matmul(h, w['w_v'], **cfg), w['b_v'])
        q = ttnn.transpose(ttnn.reshape(q, [1, T, n_heads, head_dim]), 1, 2)
        k = ttnn.transpose(ttnn.reshape(k, [1, T, n_heads, head_dim]), 1, 2)
        v = ttnn.transpose(ttnn.reshape(v, [1, T, n_heads, head_dim]), 1, 2)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, **cfg)
        merged = ttnn.transformer.concatenate_heads(attn)
        x = ttnn.add(x, ttnn.add(ttnn.matmul(merged, w['w_proj'], **cfg), w['b_proj']))
        h2 = ttnn.layer_norm(x, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
        ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc'], **cfg), w['b_fc']),
                       fast_and_approximate_mode=False)
        x = ttnn.add(x, ttnn.add(ttnn.matmul(ff, w['w_mlp'], **cfg), w['b_mlp']))

    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)
    out = from_dev(x, (1, T, d_model))
    return (out @ wte.T).reshape(1, T, vocab_size)

# ── Compare ──────────────────────────────────────────────────
print("\n" + "=" * 60)
for name, use_hifi4 in [("Default", False), ("HiFi4+fp32", True)]:
    t0 = time.perf_counter()
    logits = gpt2_forward(token_ids, use_hifi4=use_hifi4)
    dt = (time.perf_counter() - t0) * 1000

    last_cos = cosine(logits[0, -1], ref_logits[0, -1])
    top5 = np.argsort(logits[0, -1])[-5:][::-1]
    top1_match = top5[0] == ref_top5[0]
    overlap = len(set(top5.tolist()) & set(ref_top5.tolist()))

    print(f"\n{name}:")
    print(f"  Logit cosine: {last_cos:.6f}")
    print(f"  Top-1 match:  {'YES' if top1_match else 'NO'}")
    print(f"  Top-5 overlap: {overlap}/5")
    print(f"  Top-5: {[decode_tokens([t]) for t in top5]}")
    print(f"  Forward time: {dt:.0f}ms")

ttnn.close_device(device)
print("\nDone!")
