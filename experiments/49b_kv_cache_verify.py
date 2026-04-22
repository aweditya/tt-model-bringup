#!/usr/bin/env python3
"""
Experiment 49b: Verify KV-cached decode correctness.

Compare KV-cached decode output against full-recompute reference
for the first 5 generated tokens. This ensures the cache mechanism
(fill_cache_for_user_, update_cache_for_token_, attention_decode)
produces correct results.

Also adds temperature/top-k sampling for quality generation.
"""

import sys, os, argparse, time
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

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=30)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--top-k", type=int, default=50)
args = parser.parse_args()

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936
MAX_SEQ = 256

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

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
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# Upload weights
print("Uploading weights...")
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
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

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def apply_rope_np(x_4d, cos_t, sin_t):
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_t[None, None, :, :] - x_4d[..., 1::2] * sin_t[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_t[None, None, :, :] + x_4d[..., 1::2] * cos_t[None, None, :, :]
    return out

def apply_rope_single(x_4d, pos):
    angles = pos * freqs
    c = np.cos(angles).astype(np.float32).reshape(1, 1, 1, -1)
    s = np.sin(angles).astype(np.float32).reshape(1, 1, 1, -1)
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * c - x_4d[..., 1::2] * s
    out[..., 1::2] = x_4d[..., 0::2] * s + x_4d[..., 1::2] * c
    return out

# ── Full recompute forward (reference) ──────────────────────
def full_forward(token_ids):
    """Full recompute forward — no KV cache, for reference."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.cos(angles).astype(np.float32)
    sin_t = np.sin(angles).astype(np.float32)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, T, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, T, vocab_size))[0, -1]

# ── KV cache setup ───────────────────────────────────────────
k_caches = []
v_caches = []
for i in range(n_layers):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))

def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t, sin_t = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        k_4d_tt = to_dev_4d(k_4d)
        v_4d_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), k_4d_tt, v_4d_tt,
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, T, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, T, vocab_size))[0, -1]

def decode_step(token_id, pos):
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, 1, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, 1, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, 1, n_kv_heads * head_dim))

        q_4d = apply_rope_single(q_np.reshape(B, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3), pos)
        k_4d = apply_rope_single(k_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3), pos)
        v_4d = v_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        ttnn.kv_cache.update_cache_for_token_(k_caches[i], to_dev_4d(k_4d),
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], to_dev_4d(v_4d),
                                               update_index=pos, batch_offset=0)

        q_decode = to_dev_4d(q_4d.transpose(0, 2, 1, 3))
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos], compute_kernel_config=hifi4)
        attn_np = ttnn.to_torch(attn).float().numpy()
        merged = to_dev(attn_np.reshape(B, 1, hidden))

        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, 1, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, 1, vocab_size))[0, 0]

def sample_top_k(logits, temp=0.7, top_k=50):
    """Temperature + top-k sampling."""
    logits = logits / temp
    top_idx = np.argsort(logits)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    chosen = np.random.choice(top_idx, p=probs)
    return int(chosen)

# ══════════════════════════════════════════════════════════════
# Phase 1: Verify KV cache correctness
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: KV Cache Correctness Verification")
print("=" * 60)

tokens = list(tokenizer.encode(args.prompt))
print(f'Prompt: "{args.prompt}" ({len(tokens)} tokens)')

# Prefill
prefill_logits = prefill(np.array(tokens))
prefill_ref = full_forward(np.array(tokens))
cos_prefill = cosine(prefill_logits, prefill_ref)
print(f"\nPrefill logit cosine vs full recompute: {cos_prefill:.6f}")
print(f"  Top-1 match: {np.argmax(prefill_logits) == np.argmax(prefill_ref)}")

# Decode 5 tokens and compare each against full recompute
print(f"\nDecoding 5 tokens with correctness check:")
next_id = int(np.argmax(prefill_logits))
tokens.append(next_id)

for step in range(5):
    pos = len(tokens) - 1
    decode_logits = decode_step(next_id, pos)

    # Full recompute for comparison
    full_logits = full_forward(np.array(tokens))

    cos_val = cosine(decode_logits, full_logits)
    top1_match = np.argmax(decode_logits) == np.argmax(full_logits)
    decode_tok = tokenizer.decode([np.argmax(decode_logits)])
    full_tok = tokenizer.decode([np.argmax(full_logits)])

    print(f"  Step {step}: cosine={cos_val:.6f} top1={'MATCH' if top1_match else 'MISMATCH'}"
          f" (decode='{decode_tok}', full='{full_tok}')")

    next_id = int(np.argmax(decode_logits))
    tokens.append(next_id)

# ══════════════════════════════════════════════════════════════
# Phase 2: Generate with temperature sampling
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Phase 2: Generate with temp={args.temp}, top_k={args.top_k}")
print("=" * 60)

# Reset caches
for i in range(n_layers):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

tokens = list(tokenizer.encode(args.prompt))
logits = prefill(np.array(tokens))
next_id = sample_top_k(logits, args.temp, args.top_k)
tokens.append(next_id)

sys.stdout.write(f'\n{args.prompt}{tokenizer.decode([next_id])}')
sys.stdout.flush()

decode_times = []
for i in range(args.tokens - 1):
    pos = len(tokens) - 1
    t0 = time.perf_counter()
    logits = decode_step(next_id, pos)
    dt = time.perf_counter() - t0
    decode_times.append(dt)

    next_id = sample_top_k(logits, args.temp, args.top_k)
    tokens.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()

    if next_id == tokenizer.eos_token_id:
        break

if decode_times:
    avg = np.mean(decode_times) * 1000
    print(f"\n\n--- {len(decode_times)} tokens, avg {avg:.0f}ms/tok, {1000/avg:.1f} tok/sec ---")

ttnn.close_device(device)
