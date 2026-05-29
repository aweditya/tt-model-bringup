#!/usr/bin/env python3
"""
Experiment 49: Qwen2.5-0.5B with KV-cached decode.

Port the GPT-2 KV cache approach (exp 35) to Qwen:
  PREFILL: Process full prompt through 24 layers, store K/V in caches
  DECODE:  Single-token forward, update KV caches, Flash-Decode

Key differences from GPT-2:
  - GQA: 14 Q heads, 2 KV heads → KV caches only have 2 heads
  - RoPE instead of learned position embeddings
  - RMSNorm instead of LayerNorm
  - SiLU/SwiGLU instead of GELU in MLP
  - HiFi4+fp32 on all compute ops (from exp 46e)

Expected: ~10x speedup over exp 47 (full recompute) for decode.
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
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936
MAX_SEQ = 256

# ── HiFi4 config ────────────────────────────────────────────
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

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

# ── Device + helpers ─────────────────────────────────────────
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

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights...")
t0 = time.perf_counter()
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
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── RoPE ─────────────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def apply_rope_np(x_4d, cos_t, sin_t):
    """x_4d: (B, heads, T, head_dim), cos/sin: (T, head_dim//2)."""
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_t[None, None, :, :] - x_4d[..., 1::2] * sin_t[None, None, :, :]
    out[..., 1::2] = x_4d[..., 0::2] * sin_t[None, None, :, :] + x_4d[..., 1::2] * cos_t[None, None, :, :]
    return out

def apply_rope_single(x_4d, pos):
    """Apply RoPE for a single position. x_4d: (B, heads, 1, head_dim)."""
    angles = pos * freqs  # (head_dim//2,)
    cos_t = np.cos(angles).astype(np.float32).reshape(1, 1, 1, -1)
    sin_t = np.sin(angles).astype(np.float32).reshape(1, 1, 1, -1)
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_4d[..., 0::2] * cos_t - x_4d[..., 1::2] * sin_t
    out[..., 1::2] = x_4d[..., 0::2] * sin_t + x_4d[..., 1::2] * cos_t
    return out

# ── KV Caches ────────────────────────────────────────────────
print("Allocating KV caches...")
k_caches = []
v_caches = []
for i in range(n_layers):
    cache_np = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
cache_mb = n_layers * 2 * 1 * n_kv_heads * MAX_SEQ * head_dim * 2 / 1024 / 1024
print(f"  {n_layers * 2} caches ({cache_mb:.1f} MB) — only {n_kv_heads} KV heads per layer!")

# ══════════════════════════════════════════════════════════════
# PREFILL: Full prompt through all layers
# ══════════════════════════════════════════════════════════════

def prefill(token_ids):
    """Process full prompt, fill KV caches, return last-token logits."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        # Pull Q/K for RoPE, reshape
        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Store K/V in cache
        k_4d_tt = to_dev_4d(k_4d)
        v_4d_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        # Full-sequence SDPA for prefill
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

    # Final norm + logits
    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    logits = from_dev(logits_tt, (B, T, vocab_size))
    return logits[0, -1]

# ══════════════════════════════════════════════════════════════
# DECODE: Single-token forward with KV cache
# ══════════════════════════════════════════════════════════════

def decode_step(token_id, pos):
    """Single-token decode: compute new Q/K/V, update caches, Flash-Decode."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        # Pull Q/K for RoPE (single position)
        q_np = from_dev(q_tt, (B, 1, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, 1, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, 1, n_kv_heads * head_dim))

        q_4d = apply_rope_single(q_np.reshape(B, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3), pos)
        k_4d = apply_rope_single(k_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3), pos)
        v_4d = v_np.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Update KV caches
        k_new_tt = to_dev_4d(k_4d)
        v_new_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_new_tt,
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_new_tt,
                                               update_index=pos, batch_offset=0)

        # Flash-Decode: Q (1, 1, n_q_heads, head_dim) against full KV cache
        # Note: Q needs shape (1, 1, n_q_heads, head_dim) for decode
        q_decode = to_dev_4d(q_4d.transpose(0, 2, 1, 3))  # (1, 1, n_q_heads, head_dim)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos],
            compute_kernel_config=hifi4,
        )

        # attn: (1, 1, n_q_heads, head_dim) → (1, 1, hidden)
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
    logits = from_dev(logits_tt, (B, 1, vocab_size))
    return logits[0, 0]

# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════
tokens = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens))

print(f'\nPrompt: "{args.prompt}" ({len(tokens)} tokens)')
print(f"Generating {max_gen} tokens with KV-cached decode...\n")

# Phase 1: Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens))
t_prefill = time.perf_counter() - t0

next_id = int(np.argmax(logits))
tokens.append(next_id)
sys.stdout.write(args.prompt + tokenizer.decode([next_id]))
sys.stdout.flush()

print(f"\n  [prefill: {t_prefill*1000:.0f}ms for {len(tokens)-1} tokens]")

# Phase 2: Decode loop
decode_times = []
for i in range(max_gen - 1):
    pos = len(tokens) - 1
    t0 = time.perf_counter()
    logits = decode_step(next_id, pos)
    dt = time.perf_counter() - t0
    decode_times.append(dt)

    next_id = int(np.argmax(logits))
    tokens.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()

    if next_id == tokenizer.eos_token_id:
        break

# ── Summary ──────────────────────────────────────────────────
print("\n")
if decode_times:
    avg_decode = np.mean(decode_times) * 1000
    print(f"--- KV-Cached Decode ---")
    print(f"  Prefill: {t_prefill*1000:.0f}ms ({len(tokens)-max_gen} tokens)")
    print(f"  Decode: avg {avg_decode:.0f}ms/tok, {1000/avg_decode:.1f} tok/sec")
    if len(decode_times) > 1:
        print(f"    First decode: {decode_times[0]*1000:.0f}ms")
        print(f"    Subsequent:   {np.mean(decode_times[1:])*1000:.0f}ms avg")
    print(f"  Total: {len(decode_times)} decode tokens")
    print(f"\n  Baseline (exp 47, full recompute): 54ms/tok, 18.4 tok/sec")
    print(f"  Improvement: {54/avg_decode:.1f}x" if avg_decode > 0 else "")

ttnn.close_device(device)
