"""
Experiment 35: GPT-2 with KV-cached decode — unlimited token generation.

Architecture:
  PREFILL: Process full prompt through 12 layers, store K/V in caches.
           Uses regular SDPA (full sequence).
  DECODE:  For each new token, compute Q/K/V for just that token,
           update caches, run Flash-Decode against full cache.
           Fixed shape (single token) → traceable.

Key APIs proven in Experiment 33b:
  - ttnn.kv_cache.fill_cache_for_user_(cache, input, batch_index=0)
  - ttnn.kv_cache.update_cache_for_token_(cache, new_kv, update_index=pos, batch_offset=0)
  - ttnn.transformer.scaled_dot_product_attention_decode(q, k_cache, v_cache, cur_pos=[pos])
    Q: [1, batch, n_heads, head_dim], K/V: [batch, n_heads, max_seq, head_dim]

Phases:
  1. Prefill: full prompt through all layers, fill KV caches
  2. Decode loop: single-token forward, update caches, Flash-Decode
  3. Verify against full-recompute reference
  4. Benchmark decode latency
  5. Text generation demo
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch

# ── Load GPT-2 ──────────────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 small...")
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
max_seq = 1024

wte = weights["wte.weight"]      # (50257, 768)
wpe = weights["wpe.weight"]      # (1024, 768)

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

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}")

# ── Device ───────────────────────────────────────────────────
import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights (pre-split QKV)...")
t0 = time.perf_counter()

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
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── KV caches ────────────────────────────────────────────────
print("Allocating KV caches...")
k_caches = []
v_caches = []
for i in range(n_layers):
    cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
print(f"  {n_layers * 2} caches allocated ({n_layers * 2 * np.prod((1, n_heads, max_seq, head_dim)) * 2 / 1024 / 1024:.1f} MB)")


# ══════════════════════════════════════════════════════════════
# PREFILL: Process full prompt through all layers
# ══════════════════════════════════════════════════════════════

def prefill_layer(x_tt, w, layer_idx, seq_len):
    """Prefill one layer: full-sequence attention, store K/V in cache."""
    h = ttnn.layer_norm(x_tt, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)

    # Q/K/V projections
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])

    # Reshape to 4D: (1, seq, 768) → (1, 12, seq, 64)
    q = ttnn.transpose(ttnn.reshape(q, [1, seq_len, n_heads, head_dim]), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, [1, seq_len, n_heads, head_dim]), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, [1, seq_len, n_heads, head_dim]), 1, 2)

    # Store K/V in cache
    ttnn.kv_cache.fill_cache_for_user_(k_caches[layer_idx], k, batch_index=0)
    ttnn.kv_cache.fill_cache_for_user_(v_caches[layer_idx], v, batch_index=0)

    # Full-sequence attention (prefill uses regular SDPA)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    merged = ttnn.transformer.concatenate_heads(attn)

    # Output proj + residual
    x_tt = ttnn.add(x_tt, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))

    # MLP
    h2 = ttnn.layer_norm(x_tt, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x_tt, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))


def prefill(token_ids):
    """Run full prompt through all layers, fill KV caches."""
    seq_len = len(token_ids)
    pad_len = ((seq_len + 31) // 32) * 32
    ids = list(token_ids) + [50256] * (pad_len - seq_len)

    emb = (wte[ids] + wpe[:pad_len])[None, :, :]
    x = to_dev(emb)

    for i in range(n_layers):
        x = prefill_layer(x, layer_w[i], i, pad_len)

    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)
    out = from_dev(x, (1, pad_len, d_model))
    return out[0, seq_len - 1, :] @ wte.T


# ══════════════════════════════════════════════════════════════
# DECODE: Single-token forward with KV cache
# ══════════════════════════════════════════════════════════════

def decode_layer(x_tt, w, layer_idx, pos):
    """Decode one layer: single-token Q, Flash-Decode against KV cache."""
    h = ttnn.layer_norm(x_tt, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)

    # Q/K/V for single token: (1, 1, 768)
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k_new = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v_new = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])

    # Reshape K/V for cache update: (1, 1, 768) → (1, 12, 1, 64)
    k_new = ttnn.transpose(ttnn.reshape(k_new, [1, 1, n_heads, head_dim]), 1, 2)
    v_new = ttnn.transpose(ttnn.reshape(v_new, [1, 1, n_heads, head_dim]), 1, 2)

    # Update caches
    ttnn.kv_cache.update_cache_for_token_(k_caches[layer_idx], k_new,
                                           update_index=pos, batch_offset=0)
    ttnn.kv_cache.update_cache_for_token_(v_caches[layer_idx], v_new,
                                           update_index=pos, batch_offset=0)

    # Q for Flash-Decode: (1, 1, 768) → (1, 1, 12, 64)
    q = ttnn.reshape(q, [1, 1, n_heads, head_dim])

    # Flash-Decode: Q against full KV cache
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q, k_caches[layer_idx], v_caches[layer_idx], cur_pos=[pos]
    )
    # attn shape: (1, 1, 12, 64) → need (1, 1, 768)
    attn_np = ttnn.to_torch(attn).float().numpy()
    merged = to_dev(attn_np.reshape(1, 1, d_model))

    # Output proj + residual
    x_tt = ttnn.add(x_tt, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))

    # MLP
    h2 = ttnn.layer_norm(x_tt, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x_tt, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))


def decode_step(token_id, pos):
    """Single decode step: one new token → logits for next token."""
    emb = (wte[token_id] + wpe[pos])[None, None, :]  # (1, 1, 768)
    x = to_dev(emb)

    for i in range(n_layers):
        x = decode_layer(x, layer_w[i], i, pos)

    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)
    out = from_dev(x, (1, 1, d_model))
    return out[0, 0, :] @ wte.T


# ══════════════════════════════════════════════════════════════
# Phase 1: Verify correctness
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Verify KV-cached decode vs full recompute")
print("=" * 60)

prompt = "The meaning of life is"
token_ids = encode_simple(prompt)
print(f"  Prompt: '{prompt}' ({len(token_ids)} tokens)")

# Use prefill for the first forward, then compare decode step against full recompute
t0 = time.perf_counter()
prefill_logits = prefill(token_ids)
t_prefill = time.perf_counter() - t0
print(f"  Prefill: {t_prefill*1000:.0f}ms")

top5_prefill = np.argsort(prefill_logits)[-5:][::-1]
print(f"  Prefill top-5: {[decode_tokens([t]) for t in top5_prefill]}")
next_token = int(np.argmax(prefill_logits))
print(f"  Next token: '{decode_tokens([next_token])}' (id={next_token})")

# Now decode one more token
pos = len(token_ids)  # position of the token we just predicted
t0 = time.perf_counter()
decode_logits = decode_step(next_token, pos)
t_decode = time.perf_counter() - t0
print(f"\n  Decode step 1: {t_decode*1000:.0f}ms")

top5_decode = np.argsort(decode_logits)[-5:][::-1]
print(f"  Decode top-5: {[decode_tokens([t]) for t in top5_decode]}")

# Compare with full recompute of [prompt + next_token]
all_tokens = token_ids + [next_token]

# Reset caches for fresh full-recompute comparison
for i in range(n_layers):
    cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
    new_k = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    new_v = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    # Can't reassign — need to use the same cache objects.
    # For correctness check, just do a full prefill with all tokens
    pass

# Full recompute with all tokens
full_logits = prefill(all_tokens)
top5_full = np.argsort(full_logits)[-5:][::-1]
print(f"\n  Full recompute top-5: {[decode_tokens([t]) for t in top5_full]}")

# Compare
if top5_decode[0] == top5_full[0]:
    print(f"  ✓ Top-1 MATCH: '{decode_tokens([top5_decode[0]])}'")
else:
    print(f"  ✗ Top-1 MISMATCH: decode='{decode_tokens([top5_decode[0]])}' vs full='{decode_tokens([top5_full[0]])}'")

cos = np.dot(decode_logits, full_logits) / (
    np.linalg.norm(decode_logits) * np.linalg.norm(full_logits) + 1e-8)
print(f"  Logits cosine similarity: {cos:.6f}")


# ══════════════════════════════════════════════════════════════
# Phase 2: Benchmark decode latency
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Benchmark decode latency")
print("=" * 60)

# Reset caches and do a fresh prefill
for i in range(n_layers):
    cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
    k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

token_ids = encode_simple(prompt)
prefill(token_ids)
pos = len(token_ids)

# Warmup
decode_step(next_token, pos)
pos += 1

# Benchmark 10 decode steps
times = []
tok = next_token
for i in range(10):
    t0 = time.perf_counter()
    logits = decode_step(tok, pos)
    dt = time.perf_counter() - t0
    times.append(dt)
    tok = int(np.argmax(logits))
    pos += 1
    word = decode_tokens([tok])
    print(f"  Step {i+1}: '{word}' — {dt*1000:.1f}ms")

avg_ms = np.mean(times) * 1000
print(f"\n  Average decode: {avg_ms:.1f}ms/token ({1000/avg_ms:.1f} tok/sec)")
print(f"  Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms")


# ══════════════════════════════════════════════════════════════
# Phase 3: Full text generation
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Text generation with KV cache")
print("=" * 60)

# Reset caches
for i in range(n_layers):
    cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
    k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

prompts = [
    ("The meaning of life is", 30),
    ("Once upon a time", 40),
    ("Artificial intelligence will", 30),
]

for prompt_text, n_tokens in prompts:
    print(f"\n  Prompt: '{prompt_text}'")

    # Reset caches for each prompt
    for i in range(n_layers):
        cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
        k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                       dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                       dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    token_ids = encode_simple(prompt_text)

    # Prefill
    t0 = time.perf_counter()
    logits = prefill(token_ids)
    t_pf = time.perf_counter() - t0

    pos = len(token_ids)
    generated = []
    gen_times = []

    for step in range(n_tokens):
        tok = int(np.argmax(logits))
        generated.append(tok)

        if tok == 50256:
            break

        t0 = time.perf_counter()
        logits = decode_step(tok, pos)
        dt = time.perf_counter() - t0
        gen_times.append(dt)
        pos += 1

    full_text = decode_tokens(token_ids + generated)
    avg = np.mean(gen_times) * 1000 if gen_times else 0
    print(f"  Output: '{full_text}'")
    print(f"  Prefill: {t_pf*1000:.0f}ms, Decode: {avg:.1f}ms/tok avg ({len(generated)} tokens)")


# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
KV-cached GPT-2 decode on Blackhole:
  Prefill: full prompt → standard SDPA + cache fill
  Decode:  single token → Flash-Decode against KV cache

This approach:
  - Constant time per token regardless of sequence length
  - No O(n²) growth in attention computation
  - Supports up to 1024 token context (GPT-2 max)
  - Decode is fixed-shape → potentially traceable
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
