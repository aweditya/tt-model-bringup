#!/usr/bin/env python3
"""
Experiment 76: Numpy float32 reference for Llama-3.1-8B-Instruct.

Exp 75 showed 8B model degenerates to word salad at ~40-80 tokens.
Research says this is NOT expected — Llama-3.1-8B-Instruct should sustain 200+ tokens.
This means we have a bug.

This experiment runs a PURE NUMPY float32 reference to isolate:
  - If numpy also produces word salad → bug is in weight loading or architecture
  - If numpy is coherent → bug is in TT-NN precision or KV cache

Process one layer at a time to fit in memory (~16GB weights).
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

np.random.seed(42)

# Llama-3.1-8B architecture
hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
n_layers = 32; vocab_size = 128256; rms_eps = 1e-5; rope_theta = 500000.0
intermediate_size = 14336

print("="*70)
print("Experiment 76: Numpy float32 Reference — Llama-3.1-8B-Instruct")
print("="*70)

# ── Load model ──
print("\nLoading model...")
model_ids = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]
shard_paths = []
model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths
            model_id = mid
            print(f"  Found: {mid} ({n_shards} shards)")
            break
        except Exception:
            pass
    if shard_paths:
        break

# Load all weights into memory (float32)
print("  Loading weights...")
t0 = time.perf_counter()
all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()
dt = time.perf_counter() - t0
print(f"  Loaded {len(all_weights)} tensors in {dt:.0f}s")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w)

total_params = sum(v.size for v in all_weights.values())
print(f"  Total: {total_params/1e9:.1f}B params")

# Tokenizer
tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)


# ── Numpy reference functions ──

def rms_norm(x, g, eps=1e-5):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * g

def silu(x):
    return x / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def apply_rope(x, pos):
    """Apply RoPE to x at given position(s). x: (..., head_dim)"""
    if np.isscalar(pos):
        angles = pos * freqs
    else:
        angles = np.outer(pos, freqs)
    cos_t = np.repeat(np.cos(angles), 2, axis=-1)
    sin_t = np.repeat(np.sin(angles), 2, axis=-1)
    if x.ndim == 4:  # (B, H, T, D)
        cos_t = cos_t[None, None]
        sin_t = sin_t[None, None]
    elif x.ndim == 3:  # (B, H, D)
        cos_t = cos_t[None]
        sin_t = sin_t[None]
    return x * cos_t + rotate_interleaved(x) * sin_t


# ── KV cache (numpy float32) ──
kv_cache = [{'k': np.zeros((1, n_kv_heads, 512, head_dim), dtype=np.float32),
             'v': np.zeros((1, n_kv_heads, 512, head_dim), dtype=np.float32)}
            for _ in range(n_layers)]


def np_prefill(token_ids):
    """Full prefill in numpy float32."""
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)  # (1, T, hidden)

    positions = np.arange(T)

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        ln1_g = all_weights[prefix + "input_layernorm.weight"]
        q_w = all_weights[prefix + "self_attn.q_proj.weight"]
        k_w = all_weights[prefix + "self_attn.k_proj.weight"]
        v_w = all_weights[prefix + "self_attn.v_proj.weight"]
        o_w = all_weights[prefix + "self_attn.o_proj.weight"]
        ln2_g = all_weights[prefix + "post_attention_layernorm.weight"]
        gate_w = all_weights[prefix + "mlp.gate_proj.weight"]
        up_w = all_weights[prefix + "mlp.up_proj.weight"]
        down_w = all_weights[prefix + "mlp.down_proj.weight"]

        # RMS norm
        h = rms_norm(x, ln1_g, rms_eps)

        # QKV projections
        q = (h @ q_w.T).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = (h @ k_w.T).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = (h @ v_w.T).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        # Store in KV cache
        kv_cache[i]['k'][:, :, :T, :] = k
        kv_cache[i]['v'][:, :, :T, :] = v

        # GQA attention
        n_rep = n_q_heads // n_kv_heads  # 4
        k_exp = np.repeat(k, n_rep, axis=1)  # (B, 32, T, D)
        v_exp = np.repeat(v, n_rep, axis=1)

        scores = np.matmul(q, k_exp.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        # Causal mask
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        scores = scores + mask[None, None]
        attn = softmax(scores)
        out = np.matmul(attn, v_exp)  # (B, H, T, D)

        # Output projection
        out = out.transpose(0, 2, 1, 3).reshape(B, T, n_q_heads * head_dim)
        o = out @ o_w.T

        # Residual
        x2 = x + o

        # MLP
        h2 = rms_norm(x2, ln2_g, rms_eps)
        g = silu(h2 @ gate_w.T) * (h2 @ up_w.T)
        d = g @ down_w.T
        x = x2 + d

    # Final norm + LM head
    x = rms_norm(x, final_norm_g, rms_eps)
    logits = x @ lm_head_w.T  # (B, T, vocab)
    return logits[0, -1]  # last token logits


def np_decode_step(token_id, pos):
    """Single decode step in numpy float32."""
    B = 1
    x = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        ln1_g = all_weights[prefix + "input_layernorm.weight"]
        q_w = all_weights[prefix + "self_attn.q_proj.weight"]
        k_w = all_weights[prefix + "self_attn.k_proj.weight"]
        v_w = all_weights[prefix + "self_attn.v_proj.weight"]
        o_w = all_weights[prefix + "self_attn.o_proj.weight"]
        ln2_g = all_weights[prefix + "post_attention_layernorm.weight"]
        gate_w = all_weights[prefix + "mlp.gate_proj.weight"]
        up_w = all_weights[prefix + "mlp.up_proj.weight"]
        down_w = all_weights[prefix + "mlp.down_proj.weight"]

        h = rms_norm(x, ln1_g, rms_eps)

        q = (h @ q_w.T).reshape(B, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = (h @ k_w.T).reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = (h @ v_w.T).reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, np.array([pos]))
        k = apply_rope(k, np.array([pos]))

        kv_cache[i]['k'][:, :, pos:pos+1, :] = k
        kv_cache[i]['v'][:, :, pos:pos+1, :] = v

        # Attend to all cached positions [0..pos]
        k_all = kv_cache[i]['k'][:, :, :pos+1, :]
        v_all = kv_cache[i]['v'][:, :, :pos+1, :]

        n_rep = n_q_heads // n_kv_heads
        k_exp = np.repeat(k_all, n_rep, axis=1)
        v_exp = np.repeat(v_all, n_rep, axis=1)

        scores = np.matmul(q, k_exp.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn = softmax(scores)
        out = np.matmul(attn, v_exp)

        out = out.transpose(0, 2, 1, 3).reshape(B, 1, n_q_heads * head_dim)
        o = out @ o_w.T
        x2 = x + o

        h2 = rms_norm(x2, ln2_g, rms_eps)
        g = silu(h2 @ gate_w.T) * (h2 @ up_w.T)
        d = g @ down_w.T
        x = x2 + d

    x = rms_norm(x, final_norm_g, rms_eps)
    logits = x @ lm_head_w.T
    return logits[0, 0]


# ── Chat template ──
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

def make_chat_tokens(prompt, system="You are a helpful assistant."):
    return ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
            [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
            [start_header] + enc("assistant") + [end_header] + enc("\n\n"))


def sample_production(logits, generated_ids, temperature=0.7, min_p=0.05, rep_penalty=1.1, rep_window=64):
    logits = logits.copy()
    recent = generated_ids[-rep_window:] if generated_ids else []
    for tok_id in set(recent):
        if logits[tok_id] > 0:
            logits[tok_id] /= rep_penalty
        else:
            logits[tok_id] *= rep_penalty
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    top_prob = probs.max()
    threshold = min_p * top_prob
    mask = probs >= threshold
    if mask.sum() == 0:
        mask[np.argmax(probs)] = True
    probs = probs * mask
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ══════════════════════════════════════════════════════════════
# Test 1: Short factual Q&A (greedy)
# ══════════════════════════════════════════════════════════════

prompt = "What is the capital of France?"
tokens = make_chat_tokens(prompt)
print(f"\nTest 1: {prompt}")
print(f"  Prompt tokens: {len(tokens)}")

t0 = time.perf_counter()
logits = np_prefill(np.array(tokens))
dt = time.perf_counter() - t0
print(f"  Prefill: {dt:.1f}s")

next_id = int(np.argmax(logits))
gen_tokens = [next_id]
pos = len(tokens)

for step in range(50):
    t0 = time.perf_counter()
    logits = np_decode_step(next_id, pos)
    dt = time.perf_counter() - t0
    next_id = int(np.argmax(logits))
    gen_tokens.append(next_id)
    pos += 1
    if step < 3 or step % 10 == 0:
        print(f"  Step {step}: {dt:.2f}s, token={next_id} ({tokenizer.decode([next_id])})")
    if next_id in stop_ids:
        print(f"  [EOS at step {step}]")
        break

text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print(f"\n  GREEDY OUTPUT ({len(gen_tokens)} tokens):")
print(f"  {text}")

# ══════════════════════════════════════════════════════════════
# Test 2: Multi-paragraph (production sampling) — THE REAL TEST
# ══════════════════════════════════════════════════════════════

# Reset KV cache
for c in kv_cache:
    c['k'][:] = 0; c['v'][:] = 0

prompt2 = "Explain quantum computing in simple terms. Include what qubits are, how superposition works, and why quantum computers are faster."
tokens2 = make_chat_tokens(prompt2)
print(f"\n{'='*60}")
print(f"Test 2: {prompt2[:60]}...")
print(f"  Prompt tokens: {len(tokens2)}")

np.random.seed(42)
logits = np_prefill(np.array(tokens2))
next_id = sample_production(logits, [])
gen_tokens2 = [next_id]
pos2 = len(tokens2)

for step in range(200):
    logits = np_decode_step(next_id, pos2)
    next_id = sample_production(logits, gen_tokens2)
    gen_tokens2.append(next_id)
    pos2 += 1
    if step % 20 == 0:
        partial = tokenizer.decode(gen_tokens2[-10:], skip_special_tokens=True)
        print(f"  Step {step}: ...{partial}")
    if next_id in stop_ids:
        print(f"  [EOS at step {step}]")
        break

text2 = tokenizer.decode(gen_tokens2, skip_special_tokens=True)
print(f"\n  PRODUCTION SAMPLING OUTPUT ({len(gen_tokens2)} tokens):")
for line in text2.split('\n')[:30]:
    print(f"  {line}")

# ══════════════════════════════════════════════════════════════
# Test 3: Creative writing (production sampling)
# ══════════════════════════════════════════════════════════════

for c in kv_cache:
    c['k'][:] = 0; c['v'][:] = 0

prompt3 = "Write a short story about a robot that learns to paint. Include dialogue."
tokens3 = make_chat_tokens(prompt3)
print(f"\n{'='*60}")
print(f"Test 3: {prompt3}")
print(f"  Prompt tokens: {len(tokens3)}")

np.random.seed(42)
logits = np_prefill(np.array(tokens3))
next_id = sample_production(logits, [])
gen_tokens3 = [next_id]
pos3 = len(tokens3)

for step in range(200):
    logits = np_decode_step(next_id, pos3)
    next_id = sample_production(logits, gen_tokens3)
    gen_tokens3.append(next_id)
    pos3 += 1
    if step % 20 == 0:
        partial = tokenizer.decode(gen_tokens3[-10:], skip_special_tokens=True)
        print(f"  Step {step}: ...{partial}")
    if next_id in stop_ids:
        print(f"  [EOS at step {step}]")
        break

text3 = tokenizer.decode(gen_tokens3, skip_special_tokens=True)
print(f"\n  CREATIVE OUTPUT ({len(gen_tokens3)} tokens):")
for line in text3.split('\n')[:30]:
    print(f"  {line}")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"SUMMARY: Numpy float32 Reference — Llama-3.1-8B-Instruct")
print(f"{'='*70}")
print(f"Test 1 (Short Q&A, greedy): {len(gen_tokens)} tokens")
print(f"  → {tokenizer.decode(gen_tokens, skip_special_tokens=True)[:100]}")
print(f"Test 2 (Quantum, production): {len(gen_tokens2)} tokens")
print(f"  → Coherent: {'YES' if len(gen_tokens2) > 100 else 'PARTIAL'}")
print(f"Test 3 (Creative, production): {len(gen_tokens3)} tokens")
print(f"  → Coherent: {'YES' if len(gen_tokens3) > 100 else 'PARTIAL'}")
print(f"\nIf numpy produces coherent text where TT-NN doesn't → bf16 precision bug")
print(f"If numpy also produces word salad → bug in weight loading or architecture")

print("\nDone!")
