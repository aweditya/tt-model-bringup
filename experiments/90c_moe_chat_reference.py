#!/usr/bin/env python3
"""
Experiment 90c: Qwen1.5-MoE-A2.7B-Chat — Numpy reference with chat template.

Test if the Chat model produces coherent text vs the base model's garbage.
If Chat works, switch our TT-NN experiment to use it.

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/90c_moe_chat_reference.py'
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ── Architecture (same as base model) ───────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936
n_experts = 60; top_k = 4

print("=" * 60)
print("Exp 90c: Qwen1.5-MoE-A2.7B-Chat Numpy Reference")
print("=" * 60)

model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
n_shards = 8
print(f"\nDownloading {model_id}...")
shard_paths = [hf_hub_download(model_id, f"model-{i+1:05d}-of-{n_shards:05d}.safetensors")
               for i in range(n_shards)]

key_to_path = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            key_to_path[key] = path

tokenizer = AutoTokenizer.from_pretrained(model_id)

def load_np(key):
    with safe_open(key_to_path[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()

# ── Numpy ops ───────────────────────────────────────────────
def rms_norm(x, g, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * g

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def rotate_half(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def apply_rope(x_4d, positions):
    """x_4d: [B, n_heads, T, head_dim], positions: array of length T"""
    angles = np.outer(positions, freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return x_4d * cos_t[None, None] + rotate_half(x_4d) * sin_t[None, None]

def causal_attention(q, k, v):
    B, H, T, D = q.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    mask = np.triu(np.full((T, T), -1e9), k=1)
    scores = scores + mask[None, None]
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

def kv_attention(q, k_cache, v_cache, pos):
    """Single-token decode attention against KV cache."""
    # q: [B, H, 1, D], k_cache/v_cache: [B, H, MAX_SEQ, D]
    B, H, _, D = q.shape
    k = k_cache[:, :, :pos+1, :]
    v = v_cache[:, :, :pos+1, :]
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B, H, 1, pos+1]
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)  # [B, H, 1, D]

# ── Load global weights ─────────────────────────────────────
print("Loading embeddings...")
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight") if "lm_head.weight" in key_to_path else embed_w.copy()

# ── Cache layer weights to avoid reloading during decode ────
print("Loading all layer weights...")
t0 = time.perf_counter()
layer_weights = []
for L in range(n_layers):
    p = f"model.layers.{L}."
    lw = {
        "ln1_g": load_np(p + "input_layernorm.weight"),
        "q_w": load_np(p + "self_attn.q_proj.weight"),
        "q_b": load_np(p + "self_attn.q_proj.bias"),
        "k_w": load_np(p + "self_attn.k_proj.weight"),
        "k_b": load_np(p + "self_attn.k_proj.bias"),
        "v_w": load_np(p + "self_attn.v_proj.weight"),
        "v_b": load_np(p + "self_attn.v_proj.bias"),
        "o_w": load_np(p + "self_attn.o_proj.weight"),
        "ln2_g": load_np(p + "post_attention_layernorm.weight"),
        "router_w": load_np(p + "mlp.gate.weight"),
        "s_gate_w": load_np(p + "mlp.shared_expert.gate_proj.weight"),
        "s_up_w": load_np(p + "mlp.shared_expert.up_proj.weight"),
        "s_down_w": load_np(p + "mlp.shared_expert.down_proj.weight"),
    }
    # o_proj bias
    o_b_key = p + "self_attn.o_proj.bias"
    lw["o_b"] = load_np(o_b_key) if o_b_key in key_to_path else None
    # Shared expert gate
    seg_key = p + "mlp.shared_expert_gate.weight"
    if seg_key in key_to_path:
        seg_w = load_np(seg_key).flatten()[0]
        lw["seg_val"] = 1.0 / (1.0 + np.exp(-seg_w))
    else:
        lw["seg_val"] = 1.0
    # Expert weights
    experts = []
    for e in range(n_experts):
        experts.append({
            "gate_w": load_np(p + f"mlp.experts.{e}.gate_proj.weight"),
            "up_w": load_np(p + f"mlp.experts.{e}.up_proj.weight"),
            "down_w": load_np(p + f"mlp.experts.{e}.down_proj.weight"),
        })
    lw["experts"] = experts
    layer_weights.append(lw)
    if (L + 1) % 6 == 0:
        print(f"  Layer {L+1}/{n_layers}")
print(f"  All weights loaded in {time.perf_counter()-t0:.0f}s")

MAX_SEQ = 256

def moe_forward(h2_flat, lw):
    """Run MoE block: router + top-4 experts + shared expert."""
    BT = h2_flat.shape[0]
    router_logits = h2_flat @ lw["router_w"].T
    router_probs = softmax(router_logits, axis=-1)

    moe_out = np.zeros((BT, hidden), dtype=np.float32)
    for t_idx in range(BT):
        top4_idx = np.argsort(router_probs[t_idx])[-top_k:][::-1]
        top4_probs = router_probs[t_idx][top4_idx]
        for rank in range(top_k):
            e = top4_idx[rank]
            ew = lw["experts"][e]
            gate = h2_flat[t_idx:t_idx+1] @ ew["gate_w"].T
            up = h2_flat[t_idx:t_idx+1] @ ew["up_w"].T
            expert_out = (silu(gate) * up) @ ew["down_w"].T
            moe_out[t_idx] += top4_probs[rank] * expert_out[0]

    # Shared expert
    sg = h2_flat @ lw["s_gate_w"].T
    su = h2_flat @ lw["s_up_w"].T
    shared_out = (silu(sg) * su) @ lw["s_down_w"].T
    moe_out += lw["seg_val"] * shared_out
    return moe_out


def prefill(token_ids):
    """Pure numpy prefill. Returns (logits, k_caches, v_caches)."""
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)
    positions = np.arange(T, dtype=np.float32)

    k_caches = np.zeros((n_layers, B, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    v_caches = np.zeros((n_layers, B, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)

    for i in range(n_layers):
        lw = layer_weights[i]
        h = rms_norm(x, lw["ln1_g"])
        h_flat = h.reshape(B * T, hidden)
        q = (h_flat @ lw["q_w"].T + lw["q_b"]).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = (h_flat @ lw["k_w"].T + lw["k_b"]).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = (h_flat @ lw["v_w"].T + lw["v_b"]).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        # Store in KV cache
        k_caches[i, :, :, :T, :] = k
        v_caches[i, :, :, :T, :] = v

        attn_out = causal_attention(q, k, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, hidden)
        o = attn_out.reshape(B * T, hidden) @ lw["o_w"].T
        if lw["o_b"] is not None:
            o = o + lw["o_b"]
        x2 = x + o.reshape(B, T, hidden)

        h2 = rms_norm(x2, lw["ln2_g"])
        moe_out = moe_forward(h2.reshape(B * T, hidden), lw)
        x = x2 + moe_out.reshape(B, T, hidden)

    x_final = rms_norm(x.reshape(B * T, hidden), final_norm_g)
    logits = x_final @ lm_head_w.T
    return logits[-1], k_caches, v_caches


def decode_step(token_id, pos, k_caches, v_caches):
    """Single token decode."""
    x = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    positions = np.array([pos], dtype=np.float32)

    for i in range(n_layers):
        lw = layer_weights[i]
        h = rms_norm(x, lw["ln1_g"])
        h_flat = h.reshape(1, hidden)
        q = (h_flat @ lw["q_w"].T + lw["q_b"]).reshape(1, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = (h_flat @ lw["k_w"].T + lw["k_b"]).reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = (h_flat @ lw["v_w"].T + lw["v_b"]).reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        k_caches[i, :, :, pos, :] = k[:, :, 0, :]
        v_caches[i, :, :, pos, :] = v[:, :, 0, :]

        attn_out = kv_attention(q, k_caches[i], v_caches[i], pos)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, 1, hidden)
        o = attn_out.reshape(1, hidden) @ lw["o_w"].T
        if lw["o_b"] is not None:
            o = o + lw["o_b"]
        x2 = x + o.reshape(1, 1, hidden)

        h2 = rms_norm(x2, lw["ln2_g"])
        moe_out = moe_forward(h2.reshape(1, hidden), lw)
        x = x2 + moe_out.reshape(1, 1, hidden)

    x_final = rms_norm(x.reshape(1, hidden), final_norm_g)
    logits = x_final @ lm_head_w.T
    return logits[0]


# ── Test prompts ────────────────────────────────────────────
prompts = [
    # Base model style (completion)
    "The capital of France is",
    # Chat template
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
]

for prompt in prompts:
    print(f"\n{'=' * 60}")
    display = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print(f"Prompt: {display!r}")
    print(f"{'=' * 60}")

    tokens = tokenizer.encode(prompt)
    print(f"  {len(tokens)} tokens")

    t0 = time.perf_counter()
    logits, k_caches, v_caches = prefill(np.array(tokens))
    dt_pf = time.perf_counter() - t0
    print(f"  Prefill: {dt_pf:.1f}s")

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"  Top-5: {[(tokenizer.decode([t]), float(logits[t])) for t in top5]}")

    # Greedy decode
    next_id = int(np.argmax(logits))
    gen = [next_id]
    pos = len(tokens)
    eos_id = tokenizer.eos_token_id
    max_gen = 60

    for step in range(max_gen):
        t0 = time.perf_counter()
        logits = decode_step(next_id, pos, k_caches, v_caches)
        dt = time.perf_counter() - t0
        next_id = int(np.argmax(logits))
        gen.append(next_id)
        pos += 1
        if step < 5 or next_id == eos_id:
            print(f"    Step {step+1}: {tokenizer.decode([next_id])!r} ({dt:.2f}s)")
        if next_id == eos_id:
            break

    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"\n  Output ({len(gen)} tokens): {text}")

print("\nDone!")
