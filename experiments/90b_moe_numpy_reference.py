#!/usr/bin/env python3
"""
Experiment 90b: Qwen1.5-MoE-A2.7B — Pure Numpy Reference

Validate our TT-NN MoE implementation against a pure-numpy forward pass.
Compare layer-by-layer outputs to find where divergence starts.

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/90b_moe_numpy_reference.py'
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ── Architecture ─────────────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936
n_experts = 60; top_k = 4
moe_intermediate = 1408; shared_intermediate = 5632

print("=" * 60)
print("Exp 90b: Pure Numpy MoE Reference")
print("=" * 60)

# ── Download model ──────────────────────────────────────────
model_id = "Qwen/Qwen1.5-MoE-A2.7B"
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

def apply_rope(x_4d, T):
    """x_4d: [B, n_heads, T, head_dim]"""
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return x_4d * cos_t[None, None] + rotate_half(x_4d) * sin_t[None, None]

def causal_attention(q, k, v):
    """q,k,v: [B, heads, T, head_dim]"""
    B, H, T, D = q.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B, H, T, T]
    mask = np.triu(np.full((T, T), -1e9), k=1)
    scores = scores + mask[None, None]
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

# ── Load embeddings ─────────────────────────────────────────
print("Loading weights...")
t0 = time.perf_counter()
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight") if "lm_head.weight" in key_to_path else embed_w.copy()
print(f"  Embeddings loaded: {time.perf_counter()-t0:.1f}s")

# ── Forward pass ────────────────────────────────────────────
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
print(f"\nPrompt: \"{prompt}\"")
print(f"Tokens: {tokens} ({len(tokens)} tokens)")

B, T = 1, len(tokens)
x = embed_w[tokens].reshape(B, T, hidden)
print(f"Embedding norm: {np.linalg.norm(x):.4f}")

for layer_idx in range(n_layers):
    t0 = time.perf_counter()
    p = f"model.layers.{layer_idx}."

    # Load layer weights
    ln1_g = load_np(p + "input_layernorm.weight")
    q_w = load_np(p + "self_attn.q_proj.weight")   # [2048, 2048]
    q_b = load_np(p + "self_attn.q_proj.bias")
    k_w = load_np(p + "self_attn.k_proj.weight")
    k_b = load_np(p + "self_attn.k_proj.bias")
    v_w = load_np(p + "self_attn.v_proj.weight")
    v_b = load_np(p + "self_attn.v_proj.bias")
    o_w = load_np(p + "self_attn.o_proj.weight")
    has_o_bias = (p + "self_attn.o_proj.bias") in key_to_path
    o_b = load_np(p + "self_attn.o_proj.bias") if has_o_bias else None
    ln2_g = load_np(p + "post_attention_layernorm.weight")
    router_w = load_np(p + "mlp.gate.weight")  # [60, 2048]

    # Attention
    h = rms_norm(x, ln1_g)
    q = (h.reshape(B * T, hidden) @ q_w.T + q_b).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k = (h.reshape(B * T, hidden) @ k_w.T + k_b).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (h.reshape(B * T, hidden) @ v_w.T + v_b).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q = apply_rope(q, T)
    k = apply_rope(k, T)
    attn_out = causal_attention(q, k, v)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, hidden)
    o = attn_out.reshape(B * T, hidden) @ o_w.T
    if o_b is not None:
        o = o + o_b
    x2 = x + o.reshape(B, T, hidden)

    # MoE
    h2 = rms_norm(x2, ln2_g)
    h2_flat = h2.reshape(B * T, hidden)

    # Router: [B*T, hidden] @ [60, hidden].T → [B*T, 60]
    router_logits = h2_flat @ router_w.T
    router_probs = softmax(router_logits, axis=-1)

    moe_out = np.zeros((B * T, hidden), dtype=np.float32)
    for t_idx in range(B * T):
        top4_idx = np.argsort(router_probs[t_idx])[-top_k:][::-1]
        top4_probs = router_probs[t_idx][top4_idx]
        # norm_topk_prob=False: don't renormalize

        token_moe = np.zeros(hidden, dtype=np.float32)
        for rank in range(top_k):
            e = top4_idx[rank]
            gate_w = load_np(p + f"mlp.experts.{e}.gate_proj.weight")
            up_w = load_np(p + f"mlp.experts.{e}.up_proj.weight")
            down_w = load_np(p + f"mlp.experts.{e}.down_proj.weight")
            gate = h2_flat[t_idx:t_idx+1] @ gate_w.T
            up = h2_flat[t_idx:t_idx+1] @ up_w.T
            expert_out = (silu(gate) * up) @ down_w.T
            token_moe += top4_probs[rank] * expert_out[0]
        moe_out[t_idx] = token_moe

    # Shared expert
    s_gate_w = load_np(p + "mlp.shared_expert.gate_proj.weight")
    s_up_w = load_np(p + "mlp.shared_expert.up_proj.weight")
    s_down_w = load_np(p + "mlp.shared_expert.down_proj.weight")
    sg = h2_flat @ s_gate_w.T
    su = h2_flat @ s_up_w.T
    shared_out = (silu(sg) * su) @ s_down_w.T

    # Shared expert gate: linear [1, 2048] + sigmoid — per-token gating
    seg_key = p + "mlp.shared_expert_gate.weight"
    if seg_key in key_to_path:
        seg_w = load_np(seg_key)  # [1, 2048]
        seg_logit = h2_flat @ seg_w.T  # [B*T, 1]
        seg_val = 1.0 / (1.0 + np.exp(-seg_logit))  # [B*T, 1]
    else:
        seg_val = 1.0
    moe_out += seg_val * shared_out

    x = x2 + moe_out.reshape(B, T, hidden)

    dt = time.perf_counter() - t0
    norm = np.linalg.norm(x)
    print(f"  Layer {layer_idx+1}/{n_layers}: norm={norm:.2f} ({dt:.1f}s)")

# Final norm + logits
x_final = rms_norm(x.reshape(B * T, hidden), final_norm_g)
logits = x_final @ lm_head_w.T  # [B*T, vocab]
last_logits = logits[-1]

# Top-5 predictions
top5 = np.argsort(last_logits)[-5:][::-1]
print(f"\nTop-5 predictions for \"{prompt}\":")
for i, t in enumerate(top5):
    print(f"  {i+1}. {tokenizer.decode([t])!r} (id={t}, logit={last_logits[t]:.3f})")

# Generate a few tokens greedily
print(f"\nGreedy generation:")
gen = []
next_id = int(np.argmax(last_logits))
gen.append(next_id)
print(f"  Token 1: {tokenizer.decode([next_id])!r}")

# For decode, we'd need to run through all layers again — skip for now
# The key test is: does the prefill produce "Paris" as top-1?

output = tokenizer.decode(gen, skip_special_tokens=True)
print(f"\n  Full output: \"{prompt}{output}\"")
print("\nDone!")
