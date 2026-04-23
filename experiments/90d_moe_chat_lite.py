#!/usr/bin/env python3
"""
Experiment 90d: Qwen1.5-MoE-A2.7B-Chat — Lightweight Numpy Reference

LIGHTER than 90c: loads weights per-layer, discards after use.
Only runs prefill + 20 decode tokens (enough to validate quality).

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/90d_moe_chat_lite.py'
"""

import sys, os, time, gc
sys.path.insert(0, os.path.expanduser("~"))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ── Architecture ────────────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936
n_experts = 60; top_k = 4
MAX_SEQ = 128

print("=" * 60)
print("Exp 90d: Qwen1.5-MoE-A2.7B-Chat — Lite Numpy Reference")
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
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def rotate_half(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def apply_rope(x_4d, positions):
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
    k = k_cache[:, :, :pos+1, :]
    v = v_cache[:, :, :pos+1, :]
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)

# ── Load global weights ─────────────────────────────────────
print("Loading embeddings...")
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight") if "lm_head.weight" in key_to_path else embed_w.copy()

# KV caches (kept in memory — only ~150 MB for 24 layers)
k_caches = np.zeros((n_layers, 1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
v_caches = np.zeros((n_layers, 1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)


def load_layer(L):
    """Load weights for one layer. Returns dict. Caller must delete when done."""
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
    o_b_key = p + "self_attn.o_proj.bias"
    lw["o_b"] = load_np(o_b_key) if o_b_key in key_to_path else None
    seg_key = p + "mlp.shared_expert_gate.weight"
    if seg_key in key_to_path:
        lw["seg_w"] = load_np(seg_key)  # [1, 2048] — linear projection
    else:
        lw["seg_w"] = None
    return lw


def moe_forward_lazy(h2_flat, L, lw):
    """MoE block: loads expert weights on demand, discards after use."""
    p = f"model.layers.{L}."
    BT = h2_flat.shape[0]
    router_logits = h2_flat @ lw["router_w"].T
    router_probs = softmax(router_logits, axis=-1)

    moe_out = np.zeros((BT, hidden), dtype=np.float32)
    for t_idx in range(BT):
        top4_idx = np.argsort(router_probs[t_idx])[-top_k:][::-1]
        top4_probs = router_probs[t_idx][top4_idx]
        for rank in range(top_k):
            e = top4_idx[rank]
            gate_w = load_np(p + f"mlp.experts.{e}.gate_proj.weight")
            up_w = load_np(p + f"mlp.experts.{e}.up_proj.weight")
            down_w = load_np(p + f"mlp.experts.{e}.down_proj.weight")
            gate = h2_flat[t_idx:t_idx+1] @ gate_w.T
            up = h2_flat[t_idx:t_idx+1] @ up_w.T
            expert_out = (silu(gate) * up) @ down_w.T
            moe_out[t_idx] += top4_probs[rank] * expert_out[0]
            del gate_w, up_w, down_w

    # Shared expert
    sg = h2_flat @ lw["s_gate_w"].T
    su = h2_flat @ lw["s_up_w"].T
    shared_out = (silu(sg) * su) @ lw["s_down_w"].T
    if lw["seg_w"] is not None:
        seg_logit = h2_flat @ lw["seg_w"].T  # [B*T, 1]
        seg_val = 1.0 / (1.0 + np.exp(-seg_logit))  # [B*T, 1]
        moe_out += seg_val * shared_out
    else:
        moe_out += shared_out
    return moe_out


def forward_pass(x, positions, is_prefill=True):
    """Single pass through all layers. Loads/unloads weights per-layer."""
    B, T, _ = x.shape
    for L in range(n_layers):
        lw = load_layer(L)

        # Attention
        h = rms_norm(x, lw["ln1_g"])
        h_flat = h.reshape(B * T, hidden)
        q = (h_flat @ lw["q_w"].T + lw["q_b"]).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = (h_flat @ lw["k_w"].T + lw["k_b"]).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = (h_flat @ lw["v_w"].T + lw["v_b"]).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        if is_prefill:
            k_caches[L, :, :, :T, :] = k
            v_caches[L, :, :, :T, :] = v
            attn_out = causal_attention(q, k, v)
        else:
            pos = int(positions[0])
            k_caches[L, :, :, pos, :] = k[:, :, 0, :]
            v_caches[L, :, :, pos, :] = v[:, :, 0, :]
            attn_out = kv_attention(q, k_caches[L], v_caches[L], pos)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, hidden)
        o = attn_out.reshape(B * T, hidden) @ lw["o_w"].T
        if lw["o_b"] is not None:
            o = o + lw["o_b"]
        x2 = x + o.reshape(B, T, hidden)

        # MoE
        h2 = rms_norm(x2, lw["ln2_g"])
        moe_out = moe_forward_lazy(h2.reshape(B * T, hidden), L, lw)
        x = x2 + moe_out.reshape(B, T, hidden)

        del lw
        gc.collect()

        if is_prefill and (L + 1) % 6 == 0:
            print(f"    Prefill layer {L+1}/{n_layers}")
        elif not is_prefill and (L + 1) % 12 == 0:
            print(f"      Layer {L+1}/{n_layers}")

    # Final logits
    x_final = rms_norm(x.reshape(B * T, hidden), final_norm_g)
    logits = x_final @ lm_head_w.T
    return logits[-1]


# ── Generate ────────────────────────────────────────────────
prompts = [
    # Base completion
    "The capital of France is",
    # Chat template
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
]

for prompt in prompts:
    print(f"\n{'=' * 60}")
    display = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print(f"Prompt: {display!r}")
    print(f"{'=' * 60}")

    # Reset KV
    k_caches[:] = 0
    v_caches[:] = 0

    tokens = tokenizer.encode(prompt)
    T = len(tokens)
    print(f"  {T} tokens")

    # Prefill
    t0 = time.perf_counter()
    x = embed_w[tokens].reshape(1, T, hidden)
    logits = forward_pass(x, np.arange(T, dtype=np.float32), is_prefill=True)
    dt_pf = time.perf_counter() - t0
    print(f"  Prefill: {dt_pf:.1f}s")

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"  Top-5: {[(tokenizer.decode([t]), f'{logits[t]:.2f}') for t in top5]}")

    # Greedy decode (20 tokens max)
    next_id = int(np.argmax(logits))
    gen = [next_id]
    pos = T
    eos_id = tokenizer.eos_token_id
    max_gen = 20

    for step in range(max_gen):
        t0 = time.perf_counter()
        x = embed_w[next_id:next_id+1].reshape(1, 1, hidden)
        logits = forward_pass(x, np.array([pos], dtype=np.float32), is_prefill=False)
        dt = time.perf_counter() - t0
        next_id = int(np.argmax(logits))
        gen.append(next_id)
        pos += 1
        tok = tokenizer.decode([next_id])
        print(f"    Step {step+1}: {tok!r} ({dt:.1f}s)")
        if next_id == eos_id:
            break

    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"\n  Output: {text}")

print("\nDone!")
