#!/usr/bin/env python3
"""
Experiment 78: Can explicit length instructions get 8B to write longer?

Exp 77 showed the 8B model stops at ~35 tokens (EOS) on creative prompts.
Hypothesis: explicit length requirements in the prompt will make it generate
longer coherent text.

Uses numpy float32 reference (no TT-NN needed for this test — pure model behavior).
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

np.random.seed(42)

hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
rms_eps = 1e-5; rope_theta = 500000.0; n_layers = 32; vocab_size = 128256

print("Loading Llama-3.1-8B-Instruct...")
shard_paths = [hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct",
               f"model-{i+1:05d}-of-00004.safetensors") for i in range(4)]
all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w)

tok_path = hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct", "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

def rms_norm_np(x, g, eps=1e-5):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * g

def silu_np(x): return x / (1 + np.exp(-x))

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved(x):
    r = np.zeros_like(x); r[..., 0::2] = -x[..., 1::2]; r[..., 1::2] = x[..., 0::2]; return r

def apply_rope_np(x, positions):
    angles = np.outer(positions, freqs)
    cos_t = np.repeat(np.cos(angles), 2, axis=-1)
    sin_t = np.repeat(np.sin(angles), 2, axis=-1)
    return x * cos_t[None, None] + rotate_interleaved(x) * sin_t[None, None]

np_kv_cache = None

def np_prefill(token_ids):
    global np_kv_cache
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)
    positions = np.arange(T)
    np_kv_cache = []
    for i in range(n_layers):
        p = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[p + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[p + "self_attn.q_proj.weight"].T).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[p + "self_attn.k_proj.weight"].T).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[p + "self_attn.v_proj.weight"].T).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        q, k = apply_rope_np(q, positions), apply_rope_np(k, positions)
        np_kv_cache.append({"k": k.copy(), "v": v.copy()})
        n_rep = n_q_heads // n_kv_heads
        scores = np.matmul(q, np.repeat(k, n_rep, axis=1).transpose(0,1,3,2)) / np.sqrt(head_dim)
        scores += np.triu(np.full((T,T), -1e9, dtype=np.float32), k=1)[None, None]
        out = np.matmul(softmax_np(scores), np.repeat(v, n_rep, axis=1)).transpose(0,2,1,3).reshape(B,T,n_q_heads*head_dim)
        x2 = x + out @ all_weights[p + "self_attn.o_proj.weight"].T
        h2 = rms_norm_np(x2, all_weights[p + "post_attention_layernorm.weight"], rms_eps)
        x = x2 + (silu_np(h2 @ all_weights[p + "mlp.gate_proj.weight"].T) * (h2 @ all_weights[p + "mlp.up_proj.weight"].T)) @ all_weights[p + "mlp.down_proj.weight"].T
    return (rms_norm_np(x, final_norm_g, rms_eps) @ lm_head_w.T)[0, -1]

def np_decode_step(token_id, pos):
    global np_kv_cache
    x = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    for i in range(n_layers):
        p = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[p + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[p + "self_attn.q_proj.weight"].T).reshape(1,1,n_q_heads,head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[p + "self_attn.k_proj.weight"].T).reshape(1,1,n_kv_heads,head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[p + "self_attn.v_proj.weight"].T).reshape(1,1,n_kv_heads,head_dim).transpose(0,2,1,3)
        q, k = apply_rope_np(q, np.array([pos])), apply_rope_np(k, np.array([pos]))
        np_kv_cache[i]["k"] = np.concatenate([np_kv_cache[i]["k"], k], axis=2)
        np_kv_cache[i]["v"] = np.concatenate([np_kv_cache[i]["v"], v], axis=2)
        n_rep = n_q_heads // n_kv_heads
        scores = np.matmul(q, np.repeat(np_kv_cache[i]["k"], n_rep, axis=1).transpose(0,1,3,2)) / np.sqrt(head_dim)
        out = np.matmul(softmax_np(scores), np.repeat(np_kv_cache[i]["v"], n_rep, axis=1)).transpose(0,2,1,3).reshape(1,1,n_q_heads*head_dim)
        x2 = x + out @ all_weights[p + "self_attn.o_proj.weight"].T
        h2 = rms_norm_np(x2, all_weights[p + "post_attention_layernorm.weight"], rms_eps)
        x = x2 + (silu_np(h2 @ all_weights[p + "mlp.gate_proj.weight"].T) * (h2 @ all_weights[p + "mlp.up_proj.weight"].T)) @ all_weights[p + "mlp.down_proj.weight"].T
    return (rms_norm_np(x, final_norm_g, rms_eps) @ lm_head_w.T)[0, 0]

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

def make_chat_tokens(prompt, system="You are a helpful assistant."):
    return ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
            [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
            [start_header] + enc("assistant") + [end_header] + enc("\n\n"))

# Test prompts with explicit length requirements
prompts = [
    # Original short prompt (expect ~35 tokens + EOS)
    "Write a short story about a robot that learns to paint.",
    # Explicit multi-paragraph
    "Write a 3-paragraph story about a robot that learns to paint. Each paragraph must be at least 3 sentences long. Include dialogue.",
    # Explicit list with length
    "List exactly 5 benefits of exercise. For each benefit, write exactly 2 sentences explaining it.",
    # Factual with explicit length
    "Explain quantum computing in exactly 3 paragraphs. Paragraph 1: what it is. Paragraph 2: how qubits work. Paragraph 3: why it matters.",
]

for i, prompt in enumerate(prompts):
    print(f"\n{'='*70}")
    print(f"Test {i+1}: {prompt[:70]}...")
    print(f"{'='*70}")

    tokens = make_chat_tokens(prompt)
    print(f"  Prompt tokens: {len(tokens)}")

    t0 = time.perf_counter()
    logits = np_prefill(np.array(tokens))
    dt = time.perf_counter() - t0
    print(f"  Prefill: {dt:.1f}s")

    next_id = int(np.argmax(logits))
    gen = [next_id]
    pos = len(tokens)

    for step in range(200):
        logits = np_decode_step(next_id, pos)
        next_id = int(np.argmax(logits))
        gen.append(next_id)
        pos += 1
        if next_id in stop_ids:
            break

    text = tokenizer.decode(gen, skip_special_tokens=True)
    words = text.split()
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    print(f"  Tokens: {len(gen)}, Words: {len(words)}, Paragraphs: {len(paragraphs)}")
    print(f"  Hit EOS: {gen[-1] in stop_ids}")
    print(f"  ---")
    for line in text.split("\n")[:25]:
        print(f"  {line}")
    if len(text.split("\n")) > 25:
        print(f"  ... ({len(text.split(chr(10)))} total lines)")
    print(f"  ---")

print("\nDone!")
