#!/usr/bin/env python3
"""
Experiment 74: KV cache validation — is the cache corrupting over positions?

Hypothesis: paged_update_cache introduces bf16 rounding errors that accumulate
position-by-position, causing degeneration at ~30-50 tokens regardless of model size.

Test: Compare KV cache entries stored on device vs numpy reference after N decode steps.
If cosine drops below 0.99 for entries at position N, that explains degeneration.

Uses Llama-3.2-1B-Instruct (smaller, faster to test, already validated at 20/20 tokens).
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
import ttnn

np.random.seed(42)

# Llama-3.2-1B architecture
hidden = 2048; n_q_heads = 32; n_kv_heads = 8; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 16; vocab_size = 128256; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

n_kv_split = n_kv_heads // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()

# ── Load model ──
print("Loading Llama-3.2-1B-Instruct...")
model_ids = ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
model_path = None
model_id = None
for mid in model_ids:
    try:
        model_path = hf_hub_download(mid, "model.safetensors")
        model_id = mid
        break
    except: pass

all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# ── Numpy reference ──
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def numpy_rms_norm(x, g, eps):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * g

def numpy_silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def rotate_interleaved_np(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def numpy_single_step(token_id, pos, x_cache, kv_cache_np):
    """Single decode step with full KV cache tracking. Returns (logits, updated_kv_cache)."""
    B = 1
    x = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    angles = pos * freqs
    cos_t = np.repeat(np.cos(angles), 2).reshape(1, 1, 1, head_dim)
    sin_t = np.repeat(np.sin(angles), 2).reshape(1, 1, 1, head_dim)

    for i in range(n_layers):
        lw = layer_weights_np[i]
        h = numpy_rms_norm(x, lw["input_layernorm.weight"], rms_eps)

        q = h.reshape(B, hidden) @ lw["self_attn.q_proj.weight"].T
        k = h.reshape(B, hidden) @ lw["self_attn.k_proj.weight"].T
        v = h.reshape(B, hidden) @ lw["self_attn.v_proj.weight"].T

        q = q.reshape(B, 1, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE (interleaved)
        q = q * cos_t + rotate_interleaved_np(q) * sin_t
        k = k * cos_t + rotate_interleaved_np(k) * sin_t

        # Update KV cache
        kv_cache_np[i]['k'][:, :, pos:pos+1, :] = k
        kv_cache_np[i]['v'][:, :, pos:pos+1, :] = v

        # Attention over all cached positions (0..pos inclusive)
        k_all = kv_cache_np[i]['k'][:, :, :pos+1, :]
        v_all = kv_cache_np[i]['v'][:, :, :pos+1, :]

        gqa_ratio = n_q_heads // n_kv_heads
        k_exp = np.repeat(k_all, gqa_ratio, axis=1)
        v_exp = np.repeat(v_all, gqa_ratio, axis=1)

        scale = 1.0 / np.sqrt(head_dim)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * scale
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-9)
        attn_out = (attn_weights @ v_exp).transpose(0, 2, 1, 3).reshape(B, 1, hidden)

        o = attn_out.reshape(B, hidden) @ lw["self_attn.o_proj.weight"].T
        x2 = x + o.reshape(B, 1, hidden)

        h2 = numpy_rms_norm(x2, lw["post_attention_layernorm.weight"], rms_eps)
        gate = h2.reshape(B, hidden) @ lw["mlp.gate_proj.weight"].T
        up = h2.reshape(B, hidden) @ lw["mlp.up_proj.weight"].T
        down = (numpy_silu(gate) * up) @ lw["mlp.down_proj.weight"].T
        x = x2 + down.reshape(B, 1, hidden)

    x_final = numpy_rms_norm(x, final_norm_g, rms_eps)
    logits = x_final.reshape(B, hidden) @ lm_head_w
    return logits[0], kv_cache_np


def numpy_prefill(token_ids, kv_cache_np):
    """Prefill all positions in the KV cache."""
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)

    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.repeat(np.cos(angles), 2, axis=-1)
    sin_t = np.repeat(np.sin(angles), 2, axis=-1)

    for i in range(n_layers):
        lw = layer_weights_np[i]
        h = numpy_rms_norm(x, lw["input_layernorm.weight"], rms_eps)

        q = h.reshape(B*T, hidden) @ lw["self_attn.q_proj.weight"].T
        k = h.reshape(B*T, hidden) @ lw["self_attn.k_proj.weight"].T
        v = h.reshape(B*T, hidden) @ lw["self_attn.v_proj.weight"].T

        q = q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = q * cos_t[None, None] + rotate_interleaved_np(q) * sin_t[None, None]
        k = k * cos_t[None, None] + rotate_interleaved_np(k) * sin_t[None, None]

        kv_cache_np[i]['k'][:, :, :T, :] = k
        kv_cache_np[i]['v'][:, :, :T, :] = v

        gqa_ratio = n_q_heads // n_kv_heads
        k_exp = np.repeat(k, gqa_ratio, axis=1)
        v_exp = np.repeat(v, gqa_ratio, axis=1)

        scale = 1.0 / np.sqrt(head_dim)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * scale
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
        scores = scores + mask
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-9)
        attn_out = (attn_weights @ v_exp).transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o = attn_out.reshape(B*T, hidden) @ lw["self_attn.o_proj.weight"].T
        x2 = x + o.reshape(B, T, hidden)

        h2 = numpy_rms_norm(x2, lw["post_attention_layernorm.weight"], rms_eps)
        gate = h2.reshape(B*T, hidden) @ lw["mlp.gate_proj.weight"].T
        up = h2.reshape(B*T, hidden) @ lw["mlp.up_proj.weight"].T
        down = (numpy_silu(gate) * up) @ lw["mlp.down_proj.weight"].T
        x = x2 + down.reshape(B, T, hidden)

    x_final = numpy_rms_norm(x, final_norm_g, rms_eps)
    logits = x_final.reshape(B*T, hidden) @ lm_head_w
    return logits[-1], kv_cache_np


# ── TT-NN setup ──
def to_bf16(arr):
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
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)

R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

# KV caches (split)
k_caches_lo, v_caches_lo = [], []
k_caches_hi, v_caches_hi = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches_lo.append(to_dev_4d(c.copy()))
    v_caches_lo.append(to_dev_4d(c.copy()))
    k_caches_hi.append(to_dev_4d(c.copy()))
    v_caches_hi.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_split + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1), np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]


def ttnn_prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_interleaved(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q_np = apply_rope_interleaved_np(
            from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_interleaved_np(
            from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_lo[i], to_dev_4d(k_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_lo[i], to_dev_4d(v_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_hi[i], to_dev_4d(k_np[:, n_kv_split:]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_hi[i], to_dev_4d(v_np[:, n_kv_split:]), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,n_q_heads*head_dim)
        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,n_q_heads*head_dim)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)
    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


def decode_forward():
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q = ttnn.reshape(q, [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(k, [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(v, [1, n_kv_heads, 1, head_dim])
        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))
        kr_4d = ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim])
        v_4d = ttnn.reshape(v, [1, 1, n_kv_heads, head_dim])
        kr_lo = ttnn.to_memory_config(ttnn.slice(kr_4d, [0,0,0,0], [1,1,n_kv_split,head_dim]), kv_cfg)
        kr_hi = ttnn.to_memory_config(ttnn.slice(kr_4d, [0,0,n_kv_split,0], [1,1,n_kv_heads,head_dim]), kv_cfg)
        v_lo = ttnn.to_memory_config(ttnn.slice(v_4d, [0,0,0,0], [1,1,n_kv_split,head_dim]), kv_cfg)
        v_hi = ttnn.to_memory_config(ttnn.slice(v_4d, [0,0,n_kv_split,0], [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches_lo[i], kr_lo, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches_lo[i], v_lo, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(k_caches_hi[i], kr_hi, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches_hi[i], v_hi, update_idxs_tensor=pos_buf)
        n_q_s = n_q_heads // 2
        qr_4d = ttnn.reshape(qr, [1, 1, n_q_heads, head_dim])
        q_lo = ttnn.slice(qr_4d, [0,0,0,0], [1,1,n_q_s,head_dim])
        q_hi = ttnn.slice(qr_4d, [0,0,n_q_s,0], [1,1,n_q_heads,head_dim])
        attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
            q_lo, k_caches_lo[i], v_caches_lo[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
            q_hi, k_caches_hi[i], v_caches_hi[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn = ttnn.concat([attn_lo, attn_hi], dim=2)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# Run: Compare numpy vs TT-NN decode step-by-step for 60 tokens
# ══════════════════════════════════════════════════════════════

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

prompt = "List the top 5 programming languages and explain why each is popular."
tokens = ([bos, start_header] + enc("system") + [end_header] + enc("\n\nYou are a helpful assistant.") + [eot] +
          [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
          [start_header] + enc("assistant") + [end_header] + enc("\n\n"))
print(f"Prompt: {prompt} ({len(tokens)} tokens)")

# Initialize numpy KV cache
np_kv = [{'k': np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32),
           'v': np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)} for _ in range(n_layers)]

# Prefill both
print("\n--- Prefill ---")
np_logits, np_kv = numpy_prefill(np.array(tokens), np_kv)
tt_logits = ttnn_prefill(np.array(tokens))

cosine_prefill = np.dot(np_logits, tt_logits) / (np.linalg.norm(np_logits) * np.linalg.norm(tt_logits) + 1e-9)
print(f"  Prefill cosine: {cosine_prefill:.6f}")

# Compare KV caches after prefill (check a few layers)
print("\n--- KV cache comparison after prefill ---")
for layer_idx in [0, 7, 15]:
    # Read back device KV cache
    k_lo_dev = from_dev(k_caches_lo[layer_idx], (1, n_kv_split, MAX_SEQ, head_dim))
    k_hi_dev = from_dev(k_caches_hi[layer_idx], (1, n_kv_split, MAX_SEQ, head_dim))
    k_dev = np.concatenate([k_lo_dev, k_hi_dev], axis=1)  # (1, 8, MAX_SEQ, 64)

    k_ref = np_kv[layer_idx]['k']  # (1, 8, MAX_SEQ, 64)

    # Compare only the filled positions
    T = len(tokens)
    k_dev_filled = k_dev[:, :, :T, :].flatten()
    k_ref_filled = k_ref[:, :, :T, :].flatten()

    cos_k = np.dot(k_dev_filled, k_ref_filled) / (np.linalg.norm(k_dev_filled) * np.linalg.norm(k_ref_filled) + 1e-9)
    print(f"  Layer {layer_idx}: K cache cosine = {cos_k:.6f}")

# Decode step-by-step, comparing at each position
print(f"\n--- Step-by-step decode comparison (60 steps) ---")
print(f"{'Step':>4} {'NP tok':>8} {'TT tok':>8} {'Match':>5} {'Logit cos':>10} {'K cos L0':>10} {'K cos L15':>10}")

np_next = int(np.argmax(np_logits))
tt_next = int(np.argmax(tt_logits))
pos = len(tokens)
np_tokens = list(tokens) + [np_next]
tt_tokens = list(tokens) + [tt_next]

# No trace — full recompute each step for accurate comparison
for step in range(60):
    # Numpy step
    np_logits, np_kv = numpy_single_step(np_next, pos, None, np_kv)

    # TT-NN step (no trace, full recompute)
    update_buffers(tt_next, pos)
    tt_logits_dev = decode_forward()
    ttnn.synchronize_device(device)
    tt_logits_np = from_dev(tt_logits_dev, (1, 1, vocab_size))[0, 0]

    # Compare logits
    cos_logits = np.dot(np_logits, tt_logits_np) / (np.linalg.norm(np_logits) * np.linalg.norm(tt_logits_np) + 1e-9)

    # Compare KV cache at this position
    k_lo_0 = from_dev(k_caches_lo[0], (1, n_kv_split, MAX_SEQ, head_dim))
    k_hi_0 = from_dev(k_caches_hi[0], (1, n_kv_split, MAX_SEQ, head_dim))
    k_dev_0 = np.concatenate([k_lo_0, k_hi_0], axis=1)
    k_pos_dev = k_dev_0[:, :, pos, :].flatten()
    k_pos_ref = np_kv[0]['k'][:, :, pos, :].flatten()
    cos_k0 = np.dot(k_pos_dev, k_pos_ref) / (np.linalg.norm(k_pos_dev) * np.linalg.norm(k_pos_ref) + 1e-9)

    k_lo_15 = from_dev(k_caches_lo[15], (1, n_kv_split, MAX_SEQ, head_dim))
    k_hi_15 = from_dev(k_caches_hi[15], (1, n_kv_split, MAX_SEQ, head_dim))
    k_dev_15 = np.concatenate([k_lo_15, k_hi_15], axis=1)
    k_pos_dev15 = k_dev_15[:, :, pos, :].flatten()
    k_pos_ref15 = np_kv[15]['k'][:, :, pos, :].flatten()
    cos_k15 = np.dot(k_pos_dev15, k_pos_ref15) / (np.linalg.norm(k_pos_dev15) * np.linalg.norm(k_pos_ref15) + 1e-9)

    np_tok = int(np.argmax(np_logits))
    tt_tok = int(np.argmax(tt_logits_np))
    match = "OK" if np_tok == tt_tok else "MISS"

    np_word = tokenizer.decode([np_tok])
    tt_word = tokenizer.decode([tt_tok])

    print(f"  {step:3d}  {np_word:>8s}  {tt_word:>8s}  {match:>5s}  {cos_logits:>10.6f}  {cos_k0:>10.6f}  {cos_k15:>10.6f}")

    # Use numpy tokens as reference for next step (force same path)
    np_next = np_tok
    tt_next = np_tok  # Use SAME token to isolate cache effects
    pos += 1
    np_tokens.append(np_next)
    tt_tokens.append(tt_next)

    if np_next in stop_ids:
        print(f"  [EOS at step {step+1}]")
        break

np_text = tokenizer.decode(np_tokens[len(tokens):], skip_special_tokens=True)
print(f"\n  Numpy greedy text: {np_text[:500]}")

ttnn.close_device(device)
print("\nDone!")
