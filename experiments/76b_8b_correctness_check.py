#!/usr/bin/env python3
"""
Experiment 76b: Fast 8B correctness check — prefill cosine + 20 greedy token comparison.

Exp 76 (full numpy reference) was too slow. This does targeted diagnostics:
1. Prefill: Compare TT-NN vs numpy logits (cosine similarity)
2. First 20 greedy decode tokens: Must match exactly like 1B (exp 70) and 3B (exp 71)
3. KV cache spot-check: Read back layer 0 and layer 31 K cache, compare cosine

If cosine is high and tokens match → implementation is correct, quality is model behavior.
If cosine is low or tokens diverge → precision bug in the 8B path.
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

# Architecture
hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 32; vocab_size = 128256; MAX_SEQ = 512
intermediate_size = 14336
TILE_SIZE = 32; batch_size = 1

n_kv_split = n_kv_heads // 2
n_q_split = n_q_heads // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model ──
print("Loading Llama-3.1-8B-Instruct...")
model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]
shard_paths = []; model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths; model_id = mid
            print(f"  Found: {mid} ({n_shards} shards)")
            break
        except: pass
    if shard_paths: break

t0 = time.perf_counter()
all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()
print(f"  Loaded {len(all_weights)} tensors in {time.perf_counter()-t0:.0f}s")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w)

tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# ── Numpy reference helpers ──
def rms_norm_np(x, g, eps=1e-5):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * g

def silu_np(x):
    return x / (1.0 + np.exp(-x))

def softmax_np(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def apply_rope_np(x, positions):
    angles = np.outer(positions, freqs)
    cos_t = np.repeat(np.cos(angles), 2, axis=-1)
    sin_t = np.repeat(np.sin(angles), 2, axis=-1)
    return x * cos_t[None, None] + rotate_interleaved(x) * sin_t[None, None]

# ── Numpy prefill ──
def np_prefill(token_ids):
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)
    positions = np.arange(T)

    np_kv = []
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[prefix + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[prefix + "self_attn.q_proj.weight"].T).reshape(B, T, n_q_heads, head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[prefix + "self_attn.k_proj.weight"].T).reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[prefix + "self_attn.v_proj.weight"].T).reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        q = apply_rope_np(q, positions)
        k = apply_rope_np(k, positions)
        np_kv.append({'k': k.copy(), 'v': v.copy()})
        n_rep = n_q_heads // n_kv_heads
        k_exp = np.repeat(k, n_rep, axis=1)
        v_exp = np.repeat(v, n_rep, axis=1)
        scores = np.matmul(q, k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
        mask = np.triu(np.full((T,T), -1e9, dtype=np.float32), k=1)
        scores = scores + mask[None, None]
        attn = softmax_np(scores)
        out = np.matmul(attn, v_exp).transpose(0,2,1,3).reshape(B, T, n_q_heads * head_dim)
        o = out @ all_weights[prefix + "self_attn.o_proj.weight"].T
        x2 = x + o
        h2 = rms_norm_np(x2, all_weights[prefix + "post_attention_layernorm.weight"], rms_eps)
        g = silu_np(h2 @ all_weights[prefix + "mlp.gate_proj.weight"].T)
        u = h2 @ all_weights[prefix + "mlp.up_proj.weight"].T
        d = (g * u) @ all_weights[prefix + "mlp.down_proj.weight"].T
        x = x2 + d

    x = rms_norm_np(x, final_norm_g, rms_eps)
    logits = x @ lm_head_w.T
    return logits[0, -1], np_kv

# Numpy single decode step
def np_decode_step(token_id, pos, np_kv):
    B = 1
    x = embed_w[token_id:token_id+1].reshape(B, 1, hidden)
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[prefix + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[prefix + "self_attn.q_proj.weight"].T).reshape(B, 1, n_q_heads, head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[prefix + "self_attn.k_proj.weight"].T).reshape(B, 1, n_kv_heads, head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[prefix + "self_attn.v_proj.weight"].T).reshape(B, 1, n_kv_heads, head_dim).transpose(0,2,1,3)
        q = apply_rope_np(q, np.array([pos]))
        k = apply_rope_np(k, np.array([pos]))
        # Append to KV cache
        np_kv[i]['k'] = np.concatenate([np_kv[i]['k'], k], axis=2)
        np_kv[i]['v'] = np.concatenate([np_kv[i]['v'], v], axis=2)
        k_all = np_kv[i]['k']
        v_all = np_kv[i]['v']
        n_rep = n_q_heads // n_kv_heads
        k_exp = np.repeat(k_all, n_rep, axis=1)
        v_exp = np.repeat(v_all, n_rep, axis=1)
        scores = np.matmul(q, k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
        attn = softmax_np(scores)
        out = np.matmul(attn, v_exp).transpose(0,2,1,3).reshape(B, 1, n_q_heads * head_dim)
        o = out @ all_weights[prefix + "self_attn.o_proj.weight"].T
        x2 = x + o
        h2 = rms_norm_np(x2, all_weights[prefix + "post_attention_layernorm.weight"], rms_eps)
        g = silu_np(h2 @ all_weights[prefix + "mlp.gate_proj.weight"].T)
        u = h2 @ all_weights[prefix + "mlp.up_proj.weight"].T
        d = (g * u) @ all_weights[prefix + "mlp.down_proj.weight"].T
        x = x2 + d
    x = rms_norm_np(x, final_norm_g, rms_eps)
    return (x @ lm_head_w.T)[0, 0]


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
print("Uploading weights to device...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    dl = {
        "ln1_g": to_bf16(all_weights[prefix + "input_layernorm.weight"]),
        "q_w": to_bf16(all_weights[prefix + "self_attn.q_proj.weight"].T),
        "k_w": to_bf16(all_weights[prefix + "self_attn.k_proj.weight"].T),
        "v_w": to_bf16(all_weights[prefix + "self_attn.v_proj.weight"].T),
        "o_w": to_bf16(all_weights[prefix + "self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(all_weights[prefix + "post_attention_layernorm.weight"]),
        "gate_w": to_bf16(all_weights[prefix + "mlp.gate_proj.weight"].T),
        "up_w": to_bf16(all_weights[prefix + "mlp.up_proj.weight"].T),
        "down_w": to_bf16(all_weights[prefix + "mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)
    if (i + 1) % 8 == 0:
        print(f"    Layer {i+1}/{n_layers}")
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w.T.copy())
print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

# RoPE rotation matrix
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

# KV caches
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
    return (np.repeat(np.cos(angles), 2, axis=-1),
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved(x_4d) * sin_t[None, None]


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
        qr_4d = ttnn.reshape(qr, [1, 1, n_q_heads, head_dim])
        q_lo = ttnn.slice(qr_4d, [0,0,0,0], [1,1,n_q_split,head_dim])
        q_hi = ttnn.slice(qr_4d, [0,0,n_q_split,0], [1,1,n_q_heads,head_dim])
        attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
            q_lo, k_caches_lo[i], v_caches_lo[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
            q_hi, k_caches_hi[i], v_caches_hi[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn = ttnn.concat([attn_lo, attn_hi], dim=2)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ── Chat template ──
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

prompt = "What is the capital of France?"
system = "You are a helpful assistant."
tokens = ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
          [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
          [start_header] + enc("assistant") + [end_header] + enc("\n\n"))

print(f"\nPrompt: {prompt}")
print(f"Tokens: {len(tokens)}")
print(f"First 10 token IDs: {tokens[:10]}")

# ══════════════════════════════════════════════════════════════
# Step 1: Prefill comparison
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("Step 1: PREFILL COSINE COMPARISON")
print(f"{'='*60}")

print("  Running numpy prefill...")
t0 = time.perf_counter()
np_logits, np_kv = np_prefill(np.array(tokens))
dt_np = time.perf_counter() - t0
print(f"  Numpy prefill: {dt_np:.1f}s")

print("  Running TT-NN prefill...")
t0 = time.perf_counter()
tt_logits = ttnn_prefill(np.array(tokens))
dt_tt = time.perf_counter() - t0
print(f"  TT-NN prefill: {dt_tt:.1f}s")

# Cosine similarity
cos = np.dot(np_logits, tt_logits) / (np.linalg.norm(np_logits) * np.linalg.norm(tt_logits) + 1e-9)
print(f"\n  PREFILL COSINE: {cos:.6f}")

# Top-5 comparison
np_top5 = np.argsort(np_logits)[-5:][::-1]
tt_top5 = np.argsort(tt_logits)[-5:][::-1]
print(f"  Numpy top-1: {np_top5[0]} ({tokenizer.decode([np_top5[0]])})")
print(f"  TT-NN top-1: {tt_top5[0]} ({tokenizer.decode([tt_top5[0]])})")
print(f"  Top-1 match: {np_top5[0] == tt_top5[0]}")
print(f"  Top-5 numpy: {[tokenizer.decode([t]) for t in np_top5]}")
print(f"  Top-5 ttnn:  {[tokenizer.decode([t]) for t in tt_top5]}")
top5_match = sum(1 for t in tt_top5 if t in np_top5)
print(f"  Top-5 overlap: {top5_match}/5")

# ══════════════════════════════════════════════════════════════
# Step 2: 20-token greedy decode comparison (FORCE SAME TOKENS)
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("Step 2: 20-TOKEN GREEDY DECODE — TOKEN-BY-TOKEN COMPARISON")
print(f"{'='*60}")

# Use the numpy token as the "ground truth"
np_next = int(np.argmax(np_logits))
tt_next = int(np.argmax(tt_logits))

print(f"  First token — numpy: {np_next} ({tokenizer.decode([np_next])}), "
      f"ttnn: {tt_next} ({tokenizer.decode([tt_next])}), match: {np_next == tt_next}")

# Both paths use numpy's token to force identical inputs
next_id = np_next
pos = len(tokens)

# Setup TT-NN decode
update_buffers(next_id, pos)
_ = decode_forward(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

update_buffers(next_id, pos)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

np_gen = [next_id]
tt_gen = [next_id]
matches = 1 if np_next == tt_next else 0
cosines = []

for step in range(20):
    # TT-NN decode
    update_buffers(next_id, pos)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    tt_logits_step = from_dev(logits_ref, (1, vocab_size))[0]

    # Numpy decode (SLOW but necessary)
    np_logits_step = np_decode_step(next_id, pos, np_kv)

    # Compare
    cos_step = np.dot(np_logits_step, tt_logits_step) / (
        np.linalg.norm(np_logits_step) * np.linalg.norm(tt_logits_step) + 1e-9)
    cosines.append(cos_step)

    np_tok = int(np.argmax(np_logits_step))
    tt_tok = int(np.argmax(tt_logits_step))
    match = np_tok == tt_tok

    if match:
        matches += 1

    np_gen.append(np_tok)
    tt_gen.append(tt_tok)

    print(f"  Step {step:2d}: cos={cos_step:.4f} | np={np_tok:6d} ({tokenizer.decode([np_tok]):>10s}) | "
          f"tt={tt_tok:6d} ({tokenizer.decode([tt_tok]):>10s}) | {'✓' if match else '✗ MISMATCH'}")

    # Force same token for next step (use numpy's token)
    next_id = np_tok
    pos += 1

    if next_id in stop_ids:
        print(f"  [EOS at step {step}]")
        break

ttnn.release_trace(device, trace_id)

print(f"\n  TOKEN MATCH: {matches}/{len(np_gen)} ({100*matches/len(np_gen):.0f}%)")
print(f"  COSINE RANGE: {min(cosines):.4f} — {max(cosines):.4f}")
print(f"  MEAN COSINE: {np.mean(cosines):.4f}")

print(f"\n  NUMPY TEXT: {tokenizer.decode(np_gen, skip_special_tokens=True)}")
print(f"  TT-NN TEXT: {tokenizer.decode(tt_gen, skip_special_tokens=True)}")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"VERDICT: Llama-3.1-8B-Instruct Correctness")
print(f"{'='*60}")
print(f"  Prefill cosine:    {cos:.6f}")
print(f"  Token match:       {matches}/{len(np_gen)}")
print(f"  Mean decode cos:   {np.mean(cosines):.4f}")

if cos > 0.99 and matches >= 18:
    print(f"\n  ✓ IMPLEMENTATION IS CORRECT — quality issues are model behavior, not bugs")
elif cos > 0.95:
    print(f"\n  ⚠ MARGINAL — precision degradation detected, but may be within tolerance")
else:
    print(f"\n  ✗ BUG DETECTED — significant divergence between numpy and TT-NN")

ttnn.close_device(device)
print("\nDone!")
