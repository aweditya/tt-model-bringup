#!/usr/bin/env python3
"""
Experiment 77: Creative writing — numpy float32 vs TT-NN side-by-side.

Exp 76b proved factual Q&A is correct (8/8 tokens match).
But exp 75 showed creative writing degenerates at ~40 tokens.

This experiment runs the creative prompt through BOTH numpy and TT-NN
for 40 greedy tokens to see where they diverge.

If numpy also produces bad text → model behavior (not a bug).
If numpy is coherent and TT-NN diverges → bf16 precision bug.
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
intermediate_size = 14336; TILE_SIZE = 32; batch_size = 1
n_kv_split = n_kv_heads // 2; n_q_split = n_q_heads // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()

# Load model
print("Loading Llama-3.1-8B-Instruct...")
model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]
shard_paths = []; model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths; model_id = mid; break
        except: pass
    if shard_paths: break

all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()
print(f"  Loaded {len(all_weights)} tensors")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w)

tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# ── Numpy helpers ──
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

# ── Numpy full pipeline ──
def np_prefill(token_ids):
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)
    positions = np.arange(T)
    np_kv = []
    for i in range(n_layers):
        p = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[p + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[p + "self_attn.q_proj.weight"].T).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[p + "self_attn.k_proj.weight"].T).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[p + "self_attn.v_proj.weight"].T).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        q, k = apply_rope_np(q, positions), apply_rope_np(k, positions)
        np_kv.append({'k': k.copy(), 'v': v.copy()})
        n_rep = n_q_heads // n_kv_heads
        k_exp, v_exp = np.repeat(k, n_rep, axis=1), np.repeat(v, n_rep, axis=1)
        scores = np.matmul(q, k_exp.transpose(0,1,3,2)) / np.sqrt(head_dim)
        scores += np.triu(np.full((T,T), -1e9, dtype=np.float32), k=1)[None, None]
        out = np.matmul(softmax_np(scores), v_exp).transpose(0,2,1,3).reshape(B,T,n_q_heads*head_dim)
        x2 = x + out @ all_weights[p + "self_attn.o_proj.weight"].T
        h2 = rms_norm_np(x2, all_weights[p + "post_attention_layernorm.weight"], rms_eps)
        x = x2 + (silu_np(h2 @ all_weights[p + "mlp.gate_proj.weight"].T) *
                   (h2 @ all_weights[p + "mlp.up_proj.weight"].T)) @ all_weights[p + "mlp.down_proj.weight"].T
    return (rms_norm_np(x, final_norm_g, rms_eps) @ lm_head_w.T)[0, -1], np_kv

def np_decode_step(token_id, pos, np_kv):
    x = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    for i in range(n_layers):
        p = f"model.layers.{i}."
        h = rms_norm_np(x, all_weights[p + "input_layernorm.weight"], rms_eps)
        q = (h @ all_weights[p + "self_attn.q_proj.weight"].T).reshape(1,1,n_q_heads,head_dim).transpose(0,2,1,3)
        k = (h @ all_weights[p + "self_attn.k_proj.weight"].T).reshape(1,1,n_kv_heads,head_dim).transpose(0,2,1,3)
        v = (h @ all_weights[p + "self_attn.v_proj.weight"].T).reshape(1,1,n_kv_heads,head_dim).transpose(0,2,1,3)
        q, k = apply_rope_np(q, np.array([pos])), apply_rope_np(k, np.array([pos]))
        np_kv[i]['k'] = np.concatenate([np_kv[i]['k'], k], axis=2)
        np_kv[i]['v'] = np.concatenate([np_kv[i]['v'], v], axis=2)
        k_all, v_all = np_kv[i]['k'], np_kv[i]['v']
        n_rep = n_q_heads // n_kv_heads
        scores = np.matmul(q, np.repeat(k_all, n_rep, axis=1).transpose(0,1,3,2)) / np.sqrt(head_dim)
        out = np.matmul(softmax_np(scores), np.repeat(v_all, n_rep, axis=1)).transpose(0,2,1,3).reshape(1,1,n_q_heads*head_dim)
        x2 = x + out @ all_weights[p + "self_attn.o_proj.weight"].T
        h2 = rms_norm_np(x2, all_weights[p + "post_attention_layernorm.weight"], rms_eps)
        x = x2 + (silu_np(h2 @ all_weights[p + "mlp.gate_proj.weight"].T) *
                   (h2 @ all_weights[p + "mlp.up_proj.weight"].T)) @ all_weights[p + "mlp.down_proj.weight"].T
    return (rms_norm_np(x, final_norm_g, rms_eps) @ lm_head_w.T)[0, 0]

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

print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    p = f"model.layers.{i}."
    dev_layers.append({
        "ln1_g": to_bf16(all_weights[p + "input_layernorm.weight"]),
        "q_w": to_bf16(all_weights[p + "self_attn.q_proj.weight"].T),
        "k_w": to_bf16(all_weights[p + "self_attn.k_proj.weight"].T),
        "v_w": to_bf16(all_weights[p + "self_attn.v_proj.weight"].T),
        "o_w": to_bf16(all_weights[p + "self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(all_weights[p + "post_attention_layernorm.weight"]),
        "gate_w": to_bf16(all_weights[p + "mlp.gate_proj.weight"].T),
        "up_w": to_bf16(all_weights[p + "mlp.up_proj.weight"].T),
        "down_w": to_bf16(all_weights[p + "mlp.down_proj.weight"].T),
    })
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w.T.copy())
print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0; R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

k_caches_lo, v_caches_lo, k_caches_hi, v_caches_hi = [], [], [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches_lo.append(to_dev_4d(c.copy())); v_caches_lo.append(to_dev_4d(c.copy()))
    k_caches_hi.append(to_dev_4d(c.copy())); v_caches_hi.append(to_dev_4d(c.copy()))

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
    return np.repeat(np.cos(angles), 2, axis=-1), np.repeat(np.sin(angles), 2, axis=-1)

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
        q_np = apply_rope_interleaved_np(from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_interleaved_np(from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
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
    ttnn.copy(to_bf16(embed_w[token_id:token_id+1].reshape(1, 1, hidden)), embed_buf)
    angles = pos * freqs
    ttnn.copy(to_dev_4d(np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_cos_buf)
    ttnn.copy(to_dev_4d(np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_sin_buf)
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

# Test the creative writing prompt (the one that degenerates in exp 75)
prompt = "Write a short story about a robot that learns to paint. Include dialogue."
system = "You are a helpful assistant."
tokens = ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
          [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
          [start_header] + enc("assistant") + [end_header] + enc("\n\n"))

N_TOKENS = 40  # Just enough to see if it degenerates

print(f"\nPrompt: {prompt}")
print(f"Tokens: {len(tokens)}")
print(f"Generating {N_TOKENS} greedy tokens (forcing same input on both paths)")

# ═══════════════════════════════════════════════
# Prefill both paths
# ═══════════════════════════════════════════════
print("\n--- PREFILL ---")
t0 = time.perf_counter()
np_logits, np_kv = np_prefill(np.array(tokens))
print(f"  Numpy: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
tt_logits = ttnn_prefill(np.array(tokens))
print(f"  TT-NN: {time.perf_counter()-t0:.1f}s")

cos_prefill = np.dot(np_logits, tt_logits) / (np.linalg.norm(np_logits) * np.linalg.norm(tt_logits) + 1e-9)
print(f"  Prefill cosine: {cos_prefill:.6f}")

np_tok0 = int(np.argmax(np_logits))
tt_tok0 = int(np.argmax(tt_logits))
print(f"  Numpy first: {np_tok0} ({tokenizer.decode([np_tok0])})")
print(f"  TT-NN first: {tt_tok0} ({tokenizer.decode([tt_tok0])})")

# ═══════════════════════════════════════════════
# Decode N_TOKENS steps — EACH PATH USES ITS OWN TOKENS (no forcing)
# ═══════════════════════════════════════════════
print(f"\n--- DECODE (greedy, independent paths) ---")

# Setup TT-NN trace
next_tt = tt_tok0
pos = len(tokens)
update_buffers(next_tt, pos)
_ = decode_forward(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass
update_buffers(next_tt, pos)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

np_gen = [np_tok0]
tt_gen = [tt_tok0]
next_np = np_tok0
next_tt = tt_tok0

print(f"{'Step':>4} | {'Numpy tok':>30} | {'TT-NN tok':>30} | {'Match':>5} | {'Cos':>7}")
print("-" * 90)

for step in range(N_TOKENS):
    pos_step = len(tokens) + step

    # Numpy decode (using numpy's own token)
    np_logits_s = np_decode_step(next_np, pos_step, np_kv)
    next_np = int(np.argmax(np_logits_s))
    np_gen.append(next_np)

    # TT-NN decode (using TT-NN's own token)
    update_buffers(next_tt, pos_step)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    tt_logits_s = from_dev(logits_ref, (1, vocab_size))[0]
    next_tt = int(np.argmax(tt_logits_s))
    tt_gen.append(next_tt)

    match = next_np == next_tt
    # Cosine only meaningful when inputs were the same (step 0 only, after that paths diverge)
    if step == 0:
        cos_s = np.dot(np_logits_s, tt_logits_s) / (np.linalg.norm(np_logits_s) * np.linalg.norm(tt_logits_s) + 1e-9)
    else:
        cos_s = float('nan')  # inputs diverged, cosine is meaningless

    np_word = tokenizer.decode([next_np]).replace('\n', '\\n')
    tt_word = tokenizer.decode([next_tt]).replace('\n', '\\n')
    print(f"{step:4d} | {next_np:6d} {np_word:>22s} | {next_tt:6d} {tt_word:>22s} | {'✓' if match else '✗':>5} | {cos_s:>7.4f}" if not np.isnan(cos_s) else
          f"{step:4d} | {next_np:6d} {np_word:>22s} | {next_tt:6d} {tt_word:>22s} | {'✓' if match else '✗':>5} |     n/a")

    if next_np in stop_ids:
        print(f"  [Numpy hit EOS]")
    if next_tt in stop_ids:
        print(f"  [TT-NN hit EOS]")
    if next_np in stop_ids and next_tt in stop_ids:
        break

ttnn.release_trace(device, trace_id)

# ═══════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════
np_text = tokenizer.decode(np_gen, skip_special_tokens=True)
tt_text = tokenizer.decode(tt_gen, skip_special_tokens=True)

total_match = sum(1 for a, b in zip(np_gen, tt_gen) if a == b)

print(f"\n{'='*70}")
print(f"NUMPY TEXT ({len(np_gen)} tokens):")
print(f"{'='*70}")
for line in np_text.split('\n'):
    print(f"  {line}")

print(f"\n{'='*70}")
print(f"TT-NN TEXT ({len(tt_gen)} tokens):")
print(f"{'='*70}")
for line in tt_text.split('\n'):
    print(f"  {line}")

print(f"\n{'='*70}")
print(f"Token overlap: {total_match}/{max(len(np_gen), len(tt_gen))}")
print(f"Prefill cosine: {cos_prefill:.6f}")
print(f"\nVERDICT:")
if total_match > 0.8 * max(len(np_gen), len(tt_gen)):
    print("  Both paths produce similar text → quality is model behavior, not precision")
else:
    print("  Paths diverge → check which text is more coherent")
    print("  If numpy is coherent but TT-NN isn't → bf16 precision problem")
    print("  If both degenerate → model capacity limit")

ttnn.close_device(device)
print("\nDone!")
