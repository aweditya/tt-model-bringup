#!/usr/bin/env python3
"""
Experiment 54b: Temperature + top-k sampling on traced decode path.

53e achieves 132 tok/sec with correct greedy decoding via traced paged KV cache.
But greedy output is repetitive for a 0.5B model.

Key insight: sampling happens on CPU (numpy), OUTSIDE the trace.
The traced section only handles the 24-layer forward pass. So adding
temperature + top-k sampling should be essentially free.

This experiment:
  1. Same traced paged decode as 53e
  2. Adds temperature + top-k sampling after reading logits from device
  3. Compares greedy vs sampled output quality and speed
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=80)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Sampling ─────────────────────────────────────────────────
def sample_top_k(logits, temp=0.7, top_k=50):
    """Temperature-scaled top-k sampling. CPU-side, outside trace.
    Uses argpartition (O(n)) instead of argsort (O(n log n)) for speed."""
    logits = logits / temp
    # argpartition is O(n) — much faster than argsort for large vocab
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(top_idx, p=probs))

# ── Load model ────────────────────────────────────────────────
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights["lm_head.weight"].T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

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

# Rotation matrix for on-device RoPE
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# Upload weights
print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
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
del layer_weights_np
print(f"  Uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# KV caches
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

# KV memory config for paged_update_cache
kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

# Input buffers for trace
embed_buf = to_dev(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    """Update all input buffers before trace replay."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_dev(x_np), embed_buf)

    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


# ── Prefill (CPU RoPE) ───────────────────────────────────────
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np.reshape(B * T, hidden))
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_half_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_4d = apply_rope_half_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_4d), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_4d), batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim)).transpose(0,2,1,3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_np.reshape(B*T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B * T, hidden)).reshape(B, T, hidden)

    x_tt = to_dev(x_np.reshape(B * T, hidden))
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B * T, vocab_size))[-1]


# ── Decode step (traceable) ──────────────────────────────────
def decode_forward():
    """Full 24-layer decode using buffer references. All dynamic values via buffers."""
    x_tt = embed_buf

    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, n_kv_heads, 1, head_dim])

        q_rotated = ttnn.matmul(q_4d, R_tt)
        q_roped = ttnn.add(ttnn.mul(q_4d, rope_cos_buf), ttnn.mul(q_rotated, rope_sin_buf))
        k_rotated = ttnn.matmul(k_4d, R_tt)
        k_roped = ttnn.add(ttnn.mul(k_4d, rope_cos_buf), ttnn.mul(k_rotated, rope_sin_buf))

        k_for_cache = ttnn.reshape(k_roped, [1, 1, n_kv_heads, head_dim])
        v_for_cache = ttnn.reshape(v_4d, [1, 1, n_kv_heads, head_dim])
        k_sharded = ttnn.to_memory_config(k_for_cache, kv_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_for_cache, kv_mem_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], k_sharded, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_sharded, update_idxs_tensor=pos_buf)

        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)

        merged = ttnn.reshape(attn, [1, 1, 1, hidden])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return logits_tt


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
tokens_greedy = list(tokenizer.encode(args.prompt))
tokens_sampled = list(tokens_greedy)  # same prefill
max_gen = min(args.tokens, MAX_SEQ - len(tokens_greedy))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_greedy)} tokens)')
print(f"Generating {max_gen} tokens (temp={args.temp}, top_k={args.top_k}, seed={args.seed})")

# ── Prefill ──────────────────────────────────────────────────
t0 = time.perf_counter()
logits = prefill(np.array(tokens_greedy))
t_prefill = time.perf_counter() - t0

next_id_greedy = int(np.argmax(logits))
next_id_sampled = sample_top_k(logits, temp=args.temp, top_k=args.top_k)
tokens_greedy.append(next_id_greedy)
tokens_sampled.append(next_id_sampled)
print(f"Prefill: {t_prefill*1000:.0f}ms")

# ── Warmup non-traced decode ────────────────────────────────
update_buffers(next_id_greedy, len(tokens_greedy) - 1)
_ = decode_forward()
ttnn.synchronize_device(device)

# Enable program cache
try:
    ttnn.device.enable_program_cache(device)
except AttributeError:
    try:
        device.enable_program_cache()
    except:
        print("  (program cache API not found)")

# ── Capture trace ────────────────────────────────────────────
print("Capturing trace...")
update_buffers(next_id_greedy, len(tokens_greedy) - 1)
t_cap0 = time.perf_counter()
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
t_cap = time.perf_counter() - t_cap0
print(f"  Trace captured in {t_cap*1000:.0f}ms")


# ══════════════════════════════════════════════════════════════
# PASS 1: Greedy decode (baseline speed + repetitive output)
# ══════════════════════════════════════════════════════════════
print("\n--- Pass 1: Greedy decode (traced) ---")

# Reset KV caches for fresh generation
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    ttnn.copy(to_dev_4d(c), k_caches[i])
    ttnn.copy(to_dev_4d(c), v_caches[i])

# Re-prefill
tokens_greedy = list(tokenizer.encode(args.prompt))
logits = prefill(np.array(tokens_greedy))
next_id = int(np.argmax(logits))
tokens_greedy.append(next_id)

greedy_times = []
for step in range(max_gen - 1):
    pos = len(tokens_greedy) - 1
    update_buffers(next_id, pos)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt_trace = time.perf_counter() - t0

    t0s = time.perf_counter()
    logits = from_dev(logits_ref, (1, 1, vocab_size))[0, 0]
    next_id = int(np.argmax(logits))
    dt_sample = time.perf_counter() - t0s

    greedy_times.append((dt_trace, dt_sample))
    tokens_greedy.append(next_id)
    if next_id == tokenizer.eos_token_id:
        break

greedy_text = tokenizer.decode(tokens_greedy)


# ══════════════════════════════════════════════════════════════
# PASS 2: Temperature + top-k sampling (same trace!)
# ══════════════════════════════════════════════════════════════
print("--- Pass 2: Sampled decode (traced, temp={}, top_k={}) ---".format(args.temp, args.top_k))

# Release old trace, reset caches, re-capture
ttnn.release_trace(device, trace_id)

for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    ttnn.copy(to_dev_4d(c), k_caches[i])
    ttnn.copy(to_dev_4d(c), v_caches[i])

# Re-prefill
np.random.seed(args.seed)  # Reset RNG for reproducibility
tokens_sampled = list(tokenizer.encode(args.prompt))
logits = prefill(np.array(tokens_sampled))
next_id = sample_top_k(logits, temp=args.temp, top_k=args.top_k)
tokens_sampled.append(next_id)

# Warmup + re-capture trace (caches were reset)
update_buffers(next_id, len(tokens_sampled) - 1)
_ = decode_forward()
ttnn.synchronize_device(device)

update_buffers(next_id, len(tokens_sampled) - 1)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

sampled_times = []
for step in range(max_gen - 1):
    pos = len(tokens_sampled) - 1
    update_buffers(next_id, pos)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt_trace = time.perf_counter() - t0

    t0s = time.perf_counter()
    logits = from_dev(logits_ref, (1, 1, vocab_size))[0, 0]
    next_id = sample_top_k(logits, temp=args.temp, top_k=args.top_k)
    dt_sample = time.perf_counter() - t0s

    sampled_times.append((dt_trace, dt_sample))
    tokens_sampled.append(next_id)
    if next_id == tokenizer.eos_token_id:
        break

sampled_text = tokenizer.decode(tokens_sampled)


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Greedy stats (skip first token warmup)
g_trace = [t for t, s in greedy_times[1:]]
g_sample = [s for t, s in greedy_times[1:]]
g_total = [t + s for t, s in greedy_times[1:]]

print(f"\n--- GREEDY ---")
print(f"  Trace exec:  {np.mean(g_trace)*1000:.2f}ms/tok")
print(f"  Argmax:      {np.mean(g_sample)*1000:.2f}ms/tok")
print(f"  Total:       {np.mean(g_total)*1000:.2f}ms/tok ({1.0/np.mean(g_total):.1f} tok/sec)")
print(f"  Tokens: {len(tokens_greedy) - len(tokenizer.encode(args.prompt))}")

# Sampled stats
s_trace = [t for t, s in sampled_times[1:]]
s_sample = [s for t, s in sampled_times[1:]]
s_total = [t + s for t, s in sampled_times[1:]]

print(f"\n--- SAMPLED (temp={args.temp}, top_k={args.top_k}) ---")
print(f"  Trace exec:  {np.mean(s_trace)*1000:.2f}ms/tok")
print(f"  Sampling:    {np.mean(s_sample)*1000:.2f}ms/tok")
print(f"  Total:       {np.mean(s_total)*1000:.2f}ms/tok ({1.0/np.mean(s_total):.1f} tok/sec)")
print(f"  Tokens: {len(tokens_sampled) - len(tokenizer.encode(args.prompt))}")

overhead = np.mean(s_sample) - np.mean(g_sample)
print(f"\n--- SAMPLING OVERHEAD ---")
print(f"  Argmax time:   {np.mean(g_sample)*1000:.3f}ms")
print(f"  Sampling time: {np.mean(s_sample)*1000:.3f}ms")
print(f"  Overhead:      {overhead*1000:.3f}ms ({overhead/np.mean(g_total)*100:.1f}% of total)")

print(f"\n--- TEXT QUALITY ---")
print(f"\nGreedy output:")
print(f"  {greedy_text}")
print(f"\nSampled output (temp={args.temp}, top_k={args.top_k}):")
print(f"  {sampled_text}")

ttnn.close_device(device)
print("\nDone!")
