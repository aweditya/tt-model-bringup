#!/usr/bin/env python3
"""
Experiment 81: Benchmark timing audit — measure the REAL end-to-end decode speed.

AUDIT FINDING: Experiments 60-73 timed ONLY ttnn.execute_trace(), excluding:
  1. update_buffers() — ttnn.copy calls for embed, RoPE cos/sin, position
  2. from_dev() — PCIe readback of logits tensor
  3. np.argmax() — host-side token selection

This experiment measures both:
  (a) trace-only time (what we've been reporting)
  (b) end-to-end time (what actually matters for serving)

Tests on Qwen2.5-0.5B since it's the fastest model and overhead should
be the largest fraction of total time there.
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

print(f"TT-NN version: {ttnn.__version__ if hasattr(ttnn, '__version__') else 'unknown'}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")

# Qwen2.5-0.5B config
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
n_layers = 24; vocab_size = 151936; rms_eps = 1e-6
rope_theta = 1000000.0; MAX_SEQ = 512
half_dim = head_dim // 2; TILE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device grid: {grid.x}x{grid.y}")

# Load model
print("Loading Qwen2.5-0.5B-Instruct...")
sf_path = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors")
with safe_open(sf_path, framework="pt") as f:
    all_weights = {k: f.get_tensor(k).float().numpy() for k in f.keys()}

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

tok_path = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

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

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)

def get_rope_tables(T):
    angles = np.outer(np.arange(T), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos_t, sin_t

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# Upload weights
print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    })
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del all_weights
print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

# RoPE rotation matrix (half format for Qwen)
R_half = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_half[i, half_dim + i] = -1.0
    R_half[half_dim + i, i] = 1.0
R_tt = to_bf16(R_half)

# KV caches
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE - 1) // TILE) * TILE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q_np = apply_rope_np(from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_np(from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)
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
    cos_v = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    sin_v = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_v), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_v), rope_sin_buf)
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
        kr_s = ttnn.to_memory_config(kr_4d, kv_cfg)
        v_s = ttnn.to_memory_config(v_4d, kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], kr_s, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_s, update_idxs_tensor=pos_buf)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr, [1, 1, n_q_heads, head_dim]),
            k_caches[i], v_caches[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# BENCHMARK: Compare trace-only vs end-to-end timing
# ══════════════════════════════════════════════════════════════

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
tokens = enc(prompt)
print(f"\nPrompt tokens: {len(tokens)}")

# Prefill
logits = prefill(np.array(tokens))
next_id = int(np.argmax(logits))
pos = len(tokens)

# Warmup
update_buffers(next_id, pos)
_ = decode_forward(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

# Capture trace
update_buffers(next_id, pos)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)

N_STEPS = 100  # Generate 100 tokens for stable measurement
gen = [next_id]

# Method A: Time ONLY trace execution (how we've been measuring)
times_trace_only = []
# Method B: Time each component separately
times_update = []
times_exec = []
times_readback = []
times_argmax = []

for step in range(N_STEPS):
    # 1. update_buffers
    t_upd0 = time.perf_counter()
    update_buffers(next_id, pos)
    t_upd1 = time.perf_counter()

    # 2. trace execution
    t_exec0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    t_exec1 = time.perf_counter()

    # 3. readback
    t_read0 = time.perf_counter()
    logits = from_dev(logits_ref, (1, vocab_size))[0]
    t_read1 = time.perf_counter()

    # 4. argmax
    t_arg0 = time.perf_counter()
    next_id = int(np.argmax(logits))
    t_arg1 = time.perf_counter()

    times_update.append(t_upd1 - t_upd0)
    times_exec.append(t_exec1 - t_exec0)
    times_readback.append(t_read1 - t_read0)
    times_argmax.append(t_arg1 - t_arg0)

    gen.append(next_id)
    pos += 1
    if next_id in {151643, 151645}:
        break

ttnn.release_trace(device, trace_id)

# Skip first measurement for warmup
s = 1
avg_update = np.mean(times_update[s:]) * 1000
avg_exec = np.mean(times_exec[s:]) * 1000
avg_read = np.mean(times_readback[s:]) * 1000
avg_argmax = np.mean(times_argmax[s:]) * 1000
avg_total = avg_update + avg_exec + avg_read + avg_argmax

text = tokenizer.decode(gen, skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"BENCHMARK AUDIT: Qwen2.5-0.5B-Instruct on Blackhole P150")
print(f"{'='*60}")
print(f"  Tokens generated: {len(gen)} (over {len(times_exec)-1} measured steps)")
print(f"  Text: {text[:200]}")
print(f"\n  Component breakdown (avg over {len(times_exec)-1} steps):")
print(f"    update_buffers:  {avg_update:.2f}ms  ({avg_update/avg_total*100:.1f}%)")
print(f"    execute_trace:   {avg_exec:.2f}ms  ({avg_exec/avg_total*100:.1f}%)")
print(f"    from_dev:        {avg_read:.2f}ms  ({avg_read/avg_total*100:.1f}%)")
print(f"    np.argmax:       {avg_argmax:.3f}ms ({avg_argmax/avg_total*100:.1f}%)")
print(f"    ─────────────────────────────")
print(f"    TOTAL:           {avg_total:.2f}ms/tok")
print(f"\n  Reported (trace-only): {avg_exec:.1f}ms/tok = {1000/avg_exec:.0f} tok/sec")
print(f"  Actual (end-to-end):   {avg_total:.1f}ms/tok = {1000/avg_total:.0f} tok/sec")
print(f"  Overhead:              {avg_total - avg_exec:.2f}ms = {(avg_total-avg_exec)/avg_exec*100:.1f}%")
print(f"\n  Min/max execute_trace: {np.min(times_exec[s:])*1000:.1f}ms / {np.max(times_exec[s:])*1000:.1f}ms")
print(f"  Std dev:               {np.std(times_exec[s:])*1000:.2f}ms")

# Theoretical ceiling
model_bytes = 0.98e9  # ~0.98 GB at bf16
bw = 450e9  # 450 GB/s measured DRAM BW
ceiling_ms = model_bytes / bw * 1000
print(f"\n  Bandwidth ceiling:     {ceiling_ms:.1f}ms = {1000/ceiling_ms:.0f} tok/sec")
print(f"  Efficiency (trace):    {ceiling_ms/avg_exec*100:.0f}%")
print(f"  Efficiency (e2e):      {ceiling_ms/avg_total*100:.0f}%")

ttnn.close_device(device)
print("\nDone!")
