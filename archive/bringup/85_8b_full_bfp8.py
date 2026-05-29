#!/usr/bin/env python3
"""
Experiment 85: Full BFP8 weights on Llama-3.1-8B — attention + MLP

Exp 84 proved BFP8 MLP is safe (8/8 token match, 1.20x speedup).
Now test BFP8 for attention weights too (Q/K/V/O projections).

Weight memory breakdown at bf16:
  MLP:  3 x 4096 x 14336 x 32 layers = 11.3 GB
  Attn: (4096x4096 + 2x4096x1024 + 4096x4096) x 32 = 2.7 GB
  Other: embed + lm_head + norms = ~2.1 GB
  Total: ~16.1 GB

Full BFP8 (MLP + attention):
  MLP:  5.6 GB (bfp8)
  Attn: 1.3 GB (bfp8)
  Other: ~2.1 GB (bf16, keep norms/embed accurate)
  Total: ~9.0 GB → ceiling = 9.0/0.45 = 20ms = 50 tok/s

Three configs tested:
  A) BF16 baseline (exp 84 verified: 52ms trace, 56ms e2e)
  B) BFP8 MLP only (exp 84 verified: 43ms trace, 47ms e2e)
  C) Full BFP8 (MLP + attention, new)
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

# Llama-3.1-8B architecture
hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 32; vocab_size = 128256; MAX_SEQ = 512
TILE_SIZE = 32; batch_size = 1

n_kv_split = n_kv_heads // 2
n_q_split = n_q_heads // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Load model
print("Loading Llama-3.1-8B-Instruct...")
model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]
shard_paths = []
model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths; model_id = mid
            print(f"  Loaded from {mid} ({n_shards} shards)")
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
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bfp8(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved_np(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1),
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]


# ══════════════════════════════════════════════════════════════
# Upload weights: three configs
# ══════════════════════════════════════════════════════════════

print("\nUploading weights...")
t0 = time.perf_counter()

# Config C: Full BFP8 (attention + MLP)
dev_layers_full_bfp8 = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers_full_bfp8.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),        # norms stay bf16
        "q_w": to_bfp8(lw["self_attn.q_proj.weight"].T),       # attention: bfp8
        "k_w": to_bfp8(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bfp8(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bfp8(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bfp8(lw["mlp.gate_proj.weight"].T),       # MLP: bfp8
        "up_w": to_bfp8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bfp8(lw["mlp.down_proj.weight"].T),
    })
    if (i + 1) % 8 == 0: print(f"    Full BFP8 layer {i+1}/{n_layers}")

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)  # lm_head stays bf16 for accuracy
del layer_weights_np

print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

# RoPE
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


def prefill(token_ids, dev_layers):
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


def make_decode_fn(dev_layers, attn_cfg, mlp_cfg):
    """Create decode function with separate configs for attention and MLP matmuls."""
    def decode_forward():
        x = embed_buf
        for i in range(n_layers):
            dl = dev_layers[i]
            h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
            q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=attn_cfg)
            k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=attn_cfg)
            v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=attn_cfg)
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
            o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=attn_cfg)
            x = ttnn.add(x, o)
            h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
            g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=mlp_cfg)
            u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=mlp_cfg)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=mlp_cfg)
            x = ttnn.add(x, d)
        return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)
    return decode_forward


# ══════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

def make_chat_tokens(prompt, system="You are a helpful assistant."):
    return ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
            [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
            [start_header] + enc("assistant") + [end_header] + enc("\n\n"))


def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches_lo[i])
        ttnn.copy(to_dev_4d(c), v_caches_lo[i])
        ttnn.copy(to_dev_4d(c), k_caches_hi[i])
        ttnn.copy(to_dev_4d(c), v_caches_hi[i])


def benchmark_traced(decode_fn, dev_layers, label, tokens, n_steps=30):
    reset_kv_caches()
    logits_p = prefill(np.array(tokens), dev_layers)
    next_id = int(np.argmax(logits_p))
    pos = len(tokens)

    update_buffers(next_id, pos)
    _ = decode_fn(); ttnn.synchronize_device(device)
    try: device.enable_program_cache()
    except: pass

    update_buffers(next_id, pos)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref = decode_fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    gen = [next_id]
    times_exec = []
    times_total = []

    for step in range(n_steps):
        t0 = time.perf_counter()
        update_buffers(next_id, pos)
        t1 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t2 = time.perf_counter()
        lgt = from_dev(logits_ref, (1, vocab_size))[0]
        next_id = int(np.argmax(lgt))
        t3 = time.perf_counter()

        times_exec.append(t2 - t1)
        times_total.append(t3 - t0)
        gen.append(next_id)
        pos += 1
        if next_id in stop_ids: break

    ttnn.release_trace(device, tid)

    s = 1
    avg_exec = np.mean(times_exec[s:]) * 1000
    avg_total = np.mean(times_total[s:]) * 1000
    text = tokenizer.decode(gen, skip_special_tokens=True)

    print(f"\n  {label}:")
    print(f"    Trace-only: {avg_exec:.2f}ms/tok = {1000/avg_exec:.0f} tok/s")
    print(f"    End-to-end: {avg_total:.2f}ms/tok = {1000/avg_total:.0f} tok/s")
    print(f"    Generated {len(gen)} tokens")
    print(f"    Text: {text[:150]}")

    return avg_exec, avg_total, gen


# Test on multiple prompts for robustness
prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in one paragraph.",
    "Write a haiku about the ocean.",
]

print(f"\n{'='*60}")
print(f"FULL BFP8 BENCHMARK — Llama-3.1-8B-Instruct")
print(f"{'='*60}")

all_results = []
for prompt in prompts:
    tokens = make_chat_tokens(prompt)
    print(f"\nPrompt: \"{prompt}\" ({len(tokens)} tokens)")

    exec_t, total_t, gen = benchmark_traced(
        make_decode_fn(dev_layers_full_bfp8, hifi2, hifi2),
        dev_layers_full_bfp8,
        "Full BFP8 + HiFi2",
        tokens, n_steps=30)
    all_results.append((prompt, exec_t, total_t, gen))


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")

# Compare with exp 84 numbers
print(f"\n  Configuration comparison (all Llama-3.1-8B):")
print(f"    BF16 (exp 84):         trace=52.0ms  e2e=56.1ms  (18 tok/s)")
print(f"    BFP8 MLP (exp 84):     trace=43.0ms  e2e=47.0ms  (21 tok/s)")

avg_exec = np.mean([r[1] for r in all_results])
avg_total = np.mean([r[2] for r in all_results])
print(f"    Full BFP8 (this exp):  trace={avg_exec:.1f}ms  e2e={avg_total:.1f}ms  ({1000/avg_total:.0f} tok/s)")

print(f"\n  Speedup vs BF16:")
print(f"    Full BFP8 trace: {52.0/avg_exec:.3f}x")
print(f"    Full BFP8 e2e:   {56.1/avg_total:.3f}x")

print(f"\n  Speedup vs BFP8 MLP only:")
print(f"    Full BFP8 trace: {43.0/avg_exec:.3f}x")
print(f"    Full BFP8 e2e:   {47.0/avg_total:.3f}x")

print(f"\n  Weight memory:")
mlp_params = 3 * hidden * 14336 * n_layers
attn_params = (hidden*hidden + 2*hidden*(n_kv_heads*head_dim) + hidden*hidden) * n_layers
total_bf16 = (mlp_params + attn_params) * 2 / 1e9
total_full_bfp8 = (mlp_params + attn_params) * 1 / 1e9
print(f"    BF16: {total_bf16:.1f} GB")
print(f"    Full BFP8: {total_full_bfp8:.1f} GB ({(1-total_full_bfp8/total_bf16)*100:.0f}% reduction)")
print(f"    BW ceiling: {total_full_bfp8/0.450:.1f}ms = {450/total_full_bfp8:.0f} tok/s")

print(f"\n  Per-prompt results:")
for prompt, exec_t, total_t, gen in all_results:
    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"    \"{prompt[:40]}...\"")
    print(f"      {exec_t:.1f}ms trace, {total_t:.1f}ms e2e, {len(gen)} tok")
    print(f"      \"{text[:80]}\"")

ttnn.close_device(device)
print("\nDone!")
