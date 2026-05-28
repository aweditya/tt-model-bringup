#!/usr/bin/env python3
"""
Experiment 84: BFP8 MLP weights on Llama-3.1-8B — bandwidth optimization

8B model IS bandwidth-bound (52ms/tok, ceiling 36ms). BFP8 halves MLP weight reads:
- MLP: 3 x (4096 x 14336) x 32 layers = ~5.3 GB at bf16 → ~2.6 GB at bfp8
- Attention: stays at bf16 (Q/K/V/O projections)
- Total weight reads: 16.1 GB → ~13.4 GB (17% reduction)
- New ceiling: 13.4 GB / 450 GB/s = 29.8ms → 33.5 tok/s

BFP8 is proven safe from exp 61 (all 24 layers > 0.999 cosine on 0.5B).

Also tests: HiFi2 math fidelity for MLP matmuls (less accurate but faster compute).
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
intermediate_size = 14336
TILE_SIZE = 32; batch_size = 1

n_kv_split = n_kv_heads // 2  # 4 per group
n_q_split = n_q_heads // 2    # 16 per group

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
            shard_paths = paths
            model_id = mid
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

total_params = sum(v.size for v in all_weights.values())
print(f"  Total: {total_params/1e9:.1f}B params ({total_params*2/1e9:.1f} GB bf16)")

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
# Upload weights: bf16 attention, bfp8 MLP
# ══════════════════════════════════════════════════════════════

print("\nUploading weights...")
t0 = time.perf_counter()

# Baseline: all bf16
dev_layers_bf16 = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers_bf16.append({
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
    if (i + 1) % 8 == 0: print(f"    BF16 layer {i+1}/{n_layers}")

print(f"  BF16 uploaded in {time.perf_counter()-t0:.0f}s")

# BFP8 MLP: attention bf16, MLP bfp8
t1 = time.perf_counter()
dev_layers_bfp8 = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers_bfp8.append({
        "ln1_g": dev_layers_bf16[i]["ln1_g"],   # share attention weights
        "q_w": dev_layers_bf16[i]["q_w"],
        "k_w": dev_layers_bf16[i]["k_w"],
        "v_w": dev_layers_bf16[i]["v_w"],
        "o_w": dev_layers_bf16[i]["o_w"],
        "ln2_g": dev_layers_bf16[i]["ln2_g"],
        "gate_w": to_bfp8(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bfp8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bfp8(lw["mlp.down_proj.weight"].T),
    })
    if (i + 1) % 8 == 0: print(f"    BFP8 MLP layer {i+1}/{n_layers}")

print(f"  BFP8 MLP uploaded in {time.perf_counter()-t1:.0f}s")

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Total upload: {time.perf_counter()-t0:.0f}s")

# RoPE rotation matrix (interleaved for Llama)
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

# KV caches (split: 2 groups of 4 KV heads)
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


def make_decode_fn(dev_layers, mlp_cfg):
    """Create decode function with given weights and MLP math config."""
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

tokens = make_chat_tokens("What is the capital of France?")
print(f"\nPrompt tokens: {len(tokens)}")


def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches_lo[i])
        ttnn.copy(to_dev_4d(c), v_caches_lo[i])
        ttnn.copy(to_dev_4d(c), k_caches_hi[i])
        ttnn.copy(to_dev_4d(c), v_caches_hi[i])


def benchmark_traced(decode_fn, dev_layers, label, n_steps=30):
    """Benchmark traced decode with correct end-to-end timing."""
    reset_kv_caches()
    logits_p = prefill(np.array(tokens), dev_layers)
    next_id = int(np.argmax(logits_p))
    pos = len(tokens)

    # Warmup
    update_buffers(next_id, pos)
    _ = decode_fn(); ttnn.synchronize_device(device)
    try: device.enable_program_cache()
    except: pass

    # Capture trace
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


print("\n" + "="*60)
print("BENCHMARK: BF16 baseline vs BFP8 MLP on Llama-3.1-8B")
print("="*60)

# Test 1: BF16 baseline (same as exp 73/80)
bf16_exec, bf16_total, bf16_gen = benchmark_traced(
    make_decode_fn(dev_layers_bf16, hifi4), dev_layers_bf16, "BF16 (baseline)")

# Test 2: BFP8 MLP with HiFi4 math
bfp8_hifi4_exec, bfp8_hifi4_total, bfp8_hifi4_gen = benchmark_traced(
    make_decode_fn(dev_layers_bfp8, hifi4), dev_layers_bfp8, "BFP8 MLP + HiFi4")

# Test 3: BFP8 MLP with HiFi2 math (fastest expected config)
bfp8_hifi2_exec, bfp8_hifi2_total, bfp8_hifi2_gen = benchmark_traced(
    make_decode_fn(dev_layers_bfp8, hifi2), dev_layers_bfp8, "BFP8 MLP + HiFi2")


# ══════════════════════════════════════════════════════════════
# Correctness check
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"CORRECTNESS CHECK")
print(f"{'='*60}")

# Compare first 20 tokens
n = min(20, len(bf16_gen), len(bfp8_hifi4_gen), len(bfp8_hifi2_gen))
match_hifi4 = sum(1 for a, b in zip(bf16_gen[:n], bfp8_hifi4_gen[:n]) if a == b)
match_hifi2 = sum(1 for a, b in zip(bf16_gen[:n], bfp8_hifi2_gen[:n]) if a == b)

print(f"  BFP8+HiFi4 token match: {match_hifi4}/{n}")
print(f"  BFP8+HiFi2 token match: {match_hifi2}/{n}")

bf16_text = tokenizer.decode(bf16_gen[:n], skip_special_tokens=True)
bfp8_h4_text = tokenizer.decode(bfp8_hifi4_gen[:n], skip_special_tokens=True)
bfp8_h2_text = tokenizer.decode(bfp8_hifi2_gen[:n], skip_special_tokens=True)
print(f"\n  BF16:       {bf16_text[:100]}")
print(f"  BFP8+HiFi4: {bfp8_h4_text[:100]}")
print(f"  BFP8+HiFi2: {bfp8_h2_text[:100]}")


# ══════════════════════════════════════════════════════════════
# Results summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY — Llama-3.1-8B-Instruct")
print(f"{'='*60}")

print(f"\n  Speed:")
print(f"    BF16:          trace={bf16_exec:.1f}ms e2e={bf16_total:.1f}ms ({1000/bf16_total:.0f} tok/s)")
print(f"    BFP8+HiFi4:   trace={bfp8_hifi4_exec:.1f}ms e2e={bfp8_hifi4_total:.1f}ms ({1000/bfp8_hifi4_total:.0f} tok/s)")
print(f"    BFP8+HiFi2:   trace={bfp8_hifi2_exec:.1f}ms e2e={bfp8_hifi2_total:.1f}ms ({1000/bfp8_hifi2_total:.0f} tok/s)")

print(f"\n  Speedup vs baseline:")
print(f"    BFP8+HiFi4: {bf16_exec/bfp8_hifi4_exec:.3f}x trace, {bf16_total/bfp8_hifi4_total:.3f}x e2e")
print(f"    BFP8+HiFi2: {bf16_exec/bfp8_hifi2_exec:.3f}x trace, {bf16_total/bfp8_hifi2_total:.3f}x e2e")

print(f"\n  Weight memory:")
mlp_per_layer = 3 * hidden * intermediate_size  # gate + up + down
total_mlp = mlp_per_layer * n_layers
bf16_mlp_gb = total_mlp * 2 / 1e9
bfp8_mlp_gb = total_mlp * 1 / 1e9  # bfp8 = 1 byte/param
attn_per_layer = hidden * (hidden + 2 * (n_kv_heads * head_dim) + hidden)  # Q + K + V + O
total_attn = attn_per_layer * n_layers
bf16_attn_gb = total_attn * 2 / 1e9
print(f"    MLP weights: {bf16_mlp_gb:.1f} GB (bf16) → {bfp8_mlp_gb:.1f} GB (bfp8)")
print(f"    Attn weights: {bf16_attn_gb:.1f} GB (bf16, unchanged)")
total_bf16 = (total_mlp + total_attn) * 2 / 1e9
total_bfp8 = total_mlp * 1 / 1e9 + total_attn * 2 / 1e9
print(f"    Total: {total_bf16:.1f} GB → {total_bfp8:.1f} GB ({(1-total_bfp8/total_bf16)*100:.0f}% reduction)")
ceiling = total_bfp8 / 0.450 * 1000  # ms at 450 GB/s
print(f"    New BW ceiling: {ceiling:.1f}ms = {1000/ceiling:.0f} tok/s")

ttnn.close_device(device)
print("\nDone!")
