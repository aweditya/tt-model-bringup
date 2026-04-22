#!/usr/bin/env python3
"""
Experiment 82: Fused SiLU + On-device topk on Qwen2.5-0.5B

Two optimizations from research/ttnn_advanced_features.md:

1. Fused SiLU in gate matmul:
   OLD: g = ttnn.silu(ttnn.matmul(h, gate_w))          # 2 ops
   NEW: g = ttnn.matmul(h, gate_w, activation="silu")   # 1 fused op
   Expected: eliminates 1 kernel launch + 1 memory pass per layer (24 layers)

2. On-device topk(k=1) to replace CPU argmax:
   OLD: logits -> PCIe readback (3.9ms) -> np.argmax
   NEW: ttnn.topk(logits, k=1) -> read back single int
   Problem: vocab_size=151936 > 65536 (topk multicore limit)
   Workaround: split into chunks, topk each, compare

Tests each optimization independently, then combined.
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

# Also upload embedding table for on-device lookup
embed_w_tt = to_bf16(embed_w)  # [vocab_size, hidden] on device

del all_weights
print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

# RoPE rotation matrix
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


# ══════════════════════════════════════════════════════════════
# BASELINE: Original decode (same as exp 81)
# ══════════════════════════════════════════════════════════════

def decode_baseline():
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
# OPTIMIZATION A: Fused SiLU in gate matmul
# ══════════════════════════════════════════════════════════════

def decode_fused_silu():
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
        # FUSED: matmul + silu in one op (eliminates separate silu kernel)
        g = ttnn.matmul(h2, dl["gate_w"], activation="silu", compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(g, u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# FIRST: Test ttnn.topk standalone (outside trace)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 1: ttnn.topk feasibility")
print("="*60)

# Create a dummy logits tensor matching our vocab size
dummy_logits = np.random.randn(1, 1, 1, vocab_size).astype(np.float32)
# Pad to tile-aligned width
padded_vocab = ((vocab_size + TILE - 1) // TILE) * TILE
print(f"  Vocab size: {vocab_size}, padded: {padded_vocab}")

# Try topk on the full vocab
try:
    logits_tt = to_dev_4d(dummy_logits)
    print(f"  Logits tensor shape: {logits_tt.shape}")
    t0 = time.perf_counter()
    vals, idxs = ttnn.topk(logits_tt, k=1)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"  topk(k=1) on full vocab: {(t1-t0)*1000:.2f}ms")
    idx_val = from_dev(idxs, (1,))[0]
    expected = int(np.argmax(dummy_logits))
    print(f"  topk result: {int(idx_val)}, expected argmax: {expected}, match: {int(idx_val)==expected}")
    TOPK_WORKS = True
except Exception as e:
    print(f"  topk FAILED: {e}")
    TOPK_WORKS = False

# Try ttnn.argmax as fallback
print("\n  Testing ttnn.argmax...")
try:
    logits_tt = to_dev_4d(dummy_logits)
    t0 = time.perf_counter()
    result = ttnn.argmax(logits_tt, dim=-1)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"  argmax on full vocab: {(t1-t0)*1000:.2f}ms")
    idx_val = ttnn.to_torch(result).item()
    expected = int(np.argmax(dummy_logits))
    print(f"  argmax result: {idx_val}, expected: {expected}, match: {idx_val==expected}")
    ARGMAX_WORKS = True
except Exception as e:
    print(f"  argmax FAILED: {e}")
    ARGMAX_WORKS = False

# Try ttnn.embedding
print("\n  Testing ttnn.embedding...")
try:
    idx_tensor = ttnn.from_torch(torch.tensor([[42]], dtype=torch.int32), device=device)
    # embedding weight must be ROW_MAJOR
    embed_rm = ttnn.from_torch(
        torch.from_numpy(embed_w.astype(np.float32)),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    t0 = time.perf_counter()
    emb_out = ttnn.embedding(idx_tensor, embed_rm)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    emb_np = from_dev(emb_out, (1, hidden))
    expected_emb = embed_w[42]
    cos_sim = np.dot(emb_np[0], expected_emb) / (np.linalg.norm(emb_np[0]) * np.linalg.norm(expected_emb))
    print(f"  embedding lookup: {(t1-t0)*1000:.2f}ms, cosine: {cos_sim:.6f}")
    EMBEDDING_WORKS = True
except Exception as e:
    print(f"  embedding FAILED: {e}")
    EMBEDDING_WORKS = False


# ══════════════════════════════════════════════════════════════
# BENCHMARK: Baseline vs Fused SiLU (traced)
# ══════════════════════════════════════════════════════════════

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
tokens = enc(prompt)
print(f"\nPrompt tokens: {len(tokens)}")

# Prefill
logits = prefill(np.array(tokens))
next_id = int(np.argmax(logits))
pos = len(tokens)

def benchmark_decode(decode_fn, name, n_steps=50):
    """Benchmark a decode function with trace capture."""
    global next_id, pos

    # Reset KV caches and re-prefill for fair comparison
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])

    logits_p = prefill(np.array(tokens))
    next_id = int(np.argmax(logits_p))
    pos = len(tokens)

    # Warmup
    update_buffers(next_id, pos)
    _ = decode_fn()
    ttnn.synchronize_device(device)
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
        t_total0 = time.perf_counter()
        update_buffers(next_id, pos)
        t_exec0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t_exec1 = time.perf_counter()
        lgt = from_dev(logits_ref, (1, vocab_size))[0]
        next_id = int(np.argmax(lgt))
        t_total1 = time.perf_counter()

        times_exec.append(t_exec1 - t_exec0)
        times_total.append(t_total1 - t_total0)
        gen.append(next_id)
        pos += 1
        if next_id in {151643, 151645}:
            break

    ttnn.release_trace(device, tid)

    # Skip first step
    s = 1
    avg_exec = np.mean(times_exec[s:]) * 1000
    avg_total = np.mean(times_total[s:]) * 1000
    text = tokenizer.decode(gen, skip_special_tokens=True)

    print(f"\n  {name}:")
    print(f"    Trace-only: {avg_exec:.2f}ms/tok = {1000/avg_exec:.0f} tok/s")
    print(f"    End-to-end: {avg_total:.2f}ms/tok = {1000/avg_total:.0f} tok/s")
    print(f"    Generated {len(gen)} tokens")
    print(f"    Text: {text[:150]}")

    return avg_exec, avg_total, gen

print("\n" + "="*60)
print("BENCHMARK: Baseline vs Fused SiLU")
print("="*60)

base_exec, base_total, base_gen = benchmark_decode(decode_baseline, "BASELINE (separate silu)")
fused_exec, fused_total, fused_gen = benchmark_decode(decode_fused_silu, "FUSED SiLU")

# Check correctness
base_text = tokenizer.decode(base_gen, skip_special_tokens=True)
fused_text = tokenizer.decode(fused_gen, skip_special_tokens=True)
tokens_match = base_gen[:20] == fused_gen[:20]

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Fused SiLU speedup (trace): {base_exec/fused_exec:.3f}x ({base_exec:.2f}ms -> {fused_exec:.2f}ms)")
print(f"  Fused SiLU speedup (e2e):   {base_total/fused_total:.3f}x ({base_total:.2f}ms -> {fused_total:.2f}ms)")
print(f"  First 20 tokens match: {tokens_match}")
print(f"\n  topk works: {TOPK_WORKS}")
print(f"  argmax works: {ARGMAX_WORKS}")
print(f"  embedding works: {EMBEDDING_WORKS}")

ttnn.close_device(device)
print("\nDone!")
