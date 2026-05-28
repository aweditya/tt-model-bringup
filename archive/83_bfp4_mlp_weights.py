#!/usr/bin/env python3
"""
Experiment 83: BFP4 MLP weights — biggest projected optimization

TT's reference Llama implementation uses bfloat4_b for MLP weights (gate, up, down)
with LoFi math fidelity. This gave them a 22% speedup on 8B.

Our exp 10 showed 56.7% error on RANDOM data, but trained weights have structured
distributions that preserve well under block floating point quantization.

Test plan:
1. Per-layer cosine similarity: bf16 vs bfp4 MLP output
2. End-to-end token match: greedy decode bf16 vs bfp4
3. Traced decode speed: baseline vs bfp4 MLP

On Qwen2.5-0.5B first for fast iteration.
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

# LoFi config for bfp4 matmuls (what TT's reference uses)
lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True)

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

def to_bfp4(arr):
    """Convert weight array to bfloat4_b on device."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT)

def to_bfp8(arr):
    """Convert weight array to bfloat8_b on device."""
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

# ══════════════════════════════════════════════════════════════
# TEST 1: Per-layer MLP cosine similarity (bf16 vs bfp4)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 1: Per-layer MLP cosine similarity (bf16 vs bfp4)")
print("="*60)

# Use a realistic hidden state (not random) — take from embedding
test_input = embed_w[42].reshape(1, hidden)  # A real embedding vector

for layer_idx in [0, 6, 12, 18, 23]:
    prefix = f"model.layers.{layer_idx}."
    gate_np = all_weights[f"{prefix}mlp.gate_proj.weight"].T
    up_np = all_weights[f"{prefix}mlp.up_proj.weight"].T
    down_np = all_weights[f"{prefix}mlp.down_proj.weight"].T

    h_tt = to_bf16(test_input)

    # BF16 reference
    gate_bf16 = to_bf16(gate_np)
    up_bf16 = to_bf16(up_np)
    down_bf16 = to_bf16(down_np)
    g16 = ttnn.matmul(h_tt, gate_bf16, compute_kernel_config=hifi4)
    u16 = ttnn.matmul(h_tt, up_bf16, compute_kernel_config=hifi4)
    d16 = ttnn.matmul(ttnn.mul(ttnn.silu(g16), u16), down_bf16, compute_kernel_config=hifi4)
    ref = from_dev(d16, (1, hidden))[0]

    # BFP4 weights with LoFi math
    gate_bfp4 = to_bfp4(gate_np)
    up_bfp4 = to_bfp4(up_np)
    down_bfp4 = to_bfp4(down_np)
    g4 = ttnn.matmul(h_tt, gate_bfp4, compute_kernel_config=lofi)
    u4 = ttnn.matmul(h_tt, up_bfp4, compute_kernel_config=lofi)
    d4 = ttnn.matmul(ttnn.mul(ttnn.silu(g4), u4), down_bfp4, compute_kernel_config=lofi)
    bfp4_out = from_dev(d4, (1, hidden))[0]

    # BFP8 weights with HiFi4 (for comparison)
    gate_bfp8 = to_bfp8(gate_np)
    up_bfp8 = to_bfp8(up_np)
    down_bfp8 = to_bfp8(down_np)
    g8 = ttnn.matmul(h_tt, gate_bfp8, compute_kernel_config=hifi4)
    u8 = ttnn.matmul(h_tt, up_bfp8, compute_kernel_config=hifi4)
    d8 = ttnn.matmul(ttnn.mul(ttnn.silu(g8), u8), down_bfp8, compute_kernel_config=hifi4)
    bfp8_out = from_dev(d8, (1, hidden))[0]

    cos4 = np.dot(ref, bfp4_out) / (np.linalg.norm(ref) * np.linalg.norm(bfp4_out) + 1e-10)
    cos8 = np.dot(ref, bfp8_out) / (np.linalg.norm(ref) * np.linalg.norm(bfp8_out) + 1e-10)

    print(f"  Layer {layer_idx:2d}: bfp4 cosine={cos4:.6f}  bfp8 cosine={cos8:.6f}")

    # Clean up device tensors
    del gate_bf16, up_bf16, down_bf16, gate_bfp4, up_bfp4, down_bfp4, gate_bfp8, up_bfp8, down_bfp8


# ══════════════════════════════════════════════════════════════
# Upload weights for traced decode
# ══════════════════════════════════════════════════════════════

print("\nUploading weights (baseline bf16)...")
t0 = time.perf_counter()
dev_layers_bf16 = []
dev_layers_bfp4 = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}

    # Shared attention weights (always bf16)
    shared = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
    }

    # BF16 MLP weights
    dev_layers_bf16.append({**shared,
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    })

    # BFP4 MLP weights
    dev_layers_bfp4.append({**shared,
        "gate_w": to_bfp4(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bfp4(lw["mlp.up_proj.weight"].T),
        "down_w": to_bfp4(lw["mlp.down_proj.weight"].T),
    })

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
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


def prefill(token_ids, dev_layers):
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
        # Use appropriate math fidelity for MLP based on weight dtype
        mlp_cfg = hifi4  # prefill always uses hifi4 for accuracy
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=mlp_cfg)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=mlp_cfg)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=mlp_cfg)
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


def make_decode_fn(dev_layers, mlp_cfg):
    """Create a decode function using given layers and MLP compute config."""
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
            g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=mlp_cfg)
            u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=mlp_cfg)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=mlp_cfg)
            x = ttnn.add(x, d)
        return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)
    return decode_forward


# ══════════════════════════════════════════════════════════════
# TEST 2: End-to-end logit cosine + token match
# ══════════════════════════════════════════════════════════════

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
tokens = enc(prompt)
print(f"\nPrompt tokens: {len(tokens)}")

print("\n" + "="*60)
print("TEST 2: End-to-end logit cosine + token match (first 20 tokens)")
print("="*60)

# BF16 baseline prefill + 20 greedy tokens
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    ttnn.copy(to_dev_4d(c), k_caches[i])
    ttnn.copy(to_dev_4d(c), v_caches[i])

logits_bf16 = prefill(np.array(tokens), dev_layers_bf16)
bf16_tokens = [int(np.argmax(logits_bf16))]

decode_bf16 = make_decode_fn(dev_layers_bf16, hifi4)
update_buffers(bf16_tokens[0], len(tokens))
_ = decode_bf16(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

for step in range(19):
    update_buffers(bf16_tokens[-1], len(tokens) + step)
    logits_tt = from_dev(decode_bf16(), (1, vocab_size))[0]
    bf16_tokens.append(int(np.argmax(logits_tt)))
    if bf16_tokens[-1] in {151643, 151645}: break

bf16_text = tokenizer.decode(bf16_tokens, skip_special_tokens=True)
print(f"  BF16:  {bf16_text[:100]}")
print(f"  Tokens: {bf16_tokens[:20]}")

# BFP4 MLP prefill + 20 greedy tokens
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    ttnn.copy(to_dev_4d(c), k_caches[i])
    ttnn.copy(to_dev_4d(c), v_caches[i])

logits_bfp4 = prefill(np.array(tokens), dev_layers_bfp4)
bfp4_tokens = [int(np.argmax(logits_bfp4))]

decode_bfp4 = make_decode_fn(dev_layers_bfp4, lofi)
update_buffers(bfp4_tokens[0], len(tokens))
_ = decode_bfp4(); ttnn.synchronize_device(device)

for step in range(19):
    update_buffers(bfp4_tokens[-1], len(tokens) + step)
    logits_tt = from_dev(decode_bfp4(), (1, vocab_size))[0]
    bfp4_tokens.append(int(np.argmax(logits_tt)))
    if bfp4_tokens[-1] in {151643, 151645}: break

bfp4_text = tokenizer.decode(bfp4_tokens, skip_special_tokens=True)
print(f"  BFP4:  {bfp4_text[:100]}")
print(f"  Tokens: {bfp4_tokens[:20]}")

# Compare
logit_cos = np.dot(logits_bf16, logits_bfp4) / (np.linalg.norm(logits_bf16) * np.linalg.norm(logits_bfp4) + 1e-10)
match_count = sum(1 for a, b in zip(bf16_tokens[:20], bfp4_tokens[:20]) if a == b)
print(f"\n  First-token logit cosine: {logit_cos:.6f}")
print(f"  Token match: {match_count}/{min(len(bf16_tokens), len(bfp4_tokens), 20)}")


# ══════════════════════════════════════════════════════════════
# TEST 3: Traced decode speed comparison
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 3: Traced decode speed (50 tokens)")
print("="*60)

N_STEPS = 50

def benchmark_traced(decode_fn, dev_layers, mlp_cfg_name, n_steps=N_STEPS):
    """Benchmark traced decode."""
    # Reset caches
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])

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
        if next_id in {151643, 151645}: break

    ttnn.release_trace(device, tid)

    s = 1  # skip first
    avg_exec = np.mean(times_exec[s:]) * 1000
    avg_total = np.mean(times_total[s:]) * 1000
    text = tokenizer.decode(gen, skip_special_tokens=True)

    print(f"\n  {mlp_cfg_name}:")
    print(f"    Trace-only: {avg_exec:.2f}ms/tok = {1000/avg_exec:.0f} tok/s")
    print(f"    End-to-end: {avg_total:.2f}ms/tok = {1000/avg_total:.0f} tok/s")
    print(f"    Generated {len(gen)} tokens")
    print(f"    Text: {text[:120]}")

    return avg_exec, avg_total

bf16_exec, bf16_total = benchmark_traced(
    make_decode_fn(dev_layers_bf16, hifi4), dev_layers_bf16, "BF16 MLP (baseline)")

bfp4_exec, bfp4_total = benchmark_traced(
    make_decode_fn(dev_layers_bfp4, lofi), dev_layers_bfp4, "BFP4 MLP + LoFi")


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  BFP4 speedup (trace): {bf16_exec/bfp4_exec:.3f}x ({bf16_exec:.2f}ms -> {bfp4_exec:.2f}ms)")
print(f"  BFP4 speedup (e2e):   {bf16_total/bfp4_total:.3f}x ({bf16_total:.2f}ms -> {bfp4_total:.2f}ms)")
print(f"\n  MLP weight savings:")
mlp_params = 3 * hidden * (hidden * 4)  # gate + up + down, intermediate = 4x hidden for Qwen-0.5B
# Actually Qwen-0.5B intermediate = 4864
inter_size = 4864
mlp_params = 3 * hidden * inter_size
bf16_bytes = mlp_params * 2 * n_layers
bfp4_bytes = mlp_params // 2 * n_layers  # bfp4 = 0.5 bytes per param
print(f"    BF16 MLP weights: {bf16_bytes/1e6:.1f} MB")
print(f"    BFP4 MLP weights: {bfp4_bytes/1e6:.1f} MB")
print(f"    Savings: {(bf16_bytes - bfp4_bytes)/1e6:.1f} MB ({(1 - bfp4_bytes/bf16_bytes)*100:.0f}%)")

ttnn.close_device(device)
print("\nDone!")
