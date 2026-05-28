#!/usr/bin/env python3
"""
Experiment 66: Qwen3-0.6B port to Blackhole.

Direct successor to our Qwen2.5-0.5B. Key differences:
  - 28 layers (vs 24), 1024 hidden (vs 896), 16 Q heads (vs 14), 8 KV heads (vs 2)
  - head_dim = 128 (vs 64)
  - intermediate_size = 3072 (vs 4864)
  - NO biases on any projections (Qwen2.5 had Q/K/V biases)
  - QK-Norm: RMSNorm on Q and K per-head before attention (NEW OP)
  - Same vocab, same rope_theta, same RoPE format (half)
  - Tied embeddings (embed_tokens = lm_head)

New TT-NN primitive tested: QK-Norm via per-head RMSNorm.
Also uses split SDPA workaround (8 KV heads → 2× 4 KV heads).
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
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=100)
args = parser.parse_args()

# Qwen3-0.6B architecture
hidden = 1024; n_q_heads = 16; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 28; vocab_size = 151936; MAX_SEQ = 256
intermediate_size = 3072
TILE_SIZE = 32; batch_size = 1

# Split SDPA: 8 KV heads → 2 groups of 4
n_kv_split = n_kv_heads // 2  # 4
n_q_split = n_q_heads // 2    # 8

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model ──
print("Loading Qwen3-0.6B...")
model_id = "Qwen/Qwen3-0.6B"
model_path = hf_hub_download(model_id, "model.safetensors")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

print(f"  Loaded {len(all_weights)} tensors")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = embed_w.T.copy()  # Tied embeddings

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)

print(f"  Q weight: {layer_weights_np[0]['self_attn.q_proj.weight'].shape}")
print(f"  K weight: {layer_weights_np[0]['self_attn.k_proj.weight'].shape}")
print(f"  q_norm: {layer_weights_np[0]['self_attn.q_norm.weight'].shape}")
print(f"  gate_proj: {layer_weights_np[0]['mlp.gate_proj.weight'].shape}")
del all_weights

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


# ── RoPE: Qwen3 uses half format (same as Qwen2.5) ──
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    x_rot = np.concatenate([-x_4d[..., half_dim:], x_4d[..., :half_dim]], axis=-1)
    return x_4d * cos_t[None, None] + x_rot * sin_t[None, None]

# Rotation matrix for on-device decode
R_half = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_half[i, i + half_dim] = -1.0
    R_half[i + half_dim, i] = 1.0
R_tt = to_bf16(R_half)


# ── QK-Norm: RMSNorm per head ──
def qk_norm_np(x_4d, gamma, eps=rms_eps):
    """RMSNorm on last dim of (B, n_heads, T, head_dim) using gamma of shape (head_dim,)."""
    rms = np.sqrt(np.mean(x_4d ** 2, axis=-1, keepdims=True) + eps)
    return (x_4d / rms) * gamma


# ── Upload weights ──
print("Uploading weights to device...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "q_norm_g": to_bf16(lw["self_attn.q_norm.weight"]),
        "k_norm_g": to_bf16(lw["self_attn.k_norm.weight"]),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
dt_upload = time.perf_counter() - t0
print(f"  Uploaded in {dt_upload*1000:.0f}ms")

# ── KV caches (split: 2 groups of 4 KV heads) ──
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

# ── Buffers ──
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


# ── Prefill (CPU RoPE + QK-Norm) ──
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        lw_np = {  # Keep numpy weights for QK-norm
            "q_norm_g": from_dev(dl["q_norm_g"], (head_dim,)),
            "k_norm_g": from_dev(dl["k_norm_g"], (head_dim,)),
        }

        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)

        # Reshape to multi-head: (B, n_heads, T, head_dim)
        q_np = from_dev(q, (B*T, n_q_heads*head_dim)).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k_np = from_dev(k, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_np = from_dev(v, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # QK-Norm: RMSNorm per head
        q_np = qk_norm_np(q_np, lw_np["q_norm_g"])
        k_np = qk_norm_np(k_np, lw_np["k_norm_g"])

        # Apply RoPE (half format)
        q_np = apply_rope_half_np(q_np, cos_t, sin_t)
        k_np = apply_rope_half_np(k_np, cos_t, sin_t)

        # Fill split KV caches
        ttnn.kv_cache.fill_cache_for_user_(k_caches_lo[i], to_dev_4d(k_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_lo[i], to_dev_4d(v_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_hi[i], to_dev_4d(k_np[:, n_kv_split:]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_hi[i], to_dev_4d(v_np[:, n_kv_split:]), batch_index=0)

        # SDPA (regular, handles GQA natively)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, n_q_heads*head_dim)

        o = ttnn.matmul(to_bf16(a_np.reshape(B*T, n_q_heads*head_dim)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T, hidden)).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T, hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


# ── Traced decode ──
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

        # QK-Norm on device: RMSNorm per head
        # rms_norm works on last dim, gamma broadcasts across heads
        q = ttnn.rms_norm(q, weight=dl["q_norm_g"], epsilon=rms_eps)
        k = ttnn.rms_norm(k, weight=dl["k_norm_g"], epsilon=rms_eps)

        # On-device RoPE via rotation matrix (half format)
        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))

        # Split KV for paged cache update
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

        # Split SDPA decode: 2 groups of (8Q, 4KV)
        qr_4d = ttnn.reshape(qr, [1, 1, n_q_heads, head_dim])
        q_lo = ttnn.slice(qr_4d, [0,0,0,0], [1,1,n_q_split,head_dim])
        q_hi = ttnn.slice(qr_4d, [0,0,n_q_split,0], [1,1,n_q_heads,head_dim])
        attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
            q_lo, k_caches_lo[i], v_caches_lo[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
            q_hi, k_caches_hi[i], v_caches_hi[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn = ttnn.concat([attn_lo, attn_hi], dim=2)

        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)

        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)

    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
tokens_list = list(tokenizer.encode(args.prompt))
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))
print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
dt_prefill = time.perf_counter() - t0
next_id = int(np.argmax(logits))
tokens_list.append(next_id)
pos = len(tokens_list) - 1
print(f"Prefill: {dt_prefill*1000:.0f}ms")
print(f"First token: {next_id} ({tokenizer.decode([next_id])})")

# Warmup decode
update_buffers(next_id, pos)
_ = decode_forward()
ttnn.synchronize_device(device)

# Program cache + trace
try:
    device.enable_program_cache()
except:
    pass

update_buffers(next_id, pos)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
print("Trace captured")

# Generate
times = []
for step in range(max_gen - 1):
    update_buffers(next_id, pos)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    times.append(dt)

    logits = from_dev(logits_ref, (1, vocab_size))
    next_id = int(np.argmax(logits[0]))
    tokens_list.append(next_id)
    pos += 1
    if next_id == tokenizer.eos_token_id:
        break

text = tokenizer.decode(tokens_list)
sustained = times[1:] if len(times) > 1 else times
avg_ms = np.mean(sustained) * 1000
tps = 1000 / avg_ms

print(f"\n{'='*60}")
print(f"RESULTS: Qwen3-0.6B on Blackhole P150")
print(f"{'='*60}")
print(f"  Architecture: {n_layers} layers, {hidden} hidden, {n_q_heads} Q heads, {n_kv_heads} KV heads, head_dim={head_dim}")
print(f"  Parameters: ~0.6B")
print(f"  Upload: {dt_upload*1000:.0f}ms")
print(f"  Prefill: {dt_prefill*1000:.0f}ms")
print(f"  Traced decode: {avg_ms:.1f}ms/tok ({tps:.1f} tok/sec)")
print(f"  Tokens generated: {len(tokens_list) - len(tokenizer.encode(args.prompt))}")
print(f"  Text: {text}")
print(f"\n  Comparison:")
print(f"    Qwen2.5-0.5B: 7.1ms/tok (140 tok/sec) — 24 layers, 896 hidden, head_dim=64")
print(f"    Qwen3-0.6B:   {avg_ms:.1f}ms/tok ({tps:.0f} tok/sec) — 28 layers, 1024 hidden, head_dim=128")
print(f"    Llama-3.2-1B:  12.8ms/tok (78 tok/sec) — 16 layers, 2048 hidden, head_dim=64")

ttnn.close_device(device)
print("\nDone!")
