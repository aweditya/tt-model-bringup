#!/usr/bin/env python3
"""
Demo: Batch=8 Traced Decode for Qwen2.5-0.5B on Tenstorrent Blackhole
======================================================================

This demonstrates near-perfect linear batch scaling on Blackhole:
  batch=1 traced decode: ~7.6ms/step = 132 tok/sec
  batch=8 traced decode: ~7.6ms/step = 1050 tok/sec  (8x throughput!)

The key insight: a single decode step takes the same wall-clock time
regardless of batch size (up to batch=8). The Blackhole's 140 cores
have enough parallelism to process 8 sequences simultaneously with
zero overhead. This means aggregate throughput scales linearly with
batch size.

How batched decode works:
  - KV caches are 4D: (batch, n_kv_heads, MAX_SEQ, head_dim)
  - Embeddings are (1, 1, batch, hidden) — all sequences in one tensor
  - Position buffer is (batch,) — per-sequence cache positions
  - paged_update_cache writes each sequence's KV to its own cache slot
  - scaled_dot_product_attention_decode handles batch natively
  - RoPE cos/sin broadcast from (1, 1, 1, head_dim) across batch

All 8 sequences are prefilled with the same prompt and decoded greedily,
so they produce identical output — a built-in correctness check.

Based on: experiments/56_batch_traced_decode.py

Usage:
  python3 ~/tt-xla/demos/demo_batch_decode.py
  python3 ~/tt-xla/demos/demo_batch_decode.py --prompt "Once upon a time" --tokens 60
"""

import sys, os, time, argparse

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch=8 traced decode on Blackhole")
parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate per sequence")
parser.add_argument("--prompt", default="The capital of France is", help="Input prompt")
args = parser.parse_args()

# ── Model config (Qwen2.5-0.5B) ─────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32
batch_size = args.batch

# All ops use HiFi4 with fp32 accumulation — required on Blackhole
# to avoid the WormholeComputeKernelConfig mixing bug.
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# ── Device ───────────────────────────────────────────────────
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x * grid.y} cores")
print(f"Batch size: {batch_size}")

# ── Load weights ─────────────────────────────────────────────
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ── Helper functions ─────────────────────────────────────────
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

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Upload weights to device ─────────────────────────────────
print("Uploading weights to device...")
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
print(f"  Uploaded in {(time.perf_counter() - t0) * 1000:.0f}ms")


# ══════════════════════════════════════════════════════════════
# KV caches — (batch, n_kv_heads, MAX_SEQ, head_dim)
# ══════════════════════════════════════════════════════════════
k_caches, v_caches = [], []
for _ in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

# Sharded memory config: one shard per batch element for paged_update_cache.
kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)


# ══════════════════════════════════════════════════════════════
# Input buffers — updated via ttnn.copy() between trace replays
# ══════════════════════════════════════════════════════════════
embed_buf = to_dev_4d(np.zeros((1, 1, batch_size, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.zeros(batch_size, dtype=torch.int32), device=device)


def update_buffers(token_ids, positions):
    """Write new token embeddings, RoPE tables, and positions into device buffers."""
    x_np = embed_w[token_ids].reshape(1, 1, batch_size, hidden)
    ttnn.copy(to_dev_4d(x_np), embed_buf)

    # All sequences share the same position (same prompt, greedy decode)
    pos = positions[0]
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

    ttnn.copy(ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device), pos_buf)


# ══════════════════════════════════════════════════════════════
# Prefill — CPU-side attention, fills KV caches per sequence
# ══════════════════════════════════════════════════════════════
def prefill_single(token_ids, batch_idx):
    """Prefill KV cache for one sequence at the given batch index."""
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

        q_4d = apply_rope_half_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_half_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_4d), batch_index=batch_idx)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_4d), batch_index=batch_idx)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_np.reshape(B * T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
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


# ══════════════════════════════════════════════════════════════
# Batch decode forward — 24 transformer layers, fully on-device
# ══════════════════════════════════════════════════════════════
def decode_forward_batch():
    """One batch decode step. All dynamic values come from device buffers."""
    x_tt = embed_buf  # (1, 1, batch, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]

        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        # Q/K/V projections with fused bias
        q_tt = ttnn.linear(h_tt, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4)
        k_tt = ttnn.linear(h_tt, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4)
        v_tt = ttnn.linear(h_tt, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4)

        # Reshape for multi-head attention
        q_4d = ttnn.reshape(q_tt, [1, batch_size, n_q_heads, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, batch_size, n_kv_heads, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, batch_size, n_kv_heads, head_dim])

        # Native RoPE (half-format for Qwen)
        q_roped = ttnn.experimental.rotary_embedding(q_4d, rope_cos_buf, rope_sin_buf)
        k_roped = ttnn.experimental.rotary_embedding(k_4d, rope_cos_buf, rope_sin_buf)

        # Shard K/V for paged cache update (one shard per batch element)
        k_for_cache = ttnn.reshape(k_roped, [1, batch_size, n_kv_heads, head_dim])
        v_for_cache = ttnn.reshape(v_4d, [1, batch_size, n_kv_heads, head_dim])
        k_sharded = ttnn.to_memory_config(k_for_cache, kv_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_for_cache, kv_mem_cfg)

        # Update KV cache at per-sequence positions
        ttnn.experimental.paged_update_cache(k_caches[i], k_sharded, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_sharded, update_idxs_tensor=pos_buf)

        # Batched SDPA decode
        q_decode = ttnn.reshape(q_roped, [1, batch_size, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)

        # Merge heads back: (1, batch, n_q_heads, head_dim) -> (1, 1, batch, hidden)
        merged = ttnn.reshape(attn, [1, 1, batch_size, hidden])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)

        # MLP: fused gate+silu
        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.linear(h2_tt, dl["gate_w"], activation="silu", compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        down_tt = ttnn.matmul(ttnn.mul(gate_tt, up_tt), dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return logits_tt


# ══════════════════════════════════════════════════════════════
# STEP 1: Prefill all batch sequences
# ══════════════════════════════════════════════════════════════
tokens_list = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens x {batch_size} sequences\n")

print("Prefilling...")
t0 = time.perf_counter()
first_logits = []
for b in range(batch_size):
    logits = prefill_single(np.array(tokens_list), b)
    first_logits.append(logits)
t_prefill = time.perf_counter() - t0
print(f"  Prefill {batch_size} sequences: {t_prefill * 1000:.0f}ms")

# All sequences start with the same greedy next token
next_ids = [int(np.argmax(first_logits[0]))] * batch_size
positions = [len(tokens_list)] * batch_size
tokens_per_seq = [[t for t in tokens_list] + [next_ids[0]] for _ in range(batch_size)]


# ══════════════════════════════════════════════════════════════
# STEP 2: Batch=1 baseline (traced, for comparison)
# ══════════════════════════════════════════════════════════════
# We skip an actual batch=1 run and use the known baseline from
# experiment 53e: 7.6ms/step = 132 tok/sec. This keeps the demo
# fast and focused on the batch=8 result.
BASELINE_MS = 7.6
BASELINE_TPS = 132


# ══════════════════════════════════════════════════════════════
# STEP 3: Warmup + trace capture for batch decode
# ══════════════════════════════════════════════════════════════
print("\nWarming up and capturing trace...")

# Warmup: one non-traced step to JIT-compile all kernels
update_buffers(next_ids, positions)
_ = decode_forward_batch()
ttnn.synchronize_device(device)

# Enable program cache for faster trace capture
try:
    device.enable_program_cache()
except Exception:
    pass

# Capture the batch decode graph into a replayable trace
update_buffers(next_ids, positions)
t_cap0 = time.perf_counter()
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward_batch()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
t_cap = time.perf_counter() - t_cap0
print(f"  Trace captured in {t_cap * 1000:.0f}ms")

# The trace capture step consumed one decode — account for it
logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
for b in range(batch_size):
    next_ids[b] = int(np.argmax(logits[0, 0, b, :]))
    positions[b] += 1
    tokens_per_seq[b].append(next_ids[b])


# ══════════════════════════════════════════════════════════════
# STEP 4: Generate with traced batch decode
# ══════════════════════════════════════════════════════════════
print(f"Generating {max_gen - 2} tokens with traced batch={batch_size} decode...")

trace_times = []
for step in range(max_gen - 2):  # -2 for prefill token + trace capture token
    update_buffers(next_ids, positions)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    trace_times.append(dt)

    logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
    for b in range(batch_size):
        next_ids[b] = int(np.argmax(logits[0, 0, b, :]))
        positions[b] += 1
        tokens_per_seq[b].append(next_ids[b])


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  BATCH DECODE RESULTS — Qwen2.5-0.5B on Blackhole P150")
print("=" * 64)

# Compute sustained throughput (skip first step for warmth)
sustained = trace_times[1:] if len(trace_times) > 1 else trace_times
avg_ms = np.mean(sustained) * 1000
agg_tps = batch_size * 1000 / avg_ms

print(f"""
  Batch=1 baseline (from exp 53e):
    {BASELINE_MS:.1f} ms/step  |  {BASELINE_TPS} tok/sec

  Batch={batch_size} traced decode:
    {avg_ms:.1f} ms/step  |  {agg_tps:.0f} tok/sec aggregate  |  {avg_ms / batch_size:.2f} ms/tok

  Scaling: {agg_tps / BASELINE_TPS:.1f}x throughput for {batch_size}x batch
  Per-step overhead: {avg_ms - BASELINE_MS:+.1f} ms (near-zero = perfect scaling)
""")

# Show generated text
print("  Generated text (first 3 sequences):")
for b in range(min(3, batch_size)):
    text = tokenizer.decode(tokens_per_seq[b])
    display = text[:200] + ("..." if len(text) > 200 else "")
    print(f"    Seq[{b}]: {display}")

# Verify all sequences are identical (greedy from same prompt)
all_same = all(tokens_per_seq[b] == tokens_per_seq[0] for b in range(batch_size))
print(f"\n  All {batch_size} sequences identical (greedy): {all_same}")

if not all_same:
    # Show which sequences diverged (should not happen)
    for b in range(1, batch_size):
        if tokens_per_seq[b] != tokens_per_seq[0]:
            diff_pos = next(i for i, (a, c) in enumerate(zip(tokens_per_seq[0], tokens_per_seq[b])) if a != c)
            print(f"    Seq[{b}] diverged at position {diff_pos}")

print("=" * 64)

ttnn.close_device(device)
print("Done!")
