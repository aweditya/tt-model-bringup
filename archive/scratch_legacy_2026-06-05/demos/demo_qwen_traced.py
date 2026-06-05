#!/usr/bin/env python3
"""
Demo: Traced Decode for Qwen2.5-0.5B on Tenstorrent Blackhole
==============================================================

This demonstrates ttnn trace capture for autoregressive decode, achieving
~136 tok/sec (~7ms/tok) on a single Blackhole device.

How trace capture works:
  1. Run one decode step normally to warm up (JIT-compile) all kernels.
  2. Call ttnn.begin_trace_capture() to start recording the op graph.
  3. Execute the full decode graph (24 transformer layers). The ops are
     RECORDED but not truly executed.
  4. Call ttnn.end_trace_capture() to finalize the trace.
  5. For each subsequent token: update input buffers via ttnn.copy(),
     then call ttnn.execute_trace() to replay the captured graph.

The trace records the GRAPH, not the VALUES. Device tensor contents
(embedding, RoPE cos/sin) are updated between replays. This eliminates
all Python dispatch overhead (~14ms savings per step).

KNOWN LIMITATION — STALE POSITIONS:
  Two parameters are Python scalars that get BAKED into the trace:
    - update_index in ttnn.kv_cache.update_cache_for_token_() (int)
    - cur_pos in ttnn.transformer.scaled_dot_product_attention_decode() (list[int])

  Because these are frozen at capture time, every trace replay writes
  to the SAME KV cache slot and attends to the SAME position window.
  The RoPE embeddings DO update correctly (they are device tensors),
  and the token embedding updates correctly, but the KV cache does not
  advance. This means:
    - The first few tokens after prefill are often reasonable (cached
      context carries them).
    - Quality degrades rapidly as the KV cache becomes stale.
    - The output is NOT equivalent to correct autoregressive generation.

  This is a SPEED proof-of-concept. The path to correct traced decode
  requires HEIGHT_SHARDED tensors + paged_update_cache (see wiki/37).

Based on: experiments/52_traced_decode.py
"""

import sys, os, time, argparse

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn


# ── CLI ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Traced decode demo for Qwen2.5-0.5B on Tenstorrent Blackhole")
parser.add_argument("--prompt", default="The capital of France is",
                    help="Input prompt (default: 'The capital of France is')")
parser.add_argument("--max_tokens", type=int, default=30,
                    help="Maximum tokens to generate (default: 30)")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature, 0 for greedy (default: 0.7)")
parser.add_argument("--top_k", type=int, default=50,
                    help="Top-k sampling (default: 50)")
parser.add_argument("--device_id", type=int, default=0,
                    help="Tenstorrent device ID (default: 0)")
args = parser.parse_args()


# ── Model config (Qwen2.5-0.5B) ────────────────────────────
HIDDEN    = 896
N_Q_HEADS = 14
N_KV_HEADS = 2
HEAD_DIM  = 64
HALF_DIM  = HEAD_DIM // 2
RMS_EPS   = 1e-6
ROPE_THETA = 1000000.0
N_LAYERS  = 24
VOCAB_SIZE = 151936
MAX_SEQ   = 256

HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


# ── Load weights ────────────────────────────────────────────
print("Loading Qwen2.5-0.5B weights...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = (all_weights["lm_head.weight"].T
             if "lm_head.weight" in all_weights
             else embed_w.T.copy())

layer_weights_np = []
for i in range(N_LAYERS):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


# ── Device setup ────────────────────────────────────────────
device = ttnn.open_device(device_id=args.device_id)


def to_dev(arr):
    """Upload numpy array to device as bfloat16 tiled tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device,
                           layout=ttnn.TILE_LAYOUT)


def to_dev_4d(arr):
    """Upload 4D numpy array to device (no squeezing)."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)


def from_dev(tensor, shape):
    """Download device tensor to numpy with target shape."""
    t = ttnn.to_torch(tensor).float()
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)


# ── RoPE setup ──────────────────────────────────────────────
# Rotation matrix for on-device half-format RoPE
R = np.zeros((HEAD_DIM, HEAD_DIM), dtype=np.float32)
for i in range(HALF_DIM):
    R[i + HALF_DIM, i] = -1.0
    R[i, i + HALF_DIM] = 1.0
R_tt = to_dev(R)

freqs = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))


def rotate_half_np(x):
    return np.concatenate([-x[..., HALF_DIM:], x[..., :HALF_DIM]], axis=-1)


def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos_full, sin_full


def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


# Pre-allocated RoPE buffers for trace (updated via ttnn.copy before replay)
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, HEAD_DIM), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, HEAD_DIM), dtype=np.float32))


def update_rope_for_pos(pos):
    """Overwrite RoPE cos/sin buffers for a new position."""
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)])
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)])
    cos_full = cos_full.reshape(1, 1, 1, HEAD_DIM).astype(np.float32)
    sin_full = sin_full.reshape(1, 1, 1, HEAD_DIM).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)


def apply_rope_ondevice(x_tt):
    """On-device RoPE using the global cos/sin buffers."""
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, rope_cos_buf),
                    ttnn.mul(rotated, rope_sin_buf))


# ── Upload model weights ────────────────────────────────────
print("Uploading weights to device...")
t_upload_start = time.perf_counter()

dev_layers = []
for i in range(N_LAYERS):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g":   to_dev(lw["input_layernorm.weight"]),
        "q_w":     to_dev(lw["self_attn.q_proj.weight"].T),
        "q_b":     to_dev(lw["self_attn.q_proj.bias"]),
        "k_w":     to_dev(lw["self_attn.k_proj.weight"].T),
        "k_b":     to_dev(lw["self_attn.k_proj.bias"]),
        "v_w":     to_dev(lw["self_attn.v_proj.weight"].T),
        "v_b":     to_dev(lw["self_attn.v_proj.bias"]),
        "o_w":     to_dev(lw["self_attn.o_proj.weight"].T),
        "ln2_g":   to_dev(lw["post_attention_layernorm.weight"]),
        "gate_w":  to_dev(lw["mlp.gate_proj.weight"].T),
        "up_w":    to_dev(lw["mlp.up_proj.weight"].T),
        "down_w":  to_dev(lw["mlp.down_proj.weight"].T),
    })
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights_np

t_upload = time.perf_counter() - t_upload_start
print(f"  Uploaded in {t_upload*1000:.0f}ms")


# ── KV caches ───────────────────────────────────────────────
k_caches, v_caches = [], []
for i in range(N_LAYERS):
    zeros = np.zeros((1, N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
    k_caches.append(ttnn.from_torch(
        torch.from_numpy(zeros.copy()),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(
        torch.from_numpy(zeros.copy()),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))


# ── Embedding input buffer (overwritten before each replay) ─
embed_buf = to_dev(np.zeros((1, 1, HIDDEN), dtype=np.float32))


def update_embed_for_token(token_id):
    """Overwrite the embedding buffer with a new token's embedding."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, HIDDEN)
    ttnn.copy(to_dev(x_np), embed_buf)


# ── Sampling ────────────────────────────────────────────────
def sample_token(logits_1d, temperature, top_k):
    """Temperature + top-k sampling. Greedy if temperature == 0."""
    if temperature == 0:
        return int(np.argmax(logits_1d))
    logits_1d = logits_1d / temperature
    if top_k > 0:
        top_indices = np.argpartition(logits_1d, -top_k)[-top_k:]
        mask = np.full_like(logits_1d, -np.inf)
        mask[top_indices] = logits_1d[top_indices]
        logits_1d = mask
    # Stable softmax
    logits_1d = logits_1d - np.max(logits_1d)
    probs = np.exp(logits_1d)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


# ══════════════════════════════════════════════════════════════
# PREFILL — Non-traced, runs once
# ══════════════════════════════════════════════════════════════
# Prefill processes the full prompt with causal attention.
# CPU-side RoPE is fine here since prefill runs only once.

def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, HIDDEN)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(N_LAYERS):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=RMS_EPS)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=HIFI4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=HIFI4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=HIFI4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, N_Q_HEADS * HEAD_DIM))
        k_np = from_dev(k_tt, (B, T, N_KV_HEADS * HEAD_DIM))
        v_np = from_dev(v_tt, (B, T, N_KV_HEADS * HEAD_DIM))

        q_4d = apply_rope_half_np(
            q_np.reshape(B, T, N_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3), cos_t, sin_t)
        k_4d = apply_rope_half_np(
            k_np.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        k_4d_tt = to_dev_4d(k_4d)
        v_4d_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), k_4d_tt, v_4d_tt,
            is_causal=True, compute_kernel_config=HIFI4)
        attn_np = from_dev(attn_out_tt, (B, N_Q_HEADS, T, HEAD_DIM))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, HIDDEN)

        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=HIFI4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=RMS_EPS)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=HIFI4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=HIFI4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=HIFI4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, T, HIDDEN))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=RMS_EPS)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=HIFI4)
    return from_dev(logits_tt, (B, T, VOCAB_SIZE))[0, -1]


# ══════════════════════════════════════════════════════════════
# NON-TRACED DECODE — Used for warmup (JIT-compiles all kernels)
# ══════════════════════════════════════════════════════════════

def decode_step_warmup(token_id, pos):
    """Non-traced fully on-device decode. Used once to warm up kernels
    before trace capture. Same graph as the traced version."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, HIDDEN)
    x_tt = to_dev(x_np)
    update_rope_for_pos(pos)

    for i in range(N_LAYERS):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=RMS_EPS)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=HIFI4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=HIFI4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=HIFI4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, N_Q_HEADS, 1, HEAD_DIM])
        k_4d = ttnn.reshape(k_tt, [1, N_KV_HEADS, 1, HEAD_DIM])
        v_4d = ttnn.reshape(v_tt, [1, N_KV_HEADS, 1, HEAD_DIM])

        q_roped = apply_rope_ondevice(q_4d)
        k_roped = apply_rope_ondevice(k_4d)

        ttnn.kv_cache.update_cache_for_token_(
            k_caches[i], k_roped, update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(
            v_caches[i], v_4d, update_index=pos, batch_offset=0)

        q_decode = ttnn.reshape(q_roped, [1, 1, N_Q_HEADS, HEAD_DIM])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos], compute_kernel_config=HIFI4)

        merged = ttnn.reshape(attn, [1, 1, 1, HIDDEN])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=RMS_EPS)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=HIFI4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=HIFI4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=RMS_EPS)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=HIFI4)
    return from_dev(logits_tt, (B, 1, VOCAB_SIZE))[0, 0]


# ══════════════════════════════════════════════════════════════
# MAIN — Prefill, capture trace, generate
# ══════════════════════════════════════════════════════════════

def main():
    tokens = tokenizer.encode(args.prompt)
    max_gen = min(args.max_tokens, MAX_SEQ - len(tokens))

    print(f'\nPrompt: "{args.prompt}" ({len(tokens)} tokens)')
    print(f"Max tokens: {max_gen}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print()

    # ── Step 1: Prefill ─────────────────────────────────────
    print("Phase 1: Prefill...")
    t0 = time.perf_counter()
    logits = prefill(np.array(tokens))
    t_prefill = time.perf_counter() - t0

    first_id = sample_token(logits, args.temperature, args.top_k)
    tokens.append(first_id)
    print(f"  Prefill: {t_prefill*1000:.0f}ms")
    print(f"  First token: '{tokenizer.decode([first_id])}'")

    # ── Step 2: Warmup decode (JIT-compiles kernels) ────────
    # The trace captures compiled kernels, so we must run one real
    # decode step first to ensure everything is compiled.
    print("\nPhase 2: Warmup (JIT compile)...")
    t0 = time.perf_counter()
    warmup_pos = len(tokens) - 1
    warmup_logits = decode_step_warmup(first_id, warmup_pos)
    t_warmup = time.perf_counter() - t0

    warmup_id = sample_token(warmup_logits, args.temperature, args.top_k)
    tokens.append(warmup_id)
    print(f"  Warmup decode: {t_warmup*1000:.0f}ms")
    print(f"  Token: '{tokenizer.decode([warmup_id])}'")

    # ── Step 3: Capture the decode trace ────────────────────
    # Set up input buffers for the position where trace capture runs.
    # The trace_pos is baked into update_index and cur_pos -- this is
    # the fundamental limitation.
    print("\nPhase 3: Trace capture...")
    trace_pos = len(tokens)
    next_id_for_capture = warmup_id

    update_embed_for_token(next_id_for_capture)
    update_rope_for_pos(trace_pos)

    t0 = time.perf_counter()
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)

    # ---- Traced decode graph (24 layers) ----
    # This is recorded, not executed. The graph is identical to
    # decode_step_warmup but reads from the pre-allocated buffers
    # (embed_buf, rope_cos_buf, rope_sin_buf) instead of fresh tensors.
    x_tt = embed_buf

    for i in range(N_LAYERS):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=RMS_EPS)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=HIFI4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=HIFI4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=HIFI4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, N_Q_HEADS, 1, HEAD_DIM])
        k_4d = ttnn.reshape(k_tt, [1, N_KV_HEADS, 1, HEAD_DIM])
        v_4d = ttnn.reshape(v_tt, [1, N_KV_HEADS, 1, HEAD_DIM])

        q_roped = apply_rope_ondevice(q_4d)
        k_roped = apply_rope_ondevice(k_4d)

        # NOTE: update_index=trace_pos is BAKED INTO the trace.
        # Every replay writes K/V to this same cache slot.
        ttnn.kv_cache.update_cache_for_token_(
            k_caches[i], k_roped, update_index=trace_pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(
            v_caches[i], v_4d, update_index=trace_pos, batch_offset=0)

        q_decode = ttnn.reshape(q_roped, [1, 1, N_Q_HEADS, HEAD_DIM])
        # NOTE: cur_pos=[trace_pos] is BAKED INTO the trace.
        # Every replay attends to the same position window.
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[trace_pos], compute_kernel_config=HIFI4)

        merged = ttnn.reshape(attn, [1, 1, 1, HIDDEN])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=RMS_EPS)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=HIFI4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=HIFI4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=RMS_EPS)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=HIFI4)
    # ---- End traced decode graph ----

    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    t_capture = time.perf_counter() - t0
    print(f"  Trace captured in {t_capture*1000:.0f}ms (trace_id={trace_id})")

    # ── Step 4: Traced decode loop ──────────────────────────
    # For each token: update input buffers, replay trace, read logits.
    print(f"\nPhase 4: Traced decode ({max_gen - 2} tokens)...")
    print(f"  NOTE: KV cache position is frozen at {trace_pos}.")
    print(f"        Output quality will degrade. This demonstrates speed only.\n")

    # Print prompt + tokens generated so far
    generated_so_far = tokenizer.decode(tokens)
    sys.stdout.write(generated_so_far)
    sys.stdout.flush()

    current_id = warmup_id
    decode_times = []

    for step in range(max_gen - 2):
        # Update input buffers (these are NOT part of the trace)
        pos = len(tokens)
        update_embed_for_token(current_id)
        update_rope_for_pos(pos)

        # Replay the captured trace
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        # Read logits from the output tensor (same reference from capture)
        logits_np = from_dev(logits_tt, (1, 1, VOCAB_SIZE))[0, 0]
        current_id = sample_token(logits_np, args.temperature, args.top_k)
        tokens.append(current_id)

        sys.stdout.write(tokenizer.decode([current_id]))
        sys.stdout.flush()

        if current_id == tokenizer.eos_token_id:
            break

    # Clean up trace
    ttnn.release_trace(device, trace_id)

    # ── Results ─────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)

    print(f"\n  Prefill:         {t_prefill*1000:.0f}ms ({len(tokenizer.encode(args.prompt))} tokens)")
    print(f"  Warmup decode:   {t_warmup*1000:.0f}ms (JIT compile, not representative)")
    print(f"  Trace capture:   {t_capture*1000:.0f}ms")

    if decode_times:
        times_ms = [t * 1000 for t in decode_times]
        avg_ms = np.mean(times_ms)
        min_ms = np.min(times_ms)
        max_ms = np.max(times_ms)
        tok_per_sec = 1000.0 / avg_ms

        print(f"\n  Traced decode ({len(decode_times)} tokens):")
        print(f"    Average:   {avg_ms:.1f}ms/tok ({tok_per_sec:.1f} tok/sec)")
        print(f"    Min:       {min_ms:.1f}ms")
        print(f"    Max:       {max_ms:.1f}ms")
        print(f"    Per-token: {[f'{t:.0f}' for t in times_ms]}")

        total_gen_time = sum(decode_times) + t_warmup
        total_tokens = len(decode_times) + 1  # +1 for warmup token
        print(f"\n  End-to-end generation:")
        print(f"    {total_tokens} tokens in {total_gen_time*1000:.0f}ms")
        print(f"    ({total_tokens / total_gen_time:.1f} tok/sec including warmup)")

    print(f"\n  LIMITATION: KV cache position frozen at {trace_pos}.")
    print(f"  This is a speed demo, not correct autoregressive generation.")
    print(f"  See wiki/37_trace_capture_deep_dive.md for the path forward.")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
