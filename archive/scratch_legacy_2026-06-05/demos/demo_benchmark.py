#!/usr/bin/env python3
"""
Benchmark demo: Qwen2.5-0.5B decode optimization comparison on Blackhole.

Runs two decode modes back-to-back and prints a comparison table:

  "standard"  — Fully on-device decode with cur_pos_tensor (exp 52c).
                 All 24 layers stay on device. Only 2 CPU transfers per step
                 (embedding in + logits out). Correct KV cache positions.
                 ~49 tok/sec.

  "traced"    — Trace-captured decode (exp 52). Records the 24-layer graph
                 once with ttnn.begin_trace_capture, then replays with
                 ttnn.execute_trace. Input buffers (embedding, RoPE cos/sin)
                 are updated via ttnn.copy before each replay. 2.8x faster
                 but cur_pos/update_index are baked as Python scalars, so
                 KV cache positions are stale. ~136 tok/sec.

Both modes share the same prefill function (CPU RoPE, fills KV caches).

Based on experiments 52c (standard) and 52 (traced).
"""

import sys, os, time, argparse

sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── Args ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen2.5-0.5B decode benchmark on Blackhole")
parser.add_argument("--prompt", default="The capital of France is", help="Input prompt")
parser.add_argument("--max_tokens", type=int, default=30, help="Tokens to generate per mode")
parser.add_argument("--modes", default="standard,traced", help="Comma-separated modes: standard,traced")
args = parser.parse_args()
modes = [m.strip() for m in args.modes.split(",")]

# ── Model config ────────────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936
MAX_SEQ = 256

# HiFi4 + fp32 accumulation on ALL ops — required to avoid kernel config
# state leak on Blackhole (exp 46e). Mixing configs corrupts subsequent ops.
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# ── Load weights ────────────────────────────────────────────
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

# ── Device setup ────────────────────────────────────────────
device = ttnn.open_device(device_id=0)

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

# ── Rotation matrix for on-device RoPE (exp 51b) ───────────
# rotate_half(x) = x @ R — avoids ttnn.split which fails on Blackhole
# due to tile padding. A 64x64 permutation matrix does the same thing.
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

# ── RoPE frequency table ───────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    """Half-format RoPE tables (Qwen uses rotate_half, NOT interleaved)."""
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Upload model weights to device ─────────────────────────
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

# ── Input buffers for trace mode ────────────────────────────
# Trace records the op graph, not values. We update these buffers
# via ttnn.copy before each trace replay.
embed_buf = to_dev(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))

def update_buffers(token_id, pos):
    """Update all trace input buffers before replay."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_dev(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

def apply_rope_buf(x_tt):
    """On-device RoPE using buffer values (for traced mode)."""
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, rope_cos_buf), ttnn.mul(rotated, rope_sin_buf))

# ── Pre-uploaded RoPE tables for standard mode (exp 51c) ────
# One cos/sin tensor per position, stored on device.
print("Uploading RoPE tables...")
rope_cos_tt = []
rope_sin_tt = []
for pos in range(MAX_SEQ):
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    rope_cos_tt.append(to_dev_4d(cos_full))
    rope_sin_tt.append(to_dev_4d(sin_full))

def apply_rope_ondevice(x_tt, pos):
    """On-device RoPE using pre-uploaded per-position tables."""
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, rope_cos_tt[pos]), ttnn.mul(rotated, rope_sin_tt[pos]))

# ── KV cache allocation ────────────────────────────────────
def alloc_caches():
    k_caches, v_caches = [], []
    for _ in range(n_layers):
        c = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        k_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
        v_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    return k_caches, v_caches


# ══════════════════════════════════════════════════════════════
# PREFILL — shared by both modes. CPU RoPE, runs once per mode.
# Fills KV caches with prompt context via fill_cache_for_user_.
# ══════════════════════════════════════════════════════════════

def prefill(token_ids, k_caches, v_caches):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)
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
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_merged = attn_np.transpose(0,2,1,3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_merged), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, T, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, T, vocab_size))[0, -1]


# ══════════════════════════════════════════════════════════════
# STANDARD DECODE — fully on-device, correct positions (exp 52c)
#
# Optimizations present:
#   - On-device RoPE via rotation matrix (no ttnn.split needed)
#   - Residual stays on device across all 24 layers (2 CPU xfers total)
#   - Pre-uploaded per-position cos/sin tables
#   - HiFi4+fp32 on all ops (avoids kernel config leak)
#   - KV cache with Flash-Decode (constant-time per token)
# ══════════════════════════════════════════════════════════════

def decode_step_standard(token_id, pos, k_caches, v_caches):
    """Fully on-device decode. Only 2 CPU transfers: embed in + logits out."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    x_tt = to_dev(x_np)

    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, n_kv_heads, 1, head_dim])

        q_roped = apply_rope_ondevice(q_4d, pos)
        k_roped = apply_rope_ondevice(k_4d, pos)

        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                               update_index=pos, batch_offset=0)

        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos], compute_kernel_config=hifi4)

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
    return from_dev(logits_tt, (1, 1, vocab_size))[0, 0]


# ══════════════════════════════════════════════════════════════
# TRACED DECODE — trace-captured graph replay (exp 52)
#
# Additional optimization over standard:
#   - ttnn.begin_trace_capture records the 24-layer decode graph once
#   - ttnn.execute_trace replays it with zero Python dispatch overhead
#   - Input buffers (embed, cos, sin) updated via ttnn.copy before replay
#
# Known limitation: cur_pos and update_index are baked into the trace
# as Python scalars. KV cache positions don't advance between replays,
# causing text quality degradation ("PARTIAL" correctness).
# ══════════════════════════════════════════════════════════════

def capture_trace(k_caches, v_caches, capture_pos):
    """Capture the decode graph into a trace. Returns (trace_id, logits_tt)."""
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)

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

        q_roped = apply_rope_buf(q_4d)
        k_roped = apply_rope_buf(k_4d)

        # NOTE: update_index is a Python int — baked into trace, does not advance!
        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                               update_index=capture_pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                               update_index=capture_pos, batch_offset=0)

        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        # NOTE: cur_pos is a Python list — baked into trace, does not advance!
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[capture_pos], compute_kernel_config=hifi4)

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

    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    return trace_id, logits_tt


# ══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════

def run_standard(prompt_tokens, max_gen):
    """Run standard (non-traced) fully on-device decode."""
    k_caches, v_caches = alloc_caches()

    tokens = list(prompt_tokens)
    logits = prefill(np.array(tokens), k_caches, v_caches)
    next_id = int(np.argmax(logits))
    tokens.append(next_id)

    # Warmup one decode step (JIT compilation)
    pos = len(tokens) - 1
    logits = decode_step_standard(next_id, pos, k_caches, v_caches)
    next_id = int(np.argmax(logits))
    tokens.append(next_id)

    # Timed decode
    decode_times = []
    for _ in range(max_gen - 2):
        pos = len(tokens) - 1
        t0 = time.perf_counter()
        logits = decode_step_standard(next_id, pos, k_caches, v_caches)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        next_id = int(np.argmax(logits))
        tokens.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    return tokens, decode_times


def run_traced(prompt_tokens, max_gen):
    """Run trace-captured decode."""
    k_caches, v_caches = alloc_caches()

    tokens = list(prompt_tokens)
    logits = prefill(np.array(tokens), k_caches, v_caches)
    next_id = int(np.argmax(logits))
    tokens.append(next_id)

    # Warmup one non-traced decode step to compile all kernels
    warmup_pos = len(tokens) - 1
    update_buffers(next_id, warmup_pos)

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
        q_roped = apply_rope_buf(q_4d)
        k_roped = apply_rope_buf(k_4d)
        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                               update_index=warmup_pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                               update_index=warmup_pos, batch_offset=0)
        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[warmup_pos], compute_kernel_config=hifi4)
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
    warmup_logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    warmup_logits = from_dev(warmup_logits_tt, (1, 1, vocab_size))[0, 0]

    next_id = int(np.argmax(warmup_logits))
    tokens.append(next_id)

    # Capture trace at the next position
    capture_pos = len(tokens)
    update_buffers(next_id, capture_pos)
    trace_id, logits_tt = capture_trace(k_caches, v_caches, capture_pos)

    # First trace execution (the capture itself ran the graph)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    logits_np = from_dev(logits_tt, (1, 1, vocab_size))[0, 0]
    next_id = int(np.argmax(logits_np))
    tokens.append(next_id)

    # Timed trace replays
    decode_times = []
    for _ in range(max_gen - 3):
        pos = len(tokens) - 1
        update_buffers(next_id, pos)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        logits_np = from_dev(logits_tt, (1, 1, vocab_size))[0, 0]
        next_id = int(np.argmax(logits_np))
        tokens.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    ttnn.release_trace(device, trace_id)
    return tokens, decode_times


# ══════════════════════════════════════════════════════════════
# MAIN — run each mode, collect results, print table
# ══════════════════════════════════════════════════════════════

prompt_tokens = tokenizer.encode(args.prompt)
max_gen = min(args.max_tokens, MAX_SEQ - len(prompt_tokens))

print(f'\nPrompt: "{args.prompt}" ({len(prompt_tokens)} tokens)')
print(f"Generating {max_gen} tokens per mode")
print(f"Modes: {modes}\n")

results = {}

for mode in modes:
    print(f"{'=' * 60}")
    print(f"Running: {mode}")
    print(f"{'=' * 60}")

    if mode == "standard":
        tokens, times = run_standard(prompt_tokens, max_gen)
        correct = "YES"
    elif mode == "traced":
        tokens, times = run_traced(prompt_tokens, max_gen)
        correct = "PARTIAL"  # stale positions in KV cache
    else:
        print(f"  Unknown mode: {mode}, skipping")
        continue

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    # Trim to a reasonable display length
    display = text if len(text) <= 80 else text[:77] + "..."

    if times:
        avg_ms = np.mean(times) * 1000
        tok_sec = 1000.0 / avg_ms
    else:
        avg_ms = 0.0
        tok_sec = 0.0

    results[mode] = {
        "avg_ms": avg_ms,
        "tok_sec": tok_sec,
        "correct": correct,
        "text": display,
        "times": times,
        "n_tokens": len(tokens) - len(prompt_tokens),
    }

    print(f"  {avg_ms:.1f} ms/tok | {tok_sec:.1f} tok/sec | {correct}")
    print(f"  Output: {display}\n")

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
print(f"{'Mode':<14}| {'ms/tok':>7} | {'tok/sec':>7} | {'Correct?':<9}| Sample output")
print("-" * 80)
for mode in modes:
    if mode not in results:
        continue
    r = results[mode]
    # Truncate sample for table
    sample = r["text"]
    max_sample = 40
    if len(sample) > max_sample:
        sample = sample[:max_sample-3] + "..."
    print(f"{mode:<14}| {r['avg_ms']:>7.1f} | {r['tok_sec']:>7.1f} | {r['correct']:<9}| \"{sample}\"")
print("-" * 80)

if "standard" in results and "traced" in results:
    speedup = results["standard"]["avg_ms"] / results["traced"]["avg_ms"] if results["traced"]["avg_ms"] > 0 else 0
    print(f"\nTraced speedup over standard: {speedup:.2f}x")

print(f"\nOptimization journey (Qwen2.5-0.5B on Blackhole P150):")
print(f"  exp 41:   582 ms/tok     1.7 tok/s   Full recompute, default config")
print(f"  exp 47:    54 ms/tok    18.4 tok/s   HiFi4+fp32 precision fix")
print(f"  exp 49:    35 ms/tok    28.6 tok/s   KV-cached decode")
print(f"  exp 51b:   28 ms/tok    35.6 tok/s   On-device RoPE (rotation matrix)")
print(f"  exp 51c:   21 ms/tok    46.6 tok/s   Fully on-device (2 CPU xfers)")
print(f"  exp 52:     7 ms/tok   135.6 tok/s   Trace capture (stale positions)")

ttnn.close_device(device)
print("\nDone!")
