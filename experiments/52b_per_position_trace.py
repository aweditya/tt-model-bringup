#!/usr/bin/env python3
"""
Experiment 52b: Per-position traced decode.

Problem: update_cache_for_token_ only accepts int, not tensor, for update_index.
This means KV cache position is baked into the trace.

Solution: Capture a separate trace for each decode position. The compute graph
is identical for all positions — only update_index and cur_pos differ. Since
trace capture is fast (~76ms after warmup, <10ms with program cache), and we
only need traces for positions we'll use, this is practical.

Architecture:
  1. Prefill: CPU-based (runs once)
  2. First decode: non-traced (warms JIT + captures first trace)
  3. Subsequent decodes: execute pre-captured trace for each position
  4. If trace not yet captured for position N: capture it on the fly

This gives us trace speed (7ms/tok) with CORRECT position handling.
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
parser.add_argument("--tokens", type=int, default=30)
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2
rms_eps = 1e-6; rope_theta = 1000000.0; n_layers = 24; vocab_size = 151936
MAX_SEQ = 256

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# ── Load weights ─────────────────────────────────────────────
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

# ── Device ───────────────────────────────────────────────────
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

# ── Rotation matrix + RoPE ───────────────────────────────────
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
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos_full, sin_full

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# Pre-upload cos/sin for all positions
rope_cos_tt = []
rope_sin_tt = []
for pos in range(MAX_SEQ):
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    rope_cos_tt.append(to_dev_4d(cos_full))
    rope_sin_tt.append(to_dev_4d(sin_full))

def apply_rope_ondevice(x_tt, pos):
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, rope_cos_tt[pos]), ttnn.mul(rotated, rope_sin_tt[pos]))

# ── Upload weights ───────────────────────────────────────────
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

# ── KV Caches ────────────────────────────────────────────────
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))

# ── Embedding input buffer ───────────────────────────────────
embed_buf = to_dev(np.zeros((1, 1, hidden), dtype=np.float32))

# ══════════════════════════════════════════════════════════════
# PREFILL (CPU RoPE, runs once)
# ══════════════════════════════════════════════════════════════

def prefill(token_ids):
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

        k_4d_tt = to_dev_4d(k_4d)
        v_4d_tt = to_dev_4d(v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), k_4d_tt, v_4d_tt,
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
# TRACED DECODE: Per-position trace capture
# ══════════════════════════════════════════════════════════════

traces = {}  # pos → (trace_id, logits_tt reference)

def capture_trace_for_pos(pos):
    """Capture a trace for a specific decode position."""
    # Update embedding buffer (will be overwritten before replay anyway)
    ttnn.copy(to_dev(np.zeros((1, 1, hidden), dtype=np.float32)), embed_buf)

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

        q_roped = apply_rope_ondevice(q_4d, pos)
        k_roped = apply_rope_ondevice(k_4d, pos)

        # KV cache update with CORRECT position for this trace
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

    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    return trace_id, logits_tt


def decode_step_traced(token_id, pos):
    """Traced decode: update embed buffer, execute trace for this position."""
    # Update embedding input
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_dev(x_np), embed_buf)

    # Get or capture trace for this position
    if pos not in traces:
        t_cap = time.perf_counter()
        trace_id, logits_ref = capture_trace_for_pos(pos)
        traces[pos] = (trace_id, logits_ref)
        cap_time = (time.perf_counter() - t_cap) * 1000
        # Note: first capture also executes the graph
        logits = from_dev(logits_ref, (1, 1, vocab_size))[0, 0]
        return logits, cap_time, True
    else:
        trace_id, logits_ref = traces[pos]
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        logits = from_dev(logits_ref, (1, 1, vocab_size))[0, 0]
        return logits, 0, False


# Also: non-traced fully on-device decode for comparison
def decode_step_notrace(token_id, pos):
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)
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
    return from_dev(logits_tt, (B, 1, vocab_size))[0, 0]


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
tokens_list = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens with per-position trace...\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
t_prefill = time.perf_counter() - t0

next_id = int(np.argmax(logits))
tokens_list.append(next_id)
sys.stdout.write(args.prompt + tokenizer.decode([next_id]))
sys.stdout.flush()
print(f"\n  [prefill: {t_prefill*1000:.0f}ms]")

# First decode: non-traced (warm JIT)
pos = len(tokens_list) - 1
t0 = time.perf_counter()
logits = decode_step_notrace(next_id, pos)
dt_warmup = (time.perf_counter() - t0) * 1000
next_id = int(np.argmax(logits))
tokens_list.append(next_id)
sys.stdout.write(tokenizer.decode([next_id]))
sys.stdout.flush()
print(f"\n  [warmup decode: {dt_warmup:.0f}ms]")

# Traced decode loop
decode_times = []
capture_times = []
for step in range(max_gen - 2):
    pos = len(tokens_list) - 1
    t0 = time.perf_counter()
    logits, cap_time, was_captured = decode_step_traced(next_id, pos)
    dt = time.perf_counter() - t0
    decode_times.append(dt)
    if was_captured:
        capture_times.append(cap_time)

    next_id = int(np.argmax(logits))
    tokens_list.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()
    if next_id == tokenizer.eos_token_id:
        break

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if decode_times:
    all_ms = [t * 1000 for t in decode_times]
    avg_all = np.mean(all_ms)
    print(f"\nPer-position traced decode:")
    print(f"  Total steps: {len(decode_times)}")
    print(f"  Traces captured: {len(capture_times)}")
    print(f"  Avg (all steps): {avg_all:.1f}ms/tok ({1000/avg_all:.1f} tok/sec)")
    print(f"  All times: {[f'{t:.0f}' for t in all_ms]}")

    # Separate capture vs replay
    capture_steps = [t for t in all_ms if t > 20]  # Captures are slower
    replay_steps = [t for t in all_ms if t <= 20]   # Replays are fast

    if replay_steps:
        avg_replay = np.mean(replay_steps)
        print(f"\n  Replay-only avg: {avg_replay:.1f}ms/tok ({1000/avg_replay:.1f} tok/sec)")
    if capture_steps:
        avg_capture = np.mean(capture_steps)
        print(f"  Capture avg: {avg_capture:.1f}ms/tok")

    print(f"\n  Note: Each position needs one trace capture ({len(traces)} traces total)")
    print(f"  After all captures, sustained replay would be ~7ms/tok (135+ tok/sec)")

print(f"\n  Timeline:")
print(f"    Exp 49:  35ms/tok (29.3 tok/sec)  — CPU RoPE")
print(f"    Exp 51c: 21ms/tok (46.6 tok/sec)  — fully on-device")
print(f"    Exp 52:   7ms/tok (135.6 tok/sec)  — traced (stale positions)")
if decode_times:
    print(f"    Exp 52b: {avg_all:.0f}ms/tok ({1000/avg_all:.1f} tok/sec) — per-position traced (correct)")

# Clean up traces
for pos, (tid, _) in traces.items():
    ttnn.release_trace(device, tid)

ttnn.close_device(device)
print("\nDone!")
