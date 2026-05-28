#!/usr/bin/env python3
"""
Experiment 52: Trace-captured Qwen decode.

51c achieved 21.5ms/tok with fully on-device decode.
The remaining overhead is JIT kernel dispatch (each op compiled & dispatched).
Trace capture records the op graph once, then replays it without dispatch overhead.

Challenge: Position changes each step, affecting RoPE cos/sin lookups.
Solution: Use ttnn.copy (or overwrite input tensors) within trace capture.
The trace records the GRAPH, not the VALUES — we update input tensor values
before each replay.

Expected: 15ms/tok or better from eliminated dispatch overhead.
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
parser.add_argument("--tokens", type=int, default=20)
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

# ── Rotation matrix for on-device RoPE ───────────────────────
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

# ── RoPE tables ──────────────────────────────────────────────
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

# RoPE input buffers — overwritten before each trace replay
# These are the "traced inputs" that change per step
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))

def update_rope_for_pos(pos):
    """Update the RoPE cos/sin buffers for a new position."""
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    # Overwrite buffer contents in-place
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

def apply_rope_ondevice(x_tt):
    """On-device RoPE using the global cos/sin buffers."""
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, rope_cos_buf), ttnn.mul(rotated, rope_sin_buf))

# ── Upload model weights ─────────────────────────────────────
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
def alloc_caches():
    k_caches, v_caches = [], []
    for i in range(n_layers):
        c = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        k_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
        v_caches.append(ttnn.from_torch(torch.from_numpy(c.copy()),
                        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    return k_caches, v_caches

k_caches, v_caches = alloc_caches()

# ── Embedding input buffer ───────────────────────────────────
# Pre-allocate the embedding input buffer (overwritten per step)
embed_buf = to_dev(np.zeros((1, 1, hidden), dtype=np.float32))

def update_embed_for_token(token_id):
    """Update embedding buffer for new token."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_dev(x_np), embed_buf)

# ── cur_pos buffer for Flash-Decode ──────────────────────────
# Flash-Decode needs cur_pos as a tensor
cur_pos_val = [0]  # Will be updated

# ══════════════════════════════════════════════════════════════
# PREFILL (same as 51c — CPU RoPE, runs once)
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
# DECODE: Fully on-device (non-traced, for comparison + warmup)
# ══════════════════════════════════════════════════════════════

def decode_step(token_id, pos):
    """Non-traced fully on-device decode (from 51c)."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)
    x_tt = to_dev(x_np)

    # Update RoPE buffers
    update_rope_for_pos(pos)

    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, n_kv_heads, 1, head_dim])

        q_roped = apply_rope_ondevice(q_4d)
        k_roped = apply_rope_ondevice(k_4d)

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
# TRACE CAPTURE: Record decode graph, replay with updated inputs
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Testing trace capture...")

# First, check if trace APIs exist
has_trace = hasattr(ttnn, 'begin_trace_capture')
print(f"  ttnn.begin_trace_capture exists: {has_trace}")

if has_trace:
    # Test basic trace capture
    try:
        # Warmup with a real decode step first (compiles kernels)
        print("  Warming up kernels...")
        tokens_list = tokenizer.encode(args.prompt)
        logits = prefill(np.array(tokens_list))
        next_id = int(np.argmax(logits))
        tokens_list.append(next_id)

        # Do one real decode to warm JIT
        pos = len(tokens_list) - 1
        warmup_logits = decode_step(next_id, pos)
        print(f"  Warmup decode done (pos={pos})")

        # Now try trace capture
        print("  Attempting trace capture...")

        # The key insight: trace capture records the device-side op graph.
        # We need to:
        # 1. Set up input buffers (embed, cos, sin, cur_pos)
        # 2. Begin trace
        # 3. Run the decode graph
        # 4. End trace → get trace ID
        # 5. For each step: update input buffers, execute trace

        # Update inputs for next position
        next_pos = len(tokens_list)
        next_next_id = int(np.argmax(warmup_logits))

        update_embed_for_token(next_next_id)
        update_rope_for_pos(next_pos)

        # Begin trace
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)

        # Run the decode graph (this records, doesn't execute)
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

            q_roped = apply_rope_ondevice(q_4d)
            k_roped = apply_rope_ondevice(k_4d)

            ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                                   update_index=next_pos, batch_offset=0)
            ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                                   update_index=next_pos, batch_offset=0)

            q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q_decode, k_caches[i], v_caches[i],
                cur_pos=[next_pos], compute_kernel_config=hifi4)

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
        print(f"  ✓ Trace captured! ID: {trace_id}")

        # Test replay
        tokens_list.append(next_next_id)

        # Execute trace
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        t_trace = (time.perf_counter() - t0) * 1000
        traced_logits = from_dev(logits_tt, (1, 1, vocab_size))[0, 0]
        traced_next = int(np.argmax(traced_logits))
        print(f"  ✓ First trace execution: {t_trace:.1f}ms")
        print(f"    Next token: '{tokenizer.decode([traced_next])}'")

        # Compare with non-traced decode
        k_caches2, v_caches2 = alloc_caches()
        # Can't easily compare because caches differ... just check trace works

        # Replay several times with updated inputs
        print(f"\n  Running traced decode loop...")
        tokens_list.append(traced_next)
        current_id = traced_next

        traced_times = []
        for step in range(min(15, args.tokens - 3)):
            pos = len(tokens_list) - 1

            # Update inputs
            update_embed_for_token(current_id)
            update_rope_for_pos(pos)

            # Execute trace
            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            dt = time.perf_counter() - t0
            traced_times.append(dt)

            # Read output
            logits_np = from_dev(logits_tt, (1, 1, vocab_size))[0, 0]
            current_id = int(np.argmax(logits_np))
            tokens_list.append(current_id)
            sys.stdout.write(tokenizer.decode([current_id]))
            sys.stdout.flush()

            if current_id == tokenizer.eos_token_id:
                break

        ttnn.release_trace(device, trace_id)

        if traced_times:
            avg_traced = np.mean(traced_times) * 1000
            print(f"\n\n  Traced decode results:")
            print(f"    Avg: {avg_traced:.1f}ms/tok ({1000/avg_traced:.1f} tok/sec)")
            print(f"    All: {[f'{t*1000:.0f}' for t in traced_times]}")

    except Exception as e:
        print(f"  ✗ Trace capture failed: {e}")
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# ALSO: Non-traced baseline for comparison
# ══════════════════════════════════════════════════════════════
print("\n\n  [Non-traced baseline (51c style, 5 steps)...]")
k_caches, v_caches = alloc_caches()
tokens_base = tokenizer.encode(args.prompt)
logits = prefill(np.array(tokens_base))
next_base = int(np.argmax(logits))
tokens_base.append(next_base)

# Warmup
_ = decode_step(next_base, len(tokens_base) - 1)
next_base2 = int(np.argmax(_))
tokens_base.append(next_base2)

decode_times_base = []
for step in range(5):
    pos = len(tokens_base) - 1
    t0 = time.perf_counter()
    logits = decode_step(next_base2, pos)
    dt = time.perf_counter() - t0
    decode_times_base.append(dt)
    next_base2 = int(np.argmax(logits))
    tokens_base.append(next_base2)

avg_base = np.mean(decode_times_base) * 1000

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nNon-traced (51c fully on-device):")
print(f"  Sustained: {avg_base:.1f}ms/tok ({1000/avg_base:.1f} tok/sec)")

if has_trace and 'avg_traced' in dir():
    print(f"\nTraced decode:")
    print(f"  Sustained: {avg_traced:.1f}ms/tok ({1000/avg_traced:.1f} tok/sec)")
    speedup = avg_base / avg_traced
    print(f"  Speedup:   {speedup:.2f}x over non-traced")

print(f"\n  Timeline:")
print(f"    Exp 49:  35ms/tok (29.3 tok/sec)")
print(f"    Exp 51b: 28ms/tok (35.6 tok/sec)")
print(f"    Exp 51c: 21ms/tok (46.6 tok/sec)")
if has_trace and 'avg_traced' in dir():
    print(f"    Exp 52:  {avg_traced:.0f}ms/tok ({1000/avg_traced:.1f} tok/sec)")

ttnn.close_device(device)
print("\nDone!")
