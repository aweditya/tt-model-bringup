#!/usr/bin/env python3
"""
Experiment 51c: Fully on-device decode — eliminate ALL CPU round-trips.

51b achieved 28ms/tok by moving RoPE on device. But we still have 48
CPU transfers for the residual stream (from_dev/to_dev between layers).

This experiment keeps x_tt on device throughout all 24 layers.
Only CPU touches: embedding lookup (input) and logit readback (output).

Transfers per decode step:
  Exp 49 (CPU RoPE):     ~192 (8/layer × 24)
  Exp 51b (on-dev RoPE): ~48  (2/layer × 24)
  Exp 51c (fully on-dev): 2   (embed in + logits out)

Expected: >1.5x over 51b by eliminating 48 transfers.
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
    cos_tt = rope_cos_tt[pos]
    sin_tt = rope_sin_tt[pos]
    rotated = ttnn.matmul(x_tt, R_tt)
    return ttnn.add(ttnn.mul(x_tt, cos_tt), ttnn.mul(rotated, sin_tt))

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

# ══════════════════════════════════════════════════════════════
# PREFILL (CPU RoPE — runs once)
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
# DECODE: Fully on-device — x stays on device between layers
# ══════════════════════════════════════════════════════════════

def decode_step_fully_ondevice(token_id, pos):
    """Fully on-device decode. Only 2 CPU transfers: embed in + logits out."""
    B = 1
    # Single CPU → device transfer: embedding
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)
    x_tt = to_dev(x_np)

    for i in range(n_layers):
        dl = dev_layers[i]

        # Everything on device — no CPU round-trips!
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        # Reshape to 4D on device
        q_4d = ttnn.reshape(q_tt, [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, n_kv_heads, 1, head_dim])

        # On-device RoPE via rotation matrix
        q_roped = apply_rope_ondevice(q_4d, pos)
        k_roped = apply_rope_ondevice(k_4d, pos)

        # Update KV caches
        ttnn.kv_cache.update_cache_for_token_(k_caches[i], k_roped,
                                               update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_caches[i], v_4d,
                                               update_index=pos, batch_offset=0)

        # Flash-Decode
        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos=[pos], compute_kernel_config=hifi4)

        # Reshape attention → (1, 1, hidden) on device
        merged = ttnn.reshape(attn, [1, 1, 1, hidden])

        # Output projection + residual — all on device
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)  # residual stays on device!

        # MLP — all on device
        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)  # residual stays on device!

    # Final norm + logits — single device → CPU transfer
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    logits = from_dev(logits_tt, (B, 1, vocab_size))
    return logits[0, 0]


# Also: 51b-style decode (on-device RoPE, but CPU residual) for comparison
def decode_step_51b(token_id, pos):
    """51b decode: on-device RoPE but CPU round-trip between layers."""
    B = 1
    x_np = embed_w[token_id:token_id+1].reshape(B, 1, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

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
        x_tt2 = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, 1, hidden))

    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B, 1, vocab_size))[0, 0]


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
tokens_list = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens...\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
t_prefill = time.perf_counter() - t0

next_id = int(np.argmax(logits))
tokens_list.append(next_id)
sys.stdout.write(args.prompt + tokenizer.decode([next_id]))
sys.stdout.flush()
print(f"\n  [prefill: {t_prefill*1000:.0f}ms]")

# Fully on-device decode
print("  [Fully on-device decode]")
decode_times = []
for step in range(max_gen - 1):
    pos = len(tokens_list) - 1
    t0 = time.perf_counter()
    logits = decode_step_fully_ondevice(next_id, pos)
    dt = time.perf_counter() - t0
    decode_times.append(dt)

    next_id = int(np.argmax(logits))
    tokens_list.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()
    if next_id == tokenizer.eos_token_id:
        break

# 51b baseline (5 steps)
print("\n\n  [Running 51b baseline (5 steps)...]")
k_caches, v_caches = alloc_caches()
tokens_base = tokenizer.encode(args.prompt)
logits = prefill(np.array(tokens_base))
next_base = int(np.argmax(logits))
tokens_base.append(next_base)

decode_times_51b = []
for step in range(min(5, max_gen - 1)):
    pos = len(tokens_base) - 1
    t0 = time.perf_counter()
    logits = decode_step_51b(next_base, pos)
    dt = time.perf_counter() - t0
    decode_times_51b.append(dt)
    next_base = int(np.argmax(logits))
    tokens_base.append(next_base)

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if decode_times:
    first = decode_times[0] * 1000
    sustained = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = np.mean(sustained) * 1000
    print(f"\nFully on-device decode (51c):")
    print(f"  First decode:  {first:.0f}ms")
    print(f"  Sustained:     {avg:.1f}ms/tok ({1000/avg:.1f} tok/sec)")
    print(f"  All times:     {[f'{t*1000:.0f}' for t in decode_times]}")

if decode_times_51b:
    first_51b = decode_times_51b[0] * 1000
    sustained_51b = decode_times_51b[1:] if len(decode_times_51b) > 1 else decode_times_51b
    avg_51b = np.mean(sustained_51b) * 1000
    print(f"\n51b baseline (on-dev RoPE, CPU residual):")
    print(f"  First decode:  {first_51b:.0f}ms")
    print(f"  Sustained:     {avg_51b:.1f}ms/tok ({1000/avg_51b:.1f} tok/sec)")

    if decode_times and len(decode_times) > 1:
        speedup = avg_51b / avg
        print(f"\n  Speedup over 51b:  {speedup:.2f}x")

print(f"\n  Transfer comparison:")
print(f"    Exp 49 (CPU RoPE):     ~192/step")
print(f"    Exp 51b (on-dev RoPE): ~48/step")
print(f"    Exp 51c (fully on-dev): 2/step")
print(f"\n  Timeline:")
print(f"    Exp 49:  35ms/tok (29.3 tok/sec)")
print(f"    Exp 51b: 28ms/tok (35.6 tok/sec)")
if decode_times and len(decode_times) > 1:
    print(f"    Exp 51c: {avg:.0f}ms/tok ({1000/avg:.1f} tok/sec)")

ttnn.close_device(device)
print("\nDone!")
