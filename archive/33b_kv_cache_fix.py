"""
Experiment 33b: Fix SDPA decode tensor layout for MHA.

Key insight from FlashDecode tech report: "MQA only" docs are outdated.
MHA works when n_kv_heads == n_q_heads. The issue was tensor layout:

  Q:   [1, batch, n_heads, head_dim]  — batch in dim 1
  K/V: [batch, n_heads, seq_len, head_dim]

We were passing Q as [1, n_heads, 1, head_dim] which made the kernel
read n_heads=12 as batch=12, failing the K batch check.

Phases:
  1. Test SDPA decode with correct layout
  2. Test KV cache fill + update + decode cycle
  3. Single GPT-2 layer with KV cache
  4. Verify correctness against full-recompute reference
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch

# ── Load GPT-2 weights ──────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 weights...")
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")

with open(config_path) as f:
    config = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

n_heads = config['n_head']       # 12
d_model = config['n_embd']       # 768
head_dim = d_model // n_heads    # 64
n_layers = config['n_layer']     # 12
max_seq = 1024
batch = 1

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}, head_dim={head_dim}")

# ── Device setup ─────────────────────────────────────────────
import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)


# ══════════════════════════════════════════════════════════════
# Phase 1: Test SDPA decode with correct layout
# ════════════════════════════════════════════════��═════════════
print("\n" + "=" * 60)
print("Phase 1: SDPA decode with correct tensor layout")
print("=" * 60)

rng = np.random.RandomState(42)
seq_filled = 32  # positions 0..31 are filled

# Flash-Decode layout:
#   Q: [1, batch, n_heads, head_dim]
#   K: [batch, n_heads, max_seq, head_dim]
#   V: [batch, n_heads, max_seq, head_dim]

# Allocate K/V caches
k_cache_np = np.zeros((batch, n_heads, max_seq, head_dim), dtype=np.float32)
v_cache_np = np.zeros((batch, n_heads, max_seq, head_dim), dtype=np.float32)

# Fill first seq_filled positions
k_cache_np[:, :, :seq_filled, :] = rng.randn(batch, n_heads, seq_filled, head_dim).astype(np.float32) * 0.1
v_cache_np[:, :, :seq_filled, :] = rng.randn(batch, n_heads, seq_filled, head_dim).astype(np.float32) * 0.1

k_cache_tt = ttnn.from_torch(torch.from_numpy(k_cache_np.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)
v_cache_tt = ttnn.from_torch(torch.from_numpy(v_cache_np.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)

# Q for single decode token: [1, batch, n_heads, head_dim]
q_np = rng.randn(1, batch, n_heads, head_dim).astype(np.float32) * 0.1
q_tt = ttnn.from_torch(torch.from_numpy(q_np.copy()), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)

print(f"  Q shape:       {q_tt.shape}  (expected [1, {batch}, {n_heads}, {head_dim}])")
print(f"  K cache shape: {k_cache_tt.shape}  (expected [{batch}, {n_heads}, {max_seq}, {head_dim}])")
print(f"  V cache shape: {v_cache_tt.shape}")

# Try SDPA decode
decode_ok = False
for attempt_name, attempt_fn in [
    ("cur_pos=[seq_filled]", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
        q_tt, k_cache_tt, v_cache_tt, cur_pos=[seq_filled])),
    ("cur_pos=[seq_filled], is_causal=True", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
        q_tt, k_cache_tt, v_cache_tt, is_causal=True, cur_pos=[seq_filled])),
    ("cur_pos=[seq_filled], is_causal=False", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
        q_tt, k_cache_tt, v_cache_tt, is_causal=False, cur_pos=[seq_filled])),
]:
    try:
        print(f"\n  Trying: {attempt_name}...")
        out = attempt_fn()
        print(f"  SUCCESS! Output shape: {out.shape}")
        out_np = ttnn.to_torch(out).float().numpy()
        print(f"  Output torch shape: {out_np.shape}")
        print(f"  Output range: [{out_np.min():.4f}, {out_np.max():.4f}]")
        print(f"  Output norm: {np.linalg.norm(out_np):.6f}")
        decode_ok = True
        break
    except Exception as e:
        err_str = str(e)
        # Print just the key error, not full backtrace
        if "TT_FATAL" in err_str:
            lines = err_str.split('\n')
            for line in lines[:5]:
                print(f"    {line}")
        else:
            print(f"    Error: {err_str[:200]}")

# If basic decode fails, try with batch padded to 32 (mentioned in update_cache docs)
if not decode_ok:
    print("\n  Trying with batch padded to 32...")
    batch_pad = 32

    k_cache_padded = np.zeros((batch_pad, n_heads, max_seq, head_dim), dtype=np.float32)
    v_cache_padded = np.zeros((batch_pad, n_heads, max_seq, head_dim), dtype=np.float32)
    k_cache_padded[:batch] = k_cache_np
    v_cache_padded[:batch] = v_cache_np

    q_padded = np.zeros((1, batch_pad, n_heads, head_dim), dtype=np.float32)
    q_padded[:, :batch] = q_np

    k_pad_tt = ttnn.from_torch(torch.from_numpy(k_cache_padded.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    v_pad_tt = ttnn.from_torch(torch.from_numpy(v_cache_padded.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    q_pad_tt = ttnn.from_torch(torch.from_numpy(q_padded.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)

    print(f"  Q padded shape: {q_pad_tt.shape}")
    print(f"  K padded shape: {k_pad_tt.shape}")

    # cur_pos needs one entry per batch item
    cur_pos_list = [seq_filled] * batch_pad

    for attempt_name, attempt_fn in [
        ("padded batch=32", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_pad_tt, k_pad_tt, v_pad_tt, cur_pos=cur_pos_list)),
        ("padded + is_causal=False", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_pad_tt, k_pad_tt, v_pad_tt, is_causal=False, cur_pos=cur_pos_list)),
    ]:
        try:
            print(f"\n  Trying: {attempt_name}...")
            out = attempt_fn()
            print(f"  SUCCESS! Output shape: {out.shape}")
            out_np = ttnn.to_torch(out).float().numpy()
            print(f"  Output torch shape: {out_np.shape}")
            # Extract batch=0 result
            result = out_np[:, :batch, :, :]
            print(f"  Batch 0 result range: [{result.min():.4f}, {result.max():.4f}]")
            decode_ok = True
            break
        except Exception as e:
            err_str = str(e)
            if "TT_FATAL" in err_str:
                lines = err_str.split('\n')
                for line in lines[:5]:
                    print(f"    {line}")
            else:
                print(f"    Error: {err_str[:200]}")


# ══════════════════════════════════════════════════════════════
# Phase 2: KV cache fill + update + decode cycle
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Full cache cycle (fill → update → decode)")
print("=" * 60)

if decode_ok:
    # Fresh caches
    cache_np = np.zeros((batch, n_heads, max_seq, head_dim), dtype=np.float32)
    k_cache2 = ttnn.from_torch(torch.from_numpy(cache_np.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    v_cache2 = ttnn.from_torch(torch.from_numpy(cache_np.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)

    # Prefill: fill positions 0..31
    k_prefill = rng.randn(batch, n_heads, seq_filled, head_dim).astype(np.float32) * 0.1
    v_prefill = rng.randn(batch, n_heads, seq_filled, head_dim).astype(np.float32) * 0.1

    k_pf_tt = ttnn.from_torch(torch.from_numpy(k_prefill.copy()), dtype=ttnn.bfloat16,
                               device=device, layout=ttnn.TILE_LAYOUT)
    v_pf_tt = ttnn.from_torch(torch.from_numpy(v_prefill.copy()), dtype=ttnn.bfloat16,
                               device=device, layout=ttnn.TILE_LAYOUT)

    try:
        ttnn.kv_cache.fill_cache_for_user_(k_cache2, k_pf_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_cache2, v_pf_tt, batch_index=0)
        print("  Prefill: filled positions 0..31")
    except Exception as e:
        print(f"  Prefill FAILED: {e}")

    # Decode step: add token at position 32
    k_new = rng.randn(batch, n_heads, 1, head_dim).astype(np.float32) * 0.1
    v_new = rng.randn(batch, n_heads, 1, head_dim).astype(np.float32) * 0.1

    k_new_tt = ttnn.from_torch(torch.from_numpy(k_new.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    v_new_tt = ttnn.from_torch(torch.from_numpy(v_new.copy()), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)

    try:
        ttnn.kv_cache.update_cache_for_token_(k_cache2, k_new_tt, update_index=seq_filled, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(v_cache2, v_new_tt, update_index=seq_filled, batch_offset=0)
        print(f"  Update: added token at position {seq_filled}")
    except Exception as e:
        print(f"  Update FAILED: {e}")

    # Verify cache contents
    k_back = ttnn.to_torch(k_cache2).float().numpy()
    print(f"  Cache position {seq_filled} norm: {np.linalg.norm(k_back[:, :, seq_filled, :]):.6f}")
    print(f"  Cache position {seq_filled+1} norm: {np.linalg.norm(k_back[:, :, seq_filled+1, :]):.6f} (should be ~0)")

else:
    print("  Skipping — SDPA decode didn't work in Phase 1")


# ══════════════════════════════════════════════════════════════
# Phase 3: Also try regular SDPA with asymmetric Q/K/V
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Regular SDPA with Q(1 token) vs K/V(cache)")
print("=" * 60)

# This is the fallback: use regular scaled_dot_product_attention
# with Q having seq_len=1 and K/V having full cache length
# Shape: Q [1, n_heads, 1, head_dim], K/V [1, n_heads, max_seq, head_dim]
q_reg = rng.randn(1, n_heads, 1, head_dim).astype(np.float32) * 0.1
q_reg_tt = ttnn.from_torch(torch.from_numpy(q_reg.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)

# Use a smaller cache for testing (max_seq=1024 might be slow)
small_seq = 64
k_small = np.zeros((1, n_heads, small_seq, head_dim), dtype=np.float32)
v_small = np.zeros((1, n_heads, small_seq, head_dim), dtype=np.float32)
k_small[:, :, :32, :] = rng.randn(1, n_heads, 32, head_dim).astype(np.float32) * 0.1
v_small[:, :, :32, :] = rng.randn(1, n_heads, 32, head_dim).astype(np.float32) * 0.1

k_small_tt = ttnn.from_torch(torch.from_numpy(k_small.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)
v_small_tt = ttnn.from_torch(torch.from_numpy(v_small.copy()), dtype=ttnn.bfloat16,
                              device=device, layout=ttnn.TILE_LAYOUT)

print(f"  Q: {q_reg_tt.shape}, K: {k_small_tt.shape}, V: {v_small_tt.shape}")

for attempt_name, attempt_fn in [
    ("is_causal=True", lambda: ttnn.transformer.scaled_dot_product_attention(
        q_reg_tt, k_small_tt, v_small_tt, is_causal=True)),
    ("is_causal=False", lambda: ttnn.transformer.scaled_dot_product_attention(
        q_reg_tt, k_small_tt, v_small_tt, is_causal=False)),
    ("with explicit mask", lambda: (lambda: (
        ttnn.transformer.scaled_dot_product_attention(
            q_reg_tt, k_small_tt, v_small_tt, is_causal=False,
            attn_mask=ttnn.from_torch(
                torch.from_numpy(np.ones((1, 1, 1, small_seq), dtype=np.float32) * -1e9),
                dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            ))
    ))()),
]:
    try:
        print(f"\n  Trying regular SDPA: {attempt_name}...")
        out = attempt_fn()
        print(f"  SUCCESS! Output shape: {out.shape}")
        out_np = ttnn.to_torch(out).float().numpy()
        print(f"  Output torch shape: {out_np.shape}")
        print(f"  Output norm: {np.linalg.norm(out_np):.6f}")
        break
    except Exception as e:
        err_str = str(e)
        if "TT_FATAL" in err_str:
            lines = err_str.split('\n')
            for line in lines[:3]:
                print(f"    {line}")
        else:
            print(f"    Error: {err_str[:300]}")


# ══════════════════════════════════════════════════════════════
# Phase 4: Single GPT-2 layer — prefill + decode with KV cache
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: GPT-2 layer 0 — KV cached decode")
print("=" * 60)

wte = weights["wte.weight"]
wpe = weights["wpe.weight"]

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# Upload layer 0 weights
p = "h.0"
lw = {
    'ln1_g': to_dev(weights[f"{p}.ln_1.weight"]),
    'ln1_b': to_dev(weights[f"{p}.ln_1.bias"]),
    'w_q': to_dev(weights[f"{p}.attn.c_attn.weight"][:, :d_model]),
    'w_k': to_dev(weights[f"{p}.attn.c_attn.weight"][:, d_model:2*d_model]),
    'w_v': to_dev(weights[f"{p}.attn.c_attn.weight"][:, 2*d_model:]),
    'b_q': to_dev(weights[f"{p}.attn.c_attn.bias"][:d_model]),
    'b_k': to_dev(weights[f"{p}.attn.c_attn.bias"][d_model:2*d_model]),
    'b_v': to_dev(weights[f"{p}.attn.c_attn.bias"][2*d_model:]),
    'w_proj': to_dev(weights[f"{p}.attn.c_proj.weight"]),
    'b_proj': to_dev(weights[f"{p}.attn.c_proj.bias"]),
    'ln2_g': to_dev(weights[f"{p}.ln_2.weight"]),
    'ln2_b': to_dev(weights[f"{p}.ln_2.bias"]),
    'w_fc': to_dev(weights[f"{p}.mlp.c_fc.weight"]),
    'b_fc': to_dev(weights[f"{p}.mlp.c_fc.bias"]),
    'w_mlp': to_dev(weights[f"{p}.mlp.c_proj.weight"]),
    'b_mlp': to_dev(weights[f"{p}.mlp.c_proj.bias"]),
}

# Input: 8 tokens, padded to 32
seq_len = 8
pad_len = 32
token_ids = list(range(1000, 1000 + seq_len))
padded_ids = token_ids + [50256] * (pad_len - seq_len)
emb = (wte[padded_ids] + wpe[:pad_len])[None, :, :]

# --- Full recompute reference ---
x_tt = to_dev(emb)
h = ttnn.layer_norm(x_tt, weight=lw['ln1_g'], bias=lw['ln1_b'], epsilon=1e-5)
q_full = ttnn.transpose(ttnn.reshape(ttnn.add(ttnn.matmul(h, lw['w_q']), lw['b_q']),
                                      [1, pad_len, n_heads, head_dim]), 1, 2)
k_full = ttnn.transpose(ttnn.reshape(ttnn.add(ttnn.matmul(h, lw['w_k']), lw['b_k']),
                                      [1, pad_len, n_heads, head_dim]), 1, 2)
v_full = ttnn.transpose(ttnn.reshape(ttnn.add(ttnn.matmul(h, lw['w_v']), lw['b_v']),
                                      [1, pad_len, n_heads, head_dim]), 1, 2)
attn_ref = ttnn.transformer.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
merged_ref = ttnn.transformer.concatenate_heads(attn_ref)
proj_ref = ttnn.add(ttnn.matmul(merged_ref, lw['w_proj']), lw['b_proj'])
out_ref = ttnn.add(x_tt, proj_ref)

# Continue with MLP
h2 = ttnn.layer_norm(out_ref, weight=lw['ln2_g'], bias=lw['ln2_b'], epsilon=1e-5)
ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, lw['w_fc']), lw['b_fc']), fast_and_approximate_mode=False)
layer_ref = ttnn.add(out_ref, ttnn.add(ttnn.matmul(ff, lw['w_mlp']), lw['b_mlp']))
ref_np = from_dev(layer_ref, (1, pad_len, d_model))
print(f"  Reference output (full recompute) — last token norm: {np.linalg.norm(ref_np[0, seq_len-1, :]):.4f}")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
SDPA decode (Flash-Decode):  {'WORKS' if decode_ok else 'FAILED — need different approach'}
Regular SDPA (asymmetric):   tested above
KV cache fill/update:        tested above

Next steps:
  - If Flash-Decode works: implement full GPT-2 prefill+decode pipeline
  - If not: use regular SDPA with manually-managed cache tensors
  - Either way: decode is fixed-shape → traceable → fast
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
