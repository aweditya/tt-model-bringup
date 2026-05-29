"""
Experiment 33: KV cache for O(1) per-token decode on Blackhole.

Goal: Instead of recomputing attention over the full sequence each token,
cache K/V from previous tokens and only compute the new token's Q.
This makes decode fixed-shape (single token in, single token out),
enabling trace capture regardless of sequence length.

Key TT-NN APIs to test:
  - ttnn.kv_cache.fill_cache_for_user_(cache, input, batch_index)
  - ttnn.kv_cache.update_cache_for_token_(cache, token, update_index, batch_offset)
  - ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, cur_pos=[pos])
  - ttnn.update_cache(cache, input, update_index)

Phases:
  1. Probe which KV cache APIs exist in our TT-NN version
  2. Test basic cache allocation and update
  3. Single-layer prefill + decode with KV cache
  4. Verify correctness against full-recompute reference
  5. Benchmark decode latency
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

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}, head_dim={head_dim}")

# ── Device setup ─────────────────────────────────────────────
import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)

# ══════════════════════════════════════════════════════════════
# Phase 1: Probe KV cache APIs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Probe KV cache APIs")
print("=" * 60)

# Check what's available
for api_path in [
    'ttnn.kv_cache',
    'ttnn.transformer.scaled_dot_product_attention_decode',
    'ttnn.update_cache',
]:
    parts = api_path.split('.')
    obj = ttnn
    found = True
    for part in parts[1:]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            found = False
            break
    print(f"  {api_path}: {'EXISTS' if found else 'NOT FOUND'}")
    if found and callable(obj):
        try:
            sig = str(obj.__doc__[:200] if obj.__doc__ else "no docstring")
            print(f"    doc: {sig}")
        except:
            pass

# Check kv_cache submodule
if hasattr(ttnn, 'kv_cache'):
    print(f"\n  ttnn.kv_cache members:")
    for name in dir(ttnn.kv_cache):
        if not name.startswith('_'):
            print(f"    {name}")

# Check transformer submodule for attention variants
print(f"\n  ttnn.transformer attention-related:")
for name in dir(ttnn.transformer):
    if 'attention' in name.lower() or 'cache' in name.lower():
        print(f"    {name}")

# Check for update_cache at top level
for name in dir(ttnn):
    if 'cache' in name.lower() or 'update' in name.lower():
        print(f"  ttnn.{name}")


# ══════════════════════════════════════════════════════════════
# Phase 2: Test cache allocation and basic operations
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Test cache allocation")
print("=" * 60)

# KV cache shape: (batch, n_heads, max_seq, head_dim) = (1, 12, 1024, 64)
# Try allocating a cache tensor on device
cache_shape = (1, n_heads, max_seq, head_dim)
print(f"  Target cache shape: {cache_shape}")
print(f"  Cache size: {np.prod(cache_shape) * 2 / 1024:.1f} KB (bfloat16)")
print(f"  Total KV cache (12 layers x 2): {12 * 2 * np.prod(cache_shape) * 2 / 1024 / 1024:.1f} MB")

cache_ok = False
try:
    # Allocate cache as zeros on device
    cache_np = np.zeros(cache_shape, dtype=np.float32)
    k_cache = ttnn.from_torch(
        torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )
    v_cache = ttnn.from_torch(
        torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )
    print(f"  K cache allocated: {k_cache.shape}")
    print(f"  V cache allocated: {v_cache.shape}")
    cache_ok = True
except Exception as e:
    print(f"  Cache allocation FAILED: {e}")


# ══════════════════════════════════════════════════════════════
# Phase 3: Test fill_cache_for_user_ (prefill)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Test cache fill (prefill)")
print("=" * 60)

if cache_ok and hasattr(ttnn, 'kv_cache'):
    seq_len = 32
    # Simulate K/V from a prompt: (1, n_heads, seq_len, head_dim)
    k_prompt = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float32) * 0.1
    k_prompt_tt = ttnn.from_torch(
        torch.from_numpy(k_prompt), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )

    fill_ok = False
    try:
        print("  Trying: fill_cache_for_user_(k_cache, k_prompt, batch_index=0)...")
        ttnn.kv_cache.fill_cache_for_user_(k_cache, k_prompt_tt, batch_index=0)
        print("  SUCCESS!")
        fill_ok = True

        # Verify: read back cache and check first seq_len positions match
        cache_back = ttnn.to_torch(k_cache).float().numpy()
        prompt_ref = ttnn.to_torch(k_prompt_tt).float().numpy()
        cos = np.dot(
            cache_back[:, :, :seq_len, :].flatten(),
            prompt_ref.flatten()
        ) / (
            np.linalg.norm(cache_back[:, :, :seq_len, :].flatten()) *
            np.linalg.norm(prompt_ref.flatten()) + 1e-8
        )
        print(f"  Cache[:seq_len] vs prompt cosine: {cos:.6f}")

    except Exception as e:
        print(f"  fill_cache_for_user_ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Try update_cache_for_token_ (decode: add one token)
    if fill_ok:
        print("\n  Testing update_cache_for_token_...")
        # Single new token's K: (1, n_heads, 1, head_dim)
        k_new = np.random.randn(1, n_heads, 1, head_dim).astype(np.float32) * 0.1
        k_new_tt = ttnn.from_torch(
            torch.from_numpy(k_new), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT
        )
        try:
            ttnn.kv_cache.update_cache_for_token_(k_cache, k_new_tt, update_index=seq_len, batch_offset=0)
            print(f"  update_cache_for_token_ at index {seq_len}: SUCCESS!")
        except Exception as e:
            print(f"  update_cache_for_token_ FAILED: {e}")
            import traceback
            traceback.print_exc()

elif cache_ok:
    # No kv_cache module — try manual cache update via ttnn operations
    print("  No ttnn.kv_cache module. Trying manual approach...")
    print("  Manual approach: use ttnn.update_cache or slice assignment")

    if hasattr(ttnn, 'update_cache'):
        seq_len = 32
        k_prompt = np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float32) * 0.1
        k_prompt_tt = ttnn.from_torch(
            torch.from_numpy(k_prompt), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT
        )
        try:
            print("  Trying: ttnn.update_cache(k_cache, k_prompt, update_index=0)...")
            ttnn.update_cache(k_cache, k_prompt_tt, update_index=0)
            print("  SUCCESS!")
        except Exception as e:
            print(f"  FAILED: {e}")


# ══════════════════════════════════════════════════════════════
# Phase 4: Test scaled_dot_product_attention_decode
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Test attention_decode")
print("=" * 60)

has_decode = hasattr(ttnn.transformer, 'scaled_dot_product_attention_decode')
print(f"  scaled_dot_product_attention_decode: {'EXISTS' if has_decode else 'NOT FOUND'}")

if has_decode:
    # Try calling it with a single query token against the cache
    # Q shape for decode: varies by API. Try (1, n_heads, 1, head_dim)
    q_decode = np.random.randn(1, n_heads, 1, head_dim).astype(np.float32) * 0.1
    q_decode_tt = ttnn.from_torch(
        torch.from_numpy(q_decode), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )

    cur_pos = 32  # We have 32 tokens in cache
    print(f"  Q shape: {q_decode_tt.shape}")
    print(f"  K cache shape: {k_cache.shape}")
    print(f"  cur_pos: {cur_pos}")

    # Try different calling conventions
    for attempt_name, attempt_fn in [
        ("cur_pos as list", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode_tt, k_cache, v_cache, cur_pos=[cur_pos])),
        ("cur_pos as tensor", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode_tt, k_cache, v_cache,
            cur_pos_tensor=ttnn.from_torch(
                torch.tensor([cur_pos], dtype=torch.int32),
                device=device))),
        ("cur_pos as kwarg", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode_tt, k_cache, v_cache, cur_pos=cur_pos)),
        ("is_causal + cur_pos", lambda: ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode_tt, k_cache, v_cache, is_causal=True, cur_pos=[cur_pos])),
    ]:
        try:
            print(f"\n  Trying: {attempt_name}...")
            out = attempt_fn()
            print(f"  SUCCESS! Output shape: {out.shape}")
            out_np = ttnn.to_torch(out).float().numpy()
            print(f"  Output torch shape: {out_np.shape}")
            print(f"  Output range: [{out_np.min():.4f}, {out_np.max():.4f}]")
            break
        except Exception as e:
            print(f"  FAILED: {e}")

else:
    print("  Trying regular SDPA with cache tensors instead...")
    # Can we use regular scaled_dot_product_attention with the full cache?
    # Q: (1, n_heads, 1, head_dim), K/V cache: (1, n_heads, max_seq, head_dim)
    q_decode = np.random.randn(1, n_heads, 1, head_dim).astype(np.float32) * 0.1
    q_decode_tt = ttnn.from_torch(
        torch.from_numpy(q_decode), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )
    try:
        out = ttnn.transformer.scaled_dot_product_attention(
            q_decode_tt, k_cache, v_cache, is_causal=True
        )
        print(f"  Regular SDPA with cache: output shape = {out.shape}")
    except Exception as e:
        print(f"  Regular SDPA with cache FAILED: {e}")


# ══════════════════════════════════════════════════════════════
# Phase 5: Single-layer prefill + decode test
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 5: Single-layer KV cache decode")
print("=" * 60)

# Use layer 0 weights
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

# Prepare layer 0 weights (pre-split QKV)
p = "h.0"
w_q = to_dev(weights[f"{p}.attn.c_attn.weight"][:, :d_model])
w_k = to_dev(weights[f"{p}.attn.c_attn.weight"][:, d_model:2*d_model])
w_v = to_dev(weights[f"{p}.attn.c_attn.weight"][:, 2*d_model:])
b_q = to_dev(weights[f"{p}.attn.c_attn.bias"][:d_model])
b_k = to_dev(weights[f"{p}.attn.c_attn.bias"][d_model:2*d_model])
b_v = to_dev(weights[f"{p}.attn.c_attn.bias"][2*d_model:])
ln1_g = to_dev(weights[f"{p}.ln_1.weight"])
ln1_b = to_dev(weights[f"{p}.ln_1.bias"])
w_proj = to_dev(weights[f"{p}.attn.c_proj.weight"])
b_proj = to_dev(weights[f"{p}.attn.c_proj.bias"])

# Create input: 8 tokens
seq_len = 8
pad_len = 32
token_ids = list(range(1000, 1000 + seq_len))
padded_ids = token_ids + [50256] * (pad_len - seq_len)
emb = (wte[padded_ids] + wpe[:pad_len])[None, :, :]  # (1, 32, 768)
x_tt = to_dev(emb)

# --- Full recompute reference (existing approach) ---
h = ttnn.layer_norm(x_tt, weight=ln1_g, bias=ln1_b, epsilon=1e-5)
q_full = ttnn.add(ttnn.matmul(h, w_q), b_q)
k_full = ttnn.add(ttnn.matmul(h, w_k), b_k)
v_full = ttnn.add(ttnn.matmul(h, w_v), b_v)

q_full = ttnn.transpose(ttnn.reshape(q_full, [1, pad_len, n_heads, head_dim]), 1, 2)
k_full = ttnn.transpose(ttnn.reshape(k_full, [1, pad_len, n_heads, head_dim]), 1, 2)
v_full = ttnn.transpose(ttnn.reshape(v_full, [1, pad_len, n_heads, head_dim]), 1, 2)

attn_ref = ttnn.transformer.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
merged_ref = ttnn.transformer.concatenate_heads(attn_ref)
proj_ref = ttnn.add(ttnn.matmul(merged_ref, w_proj), b_proj)
out_ref = ttnn.add(x_tt, proj_ref)
ref_np = from_dev(out_ref, (1, pad_len, d_model))

# Get the reference output for the last real token
ref_last = ref_np[0, seq_len - 1, :]
print(f"  Reference (full recompute) last-token hidden: norm={np.linalg.norm(ref_last):.4f}")

# --- KV cache approach ---
# Step 1: Prefill — compute K/V for all prompt tokens, store in cache
print("\n  KV cache approach:")
print("  Step 1: Prefill (compute all K/V, store in cache)")

# We already computed k_full, v_full above. Just need to fill the cache.
k_np = from_dev(k_full, (1, n_heads, pad_len, head_dim))
v_np = from_dev(v_full, (1, n_heads, pad_len, head_dim))
print(f"  K shape from prefill: {k_np.shape}")
print(f"  V shape from prefill: {v_np.shape}")

# TODO: Once we know which cache API works (from Phase 3/4),
# implement the actual decode step here


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
KV cache APIs found:
  ttnn.kv_cache module:           {'YES' if hasattr(ttnn, 'kv_cache') else 'NO'}
  fill_cache_for_user_:           {'YES' if hasattr(ttnn, 'kv_cache') and hasattr(ttnn.kv_cache, 'fill_cache_for_user_') else 'NO'}
  update_cache_for_token_:        {'YES' if hasattr(ttnn, 'kv_cache') and hasattr(ttnn.kv_cache, 'update_cache_for_token_') else 'NO'}
  attention_decode:               {'YES' if has_decode else 'NO'}
  ttnn.update_cache:              {'YES' if hasattr(ttnn, 'update_cache') else 'NO'}

Cache allocation: {'OK' if cache_ok else 'FAILED'}
Cache shape: {cache_shape}
Total KV memory (12L): {12 * 2 * np.prod(cache_shape) * 2 / 1024 / 1024:.1f} MB
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
