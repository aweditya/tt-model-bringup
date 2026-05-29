"""
Experiment 34: Regular SDPA with KV cache for MHA decode.

The scaled_dot_product_attention_decode API appears MQA-only, which blocks
GPT-2 (12-head MHA). Can we use the REGULAR ttnn.transformer.scaled_dot_product_attention
with asymmetric Q/K/V lengths instead?

Test plan:
  1. Allocate K/V caches (1, 12, 1024, 64)
  2. Fill first 32 positions with random data
  3. Try regular SDPA with Q=(1,12,1,64) against the cache
  4. Try with explicit attention mask if is_causal doesn't work
  5. Probe paged_scaled_dot_product_attention_decode and
     chunked_scaled_dot_product_attention for MHA support

Key question: does is_causal=True produce the correct mask when
Q has seq_len=1 and K/V have seq_len=1024? The new token at position N
should attend to positions 0..N only.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import time

import ttnn

device = ttnn.open_device(device_id=0)

# ── Constants ───────────────────────────────────────────────
n_heads = 12
head_dim = 64
max_seq = 1024
batch = 1
prefill_len = 32  # simulate 32 tokens already in cache

print(f"Config: batch={batch}, n_heads={n_heads}, head_dim={head_dim}, max_seq={max_seq}")
print(f"Prefill length: {prefill_len}")


# ══════════════════════════════════════════════════════════════
# Phase 1: Probe all attention variants for MHA support
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Probe attention API variants")
print("=" * 60)

api_names = [
    'scaled_dot_product_attention',
    'scaled_dot_product_attention_decode',
    'paged_scaled_dot_product_attention_decode',
    'chunked_scaled_dot_product_attention',
]

for name in api_names:
    exists = hasattr(ttnn.transformer, name)
    print(f"  ttnn.transformer.{name}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        fn = getattr(ttnn.transformer, name)
        doc = fn.__doc__
        if doc:
            # Print first 400 chars of docstring
            print(f"    doc: {doc[:400]}")
        print()

# Also check for anything else attention-related
print("  All attention-related in ttnn.transformer:")
for name in sorted(dir(ttnn.transformer)):
    if 'attention' in name.lower() or 'sdpa' in name.lower():
        print(f"    {name}")


# ══════════════════════════════════════════════════════════════
# Phase 2: Allocate KV caches and fill with data
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Allocate KV caches")
print("=" * 60)

cache_shape = (batch, n_heads, max_seq, head_dim)
print(f"  Cache shape: {cache_shape}")

# Create cache with first prefill_len positions filled, rest zeros
np.random.seed(42)
k_cache_np = np.zeros(cache_shape, dtype=np.float32)
v_cache_np = np.zeros(cache_shape, dtype=np.float32)
k_cache_np[:, :, :prefill_len, :] = np.random.randn(batch, n_heads, prefill_len, head_dim).astype(np.float32) * 0.1
v_cache_np[:, :, :prefill_len, :] = np.random.randn(batch, n_heads, prefill_len, head_dim).astype(np.float32) * 0.1

k_cache_tt = ttnn.from_torch(
    torch.from_numpy(k_cache_np), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)
v_cache_tt = ttnn.from_torch(
    torch.from_numpy(v_cache_np), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)
print(f"  K cache on device: {k_cache_tt.shape}")
print(f"  V cache on device: {v_cache_tt.shape}")

# Query for the new token at position prefill_len
q_np = np.random.randn(batch, n_heads, 1, head_dim).astype(np.float32) * 0.1
q_tt = ttnn.from_torch(
    torch.from_numpy(q_np), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)
print(f"  Q (new token) on device: {q_tt.shape}")


# ══════════════════════════════════════════════════════════════
# Phase 3: Try regular SDPA with asymmetric Q/K/V
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Regular SDPA with Q=(1,12,1,64), K/V=(1,12,1024,64)")
print("=" * 60)

# Attempt 3a: is_causal=True
print("\n  Attempt 3a: is_causal=True")
try:
    out = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_cache_tt, v_cache_tt, is_causal=True
    )
    ttnn.synchronize_device(device)
    out_np = ttnn.to_torch(out).float().numpy()
    print(f"  SUCCESS! Output shape: {out.shape}")
    print(f"  Output torch shape: {out_np.shape}")
    print(f"  Output range: [{out_np.min():.6f}, {out_np.max():.6f}]")
    print(f"  Output norm: {np.linalg.norm(out_np):.6f}")

    # Verify: compute reference in numpy
    scale = 1.0 / np.sqrt(head_dim)
    # Q @ K^T: (1,12,1,64) @ (1,12,64,1024) -> (1,12,1,1024)
    scores = np.matmul(q_np, k_cache_np.transpose(0, 1, 3, 2)) * scale

    # With is_causal, for query position prefill_len (0-indexed),
    # it should attend to positions 0..prefill_len.
    # But is_causal typically means: position i in Q attends to positions <= i in K.
    # If Q has seq_len=1, it's position 0, so it attends only to K position 0!
    # This is likely WRONG for decode. Let's check.

    # Causal mask interpretation 1: Q position 0 -> attend to K position 0 only
    mask_causal_strict = np.zeros((1, 1024), dtype=np.float32)
    mask_causal_strict[0, 0] = 1.0  # only position 0

    # Causal mask interpretation 2: attend to all filled positions
    mask_all_filled = np.zeros((1, 1024), dtype=np.float32)
    mask_all_filled[0, :prefill_len] = 1.0

    # Check which interpretation the output matches
    for interp_name, mask in [
        ("strict causal (pos 0 only)", mask_causal_strict),
        ("all filled (pos 0..31)", mask_all_filled),
    ]:
        masked_scores = scores * mask[None, None, :, :] + (-1e10) * (1.0 - mask[None, None, :, :])
        from scipy.special import softmax as sp_softmax
        attn_w = sp_softmax(masked_scores, axis=-1)
        ref_out = np.matmul(attn_w, v_cache_np)
        cos = np.dot(out_np.flatten(), ref_out.flatten()) / (
            np.linalg.norm(out_np.flatten()) * np.linalg.norm(ref_out.flatten()) + 1e-8)
        print(f"  vs '{interp_name}': cosine={cos:.6f}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Attempt 3b: is_causal=False (no mask at all)
print("\n  Attempt 3b: is_causal=False (no mask)")
try:
    out2 = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_cache_tt, v_cache_tt, is_causal=False
    )
    ttnn.synchronize_device(device)
    out2_np = ttnn.to_torch(out2).float().numpy()
    print(f"  SUCCESS! Output shape: {out2.shape}")
    print(f"  Output range: [{out2_np.min():.6f}, {out2_np.max():.6f}]")

    # Reference: attend to all positions (no mask)
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q_np, k_cache_np.transpose(0, 1, 3, 2)) * scale
    from scipy.special import softmax as sp_softmax
    attn_w = sp_softmax(scores, axis=-1)
    ref_out = np.matmul(attn_w, v_cache_np)
    cos = np.dot(out2_np.flatten(), ref_out.flatten()) / (
        np.linalg.norm(out2_np.flatten()) * np.linalg.norm(ref_out.flatten()) + 1e-8)
    print(f"  vs no-mask reference: cosine={cos:.6f}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# Phase 4: Try with explicit attention mask
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Explicit attention mask")
print("=" * 60)

# Build mask: (1, 1, 1, 1024) — new token attends to positions 0..prefill_len-1
# TT-NN convention: 0 = attend, large negative = mask out (pre-softmax additive mask)
# Or it might be boolean. Try both.

# Attempt 4a: additive mask (0 = keep, -inf = mask)
print("\n  Attempt 4a: additive mask (0=keep, -1e9=mask)")
try:
    mask_np = np.full((batch, 1, 1, max_seq), -1e9, dtype=np.float32)
    mask_np[:, :, :, :prefill_len] = 0.0

    mask_tt = ttnn.from_torch(
        torch.from_numpy(mask_np), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )
    print(f"  Mask shape: {mask_tt.shape}")

    out3 = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_cache_tt, v_cache_tt, attn_mask=mask_tt, is_causal=False
    )
    ttnn.synchronize_device(device)
    out3_np = ttnn.to_torch(out3).float().numpy()
    print(f"  SUCCESS! Output shape: {out3.shape}")
    print(f"  Output range: [{out3_np.min():.6f}, {out3_np.max():.6f}]")

    # Reference: attend to first prefill_len positions only
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q_np, k_cache_np.transpose(0, 1, 3, 2)) * scale
    mask_ref = np.full((1, 1, 1, max_seq), -1e9, dtype=np.float32)
    mask_ref[:, :, :, :prefill_len] = 0.0
    scores = scores + mask_ref
    from scipy.special import softmax as sp_softmax
    attn_w = sp_softmax(scores, axis=-1)
    ref_out = np.matmul(attn_w, v_cache_np)
    cos = np.dot(out3_np.flatten(), ref_out.flatten()) / (
        np.linalg.norm(out3_np.flatten()) * np.linalg.norm(ref_out.flatten()) + 1e-8)
    print(f"  vs masked reference: cosine={cos:.6f}")
    max_err = np.abs(out3_np.reshape(ref_out.shape) - ref_out).max()
    print(f"  max error: {max_err:.6f}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Attempt 4b: boolean-style mask (1=keep, 0=mask) — some APIs use this
print("\n  Attempt 4b: boolean-style mask (1=keep, 0=mask) via large negative multiply")
try:
    mask_np2 = np.zeros((batch, 1, 1, max_seq), dtype=np.float32)
    mask_np2[:, :, :, :prefill_len] = 1.0
    # Convert to additive: (1-mask) * -1e9
    mask_additive = (1.0 - mask_np2) * -1e9

    mask_tt2 = ttnn.from_torch(
        torch.from_numpy(mask_additive.astype(np.float32)), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )

    out4 = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_cache_tt, v_cache_tt, attn_mask=mask_tt2, is_causal=False
    )
    ttnn.synchronize_device(device)
    out4_np = ttnn.to_torch(out4).float().numpy()
    print(f"  SUCCESS! Output shape: {out4.shape}")
    print(f"  Output range: [{out4_np.min():.6f}, {out4_np.max():.6f}]")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# Phase 5: Try with smaller cache (only prefill_len, not full 1024)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 5: SDPA with trimmed K/V (only filled positions)")
print("=" * 60)

# Maybe the trick is: don't use a full 1024-length cache.
# Just pass the actual K/V tensors with seq_len = prefill_len.
# For decode, grow them by 1 each step (or use padded + mask).

# Padded to tile boundary (32)
trim_len = 32  # prefill_len is already 32, tile-aligned
k_trim_np = k_cache_np[:, :, :trim_len, :]
v_trim_np = v_cache_np[:, :, :trim_len, :]
print(f"  Trimmed K shape: {k_trim_np.shape}")
print(f"  Trimmed V shape: {v_trim_np.shape}")

k_trim_tt = ttnn.from_torch(
    torch.from_numpy(k_trim_np), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)
v_trim_tt = ttnn.from_torch(
    torch.from_numpy(v_trim_np), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)

# Attempt 5a: is_causal=True with trimmed K/V
print("\n  Attempt 5a: is_causal=True, K/V=(1,12,32,64)")
try:
    out5a = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_trim_tt, v_trim_tt, is_causal=True
    )
    ttnn.synchronize_device(device)
    out5a_np = ttnn.to_torch(out5a).float().numpy()
    print(f"  SUCCESS! Output shape: {out5a.shape}")
    print(f"  Output range: [{out5a_np.min():.6f}, {out5a_np.max():.6f}]")

    # Reference
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q_np, k_trim_np.transpose(0, 1, 3, 2)) * scale
    # is_causal: Q pos 0 attends to K pos 0 only? Or all?
    # With Q_len=1, K_len=32:
    #   strict causal: pos 0 -> attend to pos 0
    #   "last position": pos 31 -> attend to all 32
    # Check both
    for interp, mask_fn in [
        ("Q=pos0, attend K[0] only", lambda: np.array([[1] + [0]*31])),
        ("Q=last, attend all K", lambda: np.ones((1, trim_len))),
    ]:
        mask = mask_fn().reshape(1, 1, 1, trim_len).astype(np.float32)
        masked = scores * mask + (-1e10) * (1.0 - mask)
        from scipy.special import softmax as sp_softmax
        aw = sp_softmax(masked, axis=-1)
        ref = np.matmul(aw, v_trim_np)
        cos = np.dot(out5a_np.flatten(), ref.flatten()) / (
            np.linalg.norm(out5a_np.flatten()) * np.linalg.norm(ref.flatten()) + 1e-8)
        print(f"    vs '{interp}': cosine={cos:.6f}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Attempt 5b: is_causal=False with trimmed K/V (attend to all)
print("\n  Attempt 5b: is_causal=False, K/V=(1,12,32,64)")
try:
    out5b = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_trim_tt, v_trim_tt, is_causal=False
    )
    ttnn.synchronize_device(device)
    out5b_np = ttnn.to_torch(out5b).float().numpy()
    print(f"  SUCCESS! Output shape: {out5b.shape}")
    print(f"  Output range: [{out5b_np.min():.6f}, {out5b_np.max():.6f}]")

    # Reference: no mask, attend to all
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q_np, k_trim_np.transpose(0, 1, 3, 2)) * scale
    from scipy.special import softmax as sp_softmax
    aw = sp_softmax(scores, axis=-1)
    ref = np.matmul(aw, v_trim_np)
    cos = np.dot(out5b_np.flatten(), ref.flatten()) / (
        np.linalg.norm(out5b_np.flatten()) * np.linalg.norm(ref.flatten()) + 1e-8)
    max_err = np.abs(out5b_np.reshape(ref.shape) - ref).max()
    print(f"  vs no-mask reference: cosine={cos:.6f}, max_err={max_err:.6f}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# Phase 6: Try attention_decode with n_kv_heads != n_heads (MHA test)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 6: attention_decode with MHA (n_kv_heads = n_heads = 12)")
print("=" * 60)

if hasattr(ttnn.transformer, 'scaled_dot_product_attention_decode'):
    # The decode API might accept cur_pos and have an n_kv_heads parameter.
    # Let's try it with n_kv_heads=12 (MHA, not MQA).
    q_decode_np = np.random.randn(batch, n_heads, 1, head_dim).astype(np.float32) * 0.1
    q_decode_tt = ttnn.from_torch(
        torch.from_numpy(q_decode_np), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )

    cur_pos = prefill_len  # next position to decode

    attempts = [
        ("basic: cur_pos=[pos]",
         lambda: ttnn.transformer.scaled_dot_product_attention_decode(
             q_decode_tt, k_cache_tt, v_cache_tt, cur_pos=[cur_pos])),
        ("cur_pos as tensor",
         lambda: ttnn.transformer.scaled_dot_product_attention_decode(
             q_decode_tt, k_cache_tt, v_cache_tt,
             cur_pos_tensor=ttnn.from_torch(
                 torch.tensor([cur_pos], dtype=torch.int32), device=device))),
        ("with is_causal=True",
         lambda: ttnn.transformer.scaled_dot_product_attention_decode(
             q_decode_tt, k_cache_tt, v_cache_tt, is_causal=True, cur_pos=[cur_pos])),
        ("with scale",
         lambda: ttnn.transformer.scaled_dot_product_attention_decode(
             q_decode_tt, k_cache_tt, v_cache_tt, cur_pos=[cur_pos],
             scale=1.0/np.sqrt(head_dim))),
    ]

    for name, fn in attempts:
        try:
            print(f"\n  Trying: {name}...")
            out = fn()
            ttnn.synchronize_device(device)
            out_np = ttnn.to_torch(out).float().numpy()
            print(f"  SUCCESS! Output shape: {out.shape}, torch shape: {out_np.shape}")
            print(f"  Output range: [{out_np.min():.6f}, {out_np.max():.6f}]")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
else:
    print("  scaled_dot_product_attention_decode not available.")

# Also try paged variant
if hasattr(ttnn.transformer, 'paged_scaled_dot_product_attention_decode'):
    print("\n  paged_scaled_dot_product_attention_decode exists.")
    fn = ttnn.transformer.paged_scaled_dot_product_attention_decode
    doc = fn.__doc__
    if doc:
        print(f"  doc: {doc[:500]}")

if hasattr(ttnn.transformer, 'chunked_scaled_dot_product_attention'):
    print("\n  chunked_scaled_dot_product_attention exists.")
    fn = ttnn.transformer.chunked_scaled_dot_product_attention
    doc = fn.__doc__
    if doc:
        print(f"  doc: {doc[:500]}")


# ══════════════════════════════════════════════════════════════
# Phase 7: Manual SDPA fallback (matmul + softmax on device)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 7: Manual SDPA on device (matmul + softmax)")
print("=" * 60)

# If native SDPA doesn't work with asymmetric shapes, we can do it manually:
# scores = Q @ K^T / sqrt(d) -> (1,12,1,1024)
# masked_scores = scores + mask
# weights = softmax(masked_scores, dim=-1)
# output = weights @ V -> (1,12,1,64)

try:
    scale = 1.0 / np.sqrt(head_dim)

    # Q @ K^T
    k_transposed = ttnn.transpose(k_cache_tt, -2, -1)  # (1,12,64,1024)
    print(f"  K transposed shape: {k_transposed.shape}")

    scores = ttnn.matmul(q_tt, k_transposed)  # (1,12,1,1024)
    scores = ttnn.multiply(scores, scale)
    print(f"  Scores shape: {scores.shape}")

    # Apply mask: attend to first prefill_len positions only
    mask_np = np.full((1, 1, 1, max_seq), -1e9, dtype=np.float32)
    mask_np[:, :, :, :prefill_len] = 0.0
    mask_tt = ttnn.from_torch(
        torch.from_numpy(mask_np), dtype=ttnn.bfloat16,
        device=device, layout=ttnn.TILE_LAYOUT
    )
    scores = ttnn.add(scores, mask_tt)

    # Softmax
    weights_tt = ttnn.softmax(scores, dim=-1)
    print(f"  Weights shape: {weights_tt.shape}")

    # Weights @ V
    output = ttnn.matmul(weights_tt, v_cache_tt)  # (1,12,1,64)
    ttnn.synchronize_device(device)
    out_np = ttnn.to_torch(output).float().numpy()
    print(f"  Output shape: {output.shape}, torch shape: {out_np.shape}")
    print(f"  Output range: [{out_np.min():.6f}, {out_np.max():.6f}]")

    # Verify against numpy reference
    scores_ref = np.matmul(q_np, k_cache_np.transpose(0, 1, 3, 2)) * scale
    scores_ref = scores_ref + mask_np
    from scipy.special import softmax as sp_softmax
    aw_ref = sp_softmax(scores_ref, axis=-1)
    ref_out = np.matmul(aw_ref, v_cache_np)

    cos = np.dot(out_np.flatten(), ref_out.flatten()) / (
        np.linalg.norm(out_np.flatten()) * np.linalg.norm(ref_out.flatten()) + 1e-8)
    max_err = np.abs(out_np.reshape(ref_out.shape) - ref_out).max()
    print(f"\n  Manual SDPA vs numpy reference:")
    print(f"    cosine similarity: {cos:.6f}")
    print(f"    max error: {max_err:.6f}")
    print(f"    MANUAL SDPA WORKS!" if cos > 0.99 else "    WARNING: low similarity")

    # Benchmark
    print("\n  Benchmarking manual SDPA decode...")
    # Warmup
    for _ in range(5):
        k_t = ttnn.transpose(k_cache_tt, -2, -1)
        s = ttnn.matmul(q_tt, k_t)
        s = ttnn.multiply(s, scale)
        s = ttnn.add(s, mask_tt)
        w = ttnn.softmax(s, dim=-1)
        o = ttnn.matmul(w, v_cache_tt)
        ttnn.synchronize_device(device)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        k_t = ttnn.transpose(k_cache_tt, -2, -1)
        s = ttnn.matmul(q_tt, k_t)
        s = ttnn.multiply(s, scale)
        s = ttnn.add(s, mask_tt)
        w = ttnn.softmax(s, dim=-1)
        o = ttnn.matmul(w, v_cache_tt)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    print(f"  Manual SDPA decode latency: avg={avg_ms:.3f} ms, min={min_ms:.3f} ms")

except Exception as e:
    print(f"  Manual SDPA FAILED: {e}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
  Goal: Use SDPA with KV cache for GPT-2 MHA (12 heads) decode.

  Q shape:     (1, 12, 1, 64)   — single new token
  K/V cache:   (1, 12, 1024, 64) — all past tokens

  Key findings:
  - Check stdout above for which approaches succeeded/failed.
  - The manual SDPA (matmul + mask + softmax + matmul) is the
    guaranteed fallback — it only uses basic ops.
  - If regular SDPA works with asymmetric Q/K/V, that's ideal
    since it uses the optimized FlashAttention-2 kernel.
  - If is_causal=True gives wrong mask semantics for decode,
    we need an explicit mask.
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
