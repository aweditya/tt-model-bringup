#!/usr/bin/env python3
"""
Experiment 83 — Gated Attention isolated kernel (Phase A4).

Tests the SDPA-decode block from Qwen3.6-35B-A3B's Gated Attention layer:
  - Partial RoPE (first 64 of head_dim=256 rotated, rest pass through)
  - GQA SDPA with 16 Q heads / 2 KV heads, head_dim=256
  - Output sigmoid gate (from chunk-half of q_proj)

What we EXCLUDE:
  - q_proj / k_proj / v_proj / o_proj (standard ttnn.linear)
  - paged_update_cache (existing working code in demos/generate_moe.py)

We assume Q+gate, K, V are already projected and reshaped.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/83_gated_attention.py
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np

# Qwen3.6 Gated Attention shapes
B = 1
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)   # 64
PASSTHRU_DIM = HEAD_DIM - ROTARY_DIM                  # 192
ROPE_THETA = 10_000_000.0
KV_LEN = 128       # pre-populated KV cache length
EPS = 1e-6


# ============================================================
# Numpy reference
# ============================================================

def rope_tables_np(pos: int, dim: int = ROTARY_DIM, theta: float = ROPE_THETA):
    """Compute cos/sin for a single position. Returns shape [dim]."""
    half = dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half).astype(np.float32) / half))
    angles = pos * freqs
    # cos/sin doubled to match dim (interleaved-pairs convention NOT used here:
    # we use the half-format Llama convention via rotate_half)
    cos = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    return cos, sin


def rotate_half_np(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_partial_rope_np(x: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    """Apply RoPE to first ROTARY_DIM dims; pass remaining dims through."""
    rot, passthrough = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
    rot_out = rot * cos + rotate_half_np(rot) * sin
    return np.concatenate([rot_out, passthrough], axis=-1)


def repeat_kv_np(x: np.ndarray, n_rep: int) -> np.ndarray:
    """GQA: replicate KV heads to match Q heads. x: [B, n_kv, T, d]"""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return np.broadcast_to(x[:, :, None, :, :], (b, n_kv, n_rep, t, d)).reshape(
        b, n_kv * n_rep, t, d)


def gated_attention_decode_numpy(q: np.ndarray, gate: np.ndarray,
                                  k: np.ndarray, v: np.ndarray,
                                  k_cache: np.ndarray, v_cache: np.ndarray,
                                  cos: np.ndarray, sin: np.ndarray,
                                  cur_pos: int) -> np.ndarray:
    """
    Decode-step gated attention.

    Inputs (all fp32 for reference):
      q:       [B, 1, n_q_heads,  head_dim]   current-step Q
      gate:    [B, 1, n_q_heads,  head_dim]   sigmoid gate (from q_proj other half)
      k, v:    [B, 1, n_kv_heads, head_dim]   current-step K, V
      k_cache: [B, n_kv_heads, KV_LEN, head_dim]   ALL prior K (incl. cur_pos slot)
      v_cache: [B, n_kv_heads, KV_LEN, head_dim]
      cos, sin: [ROTARY_DIM]   RoPE tables for cur_pos
      cur_pos: int (used for causal-mask length)

    Returns: out [B, 1, n_q_heads, head_dim]   (post-sigmoid-gate, pre-o_proj)
    """
    # 1) Apply partial RoPE to current-step Q and K (cache K is already RoPE'd)
    q_rot = apply_partial_rope_np(q, cos, sin)              # [B, 1, n_q, d]
    k_rot = apply_partial_rope_np(k, cos, sin)              # [B, 1, n_kv, d]

    # 2) Insert current K, V into cache at cur_pos
    kc = k_cache.copy()
    vc = v_cache.copy()
    kc[:, :, cur_pos:cur_pos + 1, :] = k_rot.transpose(0, 2, 1, 3)
    vc[:, :, cur_pos:cur_pos + 1, :] = v.transpose(0, 2, 1, 3)

    # 3) GQA: repeat K/V to match Q head count
    n_rep = N_Q_HEADS // N_KV_HEADS                          # 8
    kc_q = repeat_kv_np(kc, n_rep)                          # [B, n_q_heads, KV_LEN, d]
    vc_q = repeat_kv_np(vc, n_rep)

    # 4) SDPA: attn = softmax(QK^T / sqrt(d)) V
    #    Q shape [B, 1, n_q_heads, d]  →  rearrange to [B, n_q_heads, 1, d]
    q_for_attn = q_rot.transpose(0, 2, 1, 3)                # [B, n_q, 1, d]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    # scores: [B, n_q, 1, KV_LEN]
    scores = np.matmul(q_for_attn, kc_q.transpose(0, 1, 3, 2)) * scale

    # Causal mask: only attend to positions ≤ cur_pos
    mask = np.zeros((1, 1, 1, KV_LEN), dtype=np.float32)
    mask[..., cur_pos + 1:] = -1e9
    scores = scores + mask

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    attn = np.matmul(weights, vc_q)                         # [B, n_q, 1, d]
    attn = attn.transpose(0, 2, 1, 3)                       # [B, 1, n_q, d]

    # 5) Output sigmoid gate
    gated = attn * (1.0 / (1.0 + np.exp(-gate)))
    return gated


# ============================================================
# ttnn implementation
# ============================================================

def gated_attention_decode_ttnn(q_tt, gate_tt, k_tt, v_tt,
                                 k_cache_tt, v_cache_tt,
                                 cos_tt, sin_tt, cur_pos: int,
                                 device, ttnn):
    """Decode-step Gated Attention using ttnn.

    NOTE on partial RoPE: ttnn's slice/concat on the head_dim axis fails for
    our non-tile-aligned slice sizes (64 + 192). For A4 we apply partial
    RoPE in numpy BEFORE uploading Q/K (host-pre-rotation), and test the
    SDPA + cache-update + sigmoid-gate path on device. Partial RoPE on
    device is deferred to A4b — likely needs ttnn.experimental.rotary_embedding_llama
    or a custom 64-dim rotate kernel.

    Q and K passed in here are ALREADY partial-rope-applied on host.
    """
    # A4 isolation: skip on-device cache update. Cache is pre-populated on host
    # with cur_pos slot already containing the current K/V. SDPA decode just reads.
    # (paged_update_cache requires sharded inputs — that production-path dance
    # is well-validated in demos/generate_moe.py and not what A4 is testing.)

    pos_tensor = ttnn.from_torch(
        __import__('torch').tensor([cur_pos], dtype=__import__('torch').int32),
        device=device)

    # SDPA decode
    q_for_sdpa = ttnn.reshape(q_tt, [B, 1, N_Q_HEADS, HEAD_DIM])
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q_for_sdpa, k_cache_tt, v_cache_tt, cur_pos_tensor=pos_tensor)
    # Returns [B, 1, n_q_heads, head_dim] or [B, 1, n_q_heads*head_dim]
    attn = ttnn.reshape(attn, [B, 1, N_Q_HEADS, HEAD_DIM])

    # Output sigmoid gate
    gate_4d = ttnn.reshape(gate_tt, [B, 1, N_Q_HEADS, HEAD_DIM])
    gate_sig = ttnn.sigmoid(gate_4d)
    out = ttnn.mul(attn, gate_sig)

    return out


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Phase A4 — Gated Attention isolated kernel")
    print("=" * 64)
    print(f"  B={B}, n_q={N_Q_HEADS}, n_kv={N_KV_HEADS}, head_dim={HEAD_DIM}, KV_LEN={KV_LEN}")
    print(f"  partial_rotary={ROTARY_DIM} of {HEAD_DIM} (rest pass through)")
    print()

    # Random inputs (seeded)
    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((B, 1, N_Q_HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    gate_np = rng.standard_normal((B, 1, N_Q_HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    k_np = rng.standard_normal((B, 1, N_KV_HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    v_np = rng.standard_normal((B, 1, N_KV_HEADS, HEAD_DIM)).astype(np.float32) * 0.1
    # Pre-populate KV cache for positions 0 .. cur_pos (INCLUDING cur_pos slot
    # already filled — A4 skips on-device cache update; numpy ref will overwrite
    # the slot too, so the two paths see identical cache contents).
    cur_pos = 32     # somewhere in the middle of the cache
    k_cache_np = np.zeros((B, N_KV_HEADS, KV_LEN, HEAD_DIM), dtype=np.float32)
    v_cache_np = np.zeros((B, N_KV_HEADS, KV_LEN, HEAD_DIM), dtype=np.float32)
    k_cache_np[:, :, :cur_pos, :] = rng.standard_normal(
        (B, N_KV_HEADS, cur_pos, HEAD_DIM)).astype(np.float32) * 0.05
    v_cache_np[:, :, :cur_pos, :] = rng.standard_normal(
        (B, N_KV_HEADS, cur_pos, HEAD_DIM)).astype(np.float32) * 0.05
    # The cur_pos slot will be set with the rotated current-step K/V AFTER
    # we apply partial RoPE on host (below).

    cos_np, sin_np = rope_tables_np(cur_pos)

    # Host-side partial RoPE applied to Q and K before upload (workaround
    # for ttnn slice/concat tile-alignment issues on non-32-multiple sizes).
    q_rotated_np = apply_partial_rope_np(q_np, cos_np, sin_np)
    k_rotated_np = apply_partial_rope_np(k_np, cos_np, sin_np)

    # Pre-write the cur_pos slot into the cache so the device path
    # doesn't need paged_update_cache.
    k_cache_np[:, :, cur_pos:cur_pos+1, :] = k_rotated_np.transpose(0, 2, 1, 3)
    v_cache_np[:, :, cur_pos:cur_pos+1, :] = v_np.transpose(0, 2, 1, 3)

    print(f"[1/4] Numpy reference at cur_pos={cur_pos}")
    out_np = gated_attention_decode_numpy(q_np, gate_np, k_np, v_np,
                                           k_cache_np, v_cache_np,
                                           cos_np, sin_np, cur_pos)
    print(f"  out range [{out_np.min():.4f}, {out_np.max():.4f}], norm={np.linalg.norm(out_np):.4f}")

    try:
        import ttnn, torch
    except ImportError:
        print("\n[ttnn not available — numpy reference verified, skipping ttnn]")
        return

    print("\n[2/4] Opening device and uploading inputs")
    device = ttnn.open_device(device_id=0)

    def upload(arr, dtype=ttnn.bfloat16):
        t = torch.from_numpy(arr.astype(np.float32))
        while t.dim() < 2:
            t = t.unsqueeze(0)
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    # Upload PRE-ROTATED Q and K (partial RoPE applied on host)
    q_tt = upload(q_rotated_np)
    gate_tt = upload(gate_np)
    k_tt = upload(k_rotated_np)
    v_tt = upload(v_np)
    k_cache_tt = upload(k_cache_np)
    v_cache_tt = upload(v_cache_np)
    cos_tt = upload(cos_np)
    sin_tt = upload(sin_np)
    ttnn.synchronize_device(device)

    print("\n[3/4] Cosine check")
    out_tt = gated_attention_decode_ttnn(q_tt, gate_tt, k_tt, v_tt,
                                          k_cache_tt, v_cache_tt,
                                          cos_tt, sin_tt, cur_pos,
                                          device, ttnn)
    ttnn.synchronize_device(device)
    out_back = ttnn.to_torch(out_tt).float().numpy().reshape(B, 1, N_Q_HEADS, HEAD_DIM)

    cos_v = _cosine(out_np, out_back)
    max_abs = float(np.max(np.abs(out_np - out_back)))
    print(f"  cosine(out) = {cos_v:.6f}   max-abs-diff = {max_abs:.6f}")
    PASS = cos_v >= 0.99
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}  (gate: cosine ≥ 0.99)")

    print("\n[4/4] Performance")
    WARMUP, ITERS = 10, 200

    # ── Eager ──────────────────────────────────────────────
    for _ in range(WARMUP):
        gated_attention_decode_ttnn(q_tt, gate_tt, k_tt, v_tt,
                                     k_cache_tt, v_cache_tt,
                                     cos_tt, sin_tt, cur_pos, device, ttnn)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        gated_attention_decode_ttnn(q_tt, gate_tt, k_tt, v_tt,
                                     k_cache_tt, v_cache_tt,
                                     cos_tt, sin_tt, cur_pos, device, ttnn)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    med = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[8]

    # Memory ceiling estimate
    # KV cache read: 2 * (KV_LEN * n_kv * head_dim * 2) bytes
    kv_bytes = 2 * KV_LEN * N_KV_HEADS * HEAD_DIM * 2
    out_bytes = N_Q_HEADS * HEAD_DIM * 2
    mem_floor_us = (kv_bytes + out_bytes) / 450e9 * 1e6
    pct = mem_floor_us / med * 100
    print(f"  decode step (eager): median = {med:7.1f} µs  p90 = {p90:7.1f} µs")
    print(f"  ceiling (KV cache {kv_bytes/1024:.1f} KB read): {mem_floor_us:.2f} µs")
    print(f"  eager % of ceiling: {pct:.2f}%")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
