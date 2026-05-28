#!/usr/bin/env python3
"""
Experiment 82 — Gated DeltaNet recurrence in isolation (Phase A3).

The CORE delta-rule update at decode shape (T=1). Excludes in-projections,
conv1d, and the output gate — those are standard ops covered elsewhere.

Architecture matches Qwen3.6-35B-A3B (see research/qwen36_arch_notes.md):
  - n_v_heads = 32 V heads
  - n_k_heads = 16 K heads (GQA, each K head paired with 2 V heads)
  - d_k = d_v = 128
  - State H in fp32 (mamba_ssm_dtype="float32")

Recurrence per step (matches HF Qwen3_5MoeGatedDeltaNet.recurrent_gated_delta_rule):
  1) L2 normalize Q and K
  2) H_decayed = H_prev * exp(g)
  3) kv_mem = (H_decayed * K).sum(d_k)
  4) delta = (V - kv_mem) * beta
  5) H_new = H_decayed + outer(K, delta)
  6) out = (H_new * Q).sum(d_k)

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/82_gated_deltanet.py
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np

# --- Architecture constants (Qwen3.6-35B-A3B DeltaNet) ---
B = 1                  # batch
N_V_HEADS = 32         # number of value heads (== heads in recurrence)
N_K_HEADS = 16         # number of key heads (GQA: each k-head pairs with 2 v-heads)
D_K = 128              # key head dimension
D_V = 128              # value head dimension

EPS = 1e-6


# ============================================================
# Step 1 — Numpy fp32 reference
# ============================================================

def _l2_normalize_np(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """L2-normalize along the last dim."""
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def deltanet_step_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                        g: np.ndarray, beta: np.ndarray,
                        H_prev: np.ndarray):
    """
    Single-step delta-rule update.

    Inputs:
      q, k:    [B, n_v_heads, d_k]   fp32
      v:       [B, n_v_heads, d_v]   fp32
      g:       [B, n_v_heads]        fp32 (decay, will be exponentiated)
      beta:    [B, n_v_heads]        fp32 (delta rate, sigmoid-applied upstream)
      H_prev:  [B, n_v_heads, d_k, d_v]  fp32 (recurrent state)

    Returns:
      out:     [B, n_v_heads, d_v]   fp32
      H_new:   [B, n_v_heads, d_k, d_v]  fp32
    """
    # 1) L2-normalize Q and K (per HF: use_qk_l2norm_in_kernel=True)
    q = _l2_normalize_np(q)
    k = _l2_normalize_np(k)

    # 2) Decay state: H *= exp(g)
    decay = np.exp(g)[..., None, None]            # [B, H, 1, 1]
    H_decayed = H_prev * decay                    # [B, H, d_k, d_v]

    # 3) Read current V via K
    #    H_decayed: [B, H, d_k, d_v];  k[..., :, None]: [B, H, d_k, 1]
    #    sum over d_k -> [B, H, d_v]
    kv_mem = (H_decayed * k[..., :, None]).sum(axis=-2)   # [B, H, d_v]

    # 4) Delta correction
    delta = (v - kv_mem) * beta[..., None]        # [B, H, d_v]

    # 5) Update state: H_new = H_decayed + outer(k, delta)
    #    k[..., :, None]: [B, H, d_k, 1]
    #    delta[..., None, :]: [B, H, 1, d_v]
    #    product broadcasts to [B, H, d_k, d_v]
    H_new = H_decayed + k[..., :, None] * delta[..., None, :]

    # 6) Read output via Q
    out = (H_new * q[..., :, None]).sum(axis=-2)   # [B, H, d_v]

    return out, H_new


def _self_test_numpy():
    """Sanity: with H_prev=0 and Q=K=V=1, the math should be predictable."""
    rng = np.random.default_rng(42)
    q = rng.standard_normal((B, N_V_HEADS, D_K)).astype(np.float32) * 0.1
    k = rng.standard_normal((B, N_V_HEADS, D_K)).astype(np.float32) * 0.1
    v = rng.standard_normal((B, N_V_HEADS, D_V)).astype(np.float32) * 0.1
    g = (-0.1 * rng.uniform(size=(B, N_V_HEADS))).astype(np.float32)
    beta = rng.uniform(size=(B, N_V_HEADS)).astype(np.float32)
    H = np.zeros((B, N_V_HEADS, D_K, D_V), dtype=np.float32)

    # First step from H=0: kv_mem=0, delta = v * beta, H_new = outer(k, beta*v)
    # out = (H_new * q).sum(d_k) = q · (k * (beta * v).T).sum(d_k)
    out, H_new = deltanet_step_numpy(q, k, v, g, beta, H)

    # The state should be non-zero and finite
    assert np.all(np.isfinite(out)), "out contains non-finite values"
    assert np.all(np.isfinite(H_new)), "H_new contains non-finite values"
    assert H_new.sum() != 0, "H_new should be non-zero after step from H_prev=0"
    print(f"  numpy self-test: out range [{out.min():.4f}, {out.max():.4f}], "
          f"H_new norm = {np.linalg.norm(H_new):.4f}")


# ============================================================
# Step 2 — ttnn implementation (decode path, T=1)
# ============================================================

def _deltanet_step_ttnn(q_tt, k_tt, v_tt, g_tt, beta_tt, H_tt, device, ttnn):
    """Single-step delta-rule on ttnn tensors. All inputs already on device.

    Shapes:
      q_tt, k_tt: [B, H, d_k]    bf16
      v_tt:       [B, H, d_v]    bf16
      g_tt, beta_tt: [B, H]      bf16
      H_tt:       [B, H, d_k, d_v]  fp32 (state)

    Returns: (out_tt [B, H, d_v] bf16,  H_new_tt [B, H, d_k, d_v] fp32)
    """
    # 1) L2 normalize Q and K — compose: x / sqrt(sum(x*x) + eps)
    #    Using ttnn.rsqrt + ttnn.sum
    qq = ttnn.mul(q_tt, q_tt)                                       # [B, H, d_k]
    q_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS))
    q_n = ttnn.mul(q_tt, q_norm)                                    # broadcast scalar back

    kk = ttnn.mul(k_tt, k_tt)
    k_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS))
    k_n = ttnn.mul(k_tt, k_norm)

    # 2) Decay state: H *= exp(g). g is [B, H]; need to broadcast to [B, H, 1, 1].
    decay = ttnn.exp(g_tt)                                          # [B, H]
    # Reshape g to [B, H, 1, 1] so it broadcasts against H_tt [B, H, d_k, d_v]
    decay_4d = ttnn.reshape(decay, [B, N_V_HEADS, 1, 1])
    H_decayed = ttnn.mul(H_tt, decay_4d)                            # [B, H, d_k, d_v]

    # 3) kv_mem = sum(H_decayed * k[..., None], dim=-2)
    #    k_n is [B, H, d_k]; reshape to [B, H, d_k, 1]
    k_col = ttnn.reshape(k_n, [B, N_V_HEADS, D_K, 1])
    kv_mem = ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2)           # [B, H, 1, d_v]
    kv_mem = ttnn.reshape(kv_mem, [B, N_V_HEADS, D_V])              # squeeze sum dim

    # 4) delta = (v - kv_mem) * beta
    diff = ttnn.sub(v_tt, kv_mem)                                   # [B, H, d_v]
    beta_3d = ttnn.reshape(beta_tt, [B, N_V_HEADS, 1])
    delta = ttnn.mul(diff, beta_3d)                                 # [B, H, d_v]

    # 5) H_new = H_decayed + outer(k, delta)
    #    k_col: [B, H, d_k, 1];  delta as [B, H, 1, d_v]
    delta_row = ttnn.reshape(delta, [B, N_V_HEADS, 1, D_V])
    outer = ttnn.mul(k_col, delta_row)                              # [B, H, d_k, d_v]
    H_new = ttnn.add(H_decayed, outer)

    # 6) out = sum(H_new * q[..., None], dim=-2)
    q_col = ttnn.reshape(q_n, [B, N_V_HEADS, D_K, 1])
    out = ttnn.sum(ttnn.mul(H_new, q_col), dim=-2)                  # [B, H, 1, d_v]
    out = ttnn.reshape(out, [B, N_V_HEADS, D_V])

    return out, H_new


def _cosine(a, b):
    """Cosine similarity between two flat arrays."""
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ============================================================
# Step 3 + 4 — Cosine check + perf
# ============================================================

def main():
    print("=" * 64)
    print("Phase A3 — Gated DeltaNet isolated kernel")
    print("=" * 64)
    print(f"  B={B}, n_v_heads={N_V_HEADS}, d_k={D_K}, d_v={D_V}")
    print()

    # Self-test numpy
    print("[1/4] Numpy reference self-test")
    _self_test_numpy()

    # Random inputs (seed-fixed for reproducibility)
    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((B, N_V_HEADS, D_K)).astype(np.float32) * 0.1
    k_np = rng.standard_normal((B, N_V_HEADS, D_K)).astype(np.float32) * 0.1
    v_np = rng.standard_normal((B, N_V_HEADS, D_V)).astype(np.float32) * 0.1
    g_np = (-0.1 * rng.uniform(size=(B, N_V_HEADS))).astype(np.float32)
    beta_np = rng.uniform(size=(B, N_V_HEADS)).astype(np.float32) * 0.5
    H_prev_np = (rng.standard_normal((B, N_V_HEADS, D_K, D_V))
                 .astype(np.float32) * 0.05)

    # Numpy gold
    out_np, H_new_np = deltanet_step_numpy(q_np, k_np, v_np, g_np, beta_np, H_prev_np)

    # ttnn implementation (only if ttnn is available)
    try:
        import ttnn, torch
    except ImportError:
        print("\n[ttnn not available — numpy reference verified, skipping ttnn]")
        return

    print("\n[2/4] Opening device and uploading inputs")
    device = ttnn.open_device(device_id=0)

    def upload(arr, dtype=ttnn.bfloat16):
        t = torch.from_numpy(arr.astype(np.float32))
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    # Q, K, V, g, beta as bf16; H as fp32 per mamba_ssm_dtype config
    q_tt = upload(q_np)
    k_tt = upload(k_np)
    v_tt = upload(v_np)
    g_tt = upload(g_np)
    beta_tt = upload(beta_np)
    H_tt = upload(H_prev_np, dtype=ttnn.float32)
    ttnn.synchronize_device(device)
    print("  All inputs on device (Q,K,V,g,β bf16, H fp32)")

    # Run ttnn
    print("\n[3/4] Cosine check")
    out_tt, H_new_tt = _deltanet_step_ttnn(q_tt, k_tt, v_tt, g_tt, beta_tt,
                                            H_tt, device, ttnn)
    ttnn.synchronize_device(device)

    # Read back
    out_back = ttnn.to_torch(out_tt).float().numpy().reshape(B, N_V_HEADS, D_V)
    H_back = ttnn.to_torch(H_new_tt).float().numpy().reshape(B, N_V_HEADS, D_K, D_V)

    cos_out = _cosine(out_np, out_back)
    cos_H = _cosine(H_new_np, H_back)
    max_abs_out = float(np.max(np.abs(out_np - out_back)))
    max_abs_H = float(np.max(np.abs(H_new_np - H_back)))

    print(f"  cosine(out)  = {cos_out:.6f}   max-abs-diff = {max_abs_out:.6f}")
    print(f"  cosine(H_new)= {cos_H:.6f}   max-abs-diff = {max_abs_H:.6f}")

    PASS = cos_out >= 0.99 and cos_H >= 0.99
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}  (gate: cosine ≥ 0.99)")

    # Bench
    print("\n[4/4] Performance")
    WARMUP, ITERS = 10, 200

    def bench():
        out_tt, H_new_tt = _deltanet_step_ttnn(q_tt, k_tt, v_tt, g_tt, beta_tt,
                                                H_tt, device, ttnn)
        return out_tt, H_new_tt

    for _ in range(WARMUP):
        bench()
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        bench()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)

    med = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[8]

    # Memory ceiling: read+write H (~4 MB at fp32) plus tiny scratch
    state_bytes = B * N_V_HEADS * D_K * D_V * 4 * 2  # *2 for read+write
    mem_floor_us = state_bytes / (450e9) * 1e6   # 450 GB/s
    pct = mem_floor_us / med * 100

    print(f"  decode step (eager):   median = {med:7.1f} µs    p90 = {p90:7.1f} µs")
    print(f"  state I/O ceiling (1× P150 @ 450 GB/s, {state_bytes/1e6:.1f} MB): {mem_floor_us:.1f} µs")
    print(f"  eager % of ceiling: {pct:.2f}%   (higher is better)")

    # ── Trace capture variant ─────────────────────────────────
    # Trace captures the op sequence on-device once, replays it on
    # subsequent calls with zero per-op dispatch overhead. Real floor.
    print("\n  Capturing trace…")
    # Pre-allocate persistent input/output buffers to copy into
    q_buf = upload(np.zeros_like(q_np)); k_buf = upload(np.zeros_like(k_np))
    v_buf = upload(np.zeros_like(v_np)); g_buf = upload(np.zeros_like(g_np))
    beta_buf = upload(np.zeros_like(beta_np))
    H_buf = upload(np.zeros_like(H_prev_np), dtype=ttnn.float32)

    # Warm-up the kernels (build_cache, autotune)
    for _ in range(WARMUP):
        _deltanet_step_ttnn(q_buf, k_buf, v_buf, g_buf, beta_buf, H_buf, device, ttnn)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    _ = _deltanet_step_ttnn(q_buf, k_buf, v_buf, g_buf, beta_buf, H_buf, device, ttnn)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    samples_t = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        samples_t.append((time.perf_counter_ns() - t0) / 1000.0)

    med_t = statistics.median(samples_t)
    p90_t = statistics.quantiles(samples_t, n=10)[8]
    pct_t = mem_floor_us / med_t * 100
    print(f"  decode step (traced):  median = {med_t:7.1f} µs    p90 = {p90_t:7.1f} µs")
    print(f"  traced % of ceiling: {pct_t:.2f}%")
    print(f"  trace speedup vs eager: {med/med_t:.2f}×")

    ttnn.release_trace(device, trace_id)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
