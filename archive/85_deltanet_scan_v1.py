#!/usr/bin/env python3
"""
Experiment 85 — Gated DeltaNet PREFILL scan v1 (chunked-serial).

Phase A6 v1 — naive serial scan over time. Bootstrap to unblock prefill
correctness validation. Beats serial-on-host by running the per-chunk
math on device, but no asymptotic improvement.

Math: same per-step recurrence as A3 (`experiments/82_gated_deltanet.py`),
just applied serially over T time steps with H carried across steps.

Tests:
  1. Cosine ≥ 0.99 of (out, H_final) vs numpy fp32 reference at T=64 and T=1024
  2. Throughput: tokens/sec for various sequence lengths
  3. Compare to "T separate decode steps" baseline

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/85_deltanet_scan_v1.py
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np

B = 1
N_V_HEADS = 32
D_K = 128
D_V = 128
EPS = 1e-6


# ============================================================
# Numpy reference — multi-step serial scan
# ============================================================

def _l2_normalize_np(x):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + EPS)


def deltanet_scan_numpy(Q, K, V, G, BETA, H_init):
    """
    Multi-step delta-rule scan, serial over time.

    Inputs:
      Q, K:    [B, T, n_v_heads, d_k]   fp32
      V:       [B, T, n_v_heads, d_v]   fp32
      G, BETA: [B, T, n_v_heads]        fp32
      H_init:  [B, n_v_heads, d_k, d_v] fp32

    Returns:
      OUT:     [B, T, n_v_heads, d_v]   fp32
      H_final: [B, n_v_heads, d_k, d_v] fp32
    """
    B_, T, H_, D_K_ = Q.shape
    D_V_ = V.shape[-1]
    H = H_init.copy()
    OUT = np.zeros((B_, T, H_, D_V_), dtype=np.float32)

    for t in range(T):
        q_t = _l2_normalize_np(Q[:, t])              # [B, H, d_k]
        k_t = _l2_normalize_np(K[:, t])
        v_t = V[:, t]                                 # [B, H, d_v]
        g_t = G[:, t]
        beta_t = BETA[:, t]

        # Decay
        H = H * np.exp(g_t)[..., None, None]
        # Read kv via K
        kv_mem = (H * k_t[..., :, None]).sum(axis=-2)
        # Delta
        delta = (v_t - kv_mem) * beta_t[..., None]
        # Update
        H = H + k_t[..., :, None] * delta[..., None, :]
        # Output via Q
        OUT[:, t] = (H * q_t[..., :, None]).sum(axis=-2)

    return OUT, H


# ============================================================
# ttnn chunked-serial implementation (v1)
# ============================================================

def _deltanet_step_on_device(q, k, v, g, beta, H, ttnn):
    """One device-side step. Same as experiment 82.
    All inputs are [B, n_v_heads, d_*] or [B, n_v_heads] for g/beta.
    H is [B, n_v_heads, d_k, d_v] fp32.
    Returns: out [B, n_v_heads, d_v], H_new [B, n_v_heads, d_k, d_v]
    """
    qq = ttnn.mul(q, q)
    q_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS))
    q_n = ttnn.mul(q, q_norm)
    kk = ttnn.mul(k, k)
    k_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS))
    k_n = ttnn.mul(k, k_norm)

    decay = ttnn.exp(g)
    decay_4d = ttnn.reshape(decay, [B, N_V_HEADS, 1, 1])
    H_decayed = ttnn.mul(H, decay_4d)

    k_col = ttnn.reshape(k_n, [B, N_V_HEADS, D_K, 1])
    kv_mem = ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2)
    kv_mem = ttnn.reshape(kv_mem, [B, N_V_HEADS, D_V])

    diff = ttnn.sub(v, kv_mem)
    beta_3d = ttnn.reshape(beta, [B, N_V_HEADS, 1])
    delta = ttnn.mul(diff, beta_3d)

    delta_row = ttnn.reshape(delta, [B, N_V_HEADS, 1, D_V])
    outer = ttnn.mul(k_col, delta_row)
    H_new = ttnn.add(H_decayed, outer)

    q_col = ttnn.reshape(q_n, [B, N_V_HEADS, D_K, 1])
    out = ttnn.sum(ttnn.mul(H_new, q_col), dim=-2)
    out = ttnn.reshape(out, [B, N_V_HEADS, D_V])
    return out, H_new


def deltanet_scan_ttnn_v1(Q_dev, K_dev, V_dev, G_dev, BETA_dev,
                           H_init_dev, T: int, device, ttnn):
    """
    v1 chunked-serial: just loop over T time steps on device.

    Q_dev, K_dev, V_dev, G_dev, BETA_dev: device tensors, indexed by t.
    Returns: list of out_t device tensors, H_final device tensor.
    """
    H = H_init_dev
    outs = []
    for t in range(T):
        q_t = Q_dev[t]
        k_t = K_dev[t]
        v_t = V_dev[t]
        g_t = G_dev[t]
        beta_t = BETA_dev[t]
        out_t, H = _deltanet_step_on_device(q_t, k_t, v_t, g_t, beta_t, H, ttnn)
        outs.append(out_t)
    return outs, H


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Phase A6 v1 — DeltaNet PREFILL scan (chunked-serial)")
    print("=" * 64)

    # Run at two sequence lengths
    for T in (64, 256, 1024):
        rng = np.random.default_rng(42)
        Q_np = rng.standard_normal((B, T, N_V_HEADS, D_K)).astype(np.float32) * 0.1
        K_np = rng.standard_normal((B, T, N_V_HEADS, D_K)).astype(np.float32) * 0.1
        V_np = rng.standard_normal((B, T, N_V_HEADS, D_V)).astype(np.float32) * 0.1
        G_np = (-0.05 * rng.uniform(size=(B, T, N_V_HEADS))).astype(np.float32)
        BETA_np = rng.uniform(size=(B, T, N_V_HEADS)).astype(np.float32) * 0.5
        H_init = np.zeros((B, N_V_HEADS, D_K, D_V), dtype=np.float32)

        # Numpy gold
        out_np, H_np = deltanet_scan_numpy(Q_np, K_np, V_np, G_np, BETA_np, H_init)

        try:
            import ttnn, torch
        except ImportError:
            print(f"[T={T}] ttnn not available; skipping device run")
            continue

        if T == 64:  # only open device once
            device = ttnn.open_device(device_id=0)

        def upload(arr, dtype=ttnn.bfloat16):
            t_ = torch.from_numpy(arr.astype(np.float32))
            while t_.dim() < 2:
                t_ = t_.unsqueeze(0)
            return ttnn.from_torch(t_, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        print(f"\n[T={T}] Uploading and running…")
        # Pre-upload each time step's slice (waste of memory but simple)
        Q_dev = [upload(Q_np[:, t]) for t in range(T)]
        K_dev = [upload(K_np[:, t]) for t in range(T)]
        V_dev = [upload(V_np[:, t]) for t in range(T)]
        G_dev = [upload(G_np[:, t]) for t in range(T)]
        BETA_dev = [upload(BETA_np[:, t]) for t in range(T)]
        H_init_dev = upload(H_init, dtype=ttnn.float32)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter_ns()
        outs_dev, H_final_dev = deltanet_scan_ttnn_v1(
            Q_dev, K_dev, V_dev, G_dev, BETA_dev, H_init_dev, T, device, ttnn)
        ttnn.synchronize_device(device)
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        us_per_tok = elapsed_us / T

        # Check final state and last output
        H_back = ttnn.to_torch(H_final_dev).float().numpy().reshape(B, N_V_HEADS, D_K, D_V)
        out_last_back = ttnn.to_torch(outs_dev[-1]).float().numpy().reshape(B, N_V_HEADS, D_V)
        cos_H = _cosine(H_np, H_back)
        cos_last = _cosine(out_np[:, -1], out_last_back)

        print(f"  cosine(H_final) = {cos_H:.6f}")
        print(f"  cosine(out[T-1]) = {cos_last:.6f}")
        print(f"  total scan time: {elapsed_us / 1000:.1f} ms  ({us_per_tok:.0f} µs/token)")
        print(f"  tokens/sec: {1e6 / us_per_tok:.0f}")

    if 'device' in dir():
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
