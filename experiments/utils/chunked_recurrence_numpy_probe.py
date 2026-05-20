#!/usr/bin/env python3
"""
v4 Stage 4 math validation: chunked Neumann recurrence vs per-position.

Pure numpy. No device. Validates the C'5 chunked-prefill plan §1 math
END-TO-END against the per-position recurrence math (mirroring the
manual recurrence path inside `deltanet_step_tp` in server_tp.py).

If chunked_output ≈ per_position_output (cos ≥ 0.9999) for synthetic
inputs at production shapes, the ttnn integration in next session can
proceed with confidence that the math itself is correct — any bug
discovered there will be implementation-side, not algorithmic.

Reference:
- C'5 plan §1: research/c5_chunked_prefill_plan.md (math statement)
- HF impl: transformers/models/qwen3_next/modeling_qwen3_next.py:797
  (`torch_chunk_gated_delta_rule`)
- Per-pos impl: server_tp.py:_deltanet_step_tp_from_inproj (manual
  recurrence branch, lines 859-868).

Run:
    cd ~/Labs/stanford/cs440lx/tt-xla
    python3 experiments/utils/chunked_recurrence_numpy_probe.py
"""

import numpy as np

# Production-shape constants per V-head, per chip (qb2 4× P150 mesh).
NV_PER_CHIP = 12   # of 48 total V-heads, sharded /4
K_DIM = 128
V_DIM = 128


def _cos(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def per_position_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init):
    """Reference: process each position sequentially, matching the manual
    branch of deltanet_step_tp (server_tp.py:859-868).

    Shapes:
        q_seq, k_seq:  [C, NV, K_DIM]
        v_seq:         [C, NV, V_DIM]
        g_seq:         [C, NV]
        beta_seq:      [C, NV]
        S_init:        [NV, K_DIM, V_DIM]   (entering SSM hidden state)

    Returns:
        out_seq:       [C, NV, V_DIM]
        S_final:       [NV, K_DIM, V_DIM]

    Per-pos math (matching server_tp.py manual branch, fp64 here for
    reference accuracy):
        H_decayed[h] = exp(g[h]) * H[h]                     # (K, V)
        kv_mem[h]    = sum_K(H_decayed[h] * k[h, :, None])  # (V,)  — kᵀS_{t-1}
        delta[h]     = (v[h] - kv_mem[h]) * beta[h]         # (V,)
        H_new[h]     = H_decayed[h] + k[h, :, None] * delta[h, None, :]   # (K, V)
        out[h]       = sum_K(H_new[h] * q[h, :, None])      # (V,)  — qᵀS_t
    """
    C, NV, KD = q_seq.shape
    VD = v_seq.shape[-1]
    H = S_init.astype(np.float64).copy()   # (NV, K, V)
    out_seq = np.zeros((C, NV, VD), dtype=np.float64)
    for t in range(C):
        q = q_seq[t].astype(np.float64)       # (NV, K)
        k = k_seq[t].astype(np.float64)       # (NV, K)
        v = v_seq[t].astype(np.float64)       # (NV, V)
        g = g_seq[t].astype(np.float64)       # (NV,)
        beta = beta_seq[t].astype(np.float64)  # (NV,)
        # Per-head, vectorized over heads
        decay = np.exp(g)[:, None, None]                     # (NV, 1, 1)
        H_decayed = decay * H                                 # (NV, K, V)
        k_col = k[:, :, None]                                 # (NV, K, 1)
        kv_mem = (H_decayed * k_col).sum(axis=1)              # (NV, V)
        delta = (v - kv_mem) * beta[:, None]                  # (NV, V)
        H_new = H_decayed + k_col * delta[:, None, :]         # (NV, K, V)
        q_col = q[:, :, None]                                 # (NV, K, 1)
        out = (H_new * q_col).sum(axis=1)                     # (NV, V)
        out_seq[t] = out
        H = H_new
    return out_seq, H


def chunked_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init):
    """Chunked Neumann recurrence per the C'5 plan §1 math.

    Same input/output shapes as per_position_recurrence. Should match
    bit-equivalently in exact arithmetic; will match within fp noise here.

    Math (per V-head independently; vectorized over heads via broadcasting):
        G[a]     = cumsum(g)[a]                              # (NV, C)
        D[a, b]  = exp(G[a] - G[b])  for a >= b else 0       # (NV, C, C)
        k_beta   = beta * k                                  # (NV, C, K)
        v_beta   = beta * v                                  # (NV, C, V)
        attn     = -(k_beta @ k^T) * D  with diag zeroed     # (NV, C, C)
        T        = (I - attn)^{-1}                           # (NV, C, C)
        V_prime  = T @ v_beta                                # (NV, C, V)
        K_prime  = T @ (k_beta * exp(G)[..., None])          # (NV, C, K)
        v_prime  = K_prime @ S_prev                          # (NV, C, V)
        v_new    = V_prime - v_prime                         # (NV, C, V)
        attn_int = (q * exp(G)[..., None]) @ S_prev          # (NV, C, V)
        A        = (q @ k^T) * D                             # (NV, C, C)
        O        = attn_int + A @ v_new                      # (NV, C, V)
        S_new    = exp(G[-1]) * S_prev
                 + (k * exp(G[-1] - G)[..., None]).T @ v_new  # (NV, K, V)
    """
    C, NV, KD = q_seq.shape
    VD = v_seq.shape[-1]
    # Move axes: numpy convention here is (NV, C, K|V) for per-head matmuls
    q = q_seq.transpose(1, 0, 2).astype(np.float64)            # (NV, C, K)
    k = k_seq.transpose(1, 0, 2).astype(np.float64)
    v = v_seq.transpose(1, 0, 2).astype(np.float64)
    g = g_seq.transpose(1, 0).astype(np.float64)               # (NV, C)
    beta = beta_seq.transpose(1, 0).astype(np.float64)         # (NV, C)
    S_prev = S_init.astype(np.float64).copy()                   # (NV, K, V)

    # G = cumsum(g)
    G = np.cumsum(g, axis=-1)                                  # (NV, C)

    # D = exp(G[a] - G[b]) for a >= b else 0
    diff = G[:, :, None] - G[:, None, :]                       # (NV, C, C)
    lower_tri = np.tril(np.ones((C, C), dtype=np.float64))     # (C, C)
    D = np.exp(diff) * lower_tri                                # (NV, C, C)

    # k_beta, v_beta
    k_beta = beta[..., None] * k                                # (NV, C, K)
    v_beta = beta[..., None] * v                                # (NV, C, V)

    # attn = -(k_beta @ k^T) * D, with diagonal zeroed (strict-lower-tri)
    attn = -np.matmul(k_beta, k.transpose(0, 2, 1)) * D         # (NV, C, C)
    # Strict lower tri: zero the diagonal
    eye_C = np.eye(C, dtype=np.float64)
    attn = attn * (1.0 - eye_C)                                  # (NV, C, C)

    # T = (I - attn)^{-1}  via numpy.linalg.inv (reference; ttnn will use Neumann)
    I_C = np.broadcast_to(eye_C, (NV, C, C)).copy()              # (NV, C, C)
    T = np.linalg.inv(I_C - attn)                                # (NV, C, C)

    # V_prime, K_prime
    V_prime = np.matmul(T, v_beta)                               # (NV, C, V)
    K_prime = np.matmul(T, k_beta * np.exp(G)[..., None])        # (NV, C, K)

    # v_prime = K_prime @ S_prev  (per head)
    v_prime = np.matmul(K_prime, S_prev)                         # (NV, C, V)
    v_new = V_prime - v_prime                                    # (NV, C, V)

    # attn_int = (q * exp(G)) @ S_prev
    attn_int = np.matmul(q * np.exp(G)[..., None], S_prev)       # (NV, C, V)

    # A = (q @ k^T) * D
    A = np.matmul(q, k.transpose(0, 2, 1)) * D                   # (NV, C, C)

    # O = attn_int + A @ v_new
    O = attn_int + np.matmul(A, v_new)                           # (NV, C, V)

    # S_new = exp(G[-1]) * S_prev + (k * exp(G[-1] - G)).T @ v_new
    expG_last = np.exp(G[:, -1])                                 # (NV,)
    decay_factor = np.exp(G[:, -1:] - G)                          # (NV, C)
    k_scaled = k * decay_factor[..., None]                        # (NV, C, K)
    S_new = expG_last[:, None, None] * S_prev \
            + np.matmul(k_scaled.transpose(0, 2, 1), v_new)       # (NV, K, V)

    # Transpose O back to (C, NV, V)
    out_seq = O.transpose(1, 0, 2)                                # (C, NV, V)
    return out_seq, S_new


def main():
    print("=" * 64)
    print("v4 Stage 4 math validation — numpy chunked vs per-position")
    print("=" * 64)

    rng = np.random.default_rng(2026_05_20)
    C = 8   # chunk size — small for fast probe
    NV = NV_PER_CHIP
    KD = K_DIM
    VD = V_DIM

    # Synthetic inputs at small but realistic shapes
    q_seq = rng.standard_normal((C, NV, KD)).astype(np.float32) * 0.05
    k_seq = rng.standard_normal((C, NV, KD)).astype(np.float32) * 0.05
    v_seq = rng.standard_normal((C, NV, VD)).astype(np.float32) * 0.05

    # g: negative reals (decay), so exp(g) ∈ (0, 1]
    # In Qwen3.6: g = -exp(A_log) * softplus(a + dt_bias), typically ~ [-1, 0]
    g_seq = -np.abs(rng.standard_normal((C, NV))).astype(np.float32) * 0.5

    # beta: sigmoid output, in (0, 1)
    beta_logits = rng.standard_normal((C, NV)).astype(np.float32) * 0.5
    beta_seq = 1.0 / (1.0 + np.exp(-beta_logits))

    # S_init: zeros (fresh state) — most common case in production
    S_init = np.zeros((NV, KD, VD), dtype=np.float32)

    print(f"\nInputs:")
    print(f"  q_seq:    {q_seq.shape}  range=[{q_seq.min():.4f}, {q_seq.max():.4f}]")
    print(f"  k_seq:    {k_seq.shape}")
    print(f"  v_seq:    {v_seq.shape}")
    print(f"  g_seq:    {g_seq.shape}  exp(g) range=[{np.exp(g_seq).min():.4f}, {np.exp(g_seq).max():.4f}]")
    print(f"  beta_seq: {beta_seq.shape}  range=[{beta_seq.min():.4f}, {beta_seq.max():.4f}]")
    print(f"  S_init:   {S_init.shape}  (zeros)")

    # Run per-position reference
    print(f"\n[per-position] running C={C} sequential steps...")
    out_pp, S_pp = per_position_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init)
    print(f"  out_seq: {out_pp.shape}  range=[{out_pp.min():.6f}, {out_pp.max():.6f}]")
    print(f"  S_final: norm={np.linalg.norm(S_pp):.4f}")

    # Run chunked
    print(f"\n[chunked Neumann] running C={C} chunked recurrence...")
    out_ck, S_ck = chunked_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init)
    print(f"  out_seq: {out_ck.shape}  range=[{out_ck.min():.6f}, {out_ck.max():.6f}]")
    print(f"  S_final: norm={np.linalg.norm(S_ck):.4f}")

    # Compare
    print("\n" + "=" * 64)
    print("Comparison")
    print("=" * 64)
    cos_out = _cos(out_pp, out_ck)
    max_abs_out = float(np.abs(out_pp - out_ck).max())
    cos_state = _cos(S_pp, S_ck)
    max_abs_state = float(np.abs(S_pp - S_ck).max())

    print(f"\noutput:")
    print(f"  cos(per_pos, chunked) = {cos_out:.10f}")
    print(f"  max|Δ|                 = {max_abs_out:.6e}")
    print(f"  (gate: cos ≥ 0.9999)   {'PASS' if cos_out >= 0.9999 else 'FAIL'}")

    print(f"\nfinal state:")
    print(f"  cos(per_pos, chunked) = {cos_state:.10f}")
    print(f"  max|Δ|                 = {max_abs_state:.6e}")
    print(f"  (gate: cos ≥ 0.9999)   {'PASS' if cos_state >= 0.9999 else 'FAIL'}")

    # Test 2: non-zero initial state (more representative of mid-prefill chunk)
    print("\n" + "=" * 64)
    print("Test 2: non-zero initial state (mid-prefill chunk)")
    print("=" * 64)
    S_init_random = rng.standard_normal((NV, KD, VD)).astype(np.float32) * 0.01

    out_pp2, S_pp2 = per_position_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init_random)
    out_ck2, S_ck2 = chunked_recurrence(q_seq, k_seq, v_seq, g_seq, beta_seq, S_init_random)
    cos_out2 = _cos(out_pp2, out_ck2)
    max_abs_out2 = float(np.abs(out_pp2 - out_ck2).max())
    cos_state2 = _cos(S_pp2, S_ck2)
    max_abs_state2 = float(np.abs(S_pp2 - S_ck2).max())
    print(f"\noutput:  cos={cos_out2:.10f}  max|Δ|={max_abs_out2:.6e}  "
          f"{'PASS' if cos_out2 >= 0.9999 else 'FAIL'}")
    print(f"state:   cos={cos_state2:.10f}  max|Δ|={max_abs_state2:.6e}  "
          f"{'PASS' if cos_state2 >= 0.9999 else 'FAIL'}")

    # Verdict
    print("\n" + "=" * 64)
    print("Verdict")
    print("=" * 64)
    all_pass = (cos_out >= 0.9999 and cos_state >= 0.9999
                and cos_out2 >= 0.9999 and cos_state2 >= 0.9999)
    if all_pass:
        print("✓ Math is correct. ttnn integration can proceed.")
        print("  Per-position == chunked Neumann at fp64 precision in both")
        print("  zero-state and non-zero-state cases.")
    else:
        print("✗ Math has a bug. Debug before attempting ttnn integration.")


if __name__ == "__main__":
    main()
