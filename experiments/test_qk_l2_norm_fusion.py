#!/usr/bin/env python3
"""Isolation test for QK L2-norm fusion via ttnn.rms_norm.

The DN forward currently does L2 normalization on Q and K as a 5-op
manual chain (per head_dim, with eps=1e-6):
    x_sq = x * x
    x_sumsq = sum(x_sq, dim=-1, keepdim=True)
    x_inv = rsqrt(x_sumsq + eps)
    x_n = x * x_inv

L2 norm:   y = x / sqrt(sum(x^2) + eps)
RMS norm:  y = x / sqrt(mean(x^2) + eps_rms) * w
        = x * sqrt(d) / sqrt(sum(x^2) + d*eps_rms) * w

So if w = 1/sqrt(d) and eps_rms = eps/d:
  y = x * sqrt(d) * (1/sqrt(d)) / sqrt(sum(x^2) + eps)
    = x / sqrt(sum(x^2) + eps)  = L2 norm ✓

This isolates the math swap and gates on PCC before integration.

Run on qb1 (single device suffices for the math check):
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/test_qk_l2_norm_fusion.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# Production-realistic per-chip shapes (35B-A3B DN block on (1,4)):
#   q: [1, NK_PER_CHIP=4, HEAD_K_DIM=128]
#   k: [1, NK_PER_CHIP=4, HEAD_K_DIM=128]
NK_PER_CHIP = 4
HEAD_K_DIM = 128
EPS = 1e-6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc(a, b):
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    af -= af.mean(); bf -= bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / denom) if denom > 0 else 1.0


def manual_l2_norm_ttnn(x_tt, device):
    """Existing manual chain — 5 ops."""
    import ttnn
    x_sq = ttnn.mul(x_tt, x_tt)
    x_sumsq = ttnn.sum(x_sq, dim=-1, keepdim=True)
    ttnn.deallocate(x_sq)
    x_inv = ttnn.rsqrt(ttnn.add(x_sumsq, EPS))
    ttnn.deallocate(x_sumsq)
    x_n = ttnn.mul(x_tt, x_inv)
    ttnn.deallocate(x_inv)
    return x_n


def fused_l2_norm_ttnn(x_tt, weight_tt, eps_rms):
    """Candidate: one rms_norm call.

    weight_tt is [HEAD_K_DIM] tile-padded with value 1/sqrt(d) so the
    rms_norm output IS the L2 norm:
        y = (x / sqrt(mean(x^2) + eps_rms)) * (1/sqrt(d))
          = x / (sqrt(d) * sqrt(mean(x^2) + eps_rms))
          = x / sqrt(d * mean(x^2) + d * eps_rms)
          = x / sqrt(sum(x^2) + d*eps_rms)
    Choosing eps_rms = eps/d makes the denominator sqrt(sum(x^2) + eps).
    """
    import ttnn
    return ttnn.rms_norm(x_tt, weight=weight_tt, epsilon=eps_rms)


def numpy_l2_norm(x_np, eps=EPS):
    """Reference — what both ttnn paths should produce."""
    sumsq = (x_np ** 2).sum(axis=-1, keepdims=True)
    return x_np / np.sqrt(sumsq + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcc-threshold", type=float, default=0.999)
    args = ap.parse_args()

    import ttnn

    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        rng = np.random.default_rng(args.seed)
        # Production-like Q/K magnitudes (post-silu, post-split). Values
        # around ~0.5 magnitude per element.
        q_np = rng.normal(0.0, 0.5, size=(1, NK_PER_CHIP, HEAD_K_DIM)).astype(np.float32)
        k_np = rng.normal(0.0, 0.5, size=(1, NK_PER_CHIP, HEAD_K_DIM)).astype(np.float32)
        log(f"q/k shape: [1, {NK_PER_CHIP}, {HEAD_K_DIM}]  bf16, TILE")

        # Upload as bf16 TILE_LAYOUT — matches dn_forward_ttnn's path.
        def to_tt(x):
            return ttnn.from_torch(torch.from_numpy(x),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        q_tt = to_tt(q_np)
        k_tt = to_tt(k_np)

        # Reference (fp32 numpy).
        q_ref = numpy_l2_norm(q_np)
        k_ref = numpy_l2_norm(k_np)

        # --- Manual chain ---
        log("manual chain (5 ops)…")
        q_man = manual_l2_norm_ttnn(q_tt, device)
        k_man = manual_l2_norm_ttnn(k_tt, device)
        q_man_np = ttnn.to_torch(q_man).float().numpy()
        k_man_np = ttnn.to_torch(k_man).float().numpy()
        log(f"  manual q vs numpy:  pcc={pcc(q_man_np, q_ref):.6f}  "
            f"max_abs_diff={np.max(np.abs(q_man_np-q_ref)):.4e}")
        log(f"  manual k vs numpy:  pcc={pcc(k_man_np, k_ref):.6f}  "
            f"max_abs_diff={np.max(np.abs(k_man_np-k_ref)):.4e}")
        ttnn.deallocate(q_man); ttnn.deallocate(k_man)

        # --- Fused rms_norm ---
        log("fused ttnn.rms_norm…")
        # Weight: [HEAD_K_DIM] filled with 1/sqrt(d). bf16 TILE.
        inv_sqrt_d = 1.0 / np.sqrt(HEAD_K_DIM)
        w_np = (np.ones(HEAD_K_DIM, dtype=np.float32) * inv_sqrt_d).reshape(1, HEAD_K_DIM)
        w_tt = ttnn.from_torch(torch.from_numpy(w_np),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        eps_rms = EPS / HEAD_K_DIM

        q_fused = fused_l2_norm_ttnn(q_tt, w_tt, eps_rms)
        k_fused = fused_l2_norm_ttnn(k_tt, w_tt, eps_rms)
        q_fused_np = ttnn.to_torch(q_fused).float().numpy()
        k_fused_np = ttnn.to_torch(k_fused).float().numpy()
        log(f"  fused q vs numpy:   pcc={pcc(q_fused_np, q_ref):.6f}  "
            f"max_abs_diff={np.max(np.abs(q_fused_np-q_ref)):.4e}")
        log(f"  fused k vs numpy:   pcc={pcc(k_fused_np, k_ref):.6f}  "
            f"max_abs_diff={np.max(np.abs(k_fused_np-k_ref)):.4e}")
        log(f"  fused q vs manual:  pcc={pcc(q_fused_np, q_man_np):.6f}  "
            f"max_abs_diff={np.max(np.abs(q_fused_np-q_man_np)):.4e}")
        log(f"  fused k vs manual:  pcc={pcc(k_fused_np, k_man_np):.6f}  "
            f"max_abs_diff={np.max(np.abs(k_fused_np-k_man_np)):.4e}")

        # Verdict.
        any_fail = False
        for name, p in [
            ("manual q", pcc(q_man_np, q_ref)),
            ("manual k", pcc(k_man_np, k_ref)),
            ("fused q",  pcc(q_fused_np, q_ref)),
            ("fused k",  pcc(k_fused_np, k_ref)),
        ]:
            if p < args.pcc_threshold:
                log(f"  FAIL: {name} pcc {p:.6f} < {args.pcc_threshold}")
                any_fail = True

        # --- Timing (eager, sync-bounded) ---
        log("timing manual chain vs fused (eager, 50 iters after 5 warmup)…")
        N_WARM, N_ITER = 5, 50
        def time_fn(fn):
            for _ in range(N_WARM):
                out = fn(); ttnn.synchronize_device(device); ttnn.deallocate(out)
            ts = []
            for _ in range(N_ITER):
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                out = fn()
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - t0) * 1000.0)
                ttnn.deallocate(out)
            return np.array(ts)
        ts_manual_q = time_fn(lambda: manual_l2_norm_ttnn(q_tt, device))
        ts_fused_q  = time_fn(lambda: fused_l2_norm_ttnn(q_tt, w_tt, eps_rms))
        log(f"  manual q: mean {ts_manual_q.mean():7.4f} ms  median {np.median(ts_manual_q):7.4f}")
        log(f"  fused  q: mean {ts_fused_q.mean():7.4f} ms  median {np.median(ts_fused_q):7.4f}")
        log(f"  speedup: {ts_manual_q.mean()/ts_fused_q.mean():.2f}x  "
            f"savings per (Q+K): {2*(ts_manual_q.mean()-ts_fused_q.mean()):.4f} ms/call")
        log(f"  per-token (x30 DN layers): {2*30*(ts_manual_q.mean()-ts_fused_q.mean()):.2f} ms/tok eager")

        ttnn.deallocate(q_fused); ttnn.deallocate(k_fused)
        ttnn.deallocate(q_tt); ttnn.deallocate(k_tt); ttnn.deallocate(w_tt)
        if any_fail:
            raise SystemExit(1)
        log("PASS: both paths PCC >= threshold; fused candidate is correctness-safe.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
