#!/usr/bin/env python3
"""Isolation test for MoE router: softmax-then-topk vs topk-then-softmax.

Current production (4 ops):
  probs = softmax(logits, dim=-1)         # softmax over 256
  top_vals, top_idxs = topk(probs, K)     # top-8 of 256
  sum_v = sum(top_vals, keepdim=True)
  weights = top_vals / sum_v

Math-equivalent candidate (2 ops):
  top_logits, top_idxs = topk(logits, K)  # top-8 of 256
  weights = softmax(top_logits, dim=-1)   # softmax over 8

Why equivalent:
  top_idxs(softmax(x)) == top_idxs(x)  (softmax monotonic)
  softmax(x)[i] / sum_topk(softmax(x)) = exp(x[i]) / sum_topk(exp(x))
                                       = softmax(x_topk)[i]
  → both produce the same (top_idxs, weights) pair.

Smaller softmax (over 8 vs 256) is cheaper; drops 2 ops (sum + div).

Run on qb1 (single device suffices):
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/test_moe_router_topk_reorder.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


NUM_EXPERTS = 256
TOP_K = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc(a, b):
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    af -= af.mean(); bf -= bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / denom) if denom > 0 else 1.0


def current_path(logits_tt):
    """Production: softmax(256) -> topk -> sum -> div."""
    import ttnn
    probs = ttnn.softmax(logits_tt, dim=-1)
    top_vals, top_idxs = ttnn.topk(probs, k=TOP_K, dim=-1)
    ttnn.deallocate(probs)
    sum_v = ttnn.sum(top_vals, dim=-1, keepdim=True)
    weights = ttnn.div(top_vals, sum_v)
    ttnn.deallocate(top_vals); ttnn.deallocate(sum_v)
    return top_idxs, weights


def candidate_path(logits_tt):
    """Candidate: topk(logits) -> softmax(K)."""
    import ttnn
    top_logits, top_idxs = ttnn.topk(logits_tt, k=TOP_K, dim=-1)
    weights = ttnn.softmax(top_logits, dim=-1)
    ttnn.deallocate(top_logits)
    return top_idxs, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcc-threshold", type=float, default=0.999)
    args = ap.parse_args()

    import ttnn

    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        # Production-realistic router logits: [1, NUM_EXPERTS] bf16 TILE.
        rng = np.random.default_rng(args.seed)
        logits_np = rng.normal(0.0, 1.0, size=(1, NUM_EXPERTS)).astype(np.float32)
        logits_tt = ttnn.from_torch(
            torch.from_numpy(logits_np),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        # ---- Numpy reference ----
        # Use the production semantics: softmax(logits) -> topk -> normalize.
        probs_np = np.exp(logits_np) / np.exp(logits_np).sum(axis=-1, keepdims=True)
        idxs_np = np.argsort(-probs_np, axis=-1)[:, :TOP_K]  # descending
        vals_np = np.take_along_axis(probs_np, idxs_np, axis=-1)
        weights_np = vals_np / vals_np.sum(axis=-1, keepdims=True)
        log(f"numpy reference: idxs[0,:4]={idxs_np[0,:4].tolist()}  "
            f"weights[0,:4]={weights_np[0,:4].round(4).tolist()}")

        # ---- Production path ----
        idxs_a, w_a = current_path(logits_tt)
        idxs_a_np = ttnn.to_torch(idxs_a).int().numpy()
        w_a_np = ttnn.to_torch(w_a).float().numpy()
        log(f"current     :  idxs[0,:4]={idxs_a_np[0,:4].tolist()}  "
            f"weights[0,:4]={w_a_np[0,:4].round(4).tolist()}")
        ttnn.deallocate(idxs_a); ttnn.deallocate(w_a)

        # ---- Candidate path ----
        idxs_b, w_b = candidate_path(logits_tt)
        idxs_b_np = ttnn.to_torch(idxs_b).int().numpy()
        w_b_np = ttnn.to_torch(w_b).float().numpy()
        log(f"candidate   :  idxs[0,:4]={idxs_b_np[0,:4].tolist()}  "
            f"weights[0,:4]={w_b_np[0,:4].round(4).tolist()}")
        ttnn.deallocate(idxs_b); ttnn.deallocate(w_b)

        # ---- Correctness ----
        # top-K idx set must match (order can differ when probabilities are
        # tied; with random logits at NUM_EXPERTS=256 ties are unlikely).
        set_a = set(idxs_a_np[0].tolist())
        set_b = set(idxs_b_np[0].tolist())
        set_ref = set(idxs_np[0].tolist())
        log(f"idx set match candidate vs numpy: {len(set_b & set_ref)}/{TOP_K}")
        log(f"idx set match current   vs numpy: {len(set_a & set_ref)}/{TOP_K}")
        log(f"pcc(current weights, numpy)   = {pcc(w_a_np, weights_np):.6f}")
        log(f"pcc(candidate weights, numpy) = {pcc(w_b_np, weights_np):.6f}")
        # Compare current vs candidate weights — need to align by idx since
        # the ORDER of top_idxs from ttnn.topk can differ between paths.
        # Build a lookup: for each idx, what's the weight?
        def to_dict(idxs, ws):
            return dict(zip(idxs[0].tolist(), ws[0].tolist()))
        d_a = to_dict(idxs_a_np, w_a_np)
        d_b = to_dict(idxs_b_np, w_b_np)
        common = set(d_a) & set(d_b)
        if common:
            w_a_aligned = np.array([d_a[i] for i in sorted(common)])
            w_b_aligned = np.array([d_b[i] for i in sorted(common)])
            log(f"pcc(current vs candidate, aligned by idx) = "
                f"{pcc(w_a_aligned, w_b_aligned):.6f}  "
                f"max_abs_diff = {np.max(np.abs(w_a_aligned - w_b_aligned)):.4e}  "
                f"({len(common)} common idxs of {TOP_K})")

        # ---- Timing ----
        log(f"\ntiming current vs candidate (warmup {args.n_warmup}, "
            f"iters {args.n_iters}, sync-bounded)…")
        def time_fn(fn):
            for _ in range(args.n_warmup):
                a, b = fn(logits_tt)
                ttnn.synchronize_device(device)
                ttnn.deallocate(a); ttnn.deallocate(b)
            ts = []
            for _ in range(args.n_iters):
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                a, b = fn(logits_tt)
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - t0) * 1000.0)
                ttnn.deallocate(a); ttnn.deallocate(b)
            return np.array(ts)
        ts_a = time_fn(current_path)
        ts_b = time_fn(candidate_path)
        log(f"  current   : mean {ts_a.mean():7.4f}  median {np.median(ts_a):7.4f}  std {ts_a.std():7.4f}")
        log(f"  candidate : mean {ts_b.mean():7.4f}  median {np.median(ts_b):7.4f}  std {ts_b.std():7.4f}")
        log(f"  speedup {ts_a.mean()/ts_b.mean():.2f}x  savings {ts_a.mean()-ts_b.mean():.4f} ms/call")
        log(f"  per-token (x40 layers): {40*(ts_a.mean()-ts_b.mean()):.2f} ms/tok eager")

        # Verdict.
        idx_overlap_pass = len(set_b & set_ref) >= TOP_K - 1
        weight_pcc_ok = pcc(w_b_np, weights_np) >= args.pcc_threshold
        if not (idx_overlap_pass and weight_pcc_ok):
            log("FAIL: idx overlap or weight pcc below threshold")
            raise SystemExit(1)
        log("PASS: candidate matches reference (idx set + weight pcc).")
        ttnn.deallocate(logits_tt)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
