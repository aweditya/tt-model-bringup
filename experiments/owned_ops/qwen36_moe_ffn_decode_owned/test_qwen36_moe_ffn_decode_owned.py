#!/usr/bin/env python3
"""Isolation test for qwen36_moe_ffn_decode_owned.

G1b: gates against the full chain (h @ W1 -> silu*up -> @ W2 -> rw[e] *
expert_out[e] summed over experts). routing_weight is pre-broadcast to
shape [E, TILE, TILE] (D-G1b-01) by the caller; the kernel consumes one
TILE×TILE tile per expert and multiplies eo[j] by it. Compares the
kernel's [1, HIDDEN] output to the numpy oracle.

Run on a TT host with the kernel integrated into ttnn:
  cd ~/tt-xla
  .venv/bin/python -u experiments/owned_ops/qwen36_moe_ffn_decode_owned/test_qwen36_moe_ffn_decode_owned.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import moe_ffn_kernel_oracle as oracle  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_ttnn(arr_np, device, dtype=None, layout=None):
    import ttnn
    if dtype is None: dtype = ttnn.bfloat16
    if layout is None: layout = ttnn.TILE_LAYOUT
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr_np.astype(np.float32))),
        dtype=dtype, layout=layout, device=device,
    )


def to_host(t):
    import ttnn
    return ttnn.to_torch(t).float().numpy()


TILE = 32


def broadcast_rw_to_tiles(rw_1d):
    """Pre-broadcast a [E] rw vector to [E, TILE, TILE] (D-G1b-01).

    The kernel consumes a single TILE×TILE tile per expert as the
    scalar-broadcast operand for the eo[j] * rw[e] mul. Caller is
    responsible for the broadcast (LLK-level row-scalar bcast would
    let us drop this, see deferral list).
    """
    rw = rw_1d.astype(np.float32).reshape(-1, 1, 1)
    return np.broadcast_to(rw, (rw.shape[0], TILE, TILE)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--moe-inter", type=int, default=32)
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    args = ap.parse_args()

    if args.hidden % 32 != 0 or args.moe_inter % 32 != 0:
        raise SystemExit("--hidden and --moe-inter must be multiples of TILE=32")

    import ttnn
    if not hasattr(ttnn.experimental, "qwen36_moe_ffn_decode_owned"):
        raise SystemExit("ttnn.experimental.qwen36_moe_ffn_decode_owned not registered")

    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        H, I, E = args.hidden, args.moe_inter, args.experts
        log(f"shapes: h=[1,{H}] W1=[{E},{H},{2*I}] W2=[{E},{I},{H}] rw_bcast=[{E},{TILE},{TILE}]")

        fx = oracle.make_fixture(E=E, HIDDEN=H, MOE_INTER=I, seed=args.seed)
        expected = oracle.moe_ffn_oracle(fx, bf16=True)

        # Upload inputs. rw is pre-broadcast to [E, TILE, TILE] (D-G1b-01).
        rw_bcast = broadcast_rw_to_tiles(fx.routing_weight)
        h_tt  = to_ttnn(fx.h.reshape(1, H), device)
        W1_tt = to_ttnn(fx.W1, device)
        W2_tt = to_ttnn(fx.W2, device)
        rw_tt = to_ttnn(rw_bcast, device)

        log("calling qwen36_moe_ffn_decode_owned…")
        t0 = time.time()
        out = ttnn.experimental.qwen36_moe_ffn_decode_owned(h_tt, W1_tt, W2_tt, rw_tt)
        ttnn.synchronize_device(device)
        dt_ms = (time.time() - t0) * 1000.0
        out_np = to_host(out)[0]
        log(f"kernel returned in {dt_ms:.2f} ms (single-call, includes JIT cache lookup)")
        log(f"output:    min={out_np.min():+.6f}  max={out_np.max():+.6f}  norm={np.linalg.norm(out_np):.6f}")
        log(f"expected:  min={expected.min():+.6f}  max={expected.max():+.6f}  norm={np.linalg.norm(expected):.6f}")

        pcc_val = oracle.pcc(out_np, expected)
        diff = oracle.max_abs_diff(out_np, expected)
        log(f"pcc(out, oracle) = {pcc_val:.8f}  max_abs_diff = {diff:.6e}")

        if pcc_val < args.pcc_threshold:
            log(f"FAIL: pcc {pcc_val:.6f} < threshold {args.pcc_threshold}")
            log(f"  out_np[:8]   = {out_np[:8].tolist()}")
            log(f"  expected[:8] = {expected[:8].tolist()}")
            raise SystemExit(1)

        log(f"PASS  G1b: kernel matches oracle (rw applied, pcc={pcc_val:.6f}).")

        for t in [h_tt, W1_tt, W2_tt, rw_tt, out]:
            try: ttnn.deallocate(t)
            except Exception: pass
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
