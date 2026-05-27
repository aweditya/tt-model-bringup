#!/usr/bin/env python3
"""Isolation test for qwen36_moe_ffn_decode_owned.

G0 (current): smoke-test the scaffold. The compute kernel runs IDENTITY
(copy h to output, one tile at a time), so this test:
  1. Pre-fills the output buffer with a non-zero sentinel.
  2. Calls the kernel.
  3. Asserts output equals h within bf16 noise (pcc > 0.9999) — proves
     the reader→compute→writer pipeline plumbs data correctly AND the
     sentinel was overwritten (kernel actually wrote tiles, didn't no-op).

G1+: switches compute kernel to the real fused FFN chain; this same test
file gates against `moe_ffn_kernel_oracle.moe_ffn_oracle` instead.

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


def to_ttnn(arr_np, device, dtype, layout):
    import ttnn
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=dtype, layout=layout, device=device,
    )


def to_host(t):
    import ttnn
    return ttnn.to_torch(t).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64,
                    help="HIDDEN dim. G0 uses tiny shapes (multiples of TILE=32).")
    ap.add_argument("--moe-inter", type=int, default=32)
    ap.add_argument("--experts", type=int, default=2)
    args = ap.parse_args()

    if args.hidden % 32 != 0:
        raise SystemExit(f"--hidden must be a multiple of TILE=32; got {args.hidden}")
    if args.moe_inter % 32 != 0:
        raise SystemExit(f"--moe-inter must be a multiple of TILE=32; got {args.moe_inter}")

    import ttnn

    if not hasattr(ttnn.experimental, "qwen36_moe_ffn_decode_owned"):
        raise SystemExit("ttnn.experimental.qwen36_moe_ffn_decode_owned not registered "
                         "in this ttnn build")

    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        H = args.hidden
        I = args.moe_inter
        E = args.experts

        log(f"shapes: h=[1,{H}] W1=[{E},{H},{2*I}] W2=[{E},{I},{H}] rw=[1,{E}]")

        # Build small fixtures via the oracle so the shapes are exactly right.
        fx = oracle.make_fixture(E=E, HIDDEN=H, MOE_INTER=I, seed=0)

        # Upload h as rank-2 [1, HIDDEN].
        h_tt = to_ttnn(
            fx.h.reshape(1, H), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Upload W1 / W2 as rank-3 [E, *, *].
        W1_tt = to_ttnn(fx.W1, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        W2_tt = to_ttnn(fx.W2, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Routing weight as rank-2 [1, E] (will pad to [1, TILE] in TILE_LAYOUT).
        rw_tt = to_ttnn(
            fx.routing_weight.reshape(1, E), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        # D-G0-05 mitigation: pre-allocate output with non-zero sentinel,
        # pass via output_tensor, then assert all-zero after kernel call.
        sentinel_value = 7.0
        sentinel_np = np.full((1, H), sentinel_value, dtype=np.float32)
        output_pre = to_ttnn(sentinel_np, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Sanity: confirm the sentinel survived the upload round-trip.
        pre_host = to_host(output_pre)[0]
        log(f"pre-call sentinel: min={pre_host.min():.3f} max={pre_host.max():.3f} "
            f"(expect ~{sentinel_value})")

        log("calling qwen36_moe_ffn_decode_owned…")
        out = ttnn.experimental.qwen36_moe_ffn_decode_owned(
            h_tt, W1_tt, W2_tt, rw_tt,
            output_tensor=output_pre,
        )
        ttnn.synchronize_device(device)
        out_np = to_host(out)[0]

        log(f"post-call output:  min={out_np.min():.6f} max={out_np.max():.6f} "
            f"norm={np.linalg.norm(out_np):.6f}")

        # G0 gate (identity compute): output should equal h within bf16 noise,
        # AND must differ from the sentinel (so we know the kernel actually wrote).
        if np.allclose(out_np, sentinel_value, atol=1e-3):
            log(f"FAIL: output equals sentinel {sentinel_value} — kernel never wrote.")
            raise SystemExit(1)

        h_bf16 = torch.from_numpy(fx.h.reshape(1, H).astype(np.float32))\
            .to(torch.bfloat16).float().numpy()[0]
        diff = float(np.abs(out_np - h_bf16).max())
        pcc = oracle.pcc(out_np, h_bf16)
        log(f"G0 identity check: pcc(out, h_bf16) = {pcc:.8f}  "
            f"max_abs_diff = {diff:.6e}")
        if pcc < 0.9999:
            log(f"FAIL: identity compute didn't reproduce h (pcc {pcc:.6f} < 0.9999).")
            log(f"  out_np[:8] = {out_np[:8].tolist()}")
            log(f"  h_bf16[:8] = {h_bf16[:8].tolist()}")
            raise SystemExit(1)

        log(f"PASS  G0 smoke: pipeline plumbs h through identity compute to output.")
        log("Next: G1 swaps compute for the real fused FFN chain; same test "
            "harness will gate against the numpy oracle.")

        for t in [h_tt, W1_tt, W2_tt, rw_tt, output_pre, out]:
            try: ttnn.deallocate(t)
            except Exception: pass
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
