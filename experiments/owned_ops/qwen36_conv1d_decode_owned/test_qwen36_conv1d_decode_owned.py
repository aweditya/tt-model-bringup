#!/usr/bin/env python3
"""Standalone correctness test for ttnn.experimental.qwen36_conv1d_decode_owned.

Run on qb2 (or qb1) after `integrate_into_ttmetal.py` has installed the op
and ttnn has been rebuilt:

    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/owned_ops/qwen36_conv1d_decode_owned/test_qwen36_conv1d_decode_owned.py \\
        --d 32 --debug-fill'

Two modes:
- `--debug-fill`: kernel writes mixed -> out (no real math). Verifies the
  scaffold compiles, links, dispatches, and the writer's state shift
  works without arithmetic interference.
- default: full math. Compares against a numpy oracle. BF16-native gate.

Sync-bounded timing. Line-buffered stdout for SSH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def numpy_oracle(mixed, state0, state1, state2, w0, w1, w2, w3):
    """Reference numpy implementation of the same math the kernel computes."""
    pre_act = state0 * w0 + state1 * w1 + state2 * w2 + mixed * w3
    out = pre_act / (1.0 + np.exp(-pre_act))  # silu
    state0_next = state1.copy()
    state1_next = state2.copy()
    state2_next = mixed.copy()
    return out, state0_next, state1_next, state2_next


def make_tap_tensor(np_arr, ttnn, torch, device, dtype):
    """Build a [D, 1]-logical, [D, 32]-padded TILE_LAYOUT tensor.
    Pass logical shape [D, 1] to ttnn.from_torch; TILE_LAYOUT auto-pads the
    column dim to 32 — the kernel's validation expects exactly this layout.
    """
    assert np_arr.ndim == 1, f"tap must be 1-D, got shape {np_arr.shape}"
    D = np_arr.shape[0]
    assert D % 32 == 0, f"D={D} must be a multiple of 32 for TILE padding"
    logical = np_arr.reshape(D, 1).astype(np.float32)
    return ttnn.from_torch(
        torch.from_numpy(logical),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def readback_tap(tensor, ttnn, D):
    """Read a [D, 1] logical / [D, 32] padded tile-padded tensor as numpy [D]."""
    arr = ttnn.to_torch(tensor).float().cpu().numpy()
    # to_torch returns logical shape [D, 1]; squeeze the singleton column.
    assert arr.shape == (D, 1), f"unexpected readback shape {arr.shape}, expected ({D}, 1)"
    return arr[:, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d", type=int, default=32,
                         help="D dimension (must be multiple of 32). 32 = 1 tile.")
    parser.add_argument("--debug-fill", action="store_true",
                         help="Use kernel debug-fill mode (out = mixed, no real math).")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.01,
                         help="BF16-native gate threshold. 0.01 ≈ 1.3 BF16 ULPs at magnitude 1, "
                              "accommodates production-shape D=2560 max-element drift. PCC is the "
                              "primary correctness signal; max_abs_diff is a secondary sanity check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    D = args.d
    assert D > 0 and D % 32 == 0, f"D={D} must be a positive multiple of 32"

    # Synthetic inputs: small bounded random.
    mixed_np = rng.uniform(-0.5, 0.5, size=D).astype(np.float32)
    state0_np = rng.uniform(-0.5, 0.5, size=D).astype(np.float32)
    state1_np = rng.uniform(-0.5, 0.5, size=D).astype(np.float32)
    state2_np = rng.uniform(-0.5, 0.5, size=D).astype(np.float32)
    w0_np = rng.uniform(-1.0, 1.0, size=D).astype(np.float32)
    w1_np = rng.uniform(-1.0, 1.0, size=D).astype(np.float32)
    w2_np = rng.uniform(-1.0, 1.0, size=D).astype(np.float32)
    w3_np = rng.uniform(-1.0, 1.0, size=D).astype(np.float32)

    oracle_out, oracle_s0_next, oracle_s1_next, oracle_s2_next = numpy_oracle(
        mixed_np, state0_np, state1_np, state2_np, w0_np, w1_np, w2_np, w3_np)

    import torch  # noqa: E402  (delayed import — keeps script importable for syntax check)
    import ttnn   # noqa: E402

    dtype_map = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    ttnn_dtype = dtype_map[args.dtype]

    device = ttnn.open_device(device_id=args.device_id)

    try:
        mixed_tt = make_tap_tensor(mixed_np, ttnn, torch, device, ttnn_dtype)
        state0_tt = make_tap_tensor(state0_np, ttnn, torch, device, ttnn_dtype)
        state1_tt = make_tap_tensor(state1_np, ttnn, torch, device, ttnn_dtype)
        state2_tt = make_tap_tensor(state2_np, ttnn, torch, device, ttnn_dtype)
        w0_tt = make_tap_tensor(w0_np, ttnn, torch, device, ttnn_dtype)
        w1_tt = make_tap_tensor(w1_np, ttnn, torch, device, ttnn_dtype)
        w2_tt = make_tap_tensor(w2_np, ttnn, torch, device, ttnn_dtype)
        w3_tt = make_tap_tensor(w3_np, ttnn, torch, device, ttnn_dtype)

        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        s0_out, s1_out, s2_out, out_tt = ttnn.experimental.qwen36_conv1d_decode_owned(
            mixed_tt, state0_tt, state1_tt, state2_tt,
            w0_tt, w1_tt, w2_tt, w3_tt,
            debug_fill=args.debug_fill,
        )
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        kernel_out = readback_tap(out_tt, ttnn, D)
        kernel_s0 = readback_tap(s0_out, ttnn, D)
        kernel_s1 = readback_tap(s1_out, ttnn, D)
        kernel_s2 = readback_tap(s2_out, ttnn, D)

        if args.debug_fill:
            # In debug-fill mode the math path is skipped; out should equal mixed.
            expected_out = mixed_np
        else:
            expected_out = oracle_out

        out_max_diff = float(np.abs(kernel_out - expected_out).max())
        s0_max_diff = float(np.abs(kernel_s0 - oracle_s0_next).max())
        s1_max_diff = float(np.abs(kernel_s1 - oracle_s1_next).max())
        s2_max_diff = float(np.abs(kernel_s2 - oracle_s2_next).max())

        def pcc(a, b):
            am = a - a.mean(); bm = b - b.mean()
            denom = (np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-30
            return float(np.dot(am, bm) / denom)

        out_pcc = pcc(kernel_out, expected_out)
        s0_pcc = pcc(kernel_s0, oracle_s0_next)
        s1_pcc = pcc(kernel_s1, oracle_s1_next)
        s2_pcc = pcc(kernel_s2, oracle_s2_next)

        report = {
            "d": D,
            "dtype": args.dtype,
            "debug_fill": args.debug_fill,
            "elapsed_ms": elapsed_ms,
            "max_abs_diff_threshold": args.max_abs_diff_threshold,
            "out": {"max_abs_diff": out_max_diff, "pcc": out_pcc,
                     "pass": out_max_diff <= args.max_abs_diff_threshold},
            "state0_next": {"max_abs_diff": s0_max_diff, "pcc": s0_pcc,
                              "pass": s0_max_diff <= args.max_abs_diff_threshold},
            "state1_next": {"max_abs_diff": s1_max_diff, "pcc": s1_pcc,
                              "pass": s1_max_diff <= args.max_abs_diff_threshold},
            "state2_next": {"max_abs_diff": s2_max_diff, "pcc": s2_pcc,
                              "pass": s2_max_diff <= args.max_abs_diff_threshold},
        }
        report["pass_gate"] = all(report[k]["pass"] for k in
                                    ("out", "state0_next", "state1_next", "state2_next"))

        print(json.dumps(report, indent=2))
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2))
            print(f"[save] {args.output_json}")

        return 0 if report["pass_gate"] else 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    raise SystemExit(main())
