#!/usr/bin/env python3
"""Standalone correctness test for ttnn.experimental.qwen36_decay_gate_decode_owned.

Run on qb2 (after install + ttnn rebuild + .so sync):

    ssh qb2 'cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_tracy_gcc12_nodist \\
        ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python \\
        experiments/owned_ops/qwen36_decay_gate_decode_owned/test_qwen36_decay_gate_decode_owned.py \\
        --nv 12 --debug-fill'

Two modes:
- --debug-fill: kernel emits a -> decay, b -> beta (no real math).
- default: full chain. Compares against numpy oracle.

Sync-bounded timing. Line-buffered stdout for SSH.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def numpy_oracle(a, b, dt_bias, A_log):
    """Reference: matches server_tp.py:681-690 production-default manual chain."""
    a_biased = a + dt_bias
    softplus_a = np.log(np.exp(a_biased) + 1.0)
    g = -np.exp(A_log) * softplus_a
    decay = np.exp(g)
    beta = 1.0 / (1.0 + np.exp(-b))
    return decay, beta


def make_row_tensor(np_arr, ttnn, torch, device, dtype):
    """Build a [1, NV]-logical, [1, 32]-padded TILE_LAYOUT tensor.
    Pass logical shape [1, NV] to ttnn.from_torch; TILE_LAYOUT auto-pads to
    [32, 32] (row 0 holds the real NV values, rest is padding/zeros).
    """
    assert np_arr.ndim == 1, f"row tensor must be 1-D, got shape {np_arr.shape}"
    NV = np_arr.shape[0]
    assert NV > 0 and NV <= 32, f"NV={NV} must be in (0, 32]"
    logical = np_arr.reshape(1, NV).astype(np.float32)
    return ttnn.from_torch(
        torch.from_numpy(logical),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def readback_row(tensor, ttnn, NV):
    arr = ttnn.to_torch(tensor).float().cpu().numpy()
    assert arr.shape == (1, NV), f"unexpected readback shape {arr.shape}, expected (1, {NV})"
    return arr[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nv", type=int, default=12,
                         help="NV_PER_CHIP (must be in (0, 32]). Default 12 = qb2 TP4 production.")
    parser.add_argument("--debug-fill", action="store_true",
                         help="Use kernel debug-fill mode (decay = a, beta = b copy).")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.01,
                         help="BF16-native gate threshold. PCC is primary correctness signal.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    NV = args.nv
    assert 0 < NV <= 32

    # Inputs in production-realistic ranges:
    # a, b: pre-norm activations, ~[-1, 1]
    # dt_bias: small bias, ~[-0.5, 0.5]
    # A_log: log of decay rate, typically negative-ish (decay rates ~ 0.5-1.5)
    a_np = rng.uniform(-1.0, 1.0, size=NV).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=NV).astype(np.float32)
    dt_bias_np = rng.uniform(-0.5, 0.5, size=NV).astype(np.float32)
    A_log_np = rng.uniform(-2.0, 0.5, size=NV).astype(np.float32)

    oracle_decay, oracle_beta = numpy_oracle(a_np, b_np, dt_bias_np, A_log_np)

    import torch  # noqa: E402
    import ttnn   # noqa: E402

    dtype_map = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    ttnn_dtype = dtype_map[args.dtype]
    device = ttnn.open_device(device_id=args.device_id)

    try:
        a_tt = make_row_tensor(a_np, ttnn, torch, device, ttnn_dtype)
        b_tt = make_row_tensor(b_np, ttnn, torch, device, ttnn_dtype)
        dt_bias_tt = make_row_tensor(dt_bias_np, ttnn, torch, device, ttnn_dtype)
        A_log_tt = make_row_tensor(A_log_np, ttnn, torch, device, ttnn_dtype)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        decay_tt, beta_tt = ttnn.experimental.qwen36_decay_gate_decode_owned(
            a_tt, b_tt, dt_bias_tt, A_log_tt, debug_fill=args.debug_fill)
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        kernel_decay = readback_row(decay_tt, ttnn, NV)
        kernel_beta = readback_row(beta_tt, ttnn, NV)

        if args.debug_fill:
            expected_decay = a_np
            expected_beta = b_np
        else:
            expected_decay = oracle_decay
            expected_beta = oracle_beta

        decay_max_diff = float(np.abs(kernel_decay - expected_decay).max())
        beta_max_diff = float(np.abs(kernel_beta - expected_beta).max())

        def pcc(a, b):
            am = a - a.mean(); bm = b - b.mean()
            denom = (np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-30
            return float(np.dot(am, bm) / denom)

        decay_pcc = pcc(kernel_decay, expected_decay)
        beta_pcc = pcc(kernel_beta, expected_beta)

        report = {
            "nv": NV,
            "dtype": args.dtype,
            "debug_fill": args.debug_fill,
            "elapsed_ms": elapsed_ms,
            "max_abs_diff_threshold": args.max_abs_diff_threshold,
            "decay": {"max_abs_diff": decay_max_diff, "pcc": decay_pcc,
                       "pass": decay_max_diff <= args.max_abs_diff_threshold},
            "beta": {"max_abs_diff": beta_max_diff, "pcc": beta_pcc,
                      "pass": beta_max_diff <= args.max_abs_diff_threshold},
            "kernel_decay_first_n": kernel_decay.tolist(),
            "oracle_decay_first_n": oracle_decay.tolist(),
            "kernel_beta_first_n": kernel_beta.tolist(),
            "oracle_beta_first_n": oracle_beta.tolist(),
        }
        report["pass_gate"] = report["decay"]["pass"] and report["beta"]["pass"]

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
