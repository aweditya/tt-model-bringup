#!/usr/bin/env python3
"""Device correctness gate for the owned qwen36_gdn_outer_update op.

Run this only on a TT host after the op has been integrated into that host's
TTNN build. Do not run it locally and do not run it while the persistent
inference server owns the chips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.utils import gdn_kernel_oracle as oracle  # noqa: E402


def _load_or_make_fixture(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.fixture_npz is not None:
        with np.load(args.fixture_npz) as npz:
            return {
                "state": npz["state"].astype(np.float32),
                "alpha": npz["alpha"].astype(np.float32),
                "k": npz["k"].astype(np.float32),
                "value": npz["value"].astype(np.float32),
                "beta": npz["beta"].astype(np.float32),
            }
    fixture = oracle.make_fixture(
        slots=args.slots,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        seed=args.seed,
        scale=args.scale,
    )
    return {
        "state": fixture["state"],
        "alpha": fixture["alpha"],
        "k": fixture["k"],
        "value": fixture["value"],
        "beta": fixture["beta"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _array_stats(value: np.ndarray) -> dict[str, Any]:
    value32 = np.asarray(value, dtype=np.float32)
    flat = value32.reshape(-1)
    return {
        "min": float(np.min(flat)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "norm": float(np.linalg.norm(flat.astype(np.float64))) if flat.size else 0.0,
        "first_16": [float(x) for x in flat[:16]],
    }


def _to_bfloat16_numpy(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(torch.bfloat16).float().cpu().numpy()


def _k_col_tiles(k: np.ndarray) -> np.ndarray:
    return np.broadcast_to(k[:, :, None], (k.shape[0], k.shape[1], 32)).astype(np.float32, copy=True)


def _delta_rows(delta_value: np.ndarray) -> np.ndarray:
    return np.broadcast_to(
        delta_value[:, None, :],
        (delta_value.shape[0], 32, delta_value.shape[1]),
    ).astype(np.float32, copy=True)


def _native_expected(
    expected: np.ndarray,
    state_scaled: np.ndarray,
    k_col: np.ndarray,
    delta_rows: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.dtype == "float32" or args.oracle_mode == "fp32":
        return expected.astype(np.float32, copy=False)
    if args.debug_fill:
        return expected.astype(np.float32, copy=False)
    state_bf16 = _to_bfloat16_numpy(state_scaled)
    k_col_bf16 = _to_bfloat16_numpy(k_col)
    delta_bf16 = _to_bfloat16_numpy(delta_rows)
    k_full = np.broadcast_to(k_col_bf16[:, :, :1], state_bf16.shape).astype(np.float32, copy=True)
    delta_full = np.broadcast_to(delta_bf16[:, :1, :], state_bf16.shape).astype(np.float32, copy=True)
    return _to_bfloat16_numpy(state_bf16 + _to_bfloat16_numpy(k_full * delta_full))


def _progress(message: str) -> None:
    print(f"[qwen36_gdn_outer_update] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--debug-fill", action="store_true")
    parser.add_argument("--verify-upload", action="store_true")
    parser.add_argument("--preallocate-output-fill", type=float, default=None)
    parser.add_argument("--fixture-npz", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pcc-threshold", type=float, default=0.99999)
    parser.add_argument("--max-abs-diff-threshold", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--oracle-mode", choices=("native", "fp32"), default="native")
    args = parser.parse_args()
    if args.key_dim % 32 != 0 or args.value_dim % 32 != 0:
        raise ValueError("--key-dim and --value-dim must be multiples of 32")
    if args.key_dim <= 0 or args.value_dim <= 0 or args.key_dim > 128 or args.value_dim > 128:
        raise ValueError("--key-dim and --value-dim must be in [32, 128]")

    import ttnn

    if not hasattr(ttnn.experimental, "qwen36_gdn_outer_update"):
        raise RuntimeError("ttnn.experimental.qwen36_gdn_outer_update is not registered in this TTNN build")

    fixture = _load_or_make_fixture(args)
    state_scaled = oracle.decay_state(fixture["state"], fixture["alpha"])
    pred = oracle.prediction(state_scaled, fixture["k"])
    delta_value = oracle.delta(fixture["value"], pred, fixture["beta"])
    k_col = _k_col_tiles(fixture["k"])
    delta_rows = _delta_rows(delta_value)
    if args.debug_fill:
        fp32_expected = np.ones_like(state_scaled, dtype=np.float32)
    else:
        fp32_expected = oracle.outer_update(state_scaled, fixture["k"], delta_value)
    expected = _native_expected(fp32_expected, state_scaled, k_col, delta_rows, args)

    _progress(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        tt_dtype = ttnn.float32 if args.dtype == "float32" else ttnn.bfloat16

        _progress("uploading state_scaled tensor")
        state_tt = ttnn.from_torch(
            torch.from_numpy(state_scaled[None, :, :, :]),
            dtype=tt_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        _progress("uploading k_col tensor")
        k_col_tt = ttnn.from_torch(
            torch.from_numpy(k_col[None, :, :, :]),
            dtype=tt_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        _progress("uploading delta tensor")
        delta_tt = ttnn.from_torch(
            torch.from_numpy(delta_rows[None, :, :, :]),
            dtype=tt_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        upload_state_report = None
        upload_k_col_report = None
        upload_delta_report = None
        if args.verify_upload:
            _progress("verifying uploaded input tensors")
            upload_state = ttnn.to_torch(state_tt).float().cpu().numpy()[0, :, :, :]
            upload_k_col = ttnn.to_torch(k_col_tt).float().cpu().numpy()[0, :, :, :]
            upload_delta = ttnn.to_torch(delta_tt).float().cpu().numpy()[0, :, :, :]
            upload_state_report = oracle.diff_report(upload_state, state_scaled)
            upload_k_col_report = oracle.diff_report(upload_k_col, k_col)
            upload_delta_report = oracle.diff_report(upload_delta, delta_rows)

        output_tt = None
        if args.preallocate_output_fill is not None:
            _progress("uploading preallocated output tensor")
            output_host = np.full_like(state_scaled[None, :, :, :], args.preallocate_output_fill, dtype=np.float32)
            output_tt = ttnn.from_torch(
                torch.from_numpy(output_host),
                dtype=tt_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )

        _progress("launching owned op")
        out_tt = ttnn.experimental.qwen36_gdn_outer_update(
            state_tt,
            k_col_tt,
            delta_tt,
            debug_fill=args.debug_fill,
            output_tensor=output_tt,
        )
        if hasattr(ttnn, "synchronize_device"):
            _progress("synchronizing device")
            ttnn.synchronize_device(device)
        _progress("reading output tensor")
        actual = ttnn.to_torch(out_tt).float().cpu().numpy()[0, :, :, :]
    finally:
        _progress("closing device")
        ttnn.close_device(device)

    report = oracle.diff_report(actual, expected)
    payload: dict[str, Any] = {
        "op": "ttnn.experimental.qwen36_gdn_outer_update",
        "config": {
            "device_id": args.device_id,
            "debug_fill": args.debug_fill,
            "preallocate_output_fill": args.preallocate_output_fill,
            "slots": int(fixture["state"].shape[0]),
            "key_dim": int(fixture["state"].shape[1]),
            "value_dim": int(fixture["state"].shape[2]),
            "seed": args.seed,
            "scale": args.scale,
            "fixture_npz": str(args.fixture_npz) if args.fixture_npz is not None else None,
            "dtype": args.dtype,
            "oracle_mode": args.oracle_mode,
        },
        "state_next": report,
        "upload_state_scaled": upload_state_report,
        "upload_k_col": upload_k_col_report,
        "upload_delta": upload_delta_report,
        "actual_stats": _array_stats(actual),
        "expected_stats": _array_stats(expected),
        "state_scaled_stats": _array_stats(state_scaled),
        "k_stats": _array_stats(fixture["k"]),
        "delta_stats": _array_stats(delta_value),
        "thresholds": {
            "pcc": args.pcc_threshold,
            "max_abs_diff": args.max_abs_diff_threshold,
        },
    }

    if args.summary_json is not None:
        _write_json(args.summary_json, payload)
    print(json.dumps(payload, indent=2))

    passed = report["pcc"] >= args.pcc_threshold and report["max_abs_diff"] <= args.max_abs_diff_threshold
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
