#!/usr/bin/env python3
"""Device correctness gate for the owned fused Qwen3.6 GDN decode op.

Run only on a TT host after the op has been integrated into that host's TTNN
build. Do not run it locally or while the persistent inference server owns the
chips.
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _to_bfloat16_numpy(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(torch.bfloat16).float().cpu().numpy()


def _rows(vector: np.ndarray) -> np.ndarray:
    return np.broadcast_to(vector[:, None, :], (vector.shape[0], 32, vector.shape[1])).astype(np.float32, copy=True)


def _compact_rows(vector: np.ndarray) -> np.ndarray:
    return vector[:, None, :].astype(np.float32, copy=True)


def _scalar_tiles(scalar: np.ndarray) -> np.ndarray:
    return np.broadcast_to(scalar[:, None, None], (scalar.shape[0], 32, 32)).astype(np.float32, copy=True)


def _scalar_singletons(scalar: np.ndarray) -> np.ndarray:
    return scalar[:, None, None].astype(np.float32, copy=True)


def _k_col_tiles(k: np.ndarray) -> np.ndarray:
    return np.broadcast_to(k[:, :, None], (k.shape[0], k.shape[1], 32)).astype(np.float32, copy=True)


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


def _diff(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    report = oracle.diff_report(actual, expected)
    report["name"] = name
    return report


def _native_expected(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = _to_bfloat16_numpy(fixture["state"])
    alpha = _to_bfloat16_numpy(fixture["alpha"])
    beta = _to_bfloat16_numpy(fixture["beta"])
    k = _to_bfloat16_numpy(fixture["k"])
    q = _to_bfloat16_numpy(fixture["q"])
    value = _to_bfloat16_numpy(fixture["value"])

    state_scaled = _to_bfloat16_numpy(oracle.decay_state(state, alpha))
    pred = _to_bfloat16_numpy(oracle.prediction(state_scaled, k))
    delta = _to_bfloat16_numpy(oracle.delta(value, pred, beta))
    outer = _to_bfloat16_numpy(k[:, :, None] * delta[:, None, :])
    state_next = _to_bfloat16_numpy(state_scaled + outer)
    out = _to_bfloat16_numpy(oracle.output(state_next, q))
    return {
        "state_scaled": state_scaled,
        "prediction": pred,
        "delta": delta,
        "state_next": state_next,
        "out_rows": _rows(out),
    }


def _progress(message: str) -> None:
    print(f"[qwen36_gdn_decode_owned] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--debug-fill", action="store_true")
    parser.add_argument("--use-pretransposed-k", action="store_true")
    parser.add_argument("--compact-vectors", action="store_true")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--preallocate-output-fill", type=float, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pcc-threshold", type=float, default=0.99999)
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.001)
    args = parser.parse_args()
    if args.key_dim % 32 != 0 or args.value_dim % 32 != 0:
        raise ValueError("--key-dim and --value-dim must be multiples of 32")
    if args.key_dim <= 0 or args.value_dim <= 0 or args.key_dim > 128 or args.value_dim > 128:
        raise ValueError("--key-dim and --value-dim must be in [32, 128]")
    if (args.compact_vectors or args.native_io) and args.use_pretransposed_k:
        raise ValueError("--compact-vectors/--native-io currently do not support --use-pretransposed-k")

    import ttnn

    if not hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"):
        raise RuntimeError("ttnn.experimental.qwen36_gdn_decode_owned is not registered in this TTNN build")

    fixture = oracle.make_fixture(
        slots=args.slots,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        seed=args.seed,
        scale=args.scale,
    )
    expected = _native_expected(fixture)
    if args.debug_fill:
        expected_state = np.ones_like(expected["state_next"])
        expected_out = np.ones_like(expected["out_rows"])
    else:
        expected_state = expected["state_next"]
        expected_out = expected["out_rows"]

    state_host = fixture["state"][None, :, :, :]
    alpha_builder = _scalar_singletons if args.native_io else _scalar_tiles
    alpha_host = alpha_builder(fixture["alpha"])[None, :, :, :]
    beta_host = alpha_builder(fixture["beta"])[None, :, :, :]
    row_builder = _compact_rows if (args.compact_vectors or args.native_io) else _rows
    q_rows_host = row_builder(fixture["q"])[None, :, :, :]
    k_rows_host = row_builder(fixture["k"])[None, :, :, :]
    k_col_host = _k_col_tiles(fixture["k"])[None, :, :, :]
    value_rows_host = row_builder(fixture["value"])[None, :, :, :]

    _progress(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        dtype = ttnn.bfloat16
        _progress("uploading inputs")
        state_tt = ttnn.from_torch(torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        q_tt = ttnn.from_torch(torch.from_numpy(q_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_tt = ttnn.from_torch(torch.from_numpy(k_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_col_tt = None
        if args.use_pretransposed_k:
            k_col_tt = ttnn.from_torch(
                torch.from_numpy(k_col_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
        value_tt = ttnn.from_torch(torch.from_numpy(value_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        alpha_tt = ttnn.from_torch(torch.from_numpy(alpha_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        beta_tt = ttnn.from_torch(torch.from_numpy(beta_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        output_tt = None
        if args.preallocate_output_fill is not None:
            _progress("uploading preallocated output tensor")
            if args.native_io:
                output_shape = (1, args.slots * args.value_dim)
            else:
                output_shape = expected_out[None, :, :, :].shape
            output_host = np.full(output_shape, args.preallocate_output_fill, dtype=np.float32)
            output_tt = ttnn.from_torch(torch.from_numpy(output_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        _progress("launching fused owned op")
        returned_state_tt, out_tt = ttnn.experimental.qwen36_gdn_decode_owned(
            state_tt,
            q_tt,
            k_tt,
            value_tt,
            alpha_tt,
            beta_tt,
            k_col=k_col_tt,
            debug_fill=args.debug_fill,
            compact_vectors=args.compact_vectors,
            native_io=args.native_io,
            output_tensor=output_tt,
        )
        if hasattr(ttnn, "synchronize_device"):
            _progress("synchronizing device")
            ttnn.synchronize_device(device)
        _progress("reading outputs")
        actual_state = ttnn.to_torch(returned_state_tt).float().cpu().numpy()[0, :, :, :]
        actual_out_raw = ttnn.to_torch(out_tt).float().cpu().numpy()
        if args.native_io:
            actual_out = actual_out_raw.reshape(1, args.slots, args.value_dim)[0, :, None, :]
        else:
            actual_out = actual_out_raw[0, :, :, :]
    finally:
        _progress("closing device")
        ttnn.close_device(device)

    state_report = _diff("state_next", actual_state, expected_state)
    if (args.compact_vectors or args.native_io) and not args.debug_fill:
        out_report = _diff("out_first_row", actual_out[:, :1, :], expected_out[:, :1, :])
    else:
        out_report = _diff("out", actual_out, expected_out)
    payload: dict[str, Any] = {
        "op": "ttnn.experimental.qwen36_gdn_decode_owned",
        "config": {
            "device_id": args.device_id,
            "debug_fill": args.debug_fill,
            "use_pretransposed_k": args.use_pretransposed_k,
            "compact_vectors": args.compact_vectors,
            "native_io": args.native_io,
            "preallocate_output_fill": args.preallocate_output_fill,
            "slots": args.slots,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "seed": args.seed,
            "scale": args.scale,
            "dtype": "bfloat16",
            "oracle_mode": "native",
        },
        "state_next": state_report,
        "out": out_report,
        "actual_state_stats": _array_stats(actual_state),
        "expected_state_stats": _array_stats(expected_state),
        "actual_out_stats": _array_stats(actual_out),
        "expected_out_stats": _array_stats(expected_out),
        "thresholds": {
            "pcc": args.pcc_threshold,
            "max_abs_diff": args.max_abs_diff_threshold,
        },
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, payload)
    print(json.dumps(payload, indent=2))

    reports = [state_report, out_report]
    passed = all(
        report["pcc"] >= args.pcc_threshold and report["max_abs_diff"] <= args.max_abs_diff_threshold
        for report in reports
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
