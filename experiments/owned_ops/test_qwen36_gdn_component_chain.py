#!/usr/bin/env python3
"""Device correctness gate for the owned Qwen3.6 GDN component chain.

This chains the five isolated owned ops:

    decay_state -> prediction -> delta -> outer_update -> output

Run only on a TT host after all five ops have been integrated into that host's
TTNN build. Do not run it locally or while the persistent inference server owns
the chips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.utils import gdn_kernel_oracle as oracle  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _to_bfloat16_numpy(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(torch.bfloat16).float().cpu().numpy()


def _rows(vector: np.ndarray) -> np.ndarray:
    return np.broadcast_to(vector[:, None, :], (vector.shape[0], 32, vector.shape[1])).astype(np.float32, copy=True)


def _scalar_tiles(scalar: np.ndarray) -> np.ndarray:
    return np.broadcast_to(scalar[:, None, None], (scalar.shape[0], 32, 32)).astype(np.float32, copy=True)


def _k_col_tiles(k: np.ndarray) -> np.ndarray:
    return np.broadcast_to(k[:, :, None], (k.shape[0], k.shape[1], 32)).astype(np.float32, copy=True)


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
    state_next = _to_bfloat16_numpy(
        state_scaled + _to_bfloat16_numpy(k[:, :, None] * delta[:, None, :])
    )
    out = _to_bfloat16_numpy(oracle.output(state_next, q))
    return {
        "state_scaled": state_scaled,
        "prediction": pred,
        "delta": delta,
        "state_next": state_next,
        "out": out,
    }


def _progress(message: str) -> None:
    print(f"[qwen36_gdn_component_chain] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pcc-threshold", type=float, default=0.99999)
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.0005)
    args = parser.parse_args()
    if args.key_dim % 32 != 0 or args.value_dim % 32 != 0:
        raise ValueError("--key-dim and --value-dim must be multiples of 32")

    import ttnn

    required = [
        "qwen36_gdn_decay_state",
        "qwen36_gdn_prediction",
        "qwen36_gdn_delta",
        "qwen36_gdn_outer_update",
        "qwen36_gdn_output",
    ]
    missing = [name for name in required if not hasattr(ttnn.experimental, name)]
    if missing:
        raise RuntimeError(f"missing owned GDN symbols: {missing}")

    fixture = oracle.make_fixture(
        slots=args.slots,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        seed=args.seed,
        scale=args.scale,
    )
    expected = _native_expected(fixture)

    state_host = fixture["state"][None, :, :, :]
    alpha_host = _scalar_tiles(fixture["alpha"])[None, :, :, :]
    k_rows_host = _rows(fixture["k"])[None, :, :, :]
    value_rows_host = _rows(fixture["value"])[None, :, :, :]
    beta_host = _scalar_tiles(fixture["beta"])[None, :, :, :]
    k_col_host = _k_col_tiles(fixture["k"])[None, :, :, :]
    q_rows_host = _rows(fixture["q"])[None, :, :, :]

    _progress(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        dtype = ttnn.bfloat16
        _progress("uploading inputs")
        state_tt = ttnn.from_torch(torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        alpha_tt = ttnn.from_torch(torch.from_numpy(alpha_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_rows_tt = ttnn.from_torch(torch.from_numpy(k_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        value_rows_tt = ttnn.from_torch(torch.from_numpy(value_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        beta_tt = ttnn.from_torch(torch.from_numpy(beta_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_col_tt = ttnn.from_torch(torch.from_numpy(k_col_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        q_rows_tt = ttnn.from_torch(torch.from_numpy(q_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        _progress("launching component chain")
        state_scaled_tt = ttnn.experimental.qwen36_gdn_decay_state(state_tt, alpha_tt)
        pred_tt = ttnn.experimental.qwen36_gdn_prediction(state_scaled_tt, k_rows_tt)
        delta_tt = ttnn.experimental.qwen36_gdn_delta(value_rows_tt, pred_tt, beta_tt)
        state_next_tt = ttnn.experimental.qwen36_gdn_outer_update(state_scaled_tt, k_col_tt, delta_tt)
        out_tt = ttnn.experimental.qwen36_gdn_output(state_next_tt, q_rows_tt)
        if hasattr(ttnn, "synchronize_device"):
            _progress("synchronizing device")
            ttnn.synchronize_device(device)
        _progress("reading outputs")
        actual_state_next = ttnn.to_torch(state_next_tt).float().cpu().numpy()[0, :, :, :]
        actual_out_rows = ttnn.to_torch(out_tt).float().cpu().numpy()[0, :, :, :]
    finally:
        _progress("closing device")
        ttnn.close_device(device)

    expected_out_rows = _rows(expected["out"])
    state_next_report = _diff("state_next", actual_state_next, expected["state_next"])
    out_report = _diff("out", actual_out_rows, expected_out_rows)
    payload: dict[str, Any] = {
        "op": "owned_gdn_component_chain",
        "config": {
            "device_id": args.device_id,
            "slots": args.slots,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "seed": args.seed,
            "scale": args.scale,
            "dtype": "bfloat16",
            "oracle_mode": "native",
        },
        "state_next": state_next_report,
        "out": out_report,
        "thresholds": {
            "pcc": args.pcc_threshold,
            "max_abs_diff": args.max_abs_diff_threshold,
        },
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, payload)
    print(json.dumps(payload, indent=2))

    passed = all(
        report["pcc"] >= args.pcc_threshold and report["max_abs_diff"] <= args.max_abs_diff_threshold
        for report in [state_next_report, out_report]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
