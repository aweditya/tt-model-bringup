#!/usr/bin/env python3
"""Sync-bounded microbench for owned GDN component chain vs fused decode op.

Run only on a TT host after all owned ops are integrated into that host's TTNN
build. Do not run locally and do not run while the persistent inference server
owns the chips.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

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


def _scalar_tiles(scalar: np.ndarray) -> np.ndarray:
    return np.broadcast_to(scalar[:, None, None], (scalar.shape[0], 32, 32)).astype(np.float32, copy=True)


def _k_col_tiles(k: np.ndarray) -> np.ndarray:
    return np.broadcast_to(k[:, :, None], (k.shape[0], k.shape[1], 32)).astype(np.float32, copy=True)


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
    state_next = _to_bfloat16_numpy(state_scaled + _to_bfloat16_numpy(k[:, :, None] * delta[:, None, :]))
    out = _to_bfloat16_numpy(oracle.output(state_next, q))
    return {"state_next": state_next, "out_rows": _rows(out)}


def _summary_ms(samples: list[float]) -> dict[str, Any]:
    values = sorted(samples)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(values[0]),
        "max_ms": float(values[-1]),
        "p25_ms": float(values[len(values) // 4]),
        "p75_ms": float(values[(len(values) * 3) // 4]),
        "samples_ms": [float(x) for x in samples],
    }


def _progress(message: str) -> None:
    print(f"[qwen36_gdn_decode_owned_bench] {message}", file=sys.stderr, flush=True)


def _time_sync_bounded(ttnn: Any, device: Any, fn: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(device)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _capture_trace(ttnn: Any, device: Any, fn: Callable[[], Any]) -> Any:
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    return trace_id


def _time_trace_replay(ttnn: Any, device: Any, trace_id: Any, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trace-warmup", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=50)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--include-ablation-modes", action="store_true")
    parser.add_argument("--include-pretransposed-k", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pcc-threshold", type=float, default=0.99999)
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.001)
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
        "qwen36_gdn_decode_owned",
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
    beta_host = _scalar_tiles(fixture["beta"])[None, :, :, :]
    q_rows_host = _rows(fixture["q"])[None, :, :, :]
    k_rows_host = _rows(fixture["k"])[None, :, :, :]
    value_rows_host = _rows(fixture["value"])[None, :, :, :]
    k_col_host = _k_col_tiles(fixture["k"])[None, :, :, :]

    _progress(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        dtype = ttnn.bfloat16
        _progress("uploading inputs")
        state_component_tt = ttnn.from_torch(
            torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
        )
        state_fused_tt = ttnn.from_torch(
            torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
        )
        state_pretransposed_tt = ttnn.from_torch(
            torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
        )
        alpha_tt = ttnn.from_torch(torch.from_numpy(alpha_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        beta_tt = ttnn.from_torch(torch.from_numpy(beta_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        q_tt = ttnn.from_torch(torch.from_numpy(q_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_tt = ttnn.from_torch(torch.from_numpy(k_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        value_tt = ttnn.from_torch(torch.from_numpy(value_rows_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        k_col_tt = ttnn.from_torch(torch.from_numpy(k_col_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        def component_chain() -> Any:
            state_scaled = ttnn.experimental.qwen36_gdn_decay_state(state_component_tt, alpha_tt)
            pred = ttnn.experimental.qwen36_gdn_prediction(state_scaled, k_tt)
            delta = ttnn.experimental.qwen36_gdn_delta(value_tt, pred, beta_tt)
            state_next = ttnn.experimental.qwen36_gdn_outer_update(state_scaled, k_col_tt, delta)
            return ttnn.experimental.qwen36_gdn_output(state_next, q_tt)

        def fused_decode() -> Any:
            return ttnn.experimental.qwen36_gdn_decode_owned(
                state_fused_tt, q_tt, k_tt, value_tt, alpha_tt, beta_tt
            )

        def fused_decode_pretransposed_k() -> Any:
            return ttnn.experimental.qwen36_gdn_decode_owned(
                state_pretransposed_tt, q_tt, k_tt, value_tt, alpha_tt, beta_tt, k_col=k_col_tt
            )

        def fused_decode_mode(mode: int) -> Any:
            return ttnn.experimental.qwen36_gdn_decode_owned(
                state_fused_tt,
                q_tt,
                k_tt,
                value_tt,
                alpha_tt,
                beta_tt,
                debug_mode=mode,
            )

        _progress("running correctness pass")
        state_scaled_tt = ttnn.experimental.qwen36_gdn_decay_state(state_component_tt, alpha_tt)
        pred_tt = ttnn.experimental.qwen36_gdn_prediction(state_scaled_tt, k_tt)
        delta_tt = ttnn.experimental.qwen36_gdn_delta(value_tt, pred_tt, beta_tt)
        state_next_tt = ttnn.experimental.qwen36_gdn_outer_update(state_scaled_tt, k_col_tt, delta_tt)
        out_component_tt = ttnn.experimental.qwen36_gdn_output(state_next_tt, q_tt)
        state_fused_once_tt = ttnn.from_torch(
            torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
        )
        state_owned_tt, out_fused_tt = ttnn.experimental.qwen36_gdn_decode_owned(
            state_fused_once_tt, q_tt, k_tt, value_tt, alpha_tt, beta_tt
        )
        state_pre_once_tt = ttnn.from_torch(
            torch.from_numpy(state_host), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT
        )
        state_pre_tt, out_pre_tt = ttnn.experimental.qwen36_gdn_decode_owned(
            state_pre_once_tt, q_tt, k_tt, value_tt, alpha_tt, beta_tt, k_col=k_col_tt
        )
        ttnn.synchronize_device(device)
        actual_component_out = ttnn.to_torch(out_component_tt).float().cpu().numpy()[0, :, :, :]
        actual_fused_state = ttnn.to_torch(state_owned_tt).float().cpu().numpy()[0, :, :, :]
        actual_fused_out = ttnn.to_torch(out_fused_tt).float().cpu().numpy()[0, :, :, :]
        actual_pre_state = ttnn.to_torch(state_pre_tt).float().cpu().numpy()[0, :, :, :]
        actual_pre_out = ttnn.to_torch(out_pre_tt).float().cpu().numpy()[0, :, :, :]

        component_out_report = oracle.diff_report(actual_component_out, expected["out_rows"])
        fused_state_report = oracle.diff_report(actual_fused_state, expected["state_next"])
        fused_out_report = oracle.diff_report(actual_fused_out, expected["out_rows"])
        fused_vs_component_report = oracle.diff_report(actual_fused_out, actual_component_out)
        pre_state_report = oracle.diff_report(actual_pre_state, expected["state_next"])
        pre_out_report = oracle.diff_report(actual_pre_out, expected["out_rows"])
        pre_vs_fused_out_report = oracle.diff_report(actual_pre_out, actual_fused_out)

        _progress("measuring sync-only floor")
        sync_only_samples = []
        for _ in range(args.warmup):
            ttnn.synchronize_device(device)
        for _ in range(args.repeats):
            start = time.perf_counter()
            ttnn.synchronize_device(device)
            sync_only_samples.append((time.perf_counter() - start) * 1000.0)

        _progress("timing component chain")
        component_samples = _time_sync_bounded(ttnn, device, component_chain, args.warmup, args.repeats)
        _progress("timing fused decode")
        fused_samples = _time_sync_bounded(ttnn, device, fused_decode, args.warmup, args.repeats)
        pretransposed_samples = None
        if args.include_pretransposed_k:
            _progress("timing fused decode with pretransposed K")
            pretransposed_samples = _time_sync_bounded(
                ttnn, device, fused_decode_pretransposed_k, args.warmup, args.repeats
            )

        trace_timings: dict[str, Any] = {}
        if not args.skip_trace:
            component_trace_id = None
            fused_trace_id = None
            try:
                _progress("capturing component-chain trace")
                component_trace_id = _capture_trace(ttnn, device, component_chain)
                _progress("timing component-chain trace replay")
                trace_timings["component_chain_execute_trace"] = _summary_ms(
                    _time_trace_replay(ttnn, device, component_trace_id, args.trace_warmup, args.trace_repeats)
                )
            finally:
                if component_trace_id is not None:
                    try:
                        ttnn.release_trace(device, component_trace_id)
                    except Exception as exc:
                        trace_timings["component_chain_release_error"] = str(exc)
            try:
                _progress("capturing fused decode trace")
                fused_trace_id = _capture_trace(ttnn, device, fused_decode)
                _progress("timing fused decode trace replay")
                trace_timings["fused_decode_owned_execute_trace"] = _summary_ms(
                    _time_trace_replay(ttnn, device, fused_trace_id, args.trace_warmup, args.trace_repeats)
                )
            finally:
                if fused_trace_id is not None:
                    try:
                        ttnn.release_trace(device, fused_trace_id)
                    except Exception as exc:
                        trace_timings["fused_decode_release_error"] = str(exc)

            if args.include_pretransposed_k:
                pre_trace_id = None
                try:
                    _progress("warming fused pretransposed-K decode")
                    fused_decode_pretransposed_k()
                    ttnn.synchronize_device(device)
                    _progress("capturing fused pretransposed-K decode trace")
                    pre_trace_id = _capture_trace(ttnn, device, fused_decode_pretransposed_k)
                    _progress("timing fused pretransposed-K decode trace replay")
                    trace_timings["fused_decode_owned_pretransposed_k_execute_trace"] = _summary_ms(
                        _time_trace_replay(ttnn, device, pre_trace_id, args.trace_warmup, args.trace_repeats)
                    )
                finally:
                    if pre_trace_id is not None:
                        try:
                            ttnn.release_trace(device, pre_trace_id)
                        except Exception as exc:
                            trace_timings["fused_pretransposed_k_release_error"] = str(exc)

            if args.include_ablation_modes:
                ablation_modes = [
                    ("mode1_skeleton_read_write_fill", 1),
                    ("mode2_decay_write_state", 2),
                    ("mode3_decay_plus_prediction", 3),
                    ("mode4_decay_prediction_delta", 4),
                    ("mode5_update_without_output", 5),
                    ("mode6_output_only", 6),
                    ("mode7_transpose_only_after_delta", 7),
                    ("mode8_transpose_outer_no_add", 8),
                    ("mode9_update_state_out_only", 9),
                ]
                for label, mode in ablation_modes:
                    trace_id = None
                    try:
                        _progress(f"warming ablation mode {label}")
                        fused_decode_mode(mode)
                        ttnn.synchronize_device(device)
                        _progress(f"capturing ablation trace {label}")
                        trace_id = _capture_trace(ttnn, device, lambda mode=mode: fused_decode_mode(mode))
                        _progress(f"timing ablation trace {label}")
                        trace_timings[f"{label}_execute_trace"] = _summary_ms(
                            _time_trace_replay(ttnn, device, trace_id, args.trace_warmup, args.trace_repeats)
                        )
                    finally:
                        if trace_id is not None:
                            try:
                                ttnn.release_trace(device, trace_id)
                            except Exception as exc:
                                trace_timings[f"{label}_release_error"] = str(exc)
    finally:
        _progress("closing device")
        ttnn.close_device(device)

    sync_only = _summary_ms(sync_only_samples)
    component = _summary_ms(component_samples)
    fused = _summary_ms(fused_samples)
    pretransposed = _summary_ms(pretransposed_samples) if pretransposed_samples is not None else None
    measured_trace_only = None
    if (
        "component_chain_execute_trace" in trace_timings
        and "fused_decode_owned_execute_trace" in trace_timings
    ):
        component_trace = trace_timings["component_chain_execute_trace"]
        fused_trace = trace_timings["fused_decode_owned_execute_trace"]
        measured_trace_only = {
            "median_delta_ms": component_trace["median_ms"] - fused_trace["median_ms"],
            "median_ratio_component_over_fused": component_trace["median_ms"] / fused_trace["median_ms"],
        }
    payload: dict[str, Any] = {
        "op": "qwen36_gdn_decode_owned_microbench",
        "config": {
            "device_id": args.device_id,
            "slots": args.slots,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "seed": args.seed,
            "scale": args.scale,
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "trace_warmup": args.trace_warmup,
            "trace_repeats": args.trace_repeats,
            "skip_trace": args.skip_trace,
            "include_ablation_modes": args.include_ablation_modes,
            "include_pretransposed_k": args.include_pretransposed_k,
            "timing": "eager sync-bounded host wall time",
        },
        "correctness": {
            "component_out_vs_oracle": component_out_report,
            "fused_state_vs_oracle": fused_state_report,
            "fused_out_vs_oracle": fused_out_report,
            "fused_out_vs_component_out": fused_vs_component_report,
            "pretransposed_k_state_vs_oracle": pre_state_report,
            "pretransposed_k_out_vs_oracle": pre_out_report,
            "pretransposed_k_out_vs_fused_out": pre_vs_fused_out_report,
        },
        "timing_ms": {
            "sync_only": sync_only,
            "component_chain": component,
            "fused_decode_owned": fused,
            **({"fused_decode_owned_pretransposed_k": pretransposed} if pretransposed is not None else {}),
            **trace_timings,
        },
        "measured_component_only": {
            "median_delta_ms": component["median_ms"] - fused["median_ms"],
            "median_ratio_component_over_fused": component["median_ms"] / fused["median_ms"],
        },
        "measured_trace_only": measured_trace_only,
        "thresholds": {
            "pcc": args.pcc_threshold,
            "max_abs_diff": args.max_abs_diff_threshold,
        },
        "caveat": "Component microbench only. This is not a full-decode speedup claim.",
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, payload)
    print(json.dumps(payload, indent=2))

    reports = [
        component_out_report,
        fused_state_report,
        fused_out_report,
        fused_vs_component_report,
        pre_state_report,
        pre_out_report,
        pre_vs_fused_out_report,
    ]
    passed = all(
        report["pcc"] >= args.pcc_threshold and report["max_abs_diff"] <= args.max_abs_diff_threshold
        for report in reports
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
