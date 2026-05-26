#!/usr/bin/env python3
"""Synthetic correctness/timing probe for Qwen36 native GDN decode.

Run only when the resident qb2 server is stopped; this opens a raw TT device.
The probe compares ttnn.experimental.qwen36_gdn_decode against the current
TTNN decomposition of the DeltaNet recurrence body.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn


K_DIM = 128
V_DIM = 128


def pcc_and_maxdiff(a: np.ndarray, b: np.ndarray) -> dict:
    af = a.astype(np.float64).reshape(-1)
    bf = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    return {
        "pcc": float((af @ bf) / denom),
        "max_abs_diff": float(np.max(np.abs(af - bf))),
        "mean_abs_diff": float(np.mean(np.abs(af - bf))),
    }


def summarize_ms(samples: list[float]) -> dict:
    arr = np.array(samples, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = {
        "probe": "qb2_gdn_native_synthetic",
        "device_id": args.device_id,
        "shape": {
            "slots": args.slots,
            "state": [1, args.slots, K_DIM, V_DIM],
            "qkv": [1, args.slots, 1, K_DIM],
            "alpha_beta": [1, args.slots, 1, 1],
        },
    }

    has_prepare = hasattr(ttnn.experimental, "qwen36_gdn_prepare_decode")
    has_decode = hasattr(ttnn.experimental, "qwen36_gdn_decode")
    result["symbols"] = {
        "qwen36_gdn_prepare_decode": has_prepare,
        "qwen36_gdn_decode": has_decode,
    }
    if not has_decode:
        result["error"] = "ttnn.experimental.qwen36_gdn_decode is not exposed"
        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        return 2

    rng = np.random.default_rng(args.seed)
    state_np = (rng.standard_normal((1, args.slots, K_DIM, V_DIM)).astype(np.float32) * 0.03)
    q_np = (rng.standard_normal((1, args.slots, 1, K_DIM)).astype(np.float32) * 0.03)
    k_np = (rng.standard_normal((1, args.slots, 1, K_DIM)).astype(np.float32) * 0.03)
    value_np = (rng.standard_normal((1, args.slots, 1, V_DIM)).astype(np.float32) * 0.03)
    alpha_np = np.exp(-np.abs(rng.standard_normal((1, args.slots, 1, 1)).astype(np.float32)) * 0.05)
    beta_np = 1.0 / (1.0 + np.exp(-rng.standard_normal((1, args.slots, 1, 1)).astype(np.float32)))

    device = ttnn.open_device(device_id=args.device_id)
    tensors = []
    try:
        def upload(arr: np.ndarray):
            tensor = ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(arr)),
                dtype=ttnn.float32,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            tensors.append(tensor)
            return tensor

        state_manual = upload(state_np)
        state_native = upload(state_np)
        q = upload(q_np)
        k = upload(k_np)
        value = upload(value_np)
        alpha = upload(alpha_np)
        beta = upload(beta_np)

        def manual():
            state_scaled = ttnn.mul(state_manual, alpha)
            k_col = ttnn.reshape(k, [1, args.slots, K_DIM, 1])
            prediction = ttnn.reshape(
                ttnn.sum(ttnn.mul(state_scaled, k_col), dim=-2),
                [1, args.slots, 1, V_DIM],
            )
            delta = ttnn.mul(ttnn.sub(value, prediction), beta)
            state_next = ttnn.add(
                state_scaled,
                ttnn.mul(k_col, ttnn.reshape(delta, [1, args.slots, 1, V_DIM])),
            )
            q_col = ttnn.reshape(q, [1, args.slots, K_DIM, 1])
            output = ttnn.reshape(
                ttnn.sum(ttnn.mul(state_next, q_col), dim=-2),
                [1, args.slots, 1, V_DIM],
            )
            return state_next, output

        def native():
            return ttnn.experimental.qwen36_gdn_decode(
                state_native,
                q,
                k,
                value,
                alpha,
                beta,
                normalize_qk_l2=False,
                output_memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        manual_state, manual_out = manual()
        native_state, native_out = native()
        ttnn.synchronize_device(device)

        manual_state_np = ttnn.to_torch(manual_state).float().cpu().numpy().reshape(1, args.slots, K_DIM, V_DIM)
        native_state_np = ttnn.to_torch(native_state).float().cpu().numpy().reshape(1, args.slots, K_DIM, V_DIM)

        manual_out_full = ttnn.to_torch(manual_out).float().cpu().numpy()
        native_out_full = ttnn.to_torch(native_out).float().cpu().numpy()
        manual_out_np = manual_out_full.reshape(1, args.slots, -1, V_DIM)[:, :, :1, :]
        native_out_np = native_out_full.reshape(1, args.slots, -1, V_DIM)[:, :, :1, :]

        state_cmp = pcc_and_maxdiff(native_state_np, manual_state_np)
        out_cmp = pcc_and_maxdiff(native_out_np, manual_out_np)
        pass_gate = (
            state_cmp["pcc"] >= 0.9999 and
            out_cmp["pcc"] >= 0.9999 and
            state_cmp["max_abs_diff"] <= 1e-2 and
            out_cmp["max_abs_diff"] <= 1e-2
        )

        # native_state aliases state_native; do not deallocate it here.
        for tensor in (manual_state, manual_out, native_out):
            ttnn.deallocate(tensor)

        def timed(fn, deallocate_state: bool):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            state_tmp, out_tmp = fn()
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) * 1000.0
            if deallocate_state:
                ttnn.deallocate(state_tmp)
            ttnn.deallocate(out_tmp)
            return dt

        for _ in range(args.warmup):
            timed(manual, deallocate_state=True)
            timed(native, deallocate_state=False)
        manual_ms = [timed(manual, deallocate_state=True) for _ in range(args.iters)]
        native_ms = [timed(native, deallocate_state=False) for _ in range(args.iters)]

        result.update({
            "correctness": {
                "state_vs_manual": state_cmp,
                "output_vs_manual": out_cmp,
                "pass_gate": pass_gate,
            },
            "timing": {
                "iters": args.iters,
                "warmup": args.warmup,
                "manual_ms": summarize_ms(manual_ms),
                "native_ms": summarize_ms(native_ms),
                "samples_ms": {
                    "manual": manual_ms,
                    "native": native_ms,
                },
                "note": "Synthetic recurrence body only; not trace or full-decode timing.",
            },
        })
    finally:
        for tensor in tensors:
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass
        ttnn.close_device(device)

    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("correctness", {}).get("pass_gate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
