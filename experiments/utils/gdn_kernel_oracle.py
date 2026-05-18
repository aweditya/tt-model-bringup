#!/usr/bin/env python3
"""CPU oracle and fixture generator for Qwen3.6 GDN decode recurrence.

This script deliberately avoids TTNN.  It creates deterministic fixtures for
bringing up isolated TT-Metal components before wiring the full recurrence
kernel together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    if a64.size == 0:
        return 1.0
    a_centered = a64 - a64.mean()
    b_centered = b64 - b64.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return 1.0 if np.allclose(a64, b64) else 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def diff_report(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual32 = np.asarray(actual, dtype=np.float32)
    expected32 = np.asarray(expected, dtype=np.float32)
    diff = actual32 - expected32
    return {
        "shape": list(actual32.shape),
        "pcc": pcc(actual32, expected32),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(actual32, expected32, rtol=1e-5, atol=1e-6)),
    }


def make_fixture(slots: int, key_dim: int, value_dim: int, seed: int, scale: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, scale, size=(slots, key_dim, value_dim)).astype(np.float32)
    q = rng.normal(0.0, scale, size=(slots, key_dim)).astype(np.float32)
    k = rng.normal(0.0, scale, size=(slots, key_dim)).astype(np.float32)
    value = rng.normal(0.0, scale, size=(slots, value_dim)).astype(np.float32)

    # Decode decay/gate values are bounded so the oracle remains numerically
    # stable while still exercising non-trivial scalar broadcast paths.
    alpha = rng.uniform(0.75, 0.995, size=(slots,)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, size=(slots,)).astype(np.float32)
    return {
        "state": state,
        "q": q,
        "k": k,
        "value": value,
        "alpha": alpha,
        "beta": beta,
    }


def decay_state(state: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return state * alpha[:, None, None]


def prediction(state_scaled: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.einsum("skv,sk->sv", state_scaled, k, optimize=True).astype(np.float32)


def delta(value: np.ndarray, pred: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return ((value - pred) * beta[:, None]).astype(np.float32)


def outer_update(state_scaled: np.ndarray, k: np.ndarray, delta_value: np.ndarray) -> np.ndarray:
    return (state_scaled + k[:, :, None] * delta_value[:, None, :]).astype(np.float32)


def output(state_next: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.einsum("sk,skv->sv", q, state_next, optimize=True).astype(np.float32)


def gdn_step(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_scaled = decay_state(fixture["state"], fixture["alpha"])
    pred = prediction(state_scaled, fixture["k"])
    delta_value = delta(fixture["value"], pred, fixture["beta"])
    state_next = outer_update(state_scaled, fixture["k"], delta_value)
    out = output(state_next, fixture["q"])
    return {
        "state_scaled": state_scaled,
        "prediction": pred,
        "delta": delta_value,
        "state_next": state_next,
        "out": out,
    }


def loop_reference(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = fixture["state"]
    q = fixture["q"]
    k = fixture["k"]
    value = fixture["value"]
    alpha = fixture["alpha"]
    beta = fixture["beta"]
    slots, key_dim, value_dim = state.shape

    state_scaled = np.empty_like(state)
    pred = np.empty((slots, value_dim), dtype=np.float32)
    delta_value = np.empty((slots, value_dim), dtype=np.float32)
    state_next = np.empty_like(state)
    out = np.empty((slots, value_dim), dtype=np.float32)

    for s in range(slots):
        state_scaled[s] = alpha[s] * state[s]
        for v in range(value_dim):
            acc = np.float32(0.0)
            for kd in range(key_dim):
                acc = np.float32(acc + np.float32(k[s, kd] * state_scaled[s, kd, v]))
            pred[s, v] = acc
            delta_value[s, v] = np.float32(beta[s] * np.float32(value[s, v] - pred[s, v]))

        for kd in range(key_dim):
            for v in range(value_dim):
                state_next[s, kd, v] = np.float32(
                    state_scaled[s, kd, v] + np.float32(k[s, kd] * delta_value[s, v])
                )

        for v in range(value_dim):
            acc = np.float32(0.0)
            for kd in range(key_dim):
                acc = np.float32(acc + np.float32(q[s, kd] * state_next[s, kd, v]))
            out[s, v] = acc

    return {
        "state_scaled": state_scaled,
        "prediction": pred,
        "delta": delta_value,
        "state_next": state_next,
        "out": out,
    }


def component_report(fixture: dict[str, np.ndarray]) -> dict[str, Any]:
    vectorized = gdn_step(fixture)
    looped = loop_reference(fixture)
    return {
        name: diff_report(vectorized[name], looped[name])
        for name in ["state_scaled", "prediction", "delta", "state_next", "out"]
    }


def tile_summary(arrays: dict[str, np.ndarray], tile: int) -> dict[str, Any]:
    state = arrays["state"]
    slots, key_dim, value_dim = state.shape
    return {
        "tile": tile,
        "key_tiles": (key_dim + tile - 1) // tile,
        "value_tiles": (value_dim + tile - 1) // tile,
        "work_blocks": slots * ((value_dim + tile - 1) // tile),
        "state_tiles_per_slot": ((key_dim + tile - 1) // tile) * ((value_dim + tile - 1) // tile),
        "default_block": {
            "state_tiles_read": (key_dim + tile - 1) // tile,
            "q_tiles_read": (key_dim + tile - 1) // tile,
            "k_tiles_read": (key_dim + tile - 1) // tile,
            "value_tiles_read": 1,
            "state_tiles_written": (key_dim + tile - 1) // tile,
            "out_tiles_written": 1,
        },
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--fixture-npz", type=Path, default=None)
    parser.add_argument("--no-loop-check", action="store_true")
    args = parser.parse_args()

    fixture = make_fixture(
        slots=args.slots,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        seed=args.seed,
        scale=args.scale,
    )
    outputs = gdn_step(fixture)
    reports = {} if args.no_loop_check else component_report(fixture)

    payload: dict[str, Any] = {
        "config": {
            "slots": args.slots,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "seed": args.seed,
            "scale": args.scale,
        },
        "tile_summary": tile_summary(fixture, args.tile),
        "component_report_vs_loop_reference": reports,
        "output_norms": {
            name: float(np.linalg.norm(value.astype(np.float64)))
            for name, value in outputs.items()
        },
    }

    if args.fixture_npz is not None:
        args.fixture_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fixture_npz, **fixture, **outputs)
        payload["fixture_npz"] = str(args.fixture_npz)

    if args.summary_json is not None:
        write_summary(args.summary_json, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
