#!/usr/bin/env python3
"""Tracy/device-profile qb2 TP decode trace replay.

This is a standalone profiling harness: stop the resident server before
running it, because it opens the same four-chip mesh. It reuses
experiments/serve/server_tp.py so the profiled graph is the production TP
decode path, not a reduced single-chip surrogate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.serve import server_tp as S  # noqa: E402


def _summary(values: list[float]) -> dict:
    import numpy as np

    if not values:
        return {"median": None, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _zone(ttnn, name: str, fn):
    ttnn.start_tracy_zone("qb2_tp_tracy_profile_probe.py", name, 0)
    try:
        return fn()
    finally:
        ttnn.stop_tracy_zone(name, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["manual", "native_softplus"],
        default="manual",
        help="DeltaNet decay/gate mode to capture in the trace.",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    import ttnn

    state = S.MeshServerState()
    state.deltanet_decay_mode = args.mode
    result: dict = {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
        "mode": args.mode,
        "trace_id": None,
        "summary_ms": {},
        "samples_ms": {},
    }

    try:
        print("[profile] bootstrap production TP state", flush=True)
        S.bootstrap(state)
        print("[profile] capture production decode trace", flush=True)
        S._ensure_decode_trace(state)
        result["trace_id"] = int(state.trace_id)

        prompt_ids = state.tok.encode(args.prompt)
        if not prompt_ids:
            raise RuntimeError("prompt encoded to zero tokens")
        result["prompt_ids"] = list(prompt_ids)

        def sync():
            ttnn.synchronize_device(state.mesh)

        def timed(name: str, fn) -> float:
            sync()
            t0 = time.perf_counter()
            _zone(ttnn, name, fn)
            sync()
            return (time.perf_counter() - t0) * 1000.0

        print("[profile] seed trace output", flush=True)
        S.update_input_buffers(state, int(prompt_ids[0]), 0)
        ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
        sync()

        print(f"[profile] warmup x{args.warmup}", flush=True)
        for i in range(args.warmup):
            tid = int(prompt_ids[i % len(prompt_ids)])
            pos = i % S.MAX_POS
            S.update_input_buffers(state, tid, pos)
            ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
        sync()

        execute_ms = []
        update_execute_ms = []
        readback_ms = []
        next_ids = []
        print(f"[profile] measured replay x{args.iters}", flush=True)
        for i in range(args.iters):
            tid = int(prompt_ids[i % len(prompt_ids)])
            pos = i % S.MAX_POS
            S.update_input_buffers(state, tid, pos)
            execute_ms.append(timed(
                f"execute_trace_only_{i}",
                lambda: ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False),
            ))

            t0 = time.perf_counter()
            next_id = S._read_argmax_id(state, state.traced_argmax_tt)
            readback_ms.append((time.perf_counter() - t0) * 1000.0)
            next_ids.append(next_id)

            pos2 = (pos + 1) % S.MAX_POS
            update_execute_ms.append(timed(
                f"update_plus_execute_{i}",
                lambda next_id=next_id, pos2=pos2: (
                    S.update_input_buffers(state, next_id, pos2),
                    ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False),
                ),
            ))

        result["summary_ms"] = {
            "execute_trace": _summary(execute_ms),
            "update_plus_execute": _summary(update_execute_ms),
            "argmax_readback": _summary(readback_ms),
        }
        result["samples_ms"] = {
            "execute_trace": execute_ms,
            "update_plus_execute": update_execute_ms,
            "argmax_readback": readback_ms,
        }
        result["next_ids"] = next_ids
        print(json.dumps(result, indent=2), flush=True)

        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"[profile] wrote {out}", flush=True)
        return 0
    finally:
        if state.trace_id is not None and state.mesh is not None:
            try:
                ttnn.release_trace(state.mesh, state.trace_id)
            except Exception as exc:
                print(f"[profile] release_trace failed: {exc}", flush=True)
        if state.mesh is not None:
            try:
                ttnn.close_mesh_device(state.mesh)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                print("[profile] mesh closed, fabric disabled", flush=True)
            except Exception as exc:
                print(f"[profile] mesh cleanup failed: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
