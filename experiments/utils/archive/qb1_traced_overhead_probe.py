#!/usr/bin/env python3
"""Measure qb1 traced-decode host/I/O overhead through the resident server.

This helper intentionally does not import ttnn, open devices, or manage the
server lifecycle. It only calls the existing qb1 Unix-socket endpoint
`bench_decode_traced`, then reports the gap between full step time and
execute_trace-only time.

Run on qb1 only after coordinating with whoever owns the qb1 server:
    cd ~/tt-xla && .venv/bin/python experiments/utils/qb1_traced_overhead_probe.py \
        --tokens 32 --runs 3 --validate-steps 5

Decision gate:
  - If full-minus-exec median is >= 5 ms/token, prototype on-device argmax /
    token feedback or multi-CQ I/O overlap before touching kernels.
  - If it is < 5 ms/token, focus on traced kernel body instead.
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from experiments.serve import protocol as P

sys.stdout.reconfigure(line_buffering=True)


def _rpc(cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(P.SOCKET_PATH)
        sock.sendall(P.pack_request(cmd, args))
        resp = P.parse_response(P.read_line(sock, max_bytes=64 << 20))
    finally:
        sock.close()
    if resp.type == "error":
        raise RuntimeError(f"{cmd} failed: {resp.msg}")
    return resp.data or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32,
                        help="Timed traced decode steps per run.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Untimed traced decode steps per run.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Repeat RPCs to estimate server-side variance.")
    parser.add_argument("--validate-steps", type=int, default=5,
                        help="Eager-vs-traced validation steps per run; use 0 only for perf-only reruns.")
    parser.add_argument("--recapture-first", action="store_true",
                        help="Force trace recapture on the first run.")
    parser.add_argument("--output-json", type=str, default="",
                        help="Optional path for a durable JSON result artifact.")
    parser.add_argument("--threshold-ms", type=float, default=5.0,
                        help="Host/I/O overhead threshold for pursuing token-feedback work.")
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status = _rpc("status", {})
    print("qb1 traced overhead probe")
    print(f"  server loaded={status.get('loaded')} mock={status.get('mock')} "
          f"device_id={status.get('device_id')}")
    print(f"  tokens={args.tokens} warmup={args.warmup} runs={args.runs} "
          f"validate_steps={args.validate_steps} threshold_ms={args.threshold_ms:.2f}")

    rows = []
    for run_idx in range(args.runs):
        data = _rpc("bench_decode_traced", {
            "n_steps": args.tokens,
            "warmup": args.warmup,
            "validate_steps": args.validate_steps,
            "recapture": bool(args.recapture_first and run_idx == 0),
        })
        median_ms = float(data["median_ms"])
        median_exec_ms = float(data["median_exec_ms"])
        overhead_ms = median_ms - median_exec_ms
        rows.append((median_ms, median_exec_ms, overhead_ms))
        print(
            f"  run {run_idx + 1}: full={median_ms:.2f} ms/tok  "
            f"execute_trace={median_exec_ms:.2f}  overhead={overhead_ms:.2f}  "
            f"tok/s={data['tok_per_sec']:.2f}  min_cos={data.get('min_cosine')}"
        )

    full = [r[0] for r in rows]
    exec_only = [r[1] for r in rows]
    overhead = [r[2] for r in rows]
    summary = {
        "started_at": started_at,
        "runs": args.runs,
        "tokens": args.tokens,
        "warmup": args.warmup,
        "validate_steps": args.validate_steps,
        "status": status,
        "median_full_ms": statistics.median(full),
        "median_execute_trace_ms": statistics.median(exec_only),
        "median_overhead_ms": statistics.median(overhead),
        "overhead_fraction": statistics.median(overhead) / max(statistics.median(full), 1e-9),
        "threshold_ms": args.threshold_ms,
        "rows": [
            {"full_ms": a, "execute_trace_ms": b, "overhead_ms": c}
            for a, b, c in rows
        ],
    }
    summary["hypothesis"] = (
        "traced decode is still paying avoidable host/I/O time outside execute_trace"
    )
    summary["verdict"] = (
        "pursue_token_feedback_or_io_overlap"
        if summary["median_overhead_ms"] >= args.threshold_ms
        else "focus_kernel_body"
    )
    summary["invalidation_criteria"] = [
        f"median_overhead_ms < {args.threshold_ms:.2f} ms/tok across repeated resident-server runs",
        "validation cosine fails or server reports mock/unloaded state",
        "results require trace recapture or standalone device ownership instead of the persistent server",
    ]
    print("\nsummary")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp = args.output_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, args.output_json)
        print(f"\nwrote {args.output_json}")

    if summary["median_overhead_ms"] >= args.threshold_ms:
        print("\nnext: validate on-device argmax/token feedback or 2-CQ I/O overlap.")
    else:
        print("\nnext: host/readback overhead is small; prioritize traced kernel body probes.")


if __name__ == "__main__":
    main()
