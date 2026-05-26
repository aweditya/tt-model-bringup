#!/usr/bin/env python3
"""Repeatable qb2 TP server benchmark via the Unix-socket generate_tp endpoint.

This is intentionally a server-client benchmark, not a raw ttnn probe: it can
run while the persistent qb2 server owns the mesh device, and it measures the
production decode path that users actually exercise.

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/utils/qb2_tp_generate_bench.py \
        --runs 5 --warmup 1 --max-tokens 30

Outputs JSON to ~/tt-xla/.cache/qb2_tp_generate_bench/ unless --out is passed.
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from experiments.serve import protocol as P  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


SOCKET_PATH = os.path.expanduser("~/tt-xla/.cache/server_tp.sock")
OUT_DIR = os.path.expanduser("~/tt-xla/.cache/qb2_tp_generate_bench")


def read_frames(sock):
    buf = bytearray()
    while True:
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            if line:
                yield json.loads(line.decode("utf-8"))
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf.extend(chunk)


def run_once(prompt, max_tokens, chunk_size, timeout):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(SOCKET_PATH)
    req = P.pack_request("generate_tp", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "chunk_size": chunk_size,
        "seed": 0,
    })
    t0 = time.perf_counter()
    sock.sendall(req)
    chunks = []
    final = None
    try:
        for frame in read_frames(sock):
            typ = frame.get("type")
            if typ == "error":
                raise RuntimeError(frame.get("msg", "server error"))
            if typ == "chunk":
                chunks.append(frame.get("data", {}).get("token_text", ""))
            elif typ == "result":
                final = frame.get("data", {})
                break
    finally:
        sock.close()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if final is None:
        raise RuntimeError("server closed before final result")
    final["client_wall_ms"] = wall_ms
    final["streamed_text"] = "".join(chunks)
    return final


def summarize(values):
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    total = args.warmup + args.runs
    records = []
    for i in range(total):
        label = "warmup" if i < args.warmup else f"run{i - args.warmup}"
        print(f"[{label}] prompt={args.prompt!r} max_tokens={args.max_tokens}")
        rec = run_once(args.prompt, args.max_tokens, args.chunk_size, args.timeout)
        rec["label"] = label
        rec["is_warmup"] = i < args.warmup
        records.append(rec)
        print(f"  {rec.get('ms_per_tok', float('nan')):.2f} ms/tok "
              f"= {rec.get('tok_per_sec', 0.0):.2f} tok/s; "
              f"text={rec.get('generated_text', '')[:80]!r}")

    measured = [r for r in records if not r["is_warmup"]]
    ms = [float(r["ms_per_tok"]) for r in measured]
    tps = [float(r["tok_per_sec"]) for r in measured]
    result = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "chunk_size": args.chunk_size,
        "runs": args.runs,
        "warmup": args.warmup,
        "summary": {
            "ms_per_tok": summarize(ms),
            "tok_per_sec": summarize(tps),
        },
        "records": records,
    }

    out = args.out
    if not out:
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(OUT_DIR, f"results_{stamp}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
