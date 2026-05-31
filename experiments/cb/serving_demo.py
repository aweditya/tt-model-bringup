#!/usr/bin/env python3
"""Continuous-batching serving demo: N concurrent socket clients -> one CB server.

ONE device-owning thread runs the Orca scheduler (server_tp_cb, B slots) plus a
Unix-socket select loop (accept -> submit -> step -> stream tokens). N client
THREADS connect concurrently and each streams its own response. Device access is
single-threaded (safe); concurrency is real (separate socket connections), and the
scheduler batches the in-flight requests across slots — showing the CB throughput
win vs one request at a time.

Run on qb1 (from repo root):
  make run PY=experiments/cb/serving_demo.py        # defaults: 4 slots, 8 clients, 24 new tok
  scripts/run_remote.sh experiments/cb/serving_demo.py --slots 4 --clients 8 --max-new 24
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import threading
import time
from pathlib import Path

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
from cb_scheduler import Scheduler           # noqa: E402

SOCK = str(_PROJECT / ".cache" / "cb_demo.sock")


PROMPTS = [
    "The capital of France is",
    "Once upon a time there lived a",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "The author of Romeo and Juliet is",
    "Photosynthesis is the process by which",
    "The speed of light in a vacuum is",
    "A balanced breakfast usually includes",
]


def _client(idx, prompt, results, ready):
    """A concurrent client: connect, send prompt, stream tokens, record result."""
    ready.wait()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    t0 = time.time()
    s.sendall((json.dumps({"prompt": prompt}) + "\n").encode())
    buf = b""
    n_tok = 0
    text = ""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            obj = json.loads(line)
            if obj.get("done"):
                text = obj["text"]
                results[idx] = {"text": text, "n_tok": n_tok, "latency": time.time() - t0}
                s.close()
                return
            n_tok += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    eos_id = getattr(state.tok, "eos_token_id", None)
    sched = Scheduler(state, args.slots, args.max_new, eos_id, use_trace=True)
    log(f"CB server ready: {args.slots} slots, trace-captured. Firing {args.clients} clients…")

    # Server socket (non-blocking accept via select; this thread owns the device).
    try:
        os.unlink(SOCK)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(64); srv.setblocking(False)

    results = [None] * args.clients
    ready = threading.Event()
    threads = [threading.Thread(target=_client, args=(i, PROMPTS[i % len(PROMPTS)], results, ready),
                                daemon=True) for i in range(args.clients)]
    for t in threads:
        t.start()

    conn_by_rid = {}          # rid -> conn (admitted requests)
    sent = {}                 # rid -> tokens already streamed
    reading = {}              # conn -> partial request bytes (pre-submit)
    submitted = 0
    wall0 = time.time()
    ready.set()               # release the clients to connect

    while submitted < args.clients or conn_by_rid or reading:
        rlist = [srv] + list(reading.keys())
        rd, _, _ = select.select(rlist, [], [], 0.0)
        for s in rd:
            if s is srv:
                try:
                    conn, _ = srv.accept()
                    conn.setblocking(True)  # recv only after select-readable; sendall blocking-safe
                    reading[conn] = b""
                except BlockingIOError:
                    pass
            else:
                data = s.recv(65536)
                reading[s] += data
                if b"\n" in reading[s]:
                    line = reading[s].split(b"\n", 1)[0]
                    del reading[s]
                    prompt = json.loads(line)["prompt"]
                    rid = sched.submit(state.tok.encode(prompt))
                    conn_by_rid[rid] = s; sent[rid] = 0; submitted += 1

        if sched.waiting or any(slot is not None for slot in sched.slots):
            sched.step()
            for rid in list(conn_by_rid.keys()):
                r = sched.reqs[rid]; conn = conn_by_rid[rid]
                while sent[rid] < len(r["gen"]):
                    conn.sendall((json.dumps({"token_id": r["gen"][sent[rid]]}) + "\n").encode())
                    sent[rid] += 1
                if r["status"] == "DONE":
                    conn.sendall((json.dumps({"done": True, "text": state.tok.decode(r["gen"])}) + "\n").encode())
                    conn.close(); del conn_by_rid[rid]; del sent[rid]
        else:
            time.sleep(0.001)

    for t in threads:
        t.join(timeout=10)
    wall = time.time() - wall0
    srv.close(); sched.release()
    try:
        os.unlink(SOCK)
    except OSError:
        pass

    total_tok = sum(r["n_tok"] for r in results if r)
    log(f"=== {args.clients} clients / {args.slots} slots done in {wall:.2f}s ===")
    for i, r in enumerate(results):
        if r:
            log(f"  client {i}: {r['n_tok']} tok in {r['latency']:.2f}s — {r['text']!r}")
    agg = total_tok / wall if wall > 0 else 0.0
    log(f"=== aggregate {total_tok} tokens / {wall:.2f}s = {agg:.1f} tok/s "
        f"(B=1 prod baseline ~12.96 tok/s → ~{agg / 12.96:.1f}x via continuous batching) ===")


if __name__ == "__main__":
    main()
