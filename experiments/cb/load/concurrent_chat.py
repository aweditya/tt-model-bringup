#!/usr/bin/env python3
"""P5 — load / SLO validation: N concurrent SSE chat clients vs a running daemon.

Assumes `experiments/serve/scripts/serve_cb.sh start` has come up on the same
host and /health returns 200. Spawns N threads that each loop
POST /v1/chat/completions (stream=true) for `duration` seconds with rotating
prompts; measures per-request TTFT + total wall + tokens; scrapes /metrics
every 2s to track engine-side state. Reports aggregate throughput, latency
p50/p99, errors, then asserts the SLO floor (no errors, every client
completed >0 requests, /metrics counters advanced).

Run on qb1 (after the daemon is up):
  cd ~/tt-xla && .venv/bin/python -m experiments.cb.load.concurrent_chat \\
      --clients 8 --duration 60 --max-tokens 32 --sampling
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROMPTS = [
    "Tell me a fun fact about space.",
    "What's the best way to start learning Python?",
    "Explain photosynthesis in one paragraph.",
    "Write a haiku about silicon.",
    "Why is the sky blue?",
    "Give me a recipe for a quick pasta dinner.",
    "What are three interesting things about octopuses?",
    "Summarise the plot of Hamlet in three sentences.",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _stream_chat(host: str, port: int, prompt: str, max_tokens: int,
                 sampling: bool, timeout: float = 120.0):
    """POST /v1/chat/completions stream=true; yield (event_kind, payload).
       event_kind ∈ {'first_byte', 'delta', 'finish', 'done', 'error'}."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if sampling:
        body.update(temperature=0.8, top_p=0.95)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions",
                     json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            yield ("error", f"HTTP {resp.status} {resp.read()[:200].decode('utf-8', 'replace')}")
            return
        first = True
        while True:
            line = resp.readline()
            if not line:
                return
            line = line.rstrip(b"\r\n")
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:].decode("utf-8")
            if payload == "[DONE]":
                yield ("done", None)
                return
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = ev.get("choices", [{}])[0]
            if first:
                yield ("first_byte", None)
                first = False
            if ch.get("finish_reason"):
                yield ("finish", ch["finish_reason"])
                continue
            d = ch.get("delta", {}).get("content")
            if d:
                yield ("delta", d)
    finally:
        conn.close()


def _client_loop(idx: int, host: str, port: int, t_end: float, max_tokens: int,
                 sampling: bool, results: list, errors: list):
    n = 0
    while time.time() < t_end:
        prompt = PROMPTS[(idx + n) % len(PROMPTS)]
        t0 = time.time()
        ttft = None
        tokens = 0
        err = None
        finish = None
        for kind, payload in _stream_chat(host, port, prompt, max_tokens, sampling):
            if kind == "first_byte":
                ttft = time.time() - t0
            elif kind == "delta":
                tokens += 1
            elif kind == "finish":
                finish = payload
            elif kind == "error":
                err = payload
                break
        wall = time.time() - t0
        if err is not None:
            errors.append({"client": idx, "n": n, "error": err})
        else:
            results.append({"client": idx, "n": n, "ttft": ttft or wall,
                            "wall": wall, "tokens": tokens, "finish": finish})
        n += 1


def _scrape_metrics(host: str, port: int) -> dict | None:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=3.0)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        conn.close()
        if resp.status != 200:
            return None
        m: dict = {}
        for line in body.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and "{" not in parts[0]:
                try:
                    m[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return m
    except Exception:
        return None


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--sampling", action="store_true",
                    help="use temperature=0.8 top_p=0.95 (default greedy)")
    args = ap.parse_args()

    # preflight: daemon ready?
    m0 = _scrape_metrics(args.host, args.port)
    if m0 is None:
        log(f"FAIL: cannot reach {args.host}:{args.port}/metrics — start serve_cb.sh first")
        sys.exit(2)
    log(f"daemon up; baseline submitted={m0.get('cb_requests_submitted_total', 0):.0f} "
        f"tokens={m0.get('cb_tokens_generated_total', 0):.0f}")

    results: list = []
    errors: list = []
    t_start = time.time()
    t_end = t_start + args.duration
    threads = [threading.Thread(
        target=_client_loop,
        args=(i, args.host, args.port, t_end, args.max_tokens, args.sampling, results, errors),
        name=f"client-{i}", daemon=True,
    ) for i in range(args.clients)]
    for t in threads:
        t.start()
    log(f"firing {args.clients} clients for {args.duration}s "
        f"({'sampling t=0.8' if args.sampling else 'greedy'}, max_tokens={args.max_tokens})…")

    snapshots: list = [(0.0, m0)]
    while any(t.is_alive() for t in threads):
        time.sleep(2.0)
        m = _scrape_metrics(args.host, args.port)
        if m is not None:
            snapshots.append((time.time() - t_start, m))
    for t in threads:
        t.join(timeout=10)

    wall = time.time() - t_start
    n_req = len(results)
    n_err = len(errors)
    total_tokens = sum(r["tokens"] for r in results)
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    walls = [r["wall"] for r in results]
    per_client = {i: 0 for i in range(args.clients)}
    for r in results:
        per_client[r["client"]] += 1

    log(f"\n=== {args.clients} clients / {args.duration}s ({wall:.1f}s wall) ===")
    log(f"  requests: {n_req} done, {n_err} errored")
    log(f"  tokens:   {total_tokens} ({total_tokens / wall:.1f} tok/s aggregate)")
    log(f"  TTFT:     p50={_pct(ttfts, 0.5)*1000:.0f}ms  p99={_pct(ttfts, 0.99)*1000:.0f}ms  "
        f"mean={statistics.mean(ttfts)*1000:.0f}ms (n={len(ttfts)})")
    log(f"  request:  p50={_pct(walls, 0.5)*1000:.0f}ms  p99={_pct(walls, 0.99)*1000:.0f}ms  "
        f"mean={statistics.mean(walls)*1000:.0f}ms")
    log(f"  per-client requests: min={min(per_client.values())} "
        f"max={max(per_client.values())} median={int(statistics.median(per_client.values()))}")
    if n_err:
        for e in errors[:5]:
            log(f"  error sample: client={e['client']} n={e['n']} {e['error'][:120]!r}")

    m_end = snapshots[-1][1]
    log("\n  /metrics delta over the run:")
    for k in ("cb_requests_submitted_total", "cb_requests_done_total",
              "cb_requests_cancelled_total", "cb_requests_rejected_total",
              "cb_tokens_generated_total", "cb_step_seconds_count"):
        log(f"    {k}: {m0.get(k, 0):.0f} → {m_end.get(k, 0):.0f}  "
            f"(+{m_end.get(k, 0) - m0.get(k, 0):.0f})")

    # SLO floor for the gate: no errors, every client did >=1 request,
    # /metrics counters actually moved.
    no_errors = n_err == 0
    all_clients_active = all(v >= 1 for v in per_client.values())
    counters_moved = (m_end.get("cb_requests_submitted_total", 0) -
                      m0.get("cb_requests_submitted_total", 0)) >= n_req
    ok = no_errors and all_clients_active and counters_moved
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log(f"  no_errors={no_errors}  all_clients_active={all_clients_active}  "
        f"counters_moved={counters_moved}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
