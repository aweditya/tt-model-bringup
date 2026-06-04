#!/usr/bin/env python3
"""Concurrent client stress test for the CB-backed OpenAI-compatible
server. Fires N parallel /v1/chat/completions requests and measures
aggregate tok/s vs single-client tok/s. Under continuous batching the
aggregate should scale near-linearly with N while per-request latency
stays bounded.

Forks no existing file (one-shot stress test; lives under scripts/
not experiments/ because it's a client-side tool that runs locally
against the remote server).

Usage:
    # From local laptop; reaches qb1 via SSH tunnel or set TT_CB_URL.
    python3 scripts/stress_concurrent_chat.py \\
        --url http://qb1:8000 \\
        --clients 1,2,4 \\
        --max-tokens 64

Outputs:
    presentation/screenshots/stress_concurrent_chat_<timestamp>.json
    Stdout: live progress + final table.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "Tell me about the city of Paris in three sentences.",
    "What's a good recipe for a quick weeknight pasta?",
    "Explain how a transistor works to a high-school student.",
    "Write a one-paragraph mystery story set in a library.",
    "What is the difference between TCP and UDP?",
    "Recommend a fun day trip from San Francisco.",
    "How do photosynthesis and cellular respiration interact?",
    "Summarize the plot of Hamlet in two sentences.",
]


def one_request(url, prompt, max_tokens, model, idx):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"idx": idx, "ok": False, "err": str(e), "code": e.code}
    except Exception as e:
        return {"idx": idx, "ok": False, "err": repr(e)}
    elapsed = time.time() - t0
    usage = data.get("usage", {})
    out = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "idx": idx,
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "prompt": prompt,
        "completion": out[:200],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "tok_s_completion": (usage.get("completion_tokens", 0) / elapsed) if elapsed > 0 else None,
    }


def run_round(url, model, n_clients, max_tokens, prompts):
    print(f"\n=== {n_clients} concurrent client(s), max_tokens={max_tokens} ===", flush=True)
    sel = (prompts * ((n_clients + len(prompts) - 1) // len(prompts)))[:n_clients]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n_clients) as ex:
        futs = [ex.submit(one_request, url, p, max_tokens, model, i)
                for i, p in enumerate(sel)]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.time() - t0
    ok = [r for r in results if r.get("ok")]
    total_completion_toks = sum(r.get("completion_tokens") or 0 for r in ok)
    agg_tok_s = total_completion_toks / wall if wall > 0 else 0.0
    print(f"  wall: {wall:.2f}s   total_completion_toks: {total_completion_toks}   "
          f"aggregate tok/s: {agg_tok_s:.2f}", flush=True)
    for r in sorted(results, key=lambda x: x["idx"]):
        if r.get("ok"):
            print(f"  client {r['idx']}: {r['elapsed_s']:.2f}s, "
                  f"{r.get('completion_tokens')} tok, "
                  f"{r.get('tok_s_completion'):.2f} tok/s — {r['completion'][:80]!r}",
                  flush=True)
        else:
            print(f"  client {r['idx']}: FAIL {r.get('err')}", flush=True)
    return {
        "n_clients": n_clients,
        "wall_s": round(wall, 3),
        "total_completion_tokens": total_completion_toks,
        "aggregate_tok_s": round(agg_tok_s, 3),
        "per_client": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("TT_CB_URL", "http://qb1:8000"))
    ap.add_argument("--model", default=os.environ.get("TT_CB_MODEL", "gemma4_12b"))
    ap.add_argument("--clients", default="1,2,4",
                    help="comma-sep N values for parallel client count")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: presentation/screenshots/...)")
    args = ap.parse_args()

    clients = [int(x) for x in args.clients.split(",")]
    print(f"URL={args.url}   model={args.model}   clients={clients}", flush=True)

    # Health gate
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=5) as r:
            if r.status != 200:
                print(f"FATAL: /health -> {r.status}", flush=True)
                return 1
    except Exception as e:
        print(f"FATAL: /health unreachable: {e!r}", flush=True)
        return 1

    summary = []
    for n in clients:
        summary.append(run_round(args.url, args.model, n, args.max_tokens, PROMPTS))

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'N':>3}  {'wall':>8}  {'total_toks':>12}  {'agg_tok/s':>12}  {'speedup_over_1':>14}", flush=True)
    baseline = summary[0]["aggregate_tok_s"] if summary else 1.0
    for s in summary:
        speedup = s["aggregate_tok_s"] / baseline if baseline > 0 else 0.0
        print(f"{s['n_clients']:>3}  {s['wall_s']:>8.2f}  "
              f"{s['total_completion_tokens']:>12}  "
              f"{s['aggregate_tok_s']:>12.2f}  {speedup:>13.2f}×", flush=True)

    if args.out is None:
        out_dir = Path(__file__).resolve().parents[1] / "presentation" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f"stress_concurrent_chat_{int(time.time())}.json")
    Path(args.out).write_text(json.dumps({"args": vars(args), "rounds": summary}, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
