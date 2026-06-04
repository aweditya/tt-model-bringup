#!/usr/bin/env python3
"""Single-client streaming probe — separates prefill (TTFT) from decode (tok/s).

The concurrent stress harness conflates prefill + decode into total wall time.
For per-token rates we need streaming + clock-on-first-byte.

Usage:
    python3 scripts/stress_ttft_decode.py --url http://qb1:8000 \\
        --model 'Qwen/Qwen3.6-27B' --max-tokens 64 --trials 3

Reports per trial:
    TTFT (s)     — submit → first delta byte (≈ prefill + 1 decode step)
    decode_s     — total stream duration after first byte
    n_decode_toks — completion tokens minus 1 (the first decode token was in TTFT)
    decode tok/s — n_decode_toks / decode_s
    total wall   — submit → last byte
    aggregate tok/s — completion_tokens / total wall
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def stream_one(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t_submit = time.time()
    ttft = None
    n_delta = 0
    completion = []
    last_t = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                if ttft is None:
                    ttft = time.time() - t_submit
                n_delta += 1
                completion.append(delta["content"])
                last_t = time.time()
    total_wall = (last_t - t_submit) if last_t is not None else (time.time() - t_submit)
    text = "".join(completion)
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_wall_s": round(total_wall, 3),
        "n_delta_chunks": n_delta,  # ~= completion tokens for non-batched tokens
        "completion": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("TT_CB_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Tell me about the city of Paris in three sentences.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    print(f"URL={args.url}\nmodel={args.model}\nprompt={args.prompt!r}\n"
          f"max_tokens={args.max_tokens}\ntrials={args.trials}\n", flush=True)

    rows = []
    for t in range(args.trials):
        r = stream_one(args.url, args.model, args.prompt, args.max_tokens)
        # Approx prefill = TTFT minus one decode step.
        # Decode tok/s = (n_delta - 1) / (total_wall - TTFT) if both positive.
        ttft = r["ttft_s"] or 0.0
        wall = r["total_wall_s"]
        n = r["n_delta_chunks"]
        decode_s = max(wall - ttft, 1e-6)
        decode_toks = max(n - 1, 0)
        decode_rate = decode_toks / decode_s if decode_toks > 0 else 0.0
        agg = n / wall if wall > 0 else 0.0
        print(f"--- trial {t} ---", flush=True)
        print(f"  TTFT       : {ttft:.3f} s   (≈ prefill + 1 decode step)", flush=True)
        print(f"  decode_s   : {decode_s:.3f} s  ({decode_toks} tokens)", flush=True)
        print(f"  decode rate: {decode_rate:.2f} tok/s", flush=True)
        print(f"  total wall : {wall:.3f} s   ({n} chunks)", flush=True)
        print(f"  aggregate  : {agg:.2f} tok/s (chunks / wall)", flush=True)
        rows.append({
            "trial": t, "ttft_s": ttft, "decode_s": round(decode_s, 3),
            "decode_toks": decode_toks, "decode_tok_s": round(decode_rate, 2),
            "total_wall_s": wall, "n_chunks": n,
        })

    if rows:
        n = len(rows)
        avg_ttft = sum(r["ttft_s"] for r in rows) / n
        avg_decode = sum(r["decode_tok_s"] for r in rows) / n
        avg_wall = sum(r["total_wall_s"] for r in rows) / n
        print("\n=== SUMMARY ===", flush=True)
        print(f"  avg TTFT   : {avg_ttft:.3f} s", flush=True)
        print(f"  avg decode : {avg_decode:.2f} tok/s", flush=True)
        print(f"  avg wall   : {avg_wall:.3f} s", flush=True)


if __name__ == "__main__":
    main()
