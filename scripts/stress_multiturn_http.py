#!/usr/bin/env python3
"""Multi-turn HTTP stress test — demonstrates prefix caching across
turns of a single chat session.

Sends 3 sequential /v1/chat/completions requests where each turn
appends to the prior chat history. With TT_CB_PREFIX_CACHE=1 on the
server, turn 2's and turn 3's prefill should reclaim the cached
prefix and only prefill the new suffix (the latest user message +
assistant continuation up to the previous turn's end).

Outputs per-turn:
  - prompt_tokens (full chat-templated length, GROWS each turn)
  - completion_tokens
  - server-side TTFT proxy (wall-clock from request submit → first byte)
  - total request wall time

Compare turn 2 wall time vs turn 1 — with PC enabled, turn 2 should
NOT be 2× turn 1 even though the prompt is roughly 2× longer. The
difference is the prefix-cache benefit.

Usage:
    python3 scripts/stress_multiturn_http.py --url http://qb1:8000 \\
        --model 'google/gemma-4-12B' --turns 3 --max-tokens 48
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

TURNS = [
    "Hello! My name is Aditya. Could you suggest a 4-character "
    "abbreviation I can use as a nickname?",
    "Tell me a one-sentence interesting fact about chess.",
    "Now combine both — write a one-sentence chess strategy named "
    "after the nickname you suggested for me.",
]


def chat(url, model, messages, max_tokens, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": messages,
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    out = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return out, usage, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="google/gemma-4-12B")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    messages = []
    rows = []
    for i in range(min(args.turns, len(TURNS))):
        user = TURNS[i]
        messages.append({"role": "user", "content": user})
        out, usage, elapsed = chat(args.url, args.model, messages, args.max_tokens)
        prompt_t = usage.get("prompt_tokens", -1)
        comp_t = usage.get("completion_tokens", -1)
        # Naive throughput math: assume server prefill+decode is the
        # majority of `elapsed`; report decode tok/s if comp_t > 0.
        decode_tok_s = comp_t / elapsed if elapsed > 0 else 0.0
        print(f"=== Turn {i} ===", flush=True)
        print(f"  user: {user!r}", flush=True)
        print(f"  prompt_toks={prompt_t}  completion_toks={comp_t}  "
              f"wall={elapsed:.2f}s  raw_tok/s={decode_tok_s:.2f}", flush=True)
        print(f"  assistant: {out!r}", flush=True)
        messages.append({"role": "assistant", "content": out})
        rows.append({
            "turn": i, "prompt_tokens": prompt_t,
            "completion_tokens": comp_t, "wall_s": round(elapsed, 3),
            "user": user, "assistant": out,
        })

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'turn':>4}  {'prompt_t':>9}  {'gen_t':>6}  {'wall_s':>8}  "
          f"{'wall/prompt_t':>14}", flush=True)
    for r in rows:
        ratio = r["wall_s"] / max(1, r["prompt_tokens"])
        print(f"{r['turn']:>4}  {r['prompt_tokens']:>9}  "
              f"{r['completion_tokens']:>6}  {r['wall_s']:>8.2f}  "
              f"{ratio:>13.4f}", flush=True)
    print("\nNote: with TT_CB_PREFIX_CACHE=1 enabled, turn 2/3 wall_s "
          "should be much smaller than turn 1 wall_s × (prompt_t ratio) "
          "— the cached prefix means only the new suffix gets prefilled.",
          flush=True)

    if args.out is None:
        out_dir = Path(__file__).resolve().parents[1] / "presentation" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f"stress_multiturn_http_{int(time.time())}.json")
    Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
