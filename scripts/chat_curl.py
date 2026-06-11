#!/usr/bin/env python3
"""Send a single chat completion to the running cb_api server. Permanent
helper so we don't fight shell quoting on every smoke.

REUSE: forks the body-building pattern from scripts/stress_concurrent_chat.py
one_request(). Single-request variant for ad-hoc smoke testing.

Run on qb1 directly OR through an SSH tunnel:
    python3 scripts/chat_curl.py --prompt "What is Tenstorrent?" \\
        --max-tokens 300 --temp 0.7 --top-p 0.9 --seed 1
    # or with tunnel
    python3 scripts/chat_curl.py --url http://localhost:8000 \\
        --prompt "..." --max-tokens 300 --temp 0.7 --seed 42 \\
        --out .cache/gm4_long_decode_proxy/seed_42.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="server base URL")
    ap.add_argument("--model", default="google/gemma-4-12B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="write full JSON response to this path; "
                         "default = print to stdout")
    args = ap.parse_args()

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        body["seed"] = args.seed

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"URL error: {e}", file=sys.stderr)
        return 2

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(raw)
        # Print a short summary so the user can see what landed.
        d = json.loads(raw)
        text = d["choices"][0]["message"]["content"]
        finish = d["choices"][0]["finish_reason"]
        toks = d["usage"]["completion_tokens"]
        print(f"wrote {len(raw)} bytes → {args.out}")
        print(f"finish={finish}  completion_tokens={toks}  "
              f"content_chars={len(text)}")
        if "#" in text:
            n_hash = text.count("#")
            print(f"#-count={n_hash} ({n_hash / max(1,len(text)) * 100:.1f}%)")
        print("---")
        print(text[:400])
        if len(text) > 400:
            print(f"… <{len(text) - 400} chars omitted>")
    else:
        print(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
