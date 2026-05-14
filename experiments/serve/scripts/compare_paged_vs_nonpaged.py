#!/usr/bin/env python3
"""Compare per-step generated token IDs between paged (`generate_long`) and
non-paged (`generate`) server paths on the SAME prompt.

We have direct evidence the paged path degrades into garbage around token ~80
while the non-paged path stays coherent through 200+ tokens for the prompt
"Implement a JSON parser combinator in Rust". The paged kernels themselves
were validated bit-correct via experiments/utils/paged_write_read_iter_probe.py
(perfect bf16 writes for all 256 positions, cosines ≥0.99993 vs numpy for SDPA
reads across block boundaries).

This script asks the next-narrow question: when, in the autoregressive
sequence, do the two paths' top-1 tokens first diverge? That will give us the
position at which the paged forward starts producing different logits than
the non-paged forward, narrowing down where the (subtle, accumulating) bug
lives.

Run on qb1 (server must be running):
    ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.serve.scripts.compare_paged_vs_nonpaged'
"""
import json
import os
import socket
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import protocol from the serve package
from experiments.serve import protocol as P  # noqa


PROMPT = "Implement a JSON parser combinator in Rust"
MAX_TOKENS = 200          # short enough to be fast, long enough to span the
                          # block-1 boundary (drift onset ~token 80)


def call(cmd: str, payload: dict) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(7200.0)
    sock.connect(P.SOCKET_PATH)
    final = None
    try:
        sock.sendall(P.pack_request(cmd, payload))
        while True:
            raw = P.read_line(sock, max_bytes=64 << 20)
            if not raw:
                break
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "result":
                final = obj.get("data", {})
                break
            elif t == "error":
                final = {"error": obj.get("msg")}
                break
            # ignore chunk messages — final has the full ids list
    finally:
        sock.close()
    return final or {}


def main():
    print("=" * 72)
    print(f"  Compare paged vs non-paged generation: prompt = {PROMPT!r}")
    print(f"  max_tokens = {MAX_TOKENS}")
    print("=" * 72)

    print("[run 1/2] non-paged `generate`…")
    r1 = call("generate", {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "chunk_size": 4})
    if "error" in r1:
        print(f"non-paged error: {r1['error']}"); return 1
    ids_np = list(r1.get("generated_ids", []))
    text_np = r1.get("generated_text", "")
    print(f"  non-paged generated {len(ids_np)} tokens")

    print("[run 2/2] paged   `generate_long` (max_pos=256)…")
    r2 = call("generate_long", {"prompt": PROMPT, "max_tokens": MAX_TOKENS,
                                  "max_pos": 256, "block_size": 64, "chunk_size": 4})
    if "error" in r2:
        print(f"paged error: {r2['error']}"); return 1
    ids_p = list(r2.get("generated_ids", []))
    text_p = r2.get("generated_text", "")
    print(f"  paged     generated {len(ids_p)} tokens")

    # Find first divergence
    first_diff = None
    n = min(len(ids_np), len(ids_p))
    for i in range(n):
        if ids_np[i] != ids_p[i]:
            first_diff = i
            break

    if first_diff is None:
        if len(ids_np) == len(ids_p):
            print("\n[VERDICT] paged and non-paged produced IDENTICAL token sequences.")
            print("  This would contradict the user-report drift — re-check inputs.")
        else:
            print(f"\n[VERDICT] paged and non-paged agree on first {n} tokens "
                  f"but lengths differ ({len(ids_np)} vs {len(ids_p)})")
    else:
        print(f"\n[VERDICT] FIRST DIVERGENCE at token #{first_diff}")
        # Effective position: prompt_len + first_diff
        prompt_len = r1.get("n_prompt_tokens", 0)
        cur_pos_at_div = prompt_len + first_diff
        print(f"  At cur_pos = {cur_pos_at_div} (prompt_len={prompt_len} + step={first_diff})")
        # Block info
        block_size = 64
        print(f"  Block layout: cur_pos % {block_size} = {cur_pos_at_div % block_size}, "
              f"cur_pos // {block_size} = block #{cur_pos_at_div // block_size}")
        print(f"  non-paged tok #{first_diff}: id={ids_np[first_diff]}")
        print(f"  paged     tok #{first_diff}: id={ids_p[first_diff]}")
        # Show some context (token indices around first_diff)
        lo, hi = max(0, first_diff - 5), min(n, first_diff + 10)
        print(f"\n  Context (tok#: nonpaged_id, paged_id):")
        for i in range(lo, hi):
            mark = "  <-- DIVERGE" if i == first_diff else ""
            same = "✓" if ids_np[i] == ids_p[i] else "✗"
            print(f"    {i:3d}: {ids_np[i]:6d}  {ids_p[i]:6d}  {same}{mark}")

    # Save the artifact
    out = Path(__file__).parent / "_compare_paged_vs_nonpaged.json"
    out.write_text(json.dumps({
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "n_prompt_tokens": r1.get("n_prompt_tokens"),
        "ids_nonpaged": ids_np,
        "ids_paged":     ids_p,
        "first_diff":    first_diff,
        "text_nonpaged": text_np,
        "text_paged":    text_p,
    }, indent=2))
    print(f"\nArtifact saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
