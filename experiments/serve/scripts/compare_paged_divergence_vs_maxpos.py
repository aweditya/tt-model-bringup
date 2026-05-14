#!/usr/bin/env python3
"""Test: does the paged-vs-nonpaged divergence step depend on max_pos?

If divergence at step 132 is purely from bf16 noise compounding, it should be
INDEPENDENT of the paged cache size (max_pos) — the kernels do the same math
either way. If divergence shifts with max_pos, something about cache size is
involved (block count, page table size, etc).

We run paged with max_pos in {256, 512, 1024} and report first divergence
from the non-paged reference (using the saved nonpaged ids from
compare_paged_vs_nonpaged.py).

Run on qb1:
    ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.serve.scripts.compare_paged_divergence_vs_maxpos'
"""
import json
import socket
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from experiments.serve import protocol as P  # noqa

PROMPT = "Implement a JSON parser combinator in Rust"
MAX_TOKENS = 200


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
    finally:
        sock.close()
    return final or {}


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None


def main():
    # Get reference (non-paged) first
    print("[ref] non-paged baseline…")
    r_ref = call("generate", {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "chunk_size": 8})
    ids_ref = r_ref.get("generated_ids", [])
    print(f"  non-paged ids len={len(ids_ref)}")

    # Try paged with multiple max_pos values
    for max_pos in (256, 512, 1024):
        print(f"\n[paged max_pos={max_pos}]")
        r = call("generate_long", {"prompt": PROMPT, "max_tokens": MAX_TOKENS,
                                     "max_pos": max_pos, "block_size": 64,
                                     "chunk_size": 8})
        if "error" in r:
            print(f"  error: {r['error']}"); continue
        ids = r.get("generated_ids", [])
        prompt_len = r.get("n_prompt_tokens", 0)
        d = first_diff(ids_ref, ids)
        if d is None:
            print(f"  identical to non-paged for all {min(len(ids_ref), len(ids))} tokens")
        else:
            print(f"  first divergence at step {d} (cur_pos {prompt_len + d}, "
                  f"block {(prompt_len + d) // 64}, slot {(prompt_len + d) % 64})")
            print(f"  ref token #{d}: id={ids_ref[d]}")
            print(f"  paged token #{d}: id={ids[d]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
