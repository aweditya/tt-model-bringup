#!/usr/bin/env python3
"""Helper: call the persistent inference server's run_91r and pretty-print
the per-layer worst-position cosine."""
import json
import socket
import sys
from pathlib import Path

SOCK = Path.home() / "tt-xla" / ".cache" / "server.sock"


def main():
    layers_arg = sys.argv[1] if len(sys.argv) > 1 else "0,3,7,11,15,31,47,63"
    layers = [int(x) for x in layers_arg.split(",")]
    req = {"cmd": "run_91r", "args": {"layers": layers}}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(SOCK))
    s.sendall((json.dumps(req) + "\n").encode())
    f = s.makefile("rb")
    line = f.readline()
    if not line:
        print("no response")
        return
    msg = json.loads(line.decode())
    if msg.get("type") == "error":
        print("server error:", msg.get("msg", "?"))
        return
    data = msg.get("data", {})
    print(f"phase: server | total_sec={data.get('total_sec', '?'):.3f}")
    print(f"{'layer':>6} {'type':>18} {'worst_cos':>10}")
    print("-" * 40)
    for r in data.get("results", []):
        cs = r.get("cosines", [])
        worst = min(cs) if cs else float("nan")
        print(f"{r.get('layer'):>6} {r.get('type'):>18} {worst:>10.6f}")


if __name__ == "__main__":
    main()
