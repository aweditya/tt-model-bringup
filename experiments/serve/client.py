"""CLI client for the persistent weight server.

Usage:
    python -m experiments.serve.client status
    python -m experiments.serve.client reset_state
    python -m experiments.serve.client reload_kernels
    python -m experiments.serve.client run_91r --layers 0,3,7
    python -m experiments.serve.client shutdown
"""
import argparse
import json
import socket
import sys

from experiments.serve import protocol as P


def send(cmd: str, args: dict, timeout: float = 7200.0) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(P.SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client: cannot connect to {P.SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        sock.sendall(P.pack_request(cmd, args))
        raw = P.read_line(sock, max_bytes=64 << 20)
    finally:
        sock.close()
    if not raw:
        print("client: server returned no data (process likely died)", file=sys.stderr)
        sys.exit(3)
    resp = P.parse_response(raw)
    if resp.type == "error":
        print(f"server error: {resp.msg}", file=sys.stderr)
        sys.exit(4)
    return resp.data or {}


def cmd_status(_):
    data = send("status", {})
    print(json.dumps(data, indent=2, default=str))


def cmd_reset(_):
    print(json.dumps(send("reset_state", {}), indent=2))


def cmd_reload(_):
    print(json.dumps(send("reload_kernels", {}), indent=2))


def cmd_shutdown(_):
    print(json.dumps(send("shutdown", {}), indent=2))


def cmd_run_91r(args):
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    payload = {}
    if layers is not None:
        payload["layers"] = layers
    if args.weight_dtype:
        payload["weight_dtype"] = args.weight_dtype
    data = send("run_91r", payload)
    # Pretty summary first, then full JSON.
    print("=" * 72)
    print(f"run_91r layers={data.get('layers')} total_sec={data.get('total_sec', 0):.1f}")
    print("-" * 72)
    print(f"{'layer':>6s} {'type':>20s}  cosines (per pos)  -> worst")
    for r in data.get("results", []):
        cs = r["cosines"]
        worst = min(cs) if cs else float("nan")
        cs_str = " ".join(f"{c:.5f}" for c in cs)
        print(f"{r['layer']:6d} {r['type']:>20s}  {cs_str}  -> {worst:.5f}")
    print("=" * 72)
    print(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("reset_state").set_defaults(fn=cmd_reset)
    sub.add_parser("reload_kernels").set_defaults(fn=cmd_reload)
    sub.add_parser("shutdown").set_defaults(fn=cmd_shutdown)
    r = sub.add_parser("run_91r")
    r.add_argument("--layers", type=str, default=None,
                   help="comma-separated layer indices (default: server default)")
    r.add_argument("--weight-dtype", type=str, default=None,
                   choices=["bf8", "bf16", "fp32"])
    r.set_defaults(fn=cmd_run_91r)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
