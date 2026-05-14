"""Minimal CLI client for server_tp.py (multi-chip persistent server on qb2).

Usage:
    python -m experiments.serve.client_tp status
    python -m experiments.serve.client_tp generate_tp --prompt "..." --max-tokens 60
    python -m experiments.serve.client_tp shutdown

Points at $TT_CACHE/server_tp.sock (different from the single-chip server.sock).
"""
import argparse
import json
import os
import socket
import sys

from experiments.serve import protocol as P


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOCKET_PATH = os.path.join(PROJECT_ROOT, ".cache", "server_tp.sock")


def _send(cmd: str, args: dict, timeout: float = 7200.0) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client_tp: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        sock.sendall(P.pack_request(cmd, args))
        raw = P.read_line(sock, max_bytes=64 << 20)
    finally:
        sock.close()
    if not raw:
        print("client_tp: server returned no data (process likely died)", file=sys.stderr)
        sys.exit(3)
    resp = P.parse_response(raw)
    if resp.type == "error":
        print(f"server error: {resp.msg}", file=sys.stderr)
        sys.exit(4)
    return resp.data or {}


def cmd_status(_):
    print(json.dumps(_send("status", {}), indent=2, default=str))


def cmd_shutdown(_):
    print(json.dumps(_send("shutdown", {}), indent=2))


def cmd_bench_decode_tp_components(args):
    data = _send("bench_decode_tp_components", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_generate_tp(args):
    """Streaming generate_tp — same chunk/result protocol as client.cmd_generate."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(7200.0)
    try:
        sock.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client_tp: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens,
               "chunk_size": args.chunk_size, "seed": args.seed}
    final = None
    try:
        sock.sendall(P.pack_request("generate_tp", payload))
        print("=" * 72)
        print(f"prompt: {args.prompt}")
        print("-" * 72)
        while True:
            raw = P.read_line(sock, max_bytes=64 << 20)
            if not raw:
                print("\nclient_tp: server closed connection before final", file=sys.stderr)
                break
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                print(f"\nERROR: {obj.get('msg')}", file=sys.stderr)
                sys.exit(4)
            if t == "chunk":
                print(obj.get("data", {}).get("token_text", ""), end="", flush=True)
            elif t == "result":
                final = obj.get("data", {})
                break
    finally:
        sock.close()
    print("\n" + "=" * 72)
    if final is None:
        return
    print(f"  prompt: {final.get('n_prompt_tokens', 0)} tokens, "
          f"prefill {final.get('prefill_ms', 0):.1f} ms")
    print(f"  generated: {final.get('n_generated_tokens', 0)} tokens, "
          f"decode {final.get('ms_per_tok', 0):.2f} ms/tok = "
          f"{final.get('tok_per_sec', 0):.2f} tok/s")
    print(f"  total wall: {final.get('total_ms', 0):.1f} ms")
    if final.get('stopped_on_eos'):
        print(f"  (stopped on EOS)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("shutdown").set_defaults(fn=cmd_shutdown)
    b = sub.add_parser("bench_decode_tp_components",
                       help="server-resident TP component timing")
    b.add_argument("--prompt", default="The capital of France is")
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--warmup", type=int, default=3)
    b.set_defaults(fn=cmd_bench_decode_tp_components)
    g = sub.add_parser("generate_tp", help="multi-chip generate (streams)")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=60)
    g.add_argument("--chunk-size", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate_tp)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
