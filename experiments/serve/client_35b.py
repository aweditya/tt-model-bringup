#!/usr/bin/env python3
"""Minimal CLI for server_35b.py — status / generate_35b / shutdown."""
import argparse
import json
import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import protocol as P  # noqa: E402

SOCK_PATH = PROJECT_ROOT / ".cache" / "server_35b.sock"


def _send(cmd, args, timeout=7200.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(SOCK_PATH))
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"cannot connect to {SOCK_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        s.sendall(P.pack_request(cmd, args))
        raw = P.read_line(s, max_bytes=64 << 20)
    finally:
        s.close()
    if not raw:
        print("server returned no data", file=sys.stderr)
        sys.exit(3)
    resp = P.parse_response(raw)
    if resp.type == "error":
        print(f"server error: {resp.msg}", file=sys.stderr)
        sys.exit(4)
    return resp.data or {}


def _stream(cmd, args, timeout=7200.0):
    """Stream chunks (one per yield); print as they arrive; return final."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(SOCK_PATH))
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"cannot connect to {SOCK_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        s.sendall(P.pack_request(cmd, args))
        final = None
        while True:
            raw = P.read_line(s)
            if not raw:
                break
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                print(f"\nerror: {obj.get('msg')}", file=sys.stderr)
                sys.exit(4)
            if t == "chunk":
                txt = obj.get("data", {}).get("token_text", "")
                print(txt, end="", flush=True)
            elif t == "result":
                final = obj.get("data", {})
                break
    finally:
        s.close()
    return final or {}


def cmd_status(_):
    print(json.dumps(_send("status", {}), indent=2))


def cmd_shutdown(_):
    print(json.dumps(_send("shutdown", {}), indent=2))


def cmd_reset(_):
    print(json.dumps(_send("reset_state", {}), indent=2))


def cmd_generate(args):
    print("=" * 72)
    print(f"prompt: {args.prompt}")
    print("-" * 72)
    final = _stream("generate_35b", {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
    })
    print("\n" + "=" * 72)
    if final:
        print(f"  prompt:    {final.get('n_prompt_tokens', 0)} tok, prefill {final.get('prefill_ms', 0):.0f} ms")
        print(f"  generated: {final.get('n_generated_tokens', 0)} tok at {final.get('decode_ms_per_tok', 0):.0f} ms/tok = {final.get('tokens_per_sec', 0):.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("shutdown").set_defaults(fn=cmd_shutdown)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=32)
    g.set_defaults(fn=cmd_generate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
