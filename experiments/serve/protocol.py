"""JSON-line wire protocol for the persistent weight server."""
import json
import os
import socket
from dataclasses import dataclass
from typing import Optional

# Resolve repo root from this file's location so a fresh clone at any path
# works without hardcoding. Prod (~/tt-xla/) lands on ~/tt-xla/.cache/... —
# unchanged from the prior hardcoded behavior. Override with TT_XLA_ROOT for
# tests that want a custom cache location.
_PROJECT_ROOT = os.environ.get("TT_XLA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache")
SOCKET_PATH = os.path.join(CACHE_DIR, "server.sock")
PID_PATH = os.path.join(CACHE_DIR, "server.pid")
LOG_PATH = os.path.join(CACHE_DIR, "server.log")


@dataclass
class Request:
    cmd: str
    args: dict


@dataclass
class Response:
    type: str   # "result" | "error"
    data: Optional[dict] = None
    msg: Optional[str] = None


def pack_request(cmd: str, args: Optional[dict] = None) -> bytes:
    return (json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode("utf-8")


def pack_result(data: dict) -> bytes:
    return (json.dumps({"type": "result", "data": data}) + "\n").encode("utf-8")


def pack_chunk(data: dict) -> bytes:
    """Streaming response chunk. One per token in generate_stream. Server sends
    multiple chunks then a final pack_result with the summary."""
    return (json.dumps({"type": "chunk", "data": data}) + "\n").encode("utf-8")


def pack_error(msg: str) -> bytes:
    return (json.dumps({"type": "error", "msg": msg}) + "\n").encode("utf-8")


def read_line(conn: socket.socket, max_bytes: int = 1 << 20) -> bytes:
    """Read one newline-terminated message from a socket."""
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in chunk or len(buf) >= max_bytes:
            break
    nl = buf.find(b"\n")
    return bytes(buf if nl < 0 else buf[:nl])


def parse_request(raw: bytes) -> Request:
    obj = json.loads(raw.decode("utf-8"))
    return Request(cmd=obj.get("cmd", ""), args=obj.get("args") or {})


def parse_response(raw: bytes) -> Response:
    obj = json.loads(raw.decode("utf-8"))
    return Response(type=obj.get("type", "error"), data=obj.get("data"), msg=obj.get("msg"))


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
