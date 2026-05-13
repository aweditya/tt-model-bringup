"""Mock-mode smoke test for the persistent weight server.

Runs entirely on the local machine — no ttnn, no qb2 device.

Spawns the server in --mock mode, exercises each Phase 1 command via the
client send() helper, and asserts the wire protocol round-trips correctly.

Run:
    cd ~/tt-xla && python -m experiments.serve.tests.test_protocol_mock
"""
import json
import os
import signal
import subprocess
import sys
import time

from experiments.serve import protocol as P
from experiments.serve import client as C


def _start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = os.path.join(P.CACHE_DIR, "mock_test.log")
    P.ensure_cache_dir()
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "experiments.serve.server", "--mock"],
        stdout=log_fp, stderr=subprocess.STDOUT, env=env,
    )
    return proc


def _wait_for_socket(timeout: float = 10.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(P.SOCKET_PATH):
            return
        time.sleep(0.1)
    raise RuntimeError(f"socket {P.SOCKET_PATH} did not appear within {timeout}s")


def _assert(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main():
    # Clean any stale socket from a prior run.
    P.ensure_cache_dir()
    try:
        os.unlink(P.SOCKET_PATH)
    except FileNotFoundError:
        pass

    print("[1] launching server (mock mode)…")
    proc = _start_server()
    try:
        _wait_for_socket()
        print("[2] socket up — exercising commands…")

        st = C.send("status", {})
        _assert(st.get("loaded") is True, f"status.loaded == True (got {st})")
        _assert(st.get("mock") is True, f"status.mock == True (got {st})")
        _assert(st.get("num_layers") == 0, f"status.num_layers == 0 (mock)")
        _assert("uptime_sec" in st, "status has uptime_sec")

        rs = C.send("reset_state", {})
        _assert(rs.get("ok") is True, "reset_state returns ok=True")
        _assert(rs.get("mock") is True, "reset_state is mock-aware")

        rk = C.send("reload_kernels", {})
        _assert(rk.get("ok") is True, "reload_kernels returns ok=True")
        # In mock mode, _91f is None so nothing is reloaded.
        _assert(rk.get("reloaded_modules") == [], "reload_kernels mock list is empty")

        r91 = C.send("run_91r", {"layers": [0, 3]})
        _assert(r91.get("mock") is True, "run_91r mock=True")
        _assert(len(r91.get("results", [])) == 2, "run_91r returned 2 results")
        _assert(r91["results"][0]["layer"] == 0, "first result is layer 0")
        _assert(r91["results"][1]["layer"] == 3, "second result is layer 3")
        _assert(r91["results"][0]["type"] == "linear_attention",
                "layer 0 is linear_attention")
        _assert(r91["results"][1]["type"] == "full_attention",
                "layer 3 is full_attention")

        # Error path: unknown command.
        import socket
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(P.SOCKET_PATH)
        s.sendall(P.pack_request("does_not_exist", {}))
        raw = P.read_line(s)
        s.close()
        err = P.parse_response(raw)
        _assert(err.type == "error", "unknown cmd returns error")
        _assert("unknown cmd" in (err.msg or ""), "error mentions unknown cmd")

        # Error path: malformed JSON.
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(P.SOCKET_PATH)
        s.sendall(b"not json at all\n")
        raw = P.read_line(s)
        s.close()
        err = P.parse_response(raw)
        _assert(err.type == "error", "bad json returns error")

        print("[3] requesting shutdown…")
        sd = C.send("shutdown", {})
        _assert(sd.get("shutting_down") is True, "shutdown returns shutting_down=True")

        # Wait for process exit.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("FAIL: server did not exit after shutdown")
            proc.kill()
            sys.exit(1)
        _assert(proc.returncode == 0, f"server exit code 0 (got {proc.returncode})")

        # Socket must be cleaned up.
        _assert(not os.path.exists(P.SOCKET_PATH),
                "socket cleaned up on shutdown")
        _assert(not os.path.exists(P.PID_PATH),
                "pidfile cleaned up on shutdown")

        print("\nALL MOCK SMOKE TESTS PASSED")
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
