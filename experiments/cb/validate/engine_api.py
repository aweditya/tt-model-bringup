#!/usr/bin/env python3
"""P2 gate — async OpenAI API over the CB engine, end-to-end HTTP on qb1.

Bootstraps state + CBEngine(sampling=True) once, attaches them to a FastAPI app
built by cb_api._build_app (no lifespan), starts uvicorn in a background thread
on 127.0.0.1, and exercises the four production-relevant HTTP paths via stdlib
http.client (no httpx dep). Validates:

  (a) non-stream /v1/chat/completions returns the OpenAI body shape with usage
      and coherent content;
  (b) streaming SSE emits role -> deltas -> finish + [DONE] and the concatenated
      content equals the non-stream content for the same prompt+max_tokens
      (greedy → deterministic);
  (c) N concurrent clients (threads) each get a correct response for their own
      prompt — CB serves them out of one engine, no cross-talk;
  (d) cancel-on-disconnect: an SSE client that closes the TCP socket mid-stream
      triggers engine.cancel(rid) (the 0.5s poll picks it up); a subsequent
      request still completes correctly (the slot recycled).

Run on qb1:
  make run PY=experiments/cb/validate/engine_api.py
"""
from __future__ import annotations

import asyncio
import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base                       # noqa: E402
from cb_engine import CBEngine                  # noqa: E402

# cb_api builds a module-level app when TT_CB_API_BUILD_APP=1; the validator
# builds its own (no lifespan), so suppress the default app to avoid double
# bootstrap on import.
import os
os.environ["TT_CB_API_BUILD_APP"] = "0"
from cb_api import _build_app                   # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PORT = 18765
HOST = "127.0.0.1"
MAX_NEW = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────
def _post(path: str, body: dict, timeout: float = 120.0) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    return resp.status, (json.loads(raw) if raw else {})


def _get(path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=5.0)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    return resp.status, (json.loads(raw) if raw else {})


def _stream(path: str, body: dict, partial_max: int | None = None) -> list[dict]:
    """POST and parse SSE `data:` events line-by-line. If partial_max is set,
    close after that many events (used to test cancel-on-disconnect)."""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120.0)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    events: list[dict] = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.rstrip(b"\r\n")
        if not line:
            continue
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].decode("utf-8")
        if payload == "[DONE]":
            conn.close()
            return events
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
        if partial_max is not None and len(events) >= partial_max:
            conn.close()  # ABRUPT disconnect — triggers server is_disconnected
            return events
    conn.close()
    return events


# ── uvicorn in a thread ───────────────────────────────────────────────────────
def _start_uvicorn(app):
    import uvicorn
    cfg = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning",
                         lifespan="off")  # we own the engine; no lifespan
    server = uvicorn.Server(cfg)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, name="uvicorn", daemon=True)
    t.start()
    # wait for the listening socket
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            s = socket.create_connection((HOST, PORT), timeout=0.5)
            s.close()
            return server, t
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("uvicorn never came up")


# ── tests ─────────────────────────────────────────────────────────────────────
def main():
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    tok = state.tok
    eos_id = getattr(tok, "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else -1

    engine = CBEngine(state, slots=4, max_new_cap=128, eos_id=eos_id, sampling=True).start()
    log("=== engine up: 4 slots, sampling mode ===")

    state = {"engine": engine, "tok": tok, "eos_id": eos_id}
    app = _build_app(state)
    server, server_thread = _start_uvicorn(app)
    log(f"=== uvicorn listening on http://{HOST}:{PORT} ===")
    try:
        _run_tests(engine, server, server_thread)
    finally:
        try:
            server.should_exit = True
            server_thread.join(timeout=10)
        except Exception:
            pass
        engine.stop()
        log("=== uvicorn + engine stopped (finally) ===")


def _run_tests(engine, server, server_thread):
    # (a) non-stream chat completion
    log("--- (a) non-stream /v1/chat/completions ---")
    status, resp = _post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": MAX_NEW,
    })
    a_ok = (
        status == 200
        and resp.get("object") == "chat.completion"
        and resp["choices"][0]["message"]["role"] == "assistant"
        and isinstance(resp["choices"][0]["message"]["content"], str)
        and len(resp["choices"][0]["message"]["content"]) > 0
        and resp["usage"]["completion_tokens"] >= 1
        and resp["usage"]["prompt_tokens"] >= 1
    )
    text_a = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"  status={status} usage={resp.get('usage')} content={text_a!r}")
    log(f"  (a) {'OK' if a_ok else 'FAIL'}")

    # (b) streaming SSE — same prompt+max greedy → same text as (a)
    log("--- (b) streaming SSE /v1/chat/completions ---")
    events = _stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": MAX_NEW, "stream": True,
    })
    delta_text = "".join(ev["choices"][0]["delta"].get("content", "") for ev in events)
    has_role = any(ev["choices"][0]["delta"].get("role") == "assistant" for ev in events)
    has_finish = any(ev["choices"][0].get("finish_reason") for ev in events)
    b_ok = has_role and has_finish and delta_text == text_a and len(events) >= 3
    log(f"  {len(events)} SSE events; role={has_role} finish={has_finish}; "
        f"text matches non-stream: {delta_text == text_a}")
    log(f"  streamed: {delta_text!r}")
    log(f"  (b) {'OK' if b_ok else 'FAIL'}")

    # (c) concurrent clients — /v1/completions (raw prompts, no chat-template
    # preamble). Concurrent runs must match the serial refs for the same prompts
    # — proves the HTTP layer multiplexes correctly without crosstalk.
    log("--- (c) 6 concurrent /v1/completions clients vs serial refs ---")
    prompts_c = [
        "The capital of France is the city of",
        "Once upon a time there lived a young",
        "The largest planet in our solar system is",
        "Water boils at a temperature of one hundred",
        "The quick brown fox jumps over the",
        "Photosynthesis is the process by which",
    ]
    refs_c = []
    for p in prompts_c:
        _, r = _post("/v1/completions", {"prompt": p, "max_tokens": MAX_NEW})
        refs_c.append(r["choices"][0].get("text") or r["choices"][0].get("message", {}).get("content", ""))
    results_c: list[dict] = [None] * len(prompts_c)  # type: ignore[assignment]

    def _client(i):
        s, r = _post("/v1/completions", {"prompt": prompts_c[i], "max_tokens": MAX_NEW})
        results_c[i] = {"status": s, "resp": r}

    t0 = time.time()
    ts = [threading.Thread(target=_client, args=(i,)) for i in range(len(prompts_c))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    contents_c = [
        (r["resp"]["choices"][0].get("text") or
         r["resp"]["choices"][0].get("message", {}).get("content", ""))
        for r in results_c
    ]
    each_matches = [c == refs_c[i] for i, c in enumerate(contents_c)]
    c_ok = all(r["status"] == 200 for r in results_c) and all(each_matches)
    log(f"  6 clients done in {dt:.2f}s; all_200={all(r['status']==200 for r in results_c)}; "
        f"distinct={len(set(contents_c))}/{len(contents_c)}; "
        f"each==ref={sum(each_matches)}/{len(each_matches)}")
    for i, c in enumerate(contents_c):
        mark = "OK" if each_matches[i] else "MISMATCH"
        log(f"    client {i} [{mark}]: {c[:80]!r}…")
    log(f"  (c) {'OK' if c_ok else 'FAIL'}")

    # (d) cancel-on-disconnect — close mid-stream, then a fresh request still works
    log("--- (d) cancel-on-disconnect + slot recycle ---")
    partial = _stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Once upon a time there lived a young"}],
        "max_tokens": 128, "stream": True,
    }, partial_max=3)
    log(f"  partial SSE events received: {len(partial)} (closed mid-stream)")
    # give the server > DISCONNECT_POLL_S (+ drain margin) to process the cancel
    time.sleep(1.5)
    status2, resp2 = _post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": MAX_NEW,
    })
    content2 = resp2.get("choices", [{}])[0].get("message", {}).get("content", "")
    d_ok = (len(partial) >= 1) and (status2 == 200) and (content2 == text_a)
    log(f"  post-cancel request: status={status2} matches (a): {content2 == text_a}")
    log(f"  (d) {'OK' if d_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log(f"  (a) non-stream={a_ok}  (b) SSE-stream={b_ok}  (c) concurrent={c_ok}  (d) cancel={d_ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
