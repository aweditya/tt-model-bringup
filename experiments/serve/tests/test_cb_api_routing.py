"""Routing-only smoke test for cb_api — no device, no bootstrap.

Builds the FastAPI app with a fake engine + tokenizer attached to app.state,
then uses FastAPI's TestClient (httpx in-process) to exercise the routes. This
catches signature / parameter-binding / body-parsing bugs in ~milliseconds —
much faster than the full qb1 validator (which spends ~350s booting the model).

Run on qb1 (no env block needed; pure host code):
    cd ~/tt-xla && .venv/bin/python -m experiments.serve.tests.test_cb_api_routing
"""
import os
import queue

# Don't let the side-effecting module-level apps build (they import transformers
# and FastAPI fully, which we don't need for routing tests).
os.environ["TT_OPENAI_BUILD_APP"] = "0"
os.environ["TT_CB_API_BUILD_APP"] = "0"

import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

from cb_api import _build_app  # noqa: E402


class _Handle:
    def __init__(self, rid, toks, final="done"):
        self.rid = rid
        self.prompt_len = 1
        self._q = queue.Queue()
        for t in toks:
            self._q.put(("tok", t))
        self._q.put((final, None))
        self.final = None
        self.error = None


class _FakeEngine:
    """Just enough surface for cb_api to run end-to-end without the device."""
    def __init__(self):
        self.slots = 4
        self.sampling = True
        self._rid = 0
        # mimic the real engine's `metrics` registry so /metrics is exercised.
        from cb_metrics import Registry
        self.metrics = Registry()
        self.metrics.counter("cb_requests_submitted_total", "fake")
        self.metrics.gauge("cb_engine_slots_total", "fake", fn=lambda: self.slots)
        self.metrics.histogram("cb_step_seconds", "fake")

    def submit(self, prompt_ids, max_new=None, sampling=None):
        self._rid += 1
        # 3 stub tokens; max_new caps them.
        n = min(3, max_new or 3)
        return _Handle(self._rid, list(range(10, 10 + n)))

    def cancel(self, rid):
        pass


class _FakeTok:
    def encode(self, s, **_):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=True):
        # produce a deterministic, monotonically-extending string from token list
        return "/".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages) + "|"


def main():
    from fastapi.testclient import TestClient

    state = {"engine": _FakeEngine(), "tok": _FakeTok(), "eos_id": 0}
    app = _build_app(state)

    c = TestClient(app)

    # 1. /health (no body)
    r = c.get("/health")
    assert r.status_code == 200, f"/health: {r.status_code} {r.text}"
    assert r.json()["ok"] is True, r.json()

    # 2. /v1/models
    r = c.get("/v1/models")
    assert r.status_code == 200 and r.json()["data"][0]["id"], r.text

    # 3. non-stream chat completion — exercises body parsing
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3,
    })
    print(f"chat (non-stream): {r.status_code} {r.text[:200]}")
    assert r.status_code == 200, f"chat (non-stream): {r.status_code} {r.text}"
    body = r.json()
    assert body["object"] == "chat.completion", body
    assert body["choices"][0]["message"]["role"] == "assistant", body
    assert body["choices"][0]["message"]["content"] == "10/11/12", body

    # 4. streaming chat completion — exercises SSE generator + async run_in_executor
    with c.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3, "stream": True,
    }) as r:
        print(f"chat (stream): {r.status_code}")
        assert r.status_code == 200, r.read()
        chunks = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunks.append(payload)
        assert chunks, "no SSE chunks"

    # 5. /v1/completions (raw prompt, not chat)
    r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
    assert r.status_code == 200, r.text

    # 6. Backpressure: when the engine's submit() raises queue.Full, cb_api
    # must return 429 (not 500). Swap the engine to one that always raises.
    class _FullEngine:
        slots = 4
        sampling = True

        def submit(self, *_a, **_kw):
            raise queue.Full("engine cap reached")

        def cancel(self, *_a, **_kw):
            pass

    state["engine"] = _FullEngine()  # closure dict; handlers see the swap
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
    })
    print(f"backpressure: {r.status_code} {r.text[:120]}")
    assert r.status_code == 429, f"backpressure: expected 429, got {r.status_code} {r.text}"

    # 7. /metrics renders Prometheus text exposition; check required shape.
    # Restore a working engine so /metrics 200s.
    state["engine"] = _FakeEngine()
    r = c.get("/metrics")
    assert r.status_code == 200, f"/metrics: {r.status_code} {r.text}"
    assert r.headers.get("content-type", "").startswith("text/plain"), r.headers
    body = r.text
    for needle in ("# HELP cb_requests_submitted_total",
                   "# TYPE cb_requests_submitted_total counter",
                   "# TYPE cb_engine_slots_total gauge",
                   "# TYPE cb_step_seconds histogram",
                   "cb_step_seconds_bucket{le=\"+Inf\"}",
                   "cb_step_seconds_count"):
        assert needle in body, f"/metrics missing {needle!r}\n--- body ---\n{body}"
    print(f"metrics: {len(body.splitlines())} lines OK")

    print("test_cb_api_routing: 7/7 PASS")


if __name__ == "__main__":
    main()
