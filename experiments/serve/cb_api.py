"""OpenAI-compatible HTTP API over the CB engine (same process as the model).

P2 of the production server (research/production_server_plan.md). Drops the
Unix-socket hop the legacy openai_endpoint uses — this module runs in the SAME
process as the device-owning CBEngine, so a FastAPI request handler submits
prompts straight to the engine queue and bridges the blocking token stream to
async via run_in_executor. On client disconnect, Starlette cancels the SSE
generator → we catch CancelledError → engine.cancel(rid), freeing the slot
(CB2 admission recycles it).

  - `_build_app(state, model_id=...)` returns a FastAPI app whose handlers close
    over a mutable `state` dict (`state["engine"] / state["tok"] / state["eos_id"]`).
    The caller fills `state` before requests land — works for both the testable
    path (validator pre-builds the engine and fills `state` directly) and the
    lifespan-based path (lifespan bootstraps + fills `state` before yield).
  - Module-level `app` boots via lifespan for `uvicorn experiments.serve.cb_api:app`.

The closure-over-state pattern avoids relying on FastAPI's auto-injection of
`Request` (which broke in 0.13x: a `request: Request` parameter is rejected as a
missing query field). Handlers use the documented `body: dict` body parameter.

Reuses the pure OpenAI helpers (_messages_to_prompt, _chat_completion,
_chat_chunk) from openai_endpoint. Greedy and sampled requests share one engine
(sampling-mode); temperature<=0 normalises to greedy.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

# Suppress the legacy proxy endpoint's module-level app build — we only want its
# pure translation helpers.
os.environ.setdefault("TT_OPENAI_BUILD_APP", "0")
from openai_endpoint import _chat_chunk, _chat_completion, _messages_to_prompt  # noqa: E402

DEFAULT_MODEL_ID = os.environ.get("TT_MODEL_ID", "Qwen/Qwen3.6-27B")


def _try_get(q: queue.Queue, timeout: float):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def _build_sampling(body: dict) -> Optional[dict]:
    temperature = float(body.get("temperature", 0.0))
    if temperature <= 0.0:
        return None
    return {
        "temperature": temperature,
        "top_p": float(body.get("top_p", 1.0)),
        "top_k": int(body.get("top_k", 0)),
        "seed": body.get("seed"),
    }


def _build_app(state: dict, model_id: str = DEFAULT_MODEL_ID, lifespan=None):
    """Construct the FastAPI app whose handlers close over `state` for engine
    access. The caller MUST populate `state["engine"]`, `state["tok"]`,
    `state["eos_id"]` before requests are served."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    app = FastAPI(title="tt-model-bringup CB OpenAI API", lifespan=lifespan)

    @app.get("/health")
    def health():
        eng = state.get("engine")
        if eng is None:
            return JSONResponse(status_code=503, content={"ok": False, "ready": False})
        return {"ok": True, "ready": True, "model": model_id,
                "slots": eng.slots, "sampling": eng.sampling}

    @app.get("/metrics")
    def metrics():
        """Prometheus text exposition (text/plain; version=0.0.4). Engine-owned
        registry; gauges sampled at scrape time."""
        eng = state.get("engine")
        if eng is None:
            return PlainTextResponse("# engine not ready\n", status_code=503,
                                     media_type="text/plain; version=0.0.4")
        return PlainTextResponse(eng.metrics.format_prometheus(),
                                 media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    async def _drain_handle(handle, on_cancel=None):
        """Pull tokens off the blocking queue in the default threadpool. On
        CancelledError (Starlette cancels us when the client disconnects), invoke
        on_cancel (engine.cancel) and drain to the terminal marker so the slot
        frees in the engine's _drain_cancels."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await loop.run_in_executor(None, _try_get, handle._q, 1.0)
                if msg is None:
                    continue
                kind, payload = msg
                if kind == "tok":
                    yield payload
                else:
                    handle.final = kind
                    return
        except asyncio.CancelledError:
            if on_cancel is not None:
                on_cancel()
            while handle.final is None:
                m = await loop.run_in_executor(None, _try_get, handle._q, 2.0)
                if m is None:
                    break
                if m[0] != "tok":
                    handle.final = m[0]
                    break
            raise

    def _finish_reason(eos_id, handle, gen_ids) -> str:
        if eos_id is not None and gen_ids and gen_ids[-1] == eos_id:
            return "stop"
        if handle.final == "cancelled":
            return "stop"
        return "length"

    async def _complete(prompt: str, body: dict):
        eng = state.get("engine")
        if eng is None:
            return JSONResponse(status_code=503, content={"error": "engine not ready"})
        tok = state["tok"]
        eos_id = state["eos_id"]

        prompt_ids = tok.encode(prompt)
        max_tokens = int(body.get("max_tokens", 256))
        stream = bool(body.get("stream", False))
        model = body.get("model", model_id)
        sampling = _build_sampling(body)

        try:
            handle = eng.submit(prompt_ids, max_new=max_tokens, sampling=sampling)
        except queue.Full as e:
            return JSONResponse(status_code=429, content={"error": str(e)})
        except RuntimeError as e:
            return JSONResponse(status_code=503, content={"error": str(e)})
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if stream:
            async def sse():
                yield f"data: {json.dumps(_chat_chunk('', model, cid, role=True))}\n\n"
                gen_ids: list[int] = []
                text_so_far = ""
                async for tid in _drain_handle(handle, on_cancel=lambda: eng.cancel(handle.rid)):
                    gen_ids.append(tid)
                    full = tok.decode(gen_ids, skip_special_tokens=True)
                    delta = full[len(text_so_far):]
                    text_so_far = full
                    if delta:
                        yield f"data: {json.dumps(_chat_chunk(delta, model, cid))}\n\n"
                fin = _finish_reason(eos_id, handle, gen_ids)
                yield f"data: {json.dumps(_chat_chunk('', model, cid, finish=fin))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        gen_ids: list[int] = []
        async for tid in _drain_handle(handle):
            gen_ids.append(tid)
        text = tok.decode(gen_ids, skip_special_tokens=True)
        return _chat_completion(text, model, len(prompt_ids), len(gen_ids),
                                _finish_reason(eos_id, handle, gen_ids))

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        tok = state["tok"]
        prompt = _messages_to_prompt(tok, body.get("messages", []))
        return await _complete(prompt, body)

    @app.post("/v1/completions")
    async def completions(body: dict):
        return await _complete(body.get("prompt", ""), body)

    return app


def _build_app_with_default_lifespan():
    """uvicorn entry: bootstraps state + CBEngine(sampling=True) in lifespan;
    stops the engine on shutdown. Bootstrap is sync + slow (~350s on qb1) — runs
    in the default executor so the event loop stays responsive."""
    import server_tp as base
    from cb_engine import CBEngine

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app):
        st = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, base.bootstrap, st)
        st.deltanet_recurrence_mode = "manual"
        st.deltanet_decay_gate_mode = "manual"
        st.deltanet_decay_mode = "native_softplus"
        eos_id = getattr(st.tok, "eos_token_id", None)
        eos_id = int(eos_id) if eos_id is not None else -1
        slots = int(os.environ.get("TT_CB_SLOTS", "4"))
        max_new_cap = int(os.environ.get("TT_CB_MAX_NEW", "1024"))
        max_inflight = int(os.environ.get("TT_CB_MAX_INFLIGHT", "0")) or None
        topk_k = int(os.environ.get("TT_CB_TOPK_K", "0")) or None
        chunked_prefill = os.environ.get("TT_CB_CHUNKED_PREFILL", "0") == "1"
        engine = CBEngine(st, slots=slots, max_new_cap=max_new_cap,
                          eos_id=eos_id, sampling=True,
                          max_inflight=max_inflight, topk_k=topk_k,
                          chunked_prefill=chunked_prefill).start()
        state["engine"] = engine
        state["tok"] = st.tok
        state["eos_id"] = eos_id
        try:
            yield
        finally:
            engine.stop()

    return _build_app(state, lifespan=lifespan)


# Module-level `app` for `uvicorn experiments.serve.cb_api:app`. The env guard
# keeps unit tests free of fastapi/transformers imports.
app = _build_app_with_default_lifespan() if os.environ.get("TT_CB_API_BUILD_APP", "1") == "1" else None
