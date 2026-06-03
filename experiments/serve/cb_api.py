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

# MM1 (2026-06-02): TT_BACKEND selects which server module to import. Each
# backend exposes the same shape (`MeshServerState` / `State` + `bootstrap(state)`
# + `forward_token_tp_inner(state)` / `forward_batch_tp_inner(state)`). Adding a
# new backend = register it here + drop a `server_*.py` in this directory.
BACKENDS = {
    "27b":   ("server_tp",        "Qwen/Qwen3.6-27B"),
    "35b":   ("server_35b_ttnn",  "Qwen/Qwen3.6-35B-A3B"),
}
TT_BACKEND = os.environ.get("TT_BACKEND", "27b")
if TT_BACKEND not in BACKENDS:
    raise ValueError(
        f"unknown TT_BACKEND={TT_BACKEND!r}; valid backends: {sorted(BACKENDS)}")
_BACKEND_MODULE, _BACKEND_DEFAULT_MODEL = BACKENDS[TT_BACKEND]
DEFAULT_MODEL_ID = os.environ.get("TT_MODEL_ID", _BACKEND_DEFAULT_MODEL)


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


def _build_app(state: dict, model_id: str = DEFAULT_MODEL_ID, lifespan=None,
               bootstrap_status: Optional[dict] = None):
    """Construct the FastAPI app whose handlers close over `state` for engine
    access. The caller MUST populate `state["engine"]`, `state["tok"]`,
    `state["eos_id"]` before requests are served.

    `bootstrap_status` is the shared dict from the lifespan; when not None it
    powers `/bootstrap` and enriches `/health` so callers can see the current
    bootstrap stage instead of a bare 503.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    app = FastAPI(title="tt-model-bringup CB OpenAI API", lifespan=lifespan)

    @app.get("/health")
    def health():
        eng = state.get("engine")
        if eng is None:
            payload = {"ok": False, "ready": False}
            if bootstrap_status is not None:
                payload["bootstrap"] = dict(bootstrap_status)
            return JSONResponse(status_code=503, content=payload)
        return {"ok": True, "ready": True, "model": model_id,
                "slots": eng.slots, "sampling": eng.sampling}

    @app.get("/bootstrap")
    def bootstrap_state():
        """Per-stage bootstrap progress. Useful when /health is 503 to see
        whether the server is loading, stuck, or failed. Returns
        {stage, elapsed_s, ready, started_at} — see cb_api lifespan."""
        if bootstrap_status is None:
            return JSONResponse(status_code=404,
                                content={"detail": "bootstrap_status not wired"})
        return dict(bootstrap_status)

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
        """Pure-asyncio drain — engine pushes via call_soon_threadsafe so
        `await handle.aget()` doesn't burn an executor thread. On Starlette
        cancel (client disconnect): call on_cancel, drain to terminal, re-raise."""
        try:
            while True:
                kind, payload = await handle.aget()
                if kind == "tok":
                    yield payload
                else:
                    handle.final = kind
                    if kind == "error":
                        handle.error = payload
                    return
        except asyncio.CancelledError:
            if on_cancel is not None:
                on_cancel()
            while handle.final is None:
                try:
                    kind, payload = await asyncio.wait_for(handle.aget(), timeout=2.0)
                except asyncio.TimeoutError:
                    break
                if kind != "tok":
                    handle.final = kind
                    if kind == "error":
                        handle.error = payload
                    break
            raise

    def _finish_reason(eos_id, handle, gen_ids) -> str:
        if eos_id is not None and gen_ids and gen_ids[-1] == eos_id:
            return "stop"
        if handle.final in ("cancelled", "error"):
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
            handle = eng.submit(prompt_ids, max_new=max_tokens, sampling=sampling,
                                 loop=asyncio.get_running_loop())
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
    in the default executor so the event loop stays responsive.

    The model backend is picked by TT_BACKEND (default 27b → server_tp). See
    `BACKENDS` at module top for the registry."""
    import importlib
    base = importlib.import_module(_BACKEND_MODULE)
    from cb_engine import CBEngine

    state: dict = {}

    # Bootstrap observability: per-stage timing + a coarse status that the
    # health endpoint can return so callers can see "stuck at <stage>" instead
    # of just 503. The bootstrap thread updates `bootstrap_status[0]` after
    # each `log(...)` call.
    import time as _time
    from pathlib import Path as _Path
    bootstrap_status = {"stage": "not_started", "started_at": None,
                        "elapsed_s": None, "ready": False}
    # Side-channel: tail-able file the user can `ssh qb1 cat`/`tail -f` while
    # uvicorn isn't yet listening (lifespan hasn't yielded so /bootstrap and
    # /health aren't reachable). cb_api is the only writer; serve_cb.log is
    # the official log but suffers from print-from-worker-thread buffering
    # under uvicorn's stdout wrapping.
    _STATUS_FILE = _Path(os.environ.get("PROJECT_ROOT", str(_Path.home() / "tt-xla"))) / ".cache" / "server_cb.bootstrap.log"
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text("")  # truncate on fresh boot

    def _flush_log(msg):
        # 1) print + flush for the main log (best-effort under uvicorn).
        print(msg, flush=True)
        # 2) explicit append to side-file with fsync — bypasses any
        #    sys.stdout games and gives the user a reliable observability
        #    channel during bootstrap.
        try:
            with open(_STATUS_FILE, "a") as f:
                ts = _time.strftime("%H:%M:%S")
                f.write(f"[{ts}] {msg}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        bootstrap_status["stage"] = str(msg)[:200]
        if bootstrap_status["started_at"] is not None:
            bootstrap_status["elapsed_s"] = round(_time.time() - bootstrap_status["started_at"], 1)

    @asynccontextmanager
    async def lifespan(app):
        st = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
        loop = asyncio.get_running_loop()
        bootstrap_status["started_at"] = _time.time()
        bootstrap_status["stage"] = "bootstrap_starting"
        # base.bootstrap takes (state, log=None). Pass our flushing logger so
        # stage transitions hit the log file immediately.
        await loop.run_in_executor(None, base.bootstrap, st, _flush_log)
        bootstrap_status["stage"] = "bootstrap_done; building engine"
        bootstrap_status["elapsed_s"] = round(_time.time() - bootstrap_status["started_at"], 1)
        # 27B-only deltanet feature flags (the 35B path keys these via
        # getattr-default in its forward; setting them here is a no-op for 35B
        # but is incorrect-by-convention. Gate by backend.)
        if TT_BACKEND == "27b":
            st.deltanet_recurrence_mode = "manual"
            st.deltanet_decay_gate_mode = "manual"
            st.deltanet_decay_mode = "native_softplus"
        eos_id = getattr(st.tok, "eos_token_id", None)
        eos_id = int(eos_id) if eos_id is not None else -1
        # 35B's B>1 batched forward path through cb_scheduler produces
        # prompt-independent degenerate output when only a subset of slots is
        # active (empty-slot inputs poison batched ops — likely SDPA mask or
        # MoE routing). v1.5 was validated with all slots active in dev harness;
        # cb_scheduler admits ragged. Until that's resolved, default 35B to 1
        # slot (B=1 fast path delegates to base.step_forward_inner — bit-validated).
        # Users can override with TT_CB_SLOTS explicitly for batched serving.
        _slots_default = "1" if TT_BACKEND == "35b" else "4"
        slots = int(os.environ.get("TT_CB_SLOTS", _slots_default))
        max_new_cap = int(os.environ.get("TT_CB_MAX_NEW", "1024"))
        max_inflight = int(os.environ.get("TT_CB_MAX_INFLIGHT", "0")) or None
        # 35B has a known ttnn bulk-readback issue on its logits tensor —
        # ttnn.to_torch returns garbage that varies per run (on-device argmax
        # of the same tensor finds the right answer). cb_scheduler's logits
        # path goes through that broken readback, so route 35B through topk-mode
        # which uses smaller per-slot index readbacks (proven working in v0
        # smoke). Override default TT_CB_TOPK_K=64 when TT_BACKEND=35b. Users
        # can still set TT_CB_TOPK_K explicitly to override.
        _topk_default = "64" if TT_BACKEND == "35b" else "0"
        topk_k = int(os.environ.get("TT_CB_TOPK_K", _topk_default)) or None
        chunked_prefill = os.environ.get("TT_CB_CHUNKED_PREFILL", "0") == "1"
        prefix_cache = os.environ.get("TT_CB_PREFIX_CACHE", "0") == "1"
        prefix_ttl_s = float(os.environ.get("TT_CB_PREFIX_TTL_S", "300"))
        engine = CBEngine(st, slots=slots, max_new_cap=max_new_cap,
                          eos_id=eos_id, sampling=True,
                          max_inflight=max_inflight, topk_k=topk_k,
                          chunked_prefill=chunked_prefill,
                          prefix_cache=prefix_cache,
                          prefix_ttl_s=prefix_ttl_s).start()
        state["engine"] = engine
        state["tok"] = st.tok
        state["eos_id"] = eos_id
        bootstrap_status["stage"] = "ready"
        bootstrap_status["ready"] = True
        bootstrap_status["elapsed_s"] = round(_time.time() - bootstrap_status["started_at"], 1)
        try:
            yield
        finally:
            bootstrap_status["stage"] = "shutting_down"
            engine.stop()
            bootstrap_status["stage"] = "stopped"

    return _build_app(state, lifespan=lifespan, bootstrap_status=bootstrap_status)


# Module-level `app` for `uvicorn experiments.serve.cb_api:app`. The env guard
# keeps unit tests free of fastapi/transformers imports.
app = _build_app_with_default_lifespan() if os.environ.get("TT_CB_API_BUILD_APP", "1") == "1" else None
