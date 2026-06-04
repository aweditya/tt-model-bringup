"""OpenAI-compatible HTTP endpoint over the Unix-socket TP server.

A thin host-side proxy: HTTP /v1/chat/completions (and /v1/completions) ->
chat-template -> the server's `generate_tp` over the Unix socket -> OpenAI
response (streaming SSE or single JSON). No device code; the model runs in the
persistent server (start it with experiments/serve/scripts/serve_tp.sh).

Run on the TT host (after `serve_tp.sh start`):
    uv run --extra serve uvicorn experiments.serve.openai_endpoint:app \
        --host 0.0.0.0 --port 8000

The translation helpers (_messages_to_prompt, _chat_completion, _chat_chunk) are
pure + unit-tested in experiments/serve/tests/test_openai_endpoint.py.
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from typing import Iterator, Optional

from experiments.serve import protocol as P

MODEL_ID = os.environ.get("TT_MODEL_ID", "Qwen/Qwen3.6-27B")
PROJECT_ROOT = os.environ.get("TT_XLA_ROOT") or os.path.expanduser("~/tt-xla")
SOCKET_PATH = os.path.join(PROJECT_ROOT, ".cache", "server_tp.sock")


# ── Pure translation helpers (no FastAPI / socket; unit-tested) ───────────────
_THINK_SUFFIX = "<think>\n\n</think>\n\n"


def _messages_to_prompt(tokenizer, messages: list[dict],
                         tools: list[dict] | None = None) -> list[int]:
    """OpenAI chat `messages` -> list of token ids via the model's chat template.

    Returns tokens DIRECTLY (apply_chat_template(tokenize=True)) instead of
    rendering to a string and re-encoding. The re-encode step is where
    invisible BPE-boundary / whitespace differences between turn N and
    turn N+1 sneak in — that defeated Gemma 4's slot-level prefix cache
    (chat template applies `| trim` to past assistant content; the trimmed
    string re-tokenises to FEWER tokens than what we generated). Going
    straight to tokens makes the same template+tokenizer pair the single
    source of truth for the cache key.

    Qwen3.6-specific patches (preserve_thinking, trailing-suffix strip)
    still apply via jinja kwargs; the trailing-suffix strip is now done
    in token space.

    Quirk 1 (Qwen3.6): the ACTIVE assistant prompt always gets a
    `<think>\\n\\n</think>\\n\\n` block appended (with
    `enable_thinking=False`). We strip those tokens off the end.
    Quirk 2 (Qwen3.6): when rendering a PAST assistant message whose
    content contains `</think>`, the template by default DROPS the
    `<think>...</think>` block. `preserve_thinking=True` re-wraps the
    extracted reasoning so past renders match what the model emitted.
    """
    # Qwen-specific kwargs are silently-unused by templates that don't
    # define them (jinja forwards through).
    kw = dict(tokenize=True, add_generation_prompt=True,
              enable_thinking=False, preserve_thinking=True)
    if tools:
        kw["tools"] = tools
    raw = tokenizer.apply_chat_template(messages, **kw)
    # HF tokenizers are inconsistent: some return a bare list[int] (Qwen2/
    # PreTrainedTokenizerFast in default mode), some return a dict-like
    # BatchEncoding with 'input_ids' (Gemma's GemmaTokenizer often does).
    # Normalise.
    if isinstance(raw, dict) or hasattr(raw, "input_ids"):
        ids = list(raw["input_ids"])
    else:
        ids = list(raw)
    # Strip the Qwen3.6 trailing `<think>\n\n</think>\n\n` (active-prompt
    # suffix). This pattern WOULDN'T break Gemma 4 quality: Qwen3.6 ships a
    # well-defined "no-think" semantics for the active-prompt suffix where
    # stripping it leaves the model in clean response mode.
    #
    # Gemma 4 IT has a SECOND-tier asymmetry: a `<|channel>thought\n<channel|>`
    # active-prompt suffix (tokens [100, 45518, 107, 101]) emitted only when
    # `enable_thinking=False`. Stripping it would tell the model "no
    # suppression marker" and potentially alter response style. We document
    # that path in research/gemma4_pc_chat_template_asymmetry_2026-06-04.md
    # and don't auto-strip here. The proper fix is scheduler-side
    # (strip-from-cache-only, keep full prompt for engine).
    suffix_ids = tokenizer.encode(_THINK_SUFFIX, add_special_tokens=False)
    if suffix_ids and len(ids) >= len(suffix_ids) \
            and ids[-len(suffix_ids):] == list(suffix_ids):
        ids = ids[:-len(suffix_ids)]
    return ids


def _chat_completion(text: str, model: str, prompt_toks: int, gen_toks: int,
                     finish: str = "stop") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_toks, "completion_tokens": gen_toks,
                  "total_tokens": prompt_toks + gen_toks},
    }


def _chat_chunk(delta: str, model: str, cid: str, role: bool = False,
                finish: Optional[str] = None) -> dict:
    d = {"role": "assistant"} if role else ({"content": delta} if delta else {})
    return {
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }


# ── Unix-socket bridge to the server's generate_tp ────────────────────────────
def _server_generate(prompt: str, max_tokens: int, temperature: float,
                     top_p: float, top_k: int) -> Iterator[dict]:
    """Yield {'delta': str} per token then {'final': <server result dict>};
    {'error': msg} on failure. Streams over the Unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
    except OSError as e:
        yield {"error": f"cannot reach server at {SOCKET_PATH}: {e}"}
        return
    try:
        sock.sendall(P.pack_request("generate_tp", {
            "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "chunk_size": 1,
        }))
        while True:
            raw = P.read_line(sock, max_bytes=64 << 20)
            if not raw:
                return
            obj = json.loads(raw)
            t = obj.get("type")
            if t == "error":
                yield {"error": obj.get("msg", "server error")}
                return
            if t == "chunk":
                yield {"delta": obj["data"].get("token_text", "")}
            elif t == "result":
                yield {"final": obj["data"]}
                return
    finally:
        sock.close()


# ── FastAPI app (imported lazily so the helpers above stay dependency-free) ───
def _build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from transformers import AutoTokenizer

    app = FastAPI(title="tt-model-bringup OpenAI endpoint")
    state = {"tok": None}

    @app.on_event("startup")
    def _load():
        state["tok"] = AutoTokenizer.from_pretrained(MODEL_ID)

    @app.get("/health")
    def health():
        return {"ok": True, "socket": SOCKET_PATH, "model": MODEL_ID}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    def _complete(prompt: str, body: dict):
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        top_k = int(body.get("top_k", 0))
        stream = bool(body.get("stream", False))
        model = body.get("model", MODEL_ID)
        prompt_toks = len(state["tok"].encode(prompt))
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if stream:
            def sse():
                yield f"data: {json.dumps(_chat_chunk('', model, cid, role=True))}\n\n"
                n = 0
                for ev in _server_generate(prompt, max_tokens, temperature, top_p, top_k):
                    if "error" in ev:
                        yield f"data: {json.dumps({'error': ev['error']})}\n\n"
                        break
                    if "delta" in ev:
                        n += 1
                        yield f"data: {json.dumps(_chat_chunk(ev['delta'], model, cid))}\n\n"
                    elif "final" in ev:
                        fin = "length" if not ev["final"].get("stopped_on_eos") else "stop"
                        yield f"data: {json.dumps(_chat_chunk('', model, cid, finish=fin))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text, gen_toks, finish = "", 0, "length"
        for ev in _server_generate(prompt, max_tokens, temperature, top_p, top_k):
            if "error" in ev:
                return JSONResponse(status_code=502, content={"error": ev["error"]})
            if "delta" in ev:
                text += ev["delta"]
            elif "final" in ev:
                gen_toks = ev["final"].get("n_generated_tokens", 0)
                finish = "stop" if ev["final"].get("stopped_on_eos") else "length"
        return _chat_completion(text, model, prompt_toks, gen_toks, finish)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        prompt = _messages_to_prompt(state["tok"], body.get("messages", []))
        return _complete(prompt, body)

    @app.post("/v1/completions")
    async def completions(body: dict):
        return _complete(body.get("prompt", ""), body)

    return app


app = _build_app() if os.environ.get("TT_OPENAI_BUILD_APP", "1") == "1" else None
