"""OpenAI-compatible HTTP endpoint (chat-template + protocol translation).

Originally a thin host-side proxy over the pre-CB Unix-socket TP server
(`server.py` / `server_tp.sh`, now in
`archive/pre_cb_server_stack_2026-06-04/`). The translation helpers
(_messages_to_prompt, _chat_completion, _chat_chunk) survived and are
now imported by `experiments/serve/cb_api.py` (the live CB HTTP server).

Helpers are pure + unit-tested in
`experiments/serve/tests/test_openai_endpoint.py`.
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

# Memoised per-tokenizer active-prompt suffix. See _active_prompt_suffix
# for the detection mechanism and why we strip it.
_SUFFIX_CACHE: dict[int, list[int]] = {}


def _normalise_template_output(raw) -> list[int]:
    """`apply_chat_template(tokenize=True)` returns either a bare list[int]
    (Qwen / PreTrainedTokenizerFast default) or a dict-like BatchEncoding
    with `input_ids` (Gemma). Normalise to a plain list."""
    if isinstance(raw, dict) or hasattr(raw, "input_ids"):
        return list(raw["input_ids"])
    return list(raw)


def _active_prompt_suffix(tokenizer, base_kw: dict) -> list[int]:
    """Detect the trailing tokens that ONLY appear in the ACTIVE assistant
    prompt (`add_generation_prompt=True`) but NOT in PAST-assistant renders.

    Examples of this asymmetry:
    - Qwen3.6 + `enable_thinking=False`: active suffix is the
      `<think>\\n\\n</think>\\n\\n` "no-think" marker block.
    - Gemma 4 IT: active suffix is `<|channel>thought\\n<channel|>` —
      a channel-switch marker the past-assistant render omits.

    Both classes cause turn-1's `prompt_1` to NOT be a byte-prefix of
    turn-2's `prompt_2`, defeating slot-level prefix caching. The
    universal fix is to strip the suffix from `prompt_1` before storing
    so the next turn's prompt aligns.

    Detection: render the same user message twice — once as a turn-1
    active prompt, once as `[user, assistant, user_2]` passively with
    `add_generation_prompt=False`. The first divergence point in the
    active render is where the suffix starts. Memoised by `id(tokenizer)`
    so this is a one-shot cost per process.
    """
    key = id(tokenizer)
    cached = _SUFFIX_CACHE.get(key)
    if cached is not None:
        return cached

    # Rare-but-valid 1-char content so the template renders the
    # passive past-assistant slot to something other than the active
    # suffix's first token.
    PROBE_USR = "x"
    PROBE_ASS = "y"
    kw_active = dict(base_kw, tokenize=True, add_generation_prompt=True)
    kw_passive = dict(base_kw, tokenize=True, add_generation_prompt=False)
    try:
        active = _normalise_template_output(tokenizer.apply_chat_template(
            [{"role": "user", "content": PROBE_USR}], **kw_active))
        passive = _normalise_template_output(tokenizer.apply_chat_template(
            [{"role": "user", "content": PROBE_USR},
             {"role": "assistant", "content": PROBE_ASS}], **kw_passive))
    except Exception:
        # Template/kwarg unsupported — assume no suffix.
        _SUFFIX_CACHE[key] = []
        return []
    n = min(len(active), len(passive))
    div = n
    for i in range(n):
        if active[i] != passive[i]:
            div = i
            break
    suffix = active[div:]
    _SUFFIX_CACHE[key] = suffix
    return suffix


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

    Active-prompt suffix strip (covers Qwen3.6 + Gemma 4 IT + any future
    template with the same asymmetry): see `_active_prompt_suffix`.

    Qwen3.6-specific jinja kwarg `preserve_thinking=True` re-wraps past
    assistant `<think>` blocks so past renders match what the model
    emitted. Silently ignored by templates that don't define it.
    """
    base_kw = dict(enable_thinking=False, preserve_thinking=True)
    kw = dict(base_kw, tokenize=True, add_generation_prompt=True)
    if tools:
        kw["tools"] = tools
    ids = _normalise_template_output(
        tokenizer.apply_chat_template(messages, **kw))
    # Universal active-prompt suffix strip — handles Qwen <think>, Gemma
    # <|channel>...<channel|>, and any future template-level asymmetry.
    suffix = _active_prompt_suffix(tokenizer, base_kw)
    if suffix and len(ids) >= len(suffix) and ids[-len(suffix):] == suffix:
        ids = ids[:-len(suffix)]
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
