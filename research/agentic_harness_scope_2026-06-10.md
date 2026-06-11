# Agentic-coding harness scope — Claude-Code replacement on tt-model-bringup

**Date:** 2026-06-10
**Author:** scoping pass for the owner's "round it off with something that looks like opencode" prompt
**TL;DR:** Adopt **opencode** (sst/anomalyco fork, MIT, TypeScript) pointed at our existing `cb_api` `/v1/chat/completions` endpoint via its `@ai-sdk/openai-compatible` provider. The only blocker is making Qwen 3.6-27B's `<tool_call>` (Hermes) markers survive the cb_api decoder and convert into OpenAI `tool_calls` deltas; that is a ~1 week change isolated to `cb_api.py` + `openai_endpoint._messages_to_prompt`. Keep `scripts/chat.py` as the throwaway/development TUI and as a "wireshark" for the backend.

---

## 1. What we have today

**Backend (production-grade, on Blackhole):**
- `experiments/serve/server_tp.py` — Qwen 3.6-27B dense, traced decode, paged attention, sampling. Most mature.
- `experiments/serve/server_35b_ttnn.py` — 35B-A3B MoE (`TT_CB_SLOTS=1` for now; B>1 batched FWD pollutes empty slots).
- `experiments/serve/server_gemma4_unified_ttnn.py` — Gemma 4 12B IT (Step 2 unshippable; long-decode bf16-acc cliff).
- `experiments/serve/cb_api.py` — single FastAPI process, OpenAI-compatible `/v1/chat/completions` + `/v1/completions` + `/v1/models` + `/health` + `/metrics` + `/bootstrap`. Backend pluggable via `TT_BACKEND={27b,35b,gemma4_12b}`. SSE streaming. Slot-level prefix cache (`TT_CB_PREFIX_CACHE=1`).
- `experiments/serve/openai_endpoint.py` — pure helpers (`_messages_to_prompt`, `_chat_completion`, `_chat_chunk`) used by `cb_api.py`. `apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)` already plumbed via the `tools` kwarg.

**Frontend (developer-grade, on the laptop):**
- `scripts/chat.py` — stdlib-only TUI, multi-turn, `<think>` streaming, 4 built-in tools (`shell` allow-listed, `read_file`, `write_file`, `calc`), 4 tool-call regex patterns (` ```tool_code ``` `, `<|tool_call|>…`, `<tool_call>…</tool_call>`, ` ```json {"name":…} ``` `), CWD-jailed file ops, sensitive-path refusal. Tool-loop capped at 4 rounds per user turn.
- Streams over `ssh -L 8000:localhost:8000 qb1`.

**Tool-call plumbing already in place:**
- `cb_api._complete(prompt, body, tools_enabled=True)` flips `skip_special_tokens=False` so `<|tool_call|>` survives into the SSE stream (commit `c65b3c3`).
- `_messages_to_prompt` accepts `tools=...` and passes through to `apply_chat_template`.
- `_strip_decode_noise` hides framing specials (`<|im_start|>`, `<end_of_turn>`) but keeps tool-call markers.
- We do NOT emit OpenAI-format `tool_calls` deltas — the client parses plain text with regex. `chat.py` works because it owns both ends.

**Qwen 3.6-27B chat template + tools:** tokenizer_config.json ships Hermes-style `<tool_call>{json}</tool_call>` tags. Reliability is mixed in vLLM (empty calls, long-turn drift). Community-fixed templates exist. We have not run our own probe yet (see open question 1).

---

## 2. The harness landscape

| Harness | License | Lang | Custom OpenAI base URL? | Tool/edit surface | Maintainer | Stars |
|---|---|---|---|---|---|---|
| **opencode (sst/anomalyco)** | MIT | TypeScript (Bun) + Zig TUI | YES — `@ai-sdk/openai-compatible` provider w/ `baseURL` ([opencode.ai/docs/providers](https://opencode.ai/docs/providers/)) | Edit, Read, Write, Bash, Grep, Glob, LSP (25+ langs), MCP, plan/build agents | Active (Anomaly Innovations, ex-SST) | ~170k |
| **aider** | Apache-2.0 | Python | YES — `OPENAI_API_BASE` + `--model openai/<name>` ([aider.chat/openai-compat](https://aider.chat/docs/llms/openai-compat.html)) | Search/replace diff blocks (no OpenAI tool_calls — purely text edit format), git-integrated, architect/editor modes | Active (Paul Gauthier) | ~30k |
| **continue.dev** | Apache-2.0 | TypeScript (VSCode/JetBrains ext) | YES — `provider: openai, apiBase: http://…/v1` ([docs.continue.dev/openai](https://docs.continue.dev/customize/model-providers/top-level/openai)) | Chat, edit, autocomplete (FIM), in-IDE only | Active | ~25k |
| **Qwen-Agent** | Apache-2.0 | Python | YES (designed around Qwen) | Function calling, MCP, code interp, browser asst. ([github.com/QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)) | Active (QwenLM) | ~10k |

**Why opencode is the natural fit:** client-server split matches ours (opencode's local Bun server + UI clients ↔ our cb_api + tunnel); TUI tool set is Claude-Code-shaped (`Edit`/`Read`/`Write`/`Bash`/`Grep`/`Glob`); provider-agnostic so we can A/B Qwen-on-Blackhole vs Claude Sonnet vs local llama.cpp without harness swaps; MCP supported; Hermes/`<tool_call>` parsing is first-class because Qwen / DeepSeek users hit it constantly (vLLM ships `--tool-call-parser hermes`).

**Aider** is the runner-up: simpler deps (Python, no Bun/Zig), git-aware, but its diff format is in-band text — it'd work today against our `/v1/chat/completions` with no protocol changes. Cost: less Claude-Code-like UX.

---

## 3. Qwen 3.6-27B tool-call readiness gap analysis

Wired today: `apply_chat_template(messages, tools=[...])` (Qwen 3.6 template renders tool defs natively); Hermes `<tool_call>{json}</tool_call>` emission; `skip_special_tokens=False` when `tools=` is in body so markers survive; `chat.py` regex `<tool_call>\s*(\{.*?\})\s*</tool_call>` catches them client-side.

**Gaps for an opencode-grade client** (which consumes OpenAI-format `tool_calls` SSE deltas, not raw text):

1. **No OpenAI `tool_calls` delta emission.** Today we stream `delta.content`. opencode expects `delta.tool_calls = [{index, id, type:"function", function:{name, arguments}}]`. Need a server-side state-machine parser: watch the token stream, enter "tool-call accumulation" at `<tool_call>`, stream `function.arguments` deltas, emit `finish_reason="tool_calls"` at `</tool_call>`. vLLM `--tool-call-parser hermes` is the direct precedent.
2. **`role: "tool"` round-trip.** `chat.py` stores tool replies as `{"role":"tool","name":...,"content":...}`. Qwen 3.6 may require lifting parsed Hermes JSON back into `assistant.tool_calls=[...]` on the previous turn (OpenAI structured shape) — confirm via probe (P0).
3. **Reliability scaffolding.** Known Qwen 3.6 issues: empty tool calls, stray `</function_invocation>`, long-turn drift. Mitigations: community chat-template patches (froggeric), low-temp for tool turns, JSON-validate before round-trip, retry-with-clarification on parse failure.
4. **`finish_reason="tool_calls"`.** `_finish_reason` returns `"stop"|"length"` only. Wire `"tool_calls"` when parser closes a block.
5. **Tool surface vs Claude Code.** Today: shell/read_file/write_file/calc. opencode brings Edit (search/replace), Glob, Grep, WebFetch, LSP — all owned **client-side**; cb_api stays a pure backend.

**Estimated diff:** ~150-300 LOC in `cb_api.py` + `openai_endpoint.py` for the Hermes parser + OpenAI tool_calls SSE. No model code touched.

---

## 4. Recommendation

**Adopt opencode pointed at cb_api over a tunnel; build the Hermes→OpenAI-tool_calls translation in cb_api.**

This is **(b)** from the owner's framing — adopt an existing harness — with a small targeted backend change. NOT (a) (don't pour months into `scripts/chat.py`; it's a developer dashboard, not a product). NOT (c) (don't build our own; opencode already exists and matches our tunnel-to-cb_api shape).

Why: owner vocabulary already says "opencode" (lowest-friction); MIT-licensed (forkable if we want a tt-smi/Prometheus panel in the TUI); Hermes parsing is reusable for 35B and any future model that emits the same tag; keeps `chat.py` honest as a protocol "wireshark" for reproducing opencode bugs without the TypeScript layer; risk is bounded — if Qwen 3.6 tool calling drifts under heavy multi-turn, fallback is **aider** (text diff format, works against today's cb_api unchanged) or **Qwen-Agent** (HTTP shim required but eliminates template mismatch).

---

## 5. Phased plan

| Phase | Week | Deliverable |
|---|---|---|
| **P0 — Qwen3.6 tool probe** | wk 1 (1-2d) | Fork `gemma4_tool_call_probe.py` → `qwen36_tool_call_probe.py`. Establish: special-token IDs, `tools=` render, `role:tool` round-trip, `skip_special_tokens` behavior. |
| **P1 — Hermes parser in cb_api** | wk 1-2 (3-5d) | State-machine parser: token stream → OpenAI `tool_calls` deltas + `finish_reason="tool_calls"`. Lift `tool_calls` back to Qwen shape on the next turn in `_messages_to_prompt`. Unit tests on frozen Qwen3.6 token IDs. |
| **P2 — opencode smoke** | wk 2 (1-2d) | `opencode.json` provider pointing at `http://localhost:8000/v1`. Run Edit/Read/Bash/Grep/Glob on this repo. Screencaps. |
| **P3 — Reliability + observability** | wk 3 (3-5d) | Eval froggeric Qwen3.6 chat-template patches. Tool-loop ceiling + retry-on-malformed-JSON. opencode TUI `/metrics` panel from our Prometheus surface. |
| **P4 — Multi-model switching** | wk 4 (3-5d) | Same parser path on `gemma4_12b` / `35b` (Gemma 4 uses `<|tool_call|>` newtoken — needs a sibling parser, dispatched by `TT_BACKEND`). Multi-`provider` opencode config. |
| **P5 — Production polish** | wk 5+ | Auth (cb_api is unauthenticated today; tunnel-only blocks wider use). Persistent agent state. Optional MCP server for tt-smi / tt-perf data. |

**Critical path is P0→P1→P2** (~7-10 working days). After P2 the owner sees opencode driving Qwen3.6 on Blackhole.

---

## 6. Open questions

1. **Qwen 3.6 tool-call template shape.** Does `apply_chat_template(tools=...)` render Hermes? Does `role:tool` round-trip, or need lifting to `assistant.tool_calls`? Are `tool_call_start`/`_end` monolithic specials? — answered by P0.
2. **35B MoE tool-call reliability.** Top-k tie-break drift (host fallback) is exactly the bug that bites tool calls (one argmax flip turns `"shell"` → `"shel"`). Keep P4 scoped to dense 27B until 35B sampling is hardened?
3. **Gemma 4 `<|tool_call|>` newtoken vs Qwen Hermes `<tool_call>` text-token.** Lean: separate parser per backend, dispatched by `TT_BACKEND`.
4. **Auth + multi-user.** cb_api is tunnel-only today. Non-tunnel access (CI, web) needs keys + per-request quota — P5.
5. **opencode TUI vs IDE.** Owner said "opencode" (TUI). For VSCode users, continue.dev runs on the same cb_api with zero protocol work (it's chat/FIM, not an agent loop — doesn't need `tool_calls` SSE).

---

## Sources

- [OpenCode docs — Providers](https://opencode.ai/docs/providers/)
- [OpenCode docs — Intro](https://opencode.ai/docs/)
- [Aider — OpenAI compatible APIs](https://aider.chat/docs/llms/openai-compat.html)
- [Aider — Edit formats](https://aider.chat/docs/more/edit-formats.html)
- [Aider — Architect mode (Sep 2024 blog)](https://aider.chat/2024/09/26/architect.html)
- [Continue.dev — Configure OpenAI models](https://docs.continue.dev/customize/model-providers/top-level/openai)
- [Qwen function calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Qwen-3 chat template deep-dive (HF blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [vLLM tool calling — Hermes parser](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [QwenLM/Qwen-Agent (GitHub)](https://github.com/QwenLM/Qwen-Agent)
- [Qwen3.6-27B tool-calling drift discussion (HF)](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/13)
- [Qwen3.6 stray `</function_invocation>` issue #178](https://github.com/QwenLM/Qwen3.6/issues/178)
- [froggeric/Qwen-Fixed-Chat-Templates (HF)](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
- [allanchan339 vLLM-Qwen3 chat-template fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
