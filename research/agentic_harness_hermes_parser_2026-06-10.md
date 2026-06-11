# Hermes tool-call parser design — cb_api streaming integration

**Date:** 2026-06-10
**Scope:** P1 in [`agentic_harness_scope_2026-06-10.md`](agentic_harness_scope_2026-06-10.md) — wire Qwen 3.6's Hermes `<tool_call>{json}</tool_call>` emissions into OpenAI-shaped `delta.tool_calls` SSE events so opencode (and any other OpenAI-compat agent harness) can drive cb_api directly.
**Reference implementation:** [`vllm/tool_parsers/hermes_tool_parser.py`](https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/hermes_tool_parser.py) (Hermes2ProToolParser).

---

## 1. State machine

```
            +------------------+   no '<' seen
            |  PLAIN_TEXT      |<-----------------+
            |  flush deltas    |                  |
            +--------+---------+                  |
                     | '<' in buffer              |
                     v                            |
            +------------------+                  |
            |  MAYBE_OPEN_TAG  |  not a prefix    |
            |  hold back tail  +------------------+
            +--------+---------+
                     | full '<tool_call>' matched
                     v
            +------------------+
            |  TOOL_CALL_BODY  |  buffer + try-parse-json each delta
            |  emit name once  |  emit args diff per delta
            +--------+---------+
                     | full '</tool_call>' matched
                     v
            +------------------+
            |  TOOL_CALL_DONE  |---> finalise call i, index++
            +--------+---------+
                     | EOS (<|im_end|>) or new '<tool_call>'
                     v
              finish="tool_calls"   (PLAIN_TEXT if more calls)
```

Mirrors vLLM's Hermes2Pro: `_extract_content` holds partial-tag suffix, `_extract_tool_call_jsons` runs `r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)"` (DOTALL) per delta. Plain text outside the tag streams as `delta.content`.

---

## 2. Streaming concerns

**Partial-tag arrival.** Delta may end mid-tag (`"...sure! <tool_"`). Hold any tail that is a strict prefix of `<tool_call>` or `</tool_call>`; emit everything before it. Literal prefix check (no regex) per vLLM's `partial_tag_overlap()`.

**False-positive `<`.** If the held tail stops prefixing a real tag (`< 5`), flush on next delta. Use a `_held_since` counter; force-flush after N deltas.

**Mid-JSON closing tag.** Skip `</tool_call>` while inside a JSON string. Brace-balance counter in `TOOL_CALL_BODY`: only honour the close once JSON parses cleanly.

**Incremental name + args.** Stream `name` on first match of `r'"name"\s*:\s*"([^"]+)"'`; stream `arguments` as a string diff vs `streamed_args_for_tool[i]`. Mirrors vLLM.

---

## 3. Multi-call handling

Qwen 3.6 can emit two `<tool_call>` blocks per turn. On each open we append to `self.calls: list[CallState]` and use the list index as `delta.tool_calls[].index`; each gets a `call_<uuid4hex24>` id. Closing resets to `PLAIN_TEXT`; the next open gets `index = len(self.calls)`.

`finish_reason="tool_calls"` fires once at EOS if at least one call closed. Partial-call-at-EOS → `finish_reason="stop"`, drop the partial (matches vLLM).

---

## 4. Concrete integration into `cb_api.py`

Touch surface (3 small edits, no model code):

- **`experiments/serve/cb_api.py:239-255`** — the `sse()` inner generator inside `_complete`. Today: `tok.decode(gen_ids, skip_special_tokens=skip_specials)` per step then yield `delta.content`. Wrap the per-delta string through a parser instance and yield whatever events the parser produces (`delta.content` or `delta.tool_calls`).
- **`experiments/serve/cb_api.py:182-195`** — `_finish_reason` gains a third arm: if the parser closed at least one tool call, return `"tool_calls"` instead of `"stop"`.
- **`experiments/serve/cb_api.py:271`** — `chat_completions` already passes `tools_enabled=bool(tools)`; add `parser_kind=` so dispatch picks Hermes for Qwen and Gemma's `<|tool_call|>` newtoken parser for Gemma 4 (P4 in the scope doc).
- **`experiments/serve/openai_endpoint.py`** — `_chat_chunk` gains an optional `tool_calls=` arg to render the OpenAI delta shape; mirrors the existing `delta.content` path.

Non-streaming path (`cb_api.py:257-264`) feeds the same parser with the whole decoded string at once and emits the OpenAI `message.tool_calls` (non-delta) shape.

---

## 5. Code sketch (do NOT apply; design only)

```python
# experiments/serve/hermes_tool_parser.py  (new file in P1)
from __future__ import annotations
import json, re, uuid
from dataclasses import dataclass, field

_OPEN  = "<tool_call>"
_CLOSE = "</tool_call>"
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')

@dataclass
class _CallState:
    index: int
    id: str
    name_sent: bool = False
    args_sent_len: int = 0  # chars of 'arguments' streamed so far
    buf: str = ""

@dataclass
class HermesStreamParser:
    """Stateful per-request parser. Feed `feed(delta_text)` once per
    decoded delta; consume the yielded events (`('content', str)` or
    `('tool_call_delta', index, dict)` or `('finish', reason)`)."""
    pending: str = ""           # held tail that may be a partial tag
    in_call: bool = False
    calls: list[_CallState] = field(default_factory=list)
    saw_any_call: bool = False

    def _maybe_partial_tag(self, s: str) -> int:
        """Return len of suffix to hold (prefix of _OPEN or _CLOSE)."""
        for tag in (_OPEN, _CLOSE):
            for k in range(min(len(tag), len(s)), 0, -1):
                if s.endswith(tag[:k]) and not s.endswith(tag):
                    return k
        return 0

    def feed(self, delta: str):
        self.pending += delta
        while self.pending:
            if not self.in_call:
                idx = self.pending.find(_OPEN)
                if idx == -1:
                    hold = self._maybe_partial_tag(self.pending)
                    if hold < len(self.pending):
                        yield ("content", self.pending[:-hold] if hold else self.pending)
                    self.pending = self.pending[-hold:] if hold else ""
                    return
                if idx > 0:
                    yield ("content", self.pending[:idx])
                self.pending = self.pending[idx + len(_OPEN):]
                self.calls.append(_CallState(
                    index=len(self.calls),
                    id=f"call_{uuid.uuid4().hex[:24]}",
                ))
                self.in_call = True
                self.saw_any_call = True
            else:
                call = self.calls[-1]
                end = self.pending.find(_CLOSE)
                chunk = self.pending if end == -1 else self.pending[:end]
                call.buf += chunk
                self.pending = "" if end == -1 else self.pending[end + len(_CLOSE):]
                # Try to emit name + args diff from call.buf.
                if not call.name_sent:
                    m = _NAME_RE.search(call.buf)
                    if m:
                        yield ("tool_call_delta", call.index, {
                            "id": call.id, "type": "function",
                            "function": {"name": m.group(1)},
                        })
                        call.name_sent = True
                # Args streaming: pull whatever's inside "arguments": ...
                args_str = _extract_arguments_so_far(call.buf)
                if args_str is not None and len(args_str) > call.args_sent_len:
                    diff = args_str[call.args_sent_len:]
                    call.args_sent_len = len(args_str)
                    yield ("tool_call_delta", call.index, {
                        "function": {"arguments": diff},
                    })
                if end != -1:
                    self.in_call = False  # next iter handles trailing text

    def finish(self):
        if self.pending and not self.in_call:
            yield ("content", self.pending)
        yield ("finish", "tool_calls" if self.saw_any_call else "stop")
```

`_extract_arguments_so_far` is a brace-balance walker that returns the current substring of the `arguments` value (JSON object source) even when incomplete. Stub omitted for brevity — see vLLM `_extract_tool_args` for the canonical implementation.

---

## 6. Test cases

| # | Input deltas (chronological) | Expected events |
|---|---|---|
| 1 | `"hi there"`, `"!"` | `content "hi there"`, `content "!"`, `finish stop` |
| 2 | `"<tool_call>"`, `'{"name":"calc","arguments":{"x":1}}'`, `"</tool_call>"` | `tool_call_delta 0 {name:"calc"}`, `tool_call_delta 0 {args:'{"x":1}'}`, `finish tool_calls` |
| 3 | `'<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>'` (single delta) | calls 0+1 emitted in order, `finish tool_calls` |
| 4 | `"sure <tool_"`, `"call>{..."` | `content "sure "`, hold `<tool_`, then enter call |
| 5 | `"price < 5 and "` | `content "price "`, hold `<`, then flush `< 5 and ` once it can't prefix a tag |
| 6 | `'<tool_call>{"name":"x","arguments":{"q":"</tool_call>"}}</tool_call>'` | brace-balance keeps inner `</tool_call>` inside the JSON string; only the outer `</tool_call>` closes the call |
| 7 | `'<tool_call>{"name":"x","arguments":'` then EOS | partial JSON, no name yet maybe; emit `finish stop` and drop |

Tests live at `experiments/utils/test_hermes_parser.py`, fed with frozen Qwen 3.6 token-id sequences captured from the P0 probe.

---

## 7. Open questions

1. **Brace-counter vs strict-JSON validator.** vLLM uses `is_complete_json`; we could reuse `json.JSONDecoder.raw_decode` for the same effect with stdlib only. Trade-off: stdlib is slower per call but no new dep.
2. **`role:tool` lift-back.** When the harness replies with `{"role":"tool",...}`, does the Qwen 3.6 template render correctly, or must `_messages_to_prompt` synthesise `{"role":"assistant","tool_calls":[...]}` on the prior turn? P0 probe answers this; if lift-back is required, owner of the buffered call needs to retain the parsed JSON to reconstruct.
3. **Empty / malformed call emission.** Qwen 3.6 issue [#178](https://github.com/QwenLM/Qwen3.6/issues/178) reports stray `</function_invocation>` artefacts. Should the parser silently drop a malformed block, or surface it as `delta.content` to the client (so opencode can show the user)? Lean: drop + log under `TT_TOOL_PARSE_STRICT=0`, surface under `=1`.
4. **`<think>` block interaction.** Qwen 3.6 streams `<think>...</think>` before regular content. The parser should treat `<think>` as plain content and only react to `<tool_call>`. Verify the held-tail logic doesn't mis-prefix `<think>` against `<tool_call>` (they share `<t`).
5. **Backend dispatch.** Gemma 4 uses `<|tool_call|>` (monolithic special). Sibling parser keyed on `TT_BACKEND`; common interface (`feed → events`). Land Hermes first (P1), Gemma 4 newtoken parser later (P4).

---

## Sources

- [vllm/tool_parsers/hermes_tool_parser.py (main)](https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/hermes_tool_parser.py)
- [vLLM tool calling docs — Hermes parser](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [Qwen function calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Qwen 3 chat template deep-dive (HF blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Qwen3.6 stray `</function_invocation>` issue #178](https://github.com/QwenLM/Qwen3.6/issues/178)
- [OpenAI Chat Completions streaming reference](https://platform.openai.com/docs/api-reference/chat-streaming)
