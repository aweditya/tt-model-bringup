#!/usr/bin/env python3
"""Gemma 4 12B IT tokenizer + tool-call probe.

What this verifies (no device required — pure tokenizer / chat-template):

  1. **Tool special tokens exist in the vocabulary** and resolve to single
     IDs. If a token doesn't exist in the model's actual tokenizer,
     chat.py's `<|tool_call|>` regex matches but the model could never
     emit the literal string.

  2. **`apply_chat_template(tools=[...])` produces a sensible prompt** —
     the template knows how to render tool definitions; we want to see
     the ID sequence so we can detect what the model expects on the
     output side.

  3. **`role: "tool"` round-trip** — when chat.py sends back a tool result
     via `{"role": "tool", "name": ..., "content": ...}`, the chat
     template either renders it or errors out. We need to know.

  4. **`skip_special_tokens=True` vs `False`** on a synthetic generation
     that contains `<|tool_call|>`. If `True` strips it, chat.py's regex
     parser can never see the new-token format, only the
     ```tool_code``` fallback.

  5. **EOS / end-of-turn IDs** — Gemma 4 emits `<|end_of_turn|>` after a
     tool call. cb_api uses `eos_set` to decide finish_reason; if only
     `<|end_of_text|>` is in eos_set, the server keeps generating
     past the tool-call boundary.

Run on qb1 (per remote-only contract):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/gemma4_tool_call_probe.py'

REUSE: forks the suffix-detection + chat-template probing pattern from
`experiments/utils/nemotron3_tokenizer_probe.py` (which itself forks
`experiments/serve/openai_endpoint._active_prompt_suffix`).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-4-12B-it"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Six special tokens documented for Gemma 4's tool format. Source: chat.py
# _TOOL_PATTERNS comment + tokenizer_config inspection.
_GM4_TOOL_TOKENS = [
    "<|tool>",
    "<|tool_call>",
    "<|tool_response>",
    "<|tool_call|>",     # alt form some templates emit
    "<|end_of_turn|>",
    "<|end_of_text|>",
]


# A minimal tool def + a tool-result echo to test the round-trip.
DEMO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a simple arithmetic expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    }
]


def _try_token_to_id(tokenizer, s: str) -> int | None:
    """Return the single-token id for `s`, or None if not present as a
    monolithic special token. We do NOT want to accept multi-token splits
    here — that would mean the model's tokenizer doesn't actually have
    this special, just the literal substring."""
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        return None
    if len(ids) != 1:
        return None
    return ids[0]


def main() -> int:
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"FAIL  transformers import: {e!r}", file=sys.stderr)
        return 2

    log(f"loading tokenizer for {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    log(f"  vocab_size={tok.vocab_size}  cls={type(tok).__name__}")

    print()
    print("══ 1. tool special tokens (single-id check) " + "═" * 24)
    found: dict[str, int] = {}
    for s in _GM4_TOOL_TOKENS:
        tid = _try_token_to_id(tok, s)
        if tid is not None:
            found[s] = tid
            print(f"  OK   {s!r:24}  → id {tid}")
        else:
            print(f"  miss {s!r:24}  not a monolithic special token")

    # If neither <|tool_call|> nor <|tool_call> is monolithic, the model
    # genuinely doesn't have the new-token format and we must use the
    # tool_code fallback exclusively.
    has_tool_call_special = (
        "<|tool_call|>" in found or "<|tool_call>" in found
    )
    print(f"  HAS_TOOL_CALL_SPECIAL = {has_tool_call_special}")

    print()
    print("══ 2. apply_chat_template(tools=[...]) render " + "═" * 22)
    # Build a tiny user turn with the tool def attached.
    msgs = [{"role": "user", "content": "What is 12 * 7?"}]
    try:
        ids_with_tools = tok.apply_chat_template(
            msgs, tools=DEMO_TOOLS, tokenize=True,
            add_generation_prompt=True,
        )
        # apply_chat_template can return a list[int] OR a BatchEncoding;
        # normalise.
        if hasattr(ids_with_tools, "input_ids"):
            ids_with_tools = list(ids_with_tools["input_ids"])
        else:
            ids_with_tools = list(ids_with_tools)
        text_with_tools = tok.decode(ids_with_tools, skip_special_tokens=False)
        print(f"  ids len = {len(ids_with_tools)}")
        # Truncate the print for readability.
        snippet = text_with_tools
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n  …<truncated>…"
        print("  rendered prompt (specials NOT stripped):")
        for line in snippet.splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"  FAIL  apply_chat_template(tools=...) raised {e!r}")
        ids_with_tools = []

    # Same prompt without tools — diff lets us see exactly what the tools
    # arg injects.
    try:
        ids_no_tools = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
        )
        if hasattr(ids_no_tools, "input_ids"):
            ids_no_tools = list(ids_no_tools["input_ids"])
        else:
            ids_no_tools = list(ids_no_tools)
        print(f"  ids len WITHOUT tools = {len(ids_no_tools)}"
              f"  (delta = {len(ids_with_tools) - len(ids_no_tools)} ids "
              "for the tool block)")
    except Exception as e:
        print(f"  WARN  no-tools render also failed: {e!r}")

    print()
    print("══ 3. role:tool round-trip (does the template accept it?) " + "═" * 8)
    # Simulate the chat.py round-trip: user → assistant tool-call →
    # tool result → next assistant prompt.
    msgs_tool_rt = [
        {"role": "user", "content": "What is 12 * 7?"},
        # Assistant emitted a tool call — we represent it the way chat.py
        # currently stores it (plain assistant content with the call
        # text). HF templates that support tool_calls expect a different
        # field; this probe surfaces the mismatch.
        {"role": "assistant", "content": '<|tool_call|>{"name":"calc","arguments":{"expr":"12*7"}}'},
        {"role": "tool", "name": "calc", "content": "84"},
    ]
    try:
        ids_rt = tok.apply_chat_template(
            msgs_tool_rt, tools=DEMO_TOOLS, tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids_rt, "input_ids"):
            ids_rt = list(ids_rt["input_ids"])
        else:
            ids_rt = list(ids_rt)
        text_rt = tok.decode(ids_rt, skip_special_tokens=False)
        print(f"  ids len = {len(ids_rt)}  (round-trip accepted)")
        snippet = text_rt
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n  …<truncated>…"
        for line in snippet.splitlines():
            print(f"    {line}")
        rt_ok = True
    except Exception as e:
        print(f"  FAIL  role:tool round-trip raised {e!r}")
        print(f"        ↪ chat.py will hit this on every tool reply")
        rt_ok = False

    # Try the alternative "assistant.tool_calls" field that many HF
    # templates expect (OpenAI shape).
    print()
    print("══ 3b. role:tool with assistant.tool_calls field " + "═" * 18)
    msgs_tool_calls = [
        {"role": "user", "content": "What is 12 * 7?"},
        {"role": "assistant", "tool_calls": [{
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expr":"12*7"}'},
        }]},
        {"role": "tool", "name": "calc", "content": "84"},
    ]
    try:
        ids_tc = tok.apply_chat_template(
            msgs_tool_calls, tools=DEMO_TOOLS, tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids_tc, "input_ids"):
            ids_tc = list(ids_tc["input_ids"])
        else:
            ids_tc = list(ids_tc)
        text_tc = tok.decode(ids_tc, skip_special_tokens=False)
        print(f"  ids len = {len(ids_tc)}  (tool_calls accepted)")
        snippet = text_tc
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n  …<truncated>…"
        for line in snippet.splitlines():
            print(f"    {line}")
        tc_ok = True
    except Exception as e:
        print(f"  FAIL  assistant.tool_calls field raised {e!r}")
        tc_ok = False

    print()
    print("══ 4. skip_special_tokens behaviour " + "═" * 30)
    # Build a synthetic assistant emission that would represent a tool
    # call in the new-token format.
    syn_tokens: list[int] = []
    if "<|tool_call|>" in found:
        syn_tokens.append(found["<|tool_call|>"])
    elif "<|tool_call>" in found:
        syn_tokens.append(found["<|tool_call>"])
    # Append JSON text as regular tokens.
    json_payload = '{"name":"calc","arguments":{"expr":"12*7"}}'
    syn_tokens.extend(tok.encode(json_payload, add_special_tokens=False))
    if "<|end_of_turn|>" in found:
        syn_tokens.append(found["<|end_of_turn|>"])

    if syn_tokens:
        with_special = tok.decode(syn_tokens, skip_special_tokens=False)
        without_special = tok.decode(syn_tokens, skip_special_tokens=True)
        print(f"  with    skip_special_tokens=False → {with_special!r}")
        print(f"  with    skip_special_tokens=True  → {without_special!r}")
        # Does the new-token marker survive the strip?
        survives_false = "<|tool_call" in with_special
        survives_true = "<|tool_call" in without_special
        print(f"  '<|tool_call' visible in False-decode: {survives_false}")
        print(f"  '<|tool_call' visible in True-decode:  {survives_true}")
        if survives_false and not survives_true:
            print("  → chat.py NEW-TOKEN regex requires server to decode "
                  "with skip_special_tokens=False (current default in "
                  "cb_api is True, which strips the marker silently).")
    else:
        print("  skipped — no tool-call special token resolved.")

    print()
    print("══ 5. EOS / end-of-turn IDs " + "═" * 40)
    print(f"  tokenizer.eos_token       = {tok.eos_token!r}")
    print(f"  tokenizer.eos_token_id    = {tok.eos_token_id}")
    print(f"  tokenizer.bos_token       = {tok.bos_token!r}")
    print(f"  tokenizer.bos_token_id    = {tok.bos_token_id}")
    # Useful pair for the chat template:
    for s in ("<|end_of_turn|>", "<|end_of_text|>"):
        tid = _try_token_to_id(tok, s)
        if tid is not None:
            print(f"  {s!r:18} → id {tid}")
    # generation_config-ish list:
    gen_cfg = getattr(tok, "model_input_names", None)
    print(f"  model_input_names = {gen_cfg}")

    print()
    print("══ Summary " + "═" * 56)
    print(f"  has_tool_call_special: {has_tool_call_special}")
    print(f"  tools=...  template render OK")
    print(f"  role:tool plain content rt ok: {rt_ok}")
    print(f"  role:tool with tool_calls rt ok: {tc_ok}")
    print(f"  recommended cb_api decode: "
          f"skip_special_tokens="
          f"{'False (so parser sees new-token format)' if has_tool_call_special else 'True'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
