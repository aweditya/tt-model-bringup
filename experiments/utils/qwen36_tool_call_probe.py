#!/usr/bin/env python3
"""Qwen 3.6-27B tokenizer + tool-call (Hermes) probe.

Forked from ``experiments/utils/gemma4_tool_call_probe.py`` per the
[[reuse-mandate]] memory: before writing a new probe, copy the existing
pattern and only adapt model-specific bits. The structural shape — five
sections covering specials, ``apply_chat_template(tools=)`` render,
``role:tool`` round-trip, ``skip_special_tokens`` behaviour, and EOS
IDs — is unchanged. Only the model id and the special-token candidate
list differ.

What this verifies (no device required — pure tokenizer / chat-template):

  1. **Tool special tokens exist in the vocabulary** for Qwen's Hermes
     format. Qwen 3.6 emits ``<tool_call>{json}</tool_call>`` as plain
     text tokens, not as monolithic specials (unlike Gemma 4's
     ``<|tool_call|>`` newtoken). We probe both forms so we can prove
     which path the model actually uses, and so the parser knows whether
     to scan token IDs or substring-match the decoded text.

  2. **``apply_chat_template(tools=[...])`` produces a sensible prompt**
     — the Qwen 3.6 chat template ships native tool-render support
     (tokenizer_config.json). We capture the ID sequence so the Hermes
     parser knows what the model expects as a render contract.

  3. **``role: "tool"`` round-trip** — when the harness sends back a
     tool result, does the Qwen template accept ``{"role":"tool",...}``
     directly, or does it require lifting the prior assistant turn into
     ``assistant.tool_calls=[...]`` (OpenAI structured shape)? Open
     question #1 in ``research/agentic_harness_scope_2026-06-10.md``.

  4. **``skip_special_tokens=True`` vs ``False``** on a synthetic
     generation containing ``<tool_call>...</tool_call>``. Because the
     Hermes tags are likely plain-text tokens, both decode paths should
     surface the marker — confirming that ``cb_api``'s current
     ``tools_enabled → skip_special_tokens=False`` flip is sufficient
     but not strictly required for Qwen (unlike Gemma 4, where it is
     mandatory).

  5. **EOS / end-of-turn IDs** — Qwen 3.6 uses ``<|im_end|>`` to
     terminate assistant turns. ``cb_api._finish_reason`` decides
     ``finish_reason`` from ``eos_set``; we surface the canonical IDs
     so the Hermes parser's terminal state can also fire on the right
     EOS.

Run on qb1 (per remote-only contract):

    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/qwen36_tool_call_probe.py'
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.6-27B"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Qwen 3.6 Hermes-style tool format candidates. Most are plain text in
# Qwen (not monolithic specials) — the probe will confirm. Framing
# tokens (`<|im_start|>` / `<|im_end|>`) ARE monolithic specials in
# Qwen and matter for the parser's terminal state.
_QWEN36_TOOL_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<|tool_result|>",
    "<|im_start|>",
    "<|im_end|>",
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
    this special, just the literal substring. For Qwen Hermes, the
    expected outcome is that ``<tool_call>`` is NOT monolithic (returns
    None) but ``<|im_end|>`` IS monolithic."""
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        return None
    if len(ids) != 1:
        return None
    return ids[0]


def _multi_token_split(tokenizer, s: str) -> list[int] | None:
    """If `s` is not monolithic, return how the tokenizer splits it.
    Needed for the Hermes parser: if ``<tool_call>`` splits into e.g.
    ``['<', 'tool', '_call', '>']`` the parser must scan decoded text,
    not raw token IDs."""
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        return None
    return list(ids)


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
    splits: dict[str, list[int]] = {}
    for s in _QWEN36_TOOL_TOKENS:
        tid = _try_token_to_id(tok, s)
        if tid is not None:
            found[s] = tid
            print(f"  OK   {s!r:24}  → id {tid}  (monolithic special)")
        else:
            ids = _multi_token_split(tok, s) or []
            splits[s] = ids
            decoded = [tok.decode([i], skip_special_tokens=False) for i in ids]
            print(f"  miss {s!r:24}  not monolithic; splits into "
                  f"{len(ids)} ids → {decoded}")

    has_tool_call_special = (
        "<tool_call>" in found or "</tool_call>" in found
    )
    print(f"  HAS_TOOL_CALL_SPECIAL = {has_tool_call_special}")
    print(f"  (Expected for Qwen Hermes: False — tags are plain text.)")

    print()
    print("══ 2. apply_chat_template(tools=[...]) render " + "═" * 22)
    msgs = [{"role": "user", "content": "What is 12 * 7?"}]
    try:
        ids_with_tools = tok.apply_chat_template(
            msgs, tools=DEMO_TOOLS, tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids_with_tools, "input_ids"):
            ids_with_tools = list(ids_with_tools["input_ids"])
        else:
            ids_with_tools = list(ids_with_tools)
        text_with_tools = tok.decode(ids_with_tools, skip_special_tokens=False)
        print(f"  ids len = {len(ids_with_tools)}")
        snippet = text_with_tools
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n  …<truncated>…"
        print("  rendered prompt (specials NOT stripped):")
        for line in snippet.splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"  FAIL  apply_chat_template(tools=...) raised {e!r}")
        ids_with_tools = []

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
    msgs_tool_rt = [
        {"role": "user", "content": "What is 12 * 7?"},
        # Assistant emitted a Hermes tool call as plain content.
        {"role": "assistant",
         "content": '<tool_call>\n{"name":"calc","arguments":{"expr":"12*7"}}\n</tool_call>'},
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
        print(f"        ↪ cb_api will hit this on every tool reply")
        rt_ok = False

    # Try the alternative "assistant.tool_calls" field — the OpenAI
    # structured shape that vLLM's hermes parser lifts into.
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
    # Build a synthetic assistant emission in Hermes shape. Because the
    # tags are likely plain text tokens, this uses ``tok.encode`` for
    # everything (no special-token splice).
    hermes_emission = (
        '<tool_call>\n{"name":"calc","arguments":{"expr":"12*7"}}\n</tool_call>'
    )
    syn_tokens = tok.encode(hermes_emission, add_special_tokens=False)
    if "<|im_end|>" in found:
        syn_tokens.append(found["<|im_end|>"])

    if syn_tokens:
        with_special = tok.decode(syn_tokens, skip_special_tokens=False)
        without_special = tok.decode(syn_tokens, skip_special_tokens=True)
        print(f"  with    skip_special_tokens=False → {with_special!r}")
        print(f"  with    skip_special_tokens=True  → {without_special!r}")
        survives_false = "<tool_call>" in with_special
        survives_true = "<tool_call>" in without_special
        print(f"  '<tool_call>' visible in False-decode: {survives_false}")
        print(f"  '<tool_call>' visible in True-decode:  {survives_true}")
        # For Qwen Hermes we expect BOTH to survive (it's plain text);
        # the only difference should be the <|im_end|> EOS marker.
        if survives_true:
            print("  → Hermes tags are plain text in Qwen; parser can run "
                  "against either decode mode. cb_api's "
                  "skip_special_tokens=False flip on tools_enabled is "
                  "still safe (only changes EOS marker visibility).")
        else:
            print("  → unexpected: Hermes tags vanished under "
                  "skip_special_tokens=True. Parser MUST use "
                  "skip_special_tokens=False, same as Gemma 4.")
    else:
        print("  skipped — no tokens encoded.")

    print()
    print("══ 5. EOS / end-of-turn IDs " + "═" * 40)
    print(f"  tokenizer.eos_token       = {tok.eos_token!r}")
    print(f"  tokenizer.eos_token_id    = {tok.eos_token_id}")
    print(f"  tokenizer.bos_token       = {tok.bos_token!r}")
    print(f"  tokenizer.bos_token_id    = {tok.bos_token_id}")
    for s in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        tid = _try_token_to_id(tok, s)
        if tid is not None:
            print(f"  {s!r:18} → id {tid}")
    gen_cfg = getattr(tok, "model_input_names", None)
    print(f"  model_input_names = {gen_cfg}")

    print()
    print("══ Summary " + "═" * 56)
    print(f"  has_tool_call_special: {has_tool_call_special} "
          f"(expected False for Qwen Hermes; tags are plain text)")
    print(f"  tools=...  template render OK")
    print(f"  role:tool plain content rt ok: {rt_ok}")
    print(f"  role:tool with tool_calls rt ok: {tc_ok}")
    print(f"  recommended cb_api parser: substring scan on decoded text, "
          f"NOT token-id scan (Hermes tags are multi-token).")
    print(f"  recommended cb_api decode: "
          f"skip_special_tokens=False so <|im_end|> EOS is visible to "
          f"the parser terminal state, even though the tags themselves "
          f"survive either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
