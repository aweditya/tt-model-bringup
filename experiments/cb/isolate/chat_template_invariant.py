"""3-turn round-trip invariant test for Qwen3.6 prefix caching.

Validates the invariant that breaks prefix caching when violated:

    tokenize(_messages_to_prompt(messages_N)) + gen_N
        == tokenize(_messages_to_prompt(messages_{N+1}))[:len(cached_N)]

i.e. turn-N's cached tokens are a prefix of turn-(N+1)'s prompt tokens.
If this holds, find_longest_match in the live-slot cache returns the full
cached length and turn N+1 skips re-prefill of the history.

Gates the production `_messages_to_prompt` fix (preserve_thinking=True +
trailing strip). Catches future Qwen3.6 chat-template regressions before
they cause silent cache-miss perf cliffs in production.

Run on qb1 (needs the actual tokenizer):
  cd ~/tt-xla && .venv/bin/python experiments/cb/isolate/chat_template_invariant.py

Exits 0 on full pass; nonzero if a load-bearing case regresses.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # repo root
_SERVE = os.path.join(_ROOT, "experiments", "serve")
sys.path.insert(0, _ROOT)   # so `experiments.serve.protocol` resolves
sys.path.insert(0, _SERVE)  # so `openai_endpoint` resolves as a top-level import

from transformers import AutoTokenizer  # noqa: E402

from openai_endpoint import _messages_to_prompt  # noqa: E402

MODEL_ID = os.environ.get("TT_MODEL_ID", "Qwen/Qwen3.6-27B")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
EOS = tok.eos_token_id


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    lim = min(len(a), len(b))
    while n < lim and a[n] == b[n]:
        n += 1
    return n


def _show_divergence(cached: list[int], nxt: list[int], n: int):
    lo = max(0, n - 3)
    hi = min(min(len(cached), len(nxt)), n + 4)
    print(f"     cached  [{lo}:{hi}] = {cached[lo:hi]}")
    print(f"     next    [{lo}:{hi}] = {nxt[lo:hi]}")
    print(f"     cached  detok: {tok.decode(cached[lo:hi])!r}")
    print(f"     next    detok: {tok.decode(nxt[lo:hi])!r}")


def _check_full_prefix(cached: list[int], next_turn_ids: list[int], label: str,
                       allow_known_gap: bool = False) -> bool:
    n = longest_common_prefix(cached, next_turn_ids)
    if n != len(cached):
        marker = "⚠" if allow_known_gap else "✗"
        print(f"  {marker} {label}: {n} / {len(cached)} match")
        _show_divergence(cached, next_turn_ids, n)
        return allow_known_gap
    print(f"  ✓ {label}: {n} / {len(cached)} match")
    return True


def _cached_for_turn(messages: list[dict], gen_content: str) -> list[int]:
    """What would land in live_slots.tokens_so_far after turn finishes.

    Production `_messages_to_prompt` produces the prompt; the model emits
    `gen_content` as text (which `tok.decode(gen_ids, skip_special=True)` would
    return), then `<|im_end|>` (EOS) when the turn ends. We cache the full
    sequence including EOS (per the PC-P5 fix in cb_scheduler._finish).
    """
    prompt_ids = tok.encode(_messages_to_prompt(tok, messages))
    gen_ids = tok.encode(gen_content, add_special_tokens=False) + [EOS]
    return prompt_ids + gen_ids


# ── Cases ────────────────────────────────────────────────────────────────────

def case_empty_thinking_two_turn() -> bool:
    """The smoke-test scenario: model emits the empty `<think>\\n\\n</think>\\n\\n`
    block as text (because we strip it from the active prompt; the model fills
    it back in), then the answer. Turn 2 must pick up at the boundary."""
    sys_msg = {"role": "system",
               "content": "You are a concise, helpful assistant."}
    u1 = {"role": "user", "content": "Capital of France?"}
    a1_text = "<think>\n\n</think>\n\nParis."
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "And Germany?"}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    return _check_full_prefix(cached, turn2, "2-turn: empty-thinking response")


def case_nontrivial_thinking_two_turn() -> bool:
    """Realistic non-empty thinking trace. Tests that the template's
    `<think>` content extraction round-trips correctly via preserve_thinking."""
    sys_msg = {"role": "system", "content": "Helpful assistant."}
    u1 = {"role": "user", "content": "What's 2+2?"}
    a1_text = "<think>\nLet me think.\n2 plus 2 is 4.\n</think>\n\n4."
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "And 3+3?"}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    return _check_full_prefix(cached, turn2, "2-turn: non-empty thinking")


def case_three_turn_compound() -> bool:
    """3-turn chain: cached after turn 1 must prefix turn 2's prompt,
    AND cached after turn 2 must prefix turn 3's prompt. The real invariant
    we care about — chat clients accumulate."""
    sys_msg = {"role": "system", "content": "Assistant."}
    u1 = {"role": "user", "content": "First question."}
    a1_text = "<think>\nT1.\n</think>\n\nAnswer 1."
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "Second question."}
    a2_text = "<think>\nT2.\n</think>\n\nAnswer 2."
    a2 = {"role": "assistant", "content": a2_text}
    u3 = {"role": "user", "content": "Third question."}

    cached_1 = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    if not _check_full_prefix(cached_1, turn2, "3-turn: cached_1 ⊂ turn_2"):
        return False

    cached_2 = _cached_for_turn([sys_msg, u1, a1, u2], a2_text)
    turn3 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2, a2, u3]))
    return _check_full_prefix(cached_2, turn3, "3-turn: cached_2 ⊂ turn_3")


def case_long_system_prompt() -> bool:
    """Real chats often have a long system prompt (e.g., persona, RAG context).
    Verifies the invariant holds across the longer prefix typical of
    real chat workloads."""
    sys_msg = {"role": "system",
               "content": (
                   "You are an expert technical assistant. " * 30
                   + "Always reply concisely.")}
    u1 = {"role": "user", "content": "Hello."}
    a1_text = "<think>\n\n</think>\n\nHi there!"
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "How are you?"}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    return _check_full_prefix(cached, turn2, "long-system-prompt 2-turn")


def case_unicode_content() -> bool:
    """BPE tokenizers are usually deterministic on text→tokens, but unicode
    in user messages is a classic round-trip stress."""
    sys_msg = {"role": "system", "content": "Assistant."}
    u1 = {"role": "user", "content": "Translate '猫が好きです' please."}
    a1_text = "<think>\n\n</think>\n\nIt means 'I like cats' in Japanese."
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "How about '犬'?"}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    return _check_full_prefix(cached, turn2, "unicode round-trip")


def case_response_without_think_markers() -> bool:
    """KNOWN GAP (flagged in research/vllm_chat_template_handling.md §5):
    if the model emits content WITHOUT `<think>...</think>` markers, the
    preserve_thinking=True template WRAPS the past content with an empty
    `<think>\\n\\n</think>\\n\\n` block on re-render — adding tokens that
    weren't in cached. In our production setup the model is reliably trained
    to emit the markers, so this case is informational, not blocking."""
    sys_msg = {"role": "system", "content": "Assistant."}
    u1 = {"role": "user", "content": "Hi."}
    a1_text = "Hello!"  # no <think> markers
    a1 = {"role": "assistant", "content": a1_text}
    u2 = {"role": "user", "content": "Bye."}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2 = tok.encode(_messages_to_prompt(tok, [sys_msg, u1, a1, u2]))
    return _check_full_prefix(
        cached, turn2,
        "no-thinking-markers (KNOWN GAP — informational)",
        allow_known_gap=True)


def case_tools_present_warning() -> bool:
    """KNOWN GAP (research doc §5 caveat 2): when the messages list contains
    tool_calls, Qwen3.6's chat template enters a 'rolling checkpoint' mode
    that OVERRIDES preserve_thinking for messages before the last non-tool-
    call user turn. We don't currently expose tools in the chat API, so this
    is informational. Re-validate when tool support is added."""
    sys_msg = {"role": "system", "content": "Assistant with tools."}
    u1 = {"role": "user", "content": "Compute."}
    # A bare tool-call-shaped message — the template enters the rolling
    # checkpoint branch and treats the older `<think>` blocks differently.
    a1_text = "<think>\nNeed to call.\n</think>\n\nCalling..."
    a1 = {"role": "assistant",
          "content": a1_text,
          "tool_calls": [{"function": {"name": "calc",
                                       "arguments": {"x": 1}}}]}
    t1_resp = {"role": "tool", "content": "42"}
    u2 = {"role": "user", "content": "Continue."}

    cached = _cached_for_turn([sys_msg, u1], a1_text)
    turn2_msgs = [sys_msg, u1, a1, t1_resp, u2]
    try:
        turn2 = tok.encode(_messages_to_prompt(tok, turn2_msgs))
    except Exception as e:
        print(f"  ⚠ tools-present: template raised {type(e).__name__} "
              f"(tools-on-chat is not currently supported)")
        return True
    return _check_full_prefix(
        cached, turn2,
        "tools-present (KNOWN GAP — re-validate when adding tool support)",
        allow_known_gap=True)


def main() -> int:
    cases = [
        case_empty_thinking_two_turn,
        case_nontrivial_thinking_two_turn,
        case_three_turn_compound,
        case_long_system_prompt,
        case_unicode_content,
        case_response_without_think_markers,
        case_tools_present_warning,
    ]
    print(f"[invariant] {MODEL_ID} — running {len(cases)} cases...")
    failed: list[str] = []
    for c in cases:
        try:
            if not c():
                failed.append(c.__name__)
        except AssertionError as e:
            print(f"  ✗ {c.__name__}: {e}")
            failed.append(c.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ {c.__name__}: {type(e).__name__}: {e}")
            failed.append(c.__name__)
    if failed:
        print(f"\n{len(failed)}/{len(cases)} FAILED: {', '.join(failed)}")
        return 1
    print(f"\nALL {len(cases)} CASES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
