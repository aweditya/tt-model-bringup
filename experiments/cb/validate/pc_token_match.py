#!/usr/bin/env python3
"""Prefix-cache token-equality validator.

Mimics the multi-turn chat lifecycle described in
`research/tokenizer_chat_template_reference.md` §1:

  1. Build turn-1 prompt via `apply_chat_template(tokenize=True, ...)`.
  2. Simulate generated tokens (random or fixed).
  3. Canonicalise them the same way `cb_scheduler._finish` does:
     decode(body, skip_special_tokens=True).rstrip() → encode(add_special_tokens=False) + [EOS].
  4. Build turn-2 prompt by appending decoded assistant text + a new user message.
  5. Assert `turn_2_prompt[:len(prompt_1 + canonical)] == prompt_1 + canonical`.

If the assertion fails, prints the first divergent index and the
nearby token slices so you can see which side has the extra/missing
token. Mirrors the exact code path in `cb_scheduler.py:_finish` and
`live_slot_store.find_longest_match`.

Usage:
    python3 -m pip install transformers
    python3 experiments/cb/validate/pc_token_match.py \\
        --model google/gemma-4-12B-it
    python3 experiments/cb/validate/pc_token_match.py \\
        --model Qwen/Qwen3.6-27B

Exits 0 on pass, 1 on fail. Runs offline if the tokenizer is in HF
cache (no model weights needed; tokeniser-only checkout is enough).
"""
from __future__ import annotations

import argparse
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    print("FATAL: `transformers` not installed. `pip install transformers`.",
          file=sys.stderr)
    sys.exit(2)


# Match the kwargs in experiments/serve/openai_endpoint.py:_messages_to_prompt
_TEMPLATE_KW = dict(
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    preserve_thinking=True,
)


def _ids(raw):
    """Normalise apply_chat_template return — dict-like (Gemma BatchEncoding)
    or bare list (Qwen). Mirrors openai_endpoint._messages_to_prompt."""
    if isinstance(raw, dict) or hasattr(raw, "input_ids"):
        return list(raw["input_ids"])
    return list(raw)


def render(tok, messages):
    """messages → list[int] via apply_chat_template, normalised."""
    return _ids(tok.apply_chat_template(messages, **_TEMPLATE_KW))


def canonicalise_gen(tok, gen_ids, eos_id):
    """Replicates cb_scheduler._finish: decode(body) → rstrip() → encode + EOS.

    Args:
      gen_ids: model output, including the trailing EOS if it triggered done.
      eos_id: the EOS token id to re-append. Required because we strip it
        before decode to avoid leaving "</s>" or similar in the decoded text.
    """
    if gen_ids and gen_ids[-1] == eos_id:
        body = gen_ids[:-1]
    else:
        body = gen_ids
    text = tok.decode(body, skip_special_tokens=True).rstrip()
    out = list(tok.encode(text, add_special_tokens=False))
    out.append(int(eos_id))
    return out


def _first_diverge(a, b):
    """Index of first mismatch between two sequences, or -1 if a is a prefix
    of b (or b of a) up to the shorter length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1


def _fmt_slice(tok, ids, around, k=6):
    """Pretty-print ids[around-k:around+k] with decoded text for each id."""
    lo = max(0, around - k)
    hi = min(len(ids), around + k + 1)
    out = []
    for i in range(lo, hi):
        if i >= len(ids):
            break
        try:
            t = tok.decode([ids[i]], skip_special_tokens=False)
        except Exception:
            t = "?"
        marker = " <" if i == around else "  "
        out.append(f"  [{i:4d}] id={ids[i]:>7d} {t!r}{marker}")
    return "\n".join(out)


def _resolve_eos(tok):
    """Pick a single EOS id consistent with cb_scheduler's eos_id set.
    Prefers the chat-end token (`<|im_end|>` for Qwen, `<end_of_turn>` for
    Gemma) over the corpus-end token (`<eos>`). Falls back to tok.eos_token_id.
    """
    # Try named special tokens
    for name in ("<|im_end|>", "<end_of_turn>"):
        try:
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid != tok.unk_token_id and tid >= 0:
                return int(tid)
        except Exception:
            pass
    return int(tok.eos_token_id) if tok.eos_token_id is not None else None


def run_case(tok, user_1, fake_assistant_text, user_2):
    """One round-trip case. Returns (ok, diagnostics)."""
    eos_id = _resolve_eos(tok)
    if eos_id is None:
        return False, "tokenizer has no EOS token; cannot run PC test"

    # Simulate gen by tokenising the fake assistant text + EOS.
    fake_gen = list(tok.encode(fake_assistant_text, add_special_tokens=False))
    fake_gen.append(eos_id)

    # Turn 1: messages = [user_1]; prompt_1 = render([user_1]).
    msgs_1 = [{"role": "user", "content": user_1}]
    prompt_1 = render(tok, msgs_1)

    # _finish canonicalisation:
    canonical = canonicalise_gen(tok, fake_gen, eos_id)
    tokens_so_far = prompt_1 + canonical

    # Turn 2: messages = [user_1, asst_decoded, user_2]; prompt_2 = render.
    asst_text = tok.decode(fake_gen, skip_special_tokens=True)
    msgs_2 = [
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": asst_text},
        {"role": "user", "content": user_2},
    ]
    prompt_2 = render(tok, msgs_2)

    n = len(tokens_so_far)
    if len(prompt_2) < n:
        return False, (f"prompt_2 is shorter ({len(prompt_2)}) than "
                       f"tokens_so_far ({n}); cannot be a prefix")
    prefix = prompt_2[:n]

    if prefix == tokens_so_far:
        return True, (f"OK: prompt_1 ({len(prompt_1)}) + canonical "
                      f"({len(canonical)}) = {n} tokens is a byte-exact "
                      f"prefix of prompt_2 ({len(prompt_2)} tokens)")
    div = _first_diverge(tokens_so_far, prefix)
    diag = [f"MISMATCH at index {div} / {n}",
            f"  prompt_1 length: {len(prompt_1)}",
            f"  canonical gen length: {len(canonical)} (incl EOS)",
            f"  prompt_2 length: {len(prompt_2)}",
            "  tokens_so_far[around]:",
            _fmt_slice(tok, tokens_so_far, div),
            "  prompt_2[around]:",
            _fmt_slice(tok, prompt_2, div)]
    return False, "\n".join(diag)


CASES = [
    ("Hello! My name is Aditya. What is the capital of France?",
     "The capital of France is Paris.\n",  # trailing newline triggers Gemma's | trim
     "Tell me a one-sentence fact about that city."),
    ("Write a short hello world program in python.",
     "```python\nprint('hello world')\n```",
     "Now make it print my name instead."),
    ("Explain GQA briefly.",
     "Grouped-Query Attention shares K/V across multiple Q heads.",
     "How is that different from MQA?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. google/gemma-4-12B-it or Qwen/Qwen3.6-27B")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"  class={tok.__class__.__name__}  eos_token={tok.eos_token!r}  "
          f"chat_template_installed={tok.chat_template is not None}")

    n_pass = 0
    n_total = 0
    for i, (u1, fake_a, u2) in enumerate(CASES):
        n_total += 1
        u1s, fas, u2s = repr(u1)[:40], repr(fake_a)[:30], repr(u2)[:30]
        print(f"\n--- Case {i}: u1={u1s}...  fake_a={fas}...  u2={u2s}...")
        ok, diag = run_case(tok, u1, fake_a, u2)
        prefix = "  PASS" if ok else "  FAIL"
        print(f"{prefix}: {diag}" if args.verbose or not ok
              else f"{prefix} ({diag.splitlines()[0] if diag else ''})")
        if ok:
            n_pass += 1

    print(f"\n=== {n_pass}/{n_total} cases PASS ===")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
