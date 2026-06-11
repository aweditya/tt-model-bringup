"""Multi-turn Q&A stress test for 35B — runs in the resident cb35 dev
harness. Validates that the model retains context across multiple
chat turns and produces coherent responses without drift collapse.

Methodology:
  - Round 1: ask a factual question with a memorable answer.
  - Round 2: ask a follow-up that requires the round-1 answer in
    context (tests KV retention across turns).
  - Round 3: ask an unrelated question (tests that the model can
    switch topics cleanly without contamination from prior turns).

Each turn appends to a growing chat-template prompt. We greedy-decode
up to 64 tokens per turn (with EOS short-circuit). Output is logged
verbatim so we can visually grade coherence.

This is the corollary to `cb35_needle_haystack.py`: needle haystack
tests long single-prompt retrieval; this tests stateful multi-turn.
For the user-facing chat demo, multi-turn is the more important gate.

Trigger:
    ssh qb1 'touch /tmp/cb35_trig/multiturn_qa'

Result:
    ssh qb1 'cat /tmp/cb35_trig/last.log'
"""
from __future__ import annotations

import time

import server_35b_ttnn as srv

TURNS = [
    "Hello! My name is Aditya. What is the capital of France?",
    "I'm working on bringing up large language models on Tenstorrent "
    "Blackhole hardware. Do you know what a P150 is in that context?",
    "Now please tell me a one-sentence interesting fact about the city "
    "you mentioned in your first answer.",
]
MAX_GEN = 64


def _log(msg: str) -> None:
    print(f"[multiturn_qa] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _apply_chat(tok, messages):
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False)


def _decode_turn(state, srv, tok, prompt_ids, max_gen):
    state.reset_caches_ttnn()
    last_argmax = None
    t_prefill = time.time()
    for p, tid in enumerate(prompt_ids):
        last_argmax = srv.step_forward_ttnn(state, int(tid), p)
    prefill_dt = time.time() - t_prefill

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if im_end is not None and im_end >= 0:
        eos_ids.add(int(im_end))

    generated = [int(last_argmax)]
    pos = len(prompt_ids)
    t_decode = time.time()
    for _ in range(max_gen - 1):
        if int(generated[-1]) in eos_ids:
            break
        next_id = srv.step_forward_ttnn(state, int(generated[-1]), pos)
        generated.append(int(next_id))
        pos += 1
    decode_dt = time.time() - t_decode
    text = tok.decode(generated, skip_special_tokens=False)
    return text, prefill_dt, decode_dt, generated


def main(state):
    tok = state.tokenizer
    _log(f"starting {len(TURNS)}-turn Q&A. tokenizer={tok.__class__.__name__}")
    messages: list[dict] = []
    results: list[dict] = []
    for i, user_msg in enumerate(TURNS):
        messages.append({"role": "user", "content": user_msg})
        rendered = _apply_chat(tok, messages)
        prompt_ids = tok.encode(rendered, add_special_tokens=False)
        if len(prompt_ids) > srv.MAX_KV:
            _log(f"  SKIP turn {i}: prompt_len={len(prompt_ids)} > MAX_KV={srv.MAX_KV}")
            break
        _log(f"--- turn {i} (prompt_tokens={len(prompt_ids)}) ---")
        _log(f"  user: {user_msg!r}")
        text, prefill_dt, decode_dt, gen_ids = _decode_turn(
            state, srv, tok, prompt_ids, MAX_GEN)
        assistant_text = tok.decode(gen_ids, skip_special_tokens=True)
        # strip trailing <|im_end|>/whitespace
        assistant_text = assistant_text.strip()
        _log(f"  assistant ({len(gen_ids)} toks, prefill {prefill_dt:.1f}s, "
             f"decode {decode_dt:.1f}s, {decode_dt*1000/max(1,len(gen_ids)-1):.0f} ms/tok):")
        _log(f"    {assistant_text!r}")
        messages.append({"role": "assistant", "content": assistant_text})
        results.append({
            "turn": i,
            "user": user_msg,
            "assistant": assistant_text,
            "prefill_s": round(prefill_dt, 2),
            "decode_s": round(decode_dt, 2),
            "n_gen_toks": len(gen_ids),
            "n_prompt_toks": len(prompt_ids),
        })

    _log("=" * 78)
    _log("MULTI-TURN SUMMARY (visual-grade for coherence + retention)")
    _log("=" * 78)
    for r in results:
        _log(f"  T{r['turn']}: {r['n_prompt_toks']:5d} prompt → "
             f"{r['n_gen_toks']:3d} tok, "
             f"prefill {r['prefill_s']:.1f}s decode {r['decode_s']:.1f}s")
    _log(f"completed {len(results)}/{len(TURNS)} turns")
    return 0
