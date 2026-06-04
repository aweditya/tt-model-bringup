"""Long-context needle-haystack gate for 35B — runs in the resident
cb35 dev harness so we don't pay the 14-min bootstrap.

Forks `experiments/needle_haystack_35b_ttnn_inproc.py` (the bootstraps-
its-own-state version that gated A002+A003+A004+A008 perf wins). Same
prompt construction, same classify(); only difference is the
`main(state)` signature that reuses the resident bootstrap.

This is the user-facing gate for the 2026-06-04 drift-resolution
finding ([[35b-drift-resolved-2026-06-04]]). Teacher-forced cosine
ladder went clean; this confirms free-run/long-context retrieval is
also clean.

Env (all optional):
    CB35_NEEDLE_LENGTHS    default "100,460,1024"  comma-sep L values
    CB35_NEEDLE_FRAC       default 0.5             needle position fraction
    CB35_NEEDLE_TRIALS     default 1               seeds per (L, frac)
    CB35_NEEDLE_MAX_GEN    default 24              tokens to decode

Trigger:
    ssh qb1 'touch /tmp/cb35_trig/needle_haystack'

Result: `tmux capture-pane -t cb35 -p -S -200 | grep needle` or
`/tmp/cb35_trig/last.log`.

Gate: at L=100, MUST see Y (memory says pre-A008 baseline did).
At L=460+, Y or P is acceptable; N at L=460 = real problem.
"""
from __future__ import annotations

import os
import random as _r
import time

import ttnn  # noqa: F401  (kept for harness consistency)

import server_35b_ttnn as srv

# Reuse the same haystack distractor for apples-to-apples comparison
# against `experiments/needle_haystack_35b_ttnn_inproc.py`.
DISTRACTOR = (
    "The history of computing spans many centuries from the abacus to "
    "modern silicon chips. Early mechanical calculators like the "
    "Pascaline gave way to electromechanical machines and eventually "
    "vacuum-tube computers in the mid 20th century. The invention of "
    "the transistor in 1947 and the integrated circuit shortly after "
    "marked a paradigm shift that enabled rapid miniaturization. "
    "Moore's law described the doubling of transistors on a chip every "
    "two years, a trend that held for decades and drove the personal "
    "computer revolution of the 1980s. Networking and the internet "
    "transformed computing from isolated machines into a global system "
    "of communication, while mobile devices and cloud platforms gave "
    "rise to today's always-connected era. "
)
ALPHABET = "ABCDEFGHIJKLMNPQRSTUVWXYZ23456789"  # no ambiguous 0/O, 1/I/L


def _log(msg: str) -> None:
    print(f"[needle_haystack] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _make_needle(seed: int) -> str:
    rng = _r.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(8))


def _apply_chat(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)


def _build_prompt(tok, target_tokens: int, frac: float, needle: str):
    needle_sentence = (
        f"REMEMBER THIS: The magic password is {needle}. END REMEMBER.")
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password."
                "\n\nAnswer: ")

    def count(user):
        return len(tok.encode(_apply_chat(tok, user), add_special_tokens=False))

    n_para = max(1, target_tokens // 220 + 2)
    distractor = DISTRACTOR * n_para
    for _ in range(40):
        body = distractor + " " + needle_sentence + " " + question
        n = count(body)
        if n < target_tokens - 5:
            distractor = distractor + DISTRACTOR
        elif n > target_tokens + 5:
            cut = max(40, int(len(distractor) * 0.04))
            distractor = distractor[:-cut]
            if len(distractor) < 80:
                break
        else:
            break

    d_ids = tok.encode(distractor, add_special_tokens=False)
    idx = int(len(d_ids) * frac)
    prefix = tok.decode(d_ids[:idx], skip_special_tokens=True)
    suffix = tok.decode(d_ids[idx:], skip_special_tokens=True)
    user_body = prefix + " " + needle_sentence + " " + suffix + question
    rendered = _apply_chat(tok, user_body)
    actual = len(tok.encode(rendered, add_special_tokens=False))
    return rendered, actual


def _classify(output_text: str, needle: str) -> str:
    if needle in output_text:
        return "Y"
    for n in range(8, 3, -1):
        for i in range(0, 9 - n):
            if needle[i:i + n] in output_text:
                return "P"
    return "N"


def main(state):
    lengths = [int(x) for x in os.environ.get(
        "CB35_NEEDLE_LENGTHS", "100,460,1024").split(",")]
    frac = float(os.environ.get("CB35_NEEDLE_FRAC", "0.5"))
    trials = int(os.environ.get("CB35_NEEDLE_TRIALS", "1"))
    max_gen = int(os.environ.get("CB35_NEEDLE_MAX_GEN", "24"))
    tok = state.tokenizer

    _log(f"config: lengths={lengths} frac={frac} trials={trials} "
         f"max_gen={max_gen}")

    results = []
    for L in lengths:
        for trial in range(trials):
            seed = L * 1000 + int(frac * 100) * 10 + trial
            needle = _make_needle(seed)
            prompt, actual_L = _build_prompt(tok, L, frac, needle)
            if actual_L > srv.MAX_KV:
                _log(f"  SKIP L={L} actual_L={actual_L} > MAX_KV={srv.MAX_KV}")
                continue
            _log(f"L={L} trial={trial} needle={needle!r} actual_L={actual_L}")

            state.reset_caches_ttnn()
            prompt_ids = tok.encode(prompt, add_special_tokens=False)

            t0 = time.time()
            last_argmax = None
            for p, tid in enumerate(prompt_ids):
                last_argmax = srv.step_forward_ttnn(state, int(tid), p)
            prefill_dt = time.time() - t0

            generated = [int(last_argmax)]
            pos = len(prompt_ids)
            t0 = time.time()
            for _ in range(max_gen - 1):
                next_id = srv.step_forward_ttnn(state, int(generated[-1]), pos)
                generated.append(int(next_id))
                pos += 1
            decode_dt = time.time() - t0
            out_text = tok.decode(generated)
            grade = _classify(out_text, needle)
            _log(f"  prefill {prefill_dt:.1f}s  decode {decode_dt:.1f}s  "
                 f"({decode_dt*1000/(max_gen-1):.0f} ms/tok)")
            _log(f"  output: {out_text!r}")
            _log(f"  grade:  {grade}")
            results.append({
                "L": L, "trial": trial, "needle": needle,
                "actual_L": actual_L, "output": out_text, "grade": grade,
            })

    _log("=" * 78)
    _log("SUMMARY")
    by_L: dict[int, list[str]] = {}
    for r in results:
        by_L.setdefault(r["L"], []).append(r["grade"])
    any_fail = False
    for L, grades in by_L.items():
        y = grades.count("Y"); p = grades.count("P"); n = grades.count("N")
        _log(f"  L={L:5d}: Y={y}  P={p}  N={n}  ({len(grades)} trials)")
        if L == 100 and y < 1:
            any_fail = True
    verdict = "FAIL" if any_fail else "PASS"
    _log(f"VERDICT: {verdict}  (gate: L=100 must produce at least one Y)")
    return 0 if not any_fail else 1
