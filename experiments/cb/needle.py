#!/usr/bin/env python3
"""Long-context needle-in-haystack on the CB SERVING path (the real chat path).

The whole point of continuous batching is multi-user chat → long context must
work. This runs the needle test THROUGH cb_scheduler (prefill one-token/step +
greedy decode, exactly what a chat slot does): build a haystack of ~L tokens,
insert "The magic password is XXXXXXXX." at fractional position `frac`, ask for
it, and check the model retrieves it.

  Y = full 8-char code in output, P = ≥4-char contiguous substring, N = no.

`--conv kdim` (default) is the bit-identical-to-production path (expected Y).
`--conv shiftacc` is the fast path (DNK-G4) — this is the gate that decides if
its op-order drift breaks retrieval.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/needle.py --length 200 --frac 0.5 --conv kdim
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
import cb_scheduler as sched                 # noqa: E402

DISTRACTOR = (
    "The history of computing spans many centuries from the abacus to modern "
    "silicon chips. Early mechanical calculators gave way to electromechanical "
    "machines and eventually to fully electronic computers. The transistor "
    "revolutionized the field in the late 1940s enabling smaller faster devices. "
    "Integrated circuits packed thousands then millions of transistors onto a "
    "single chip. Today processors contain billions of transistors and execute "
    "instructions in parallel across many cores. ")
ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"


def build_prompt(tok, target_len, frac, code):
    needle = f" REMEMBER THIS: The magic password is {code}. "
    # grow distractor token count to ~target_len, insert needle at `frac`
    body = ""
    while len(tok.encode(body)) < target_len:
        body += DISTRACTOR
    ids = tok.encode(body)[:target_len]
    body = tok.decode(ids)
    cut = int(len(body) * frac)
    haystack = body[:cut] + needle + body[cut:]
    user = haystack + "\n\nWhat is the magic password? Answer with only the password."
    msgs = [{"role": "user", "content": user}]
    try:  # disable Qwen3.6 thinking so the answer is immediate (else it reasons first)
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        return tok.encode(rendered)
    except Exception:
        pass
    try:
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok.encode(rendered)
    except Exception:
        return tok.encode(user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200, help="haystack token length")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--conv", choices=["kdim", "shiftacc"], default="kdim")
    ap.add_argument("--owned-gdn", action="store_true", default=True)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    if args.owned_gdn:
        state.deltanet_recurrence_mode = "owned_gdn"
    state.cb_dn_recurrence_mode = "owned_gdn" if args.owned_gdn else "manual"
    state.cb_conv_mode = args.conv
    tok = state.tok
    eos = getattr(tok, 'eos_token_id', None)
    eos = int(eos) if eos is not None else -1

    random.seed(0)
    code = "".join(random.choice(ALPHABET) for _ in range(8))
    prompt = build_prompt(tok, args.length, args.frac, code)
    log(f"conv={args.conv}  recurrence={state.cb_dn_recurrence_mode}  "
        f"haystack≈{args.length} tok, prompt={len(prompt)} tok, code={code}, frac={args.frac}")

    # serve it through the CB scheduler (B=1 slot), greedy decode
    s = sched.Scheduler(state, 1, args.max_new, eos)
    s.submit(prompt)
    t0 = time.perf_counter()
    iters = s.run()
    dt = time.perf_counter() - t0
    gen = s.reqs[0]['gen']
    out = tok.decode(gen)
    log(f"generated {len(gen)} tokens in {iters} iters / {dt:.1f}s")
    log(f"OUTPUT: {out!r}")

    verdict = "N"
    if code in out:
        verdict = "Y"
    else:
        for n in range(len(code), 3, -1):
            if any(code[i:i + n] in out for i in range(len(code) - n + 1)):
                verdict = f"P({n})"
                break
    log(f"=== needle verdict ({args.conv}): {verdict}  (Y=full code retrieved) ===")
    if verdict == "N":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
