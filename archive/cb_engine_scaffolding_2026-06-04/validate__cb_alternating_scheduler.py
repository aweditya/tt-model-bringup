#!/usr/bin/env python3
"""S2.5 gate — Scheduler(chunked_prefill=True) matches manual S1a+prod-decode.

Builds on S2.4 (prefill_transplant validator). The S2.4 reference — S1a chunked
prefill on production state + N decode steps via forward_token_tp_inner — is the
authoritative answer for "what tokens come out of S1a + greedy decode". This
test runs the same prompt through the new chunked-prefill scheduler and
checks the scheduler produces those same tokens.

Why this gate: S2.4 proves a SINGLE manual transplant+CB-decode == prod decode.
S2.5 proves the SCHEDULER's automated PREFILL_ONLY step does the same thing.

Run on qb1:
  make run PY=experiments/cb/validate/cb_alternating_scheduler.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
import server_tp as base                     # noqa: E402
from cb_scheduler import Scheduler           # noqa: E402


def _chip0_logits(state, rm):
    import ttnn
    t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return t.float().numpy()[0][:state.vocab_size]


def _reference(state, prompt_ids, n_decode):
    """S2.4 reference: S1a prefill + N prod decode (NO transplant).
    Same code path as experiments/cb/validate/prefill_transplant.py:_reference_decode.
    """
    import ttnn
    base._reset_state_buffers(state)
    cap = base.forward_prefill_chunked_tp(state, prompt_ids, capture_logits=True)
    first_tok = int(np.argmax(cap[-1]))
    gen = [first_tok]
    pos = len(prompt_ids)
    tid = first_tok
    for _ in range(n_decode):
        base.update_input_buffers(state, tid, pos)
        rm = base.forward_token_tp_inner(state, return_logits=True)
        tid = int(np.argmax(_chip0_logits(state, rm)))
        ttnn.deallocate(rm)
        gen.append(tid)
        pos += 1
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=64, help="prompt token count")
    ap.add_argument("--decode", type=int, default=3, help="decode steps after prefill")
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    tok = state.tok

    prompt_text = ("The capital of France is the city of Paris, which has long been a "
                   "center of art, science, philosophy, and political history in Europe, "
                   "drawing scholars and travelers from every corner of the wider world "
                   "for many centuries of recorded human civilization and culture, "
                   "blending tradition and reinvention across countless generations.")
    ids = tok.encode(prompt_text)[:args.length]
    L = len(ids)
    log(f"prompt L={L} tokens; decoding {args.decode} after prefill")

    log("=== reference (S2.4 path: S1a + prod decode) ===")
    ref_tokens = _reference(state, ids, args.decode)
    log(f"  reference tokens: {ref_tokens}  text: {tok.decode(ref_tokens)!r}")

    log("=== scheduler (chunked_prefill=True) — alternating PREFILL/DECODE ===")
    max_new = args.decode + 1     # +1 for the prefill's first generated token
    eos = getattr(tok, 'eos_token_id', None)
    eos = int(eos) if eos is not None else -1
    sched = Scheduler(state, args.slots, max_new, eos,
                       use_trace=False, chunked_prefill=True)
    rid = sched.submit(ids)
    iters = sched.run()
    gen = sched.reqs[rid]['gen']
    log(f"  completed in {iters} scheduler iterations")
    log(f"  scheduler tokens: {gen}  text: {tok.decode(gen)!r}")
    sched.release()

    if gen == ref_tokens:
        log(f"PASS: scheduler chunked-prefill output == S2.4 reference "
            f"({len(gen)} tokens identical). S2.5 gate green; S2.6 (TTFT + needle) unblocked.")
    else:
        first_diff = next((i for i, (a, b) in enumerate(zip(ref_tokens, gen))
                           if a != b), min(len(ref_tokens), len(gen)))
        log(f"FAIL: tokens diverge at pos {first_diff}  "
            f"ref={ref_tokens[first_diff] if first_diff < len(ref_tokens) else None!r}  "
            f"sched={gen[first_diff] if first_diff < len(gen) else None!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
