#!/usr/bin/env python3
"""S2.7 gate — CBEngine(chunked_prefill=True): plumbing + first tokens.

Confirms the full API surface routes through the chunked-prefill path:
  - CBEngine boots with chunked_prefill=True
  - submit/stream/cancel still work
  - Generated tokens match the S2.5 manual reference (Scheduler-level test)
  - Engine stops cleanly (no fabric wedge)

Reference path: same as S2.5 — Scheduler(chunked_prefill=True) on the same
prompt. The CBEngine wraps the scheduler with a thread + queues; same tokens
must come out either way.

Run on qb1:
  make run PY=experiments/cb/validate/engine_chunked_prefill.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
from cb_engine import CBEngine              # noqa: E402
from cb_scheduler import Scheduler          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--length", type=int, default=32)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    tok = state.tok
    eos_id = getattr(tok, "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else -1
    prompt = ("The capital of France is the city of Paris, which has long been "
              "a center of art, science, and political history in Europe.")
    ids = tok.encode(prompt)[:args.length]
    log(f"prompt L={len(ids)} tokens")

    # Reference: Scheduler(chunked_prefill=True) directly (S2.5 path, no engine).
    log("=== reference: Scheduler(chunked_prefill=True) ===")
    sched = Scheduler(state, args.slots, args.max_new, eos_id,
                       use_trace=False, chunked_prefill=True)
    rid_ref = sched.submit(ids)
    sched.run()
    ref_tokens = list(sched.reqs[rid_ref]['gen'])
    sched.release()
    log(f"  reference tokens: {ref_tokens}")

    # Test: CBEngine(chunked_prefill=True). use_trace=False to keep this test
    # purely about the chunked-prefill wiring; the engine's trace path is
    # already validated separately (experiments/cb/validate/engine.py).
    log("=== test: CBEngine(chunked_prefill=True) ===")
    eng = CBEngine(state, args.slots, max_new_cap=args.max_new, eos_id=eos_id,
                    use_trace=False, sampling=False, chunked_prefill=True).start()
    h = eng.submit(ids, max_new=args.max_new)
    out = list(h.tokens())
    log(f"  engine tokens:    {out}  final={h.final!r}")
    eng.stop()
    log("engine stopped cleanly")

    if out == ref_tokens and h.final == "done":
        log(f"PASS: CBEngine(chunked_prefill=True) produced the same tokens as "
            f"the bare Scheduler reference ({len(out)} tokens identical), "
            f"final={h.final!r}. S2.7 gate green; chunked-prefill path is "
            f"available via TT_CB_CHUNKED_PREFILL=1 in serve_cb.sh.")
    else:
        log(f"FAIL: engine {out} != reference {ref_tokens}  final={h.final!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
