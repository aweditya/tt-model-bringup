#!/usr/bin/env python3
"""Reproduce the chunked-prefill wedge at the engine API layer (no HTTP).

Two threads each call eng.submit(...) + handle.tokens(), iterating turns. With
chunked_prefill=True the production server wedges under concurrent traffic; the
legacy 1-tok/iter prefill path does not. This test isolates the bug to the
engine, eliminating FastAPI/asyncio as a variable.

Pass: both threads complete N turns each, no timeouts.
Fail: a thread stalls (no token for >stall_seconds), test dumps engine state and
exits non-zero.

Run on qb1:
  make run PY=experiments/cb/validate/chunked_concurrent.py
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
from cb_engine import CBEngine              # noqa: E402

PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "List three primary colors.",
    "What is 7 times 8?",
    "Name two planets in our solar system.",
    "What is the speed of light, approximately?",
    "Who wrote Romeo and Juliet?",
]


def _run_client(idx, eng, tok, n_turns, max_new, stall_s, results, errors):
    try:
        for turn in range(n_turns):
            text = PROMPTS[(idx * n_turns + turn) % len(PROMPTS)]
            prompt_ids = tok.encode(text)
            t0 = time.time()
            handle = eng.submit(prompt_ids, max_new=max_new)
            n_tok = 0
            for _tid in handle.tokens(timeout=stall_s):
                n_tok += 1
            elapsed = time.time() - t0
            results.append({"client": idx, "turn": turn, "tok": n_tok,
                             "elapsed": elapsed, "final": handle.final})
            log(f"  client {idx} turn {turn}: {n_tok} tok in {elapsed:.2f}s "
                f"({handle.final}) — {text!r}")
    except queue.Empty:
        # stall — token didn't arrive within stall_s
        errors.append({"client": idx, "kind": "stall",
                        "stall_after_seconds": stall_s,
                        "tokens_received": n_tok if 'n_tok' in dir() else 0})
        log(f"  client {idx}: STALL — no token for {stall_s}s "
            f"(this is the wedge signature)")
    except Exception as e:
        errors.append({"client": idx, "kind": "exception",
                        "exception": f"{type(e).__name__}: {e}"})
        log(f"  client {idx}: EXCEPTION {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=2,
                    help="number of concurrent client threads")
    ap.add_argument("--turns", type=int, default=2,
                    help="sequential turns per client")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--stall-seconds", type=float, default=60.0,
                    help="if a client gets no token for this long, declare wedge")
    ap.add_argument("--legacy", action="store_true",
                    help="run with chunked_prefill=False (control: should pass)")
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    tok = state.tok
    eos_id = getattr(tok, "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else -1

    mode = "LEGACY 1-tok/iter" if args.legacy else "CHUNKED (S2 path)"
    log(f"=== {mode} prefill, {args.clients} clients × {args.turns} turns, "
        f"max_new={args.max_new}, stall={args.stall_seconds}s ===")

    eng = CBEngine(state, slots=4, max_new_cap=args.max_new, eos_id=eos_id,
                    use_trace=True, sampling=False,
                    chunked_prefill=not args.legacy).start()

    results: list = []
    errors: list = []
    threads = [threading.Thread(
        target=_run_client,
        args=(i, eng, tok, args.turns, args.max_new, args.stall_seconds,
              results, errors),
        daemon=True,
    ) for i in range(args.clients)]
    t0 = time.time()
    for t in threads:
        t.start()
    # Stagger slightly so admissions don't race the engine warmup
    deadline = time.time() + args.clients * args.turns * args.stall_seconds + 60
    while any(t.is_alive() for t in threads):
        if time.time() > deadline:
            log("FAIL: clients did not finish before global deadline (test wedge)")
            break
        time.sleep(0.5)
    for t in threads:
        t.join(timeout=1.0)
    elapsed = time.time() - t0
    eng.stop(timeout=10)

    n_expected = args.clients * args.turns
    n_done = sum(1 for r in results if r["final"] == "done")
    log(f"=== {n_done}/{n_expected} turns completed in {elapsed:.1f}s "
        f"({len(errors)} errors) ===")
    for e in errors:
        log(f"  error: {e}")

    if errors or n_done < n_expected:
        log(f"FAIL: wedge reproduced — {mode}")
        raise SystemExit(1)
    log(f"PASS: all turns completed cleanly under concurrent load ({mode})")


if __name__ == "__main__":
    main()
