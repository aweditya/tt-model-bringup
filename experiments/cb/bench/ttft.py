#!/usr/bin/env python3
"""S2.6 — TTFT measurement: chunked prefill (via transplant) vs 1-tok/iter.

The S2 payoff is supposed to be: TTFT drops from O(L × decode_step) to
O(S1a_prefill + transplant). Whether that's actually a win depends on the
transplant cost, which is currently a host round-trip. This bench measures it
honestly at several L so we know where the crossover is and where the
optimization headroom sits.

Per L: time three components separately —
  - S1a chunked prefill (forward_prefill_chunked_tp on production state)
  - cb_prefill_transplant (host round-trip per layer × 64 layers)
  - one CB decode step on the transplanted slot
Sum = chunked-path TTFT.

Reference: time one CB step at B=1 with the existing decode forward, then
project L × step + 1 step = tok-by-tok TTFT. (Cheap proxy; the real loop is
identical-cost iteration-over-iteration.)

Gate target (from archive/superseded_research_2026-06-04/27b_s2_chunked_prefill_milestones.md S2.6): at
L=200, chunked path << 200 × decode_step. Report any L where chunked WINS.

Run on qb1:
  make run PY=experiments/cb/bench/ttft.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
import server_tp as base                     # noqa: E402
import server_tp_cb as cb                    # noqa: E402


def _ms(t):
    return f"{t * 1000.0:8.1f} ms"


def main():
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="64,200,500,1000",
                    help="prompt token counts to sweep")
    ap.add_argument("--slots", type=int, default=4,
                    help="CB slot pool (TTFT itself runs in slot 0)")
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    tok = state.tok

    text = ("The history of computing spans many centuries from the abacus to "
            "modern silicon chips. Early mechanical calculators gave way to "
            "electromechanical machines and eventually to fully electronic "
            "computers. The transistor revolutionized the field in the late "
            "1940s enabling smaller faster devices. Integrated circuits packed "
            "thousands then millions of transistors onto a single chip. Today "
            "processors contain billions of transistors and execute instructions "
            "in parallel across many cores. ") * 32
    all_ids = tok.encode(text)
    if max(lens) > len(all_ids):
        raise SystemExit(f"prompt seed too short ({len(all_ids)} tok); need "
                         f"{max(lens)}")

    log("CB setup (B=4, kdim conv, manual recurrence)…")
    cb.setup_cb_state(state, args.slots)
    state.cb_conv_mode = 'kdim'
    state.cb_dn_recurrence_mode = 'manual'

    # Reference decode-step latency: one CB B=1-active step (matches what
    # 1-tok/iter prefill would pay per token in the existing scheduler).
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [int(all_ids[0])] + [0]*(args.slots-1),
                                          [0] + [-1]*(args.slots-1))
    # JIT warmup
    for _ in range(2):
        am = cb.forward_batch_tp_inner(state); ttnn.deallocate(am)
    ttnn.synchronize_device(state.mesh)
    # Measure
    N = 20
    t0 = time.perf_counter()
    for i in range(N):
        cb.update_input_buffers_batched(state, [int(all_ids[i % len(all_ids)])] + [0]*(args.slots-1),
                                              [i] + [-1]*(args.slots-1))
        am = cb.forward_batch_tp_inner(state); ttnn.deallocate(am)
    ttnn.synchronize_device(state.mesh)
    step_ms = (time.perf_counter() - t0) / N
    log(f"baseline CB step (eager, B={args.slots}, 1 active slot): {_ms(step_ms)}")

    log(f"=== TTFT sweep over L={lens} ===")
    rows = []
    for L in lens:
        ids = all_ids[:L]
        # 1. S1a chunked prefill
        base._reset_state_buffers(state)
        cb.cb_reset_states(state)
        ttnn.synchronize_device(state.mesh)
        t0 = time.perf_counter()
        cap = base.forward_prefill_chunked_tp(state, ids, capture_logits=True)
        ttnn.synchronize_device(state.mesh)
        t_prefill = time.perf_counter() - t0
        first_tok = int(np.argmax(cap[-1]))

        # 2. Transplant
        t0 = time.perf_counter()
        cb.cb_prefill_transplant(state, 0, L)
        ttnn.synchronize_device(state.mesh)
        t_transplant = time.perf_counter() - t0

        # 3. One CB decode step on the transplanted slot
        toks = [first_tok] + [0] * (args.slots - 1)
        curs = [L] + [-1] * (args.slots - 1)
        cb.update_input_buffers_batched(state, toks, curs)
        t0 = time.perf_counter()
        am = cb.forward_batch_tp_inner(state)
        ttnn.synchronize_device(state.mesh)
        t_decode = time.perf_counter() - t0
        ttnn.deallocate(am)

        ttft_chunked = t_prefill + t_transplant + t_decode
        ttft_tokbytok = (L + 1) * step_ms
        speedup = ttft_tokbytok / ttft_chunked if ttft_chunked else 0.0
        rows.append((L, t_prefill, t_transplant, t_decode, ttft_chunked, ttft_tokbytok, speedup))
        log(f"  L={L:5d}: prefill {_ms(t_prefill)}  transplant {_ms(t_transplant)}  "
            f"decode {_ms(t_decode)}  =>  chunked TTFT {_ms(ttft_chunked)}  "
            f"vs tok-by-tok {_ms(ttft_tokbytok)}  ({speedup:.2f}× speedup)")

    log("=== summary (chunked TTFT vs 1-tok/iter projection) ===")
    log(f"  {'L':>6s}  {'S1a':>10s}  {'transplant':>12s}  {'decode':>10s}  "
        f"{'chunked':>10s}  {'tok-by-tok':>12s}  speedup")
    for L, tp, tt, td, ttft, ref, sp in rows:
        log(f"  {L:>6d}  {_ms(tp):>10s}  {_ms(tt):>12s}  {_ms(td):>10s}  "
            f"{_ms(ttft):>10s}  {_ms(ref):>12s}  {sp:>5.2f}x")
    crossover = next((L for L, _, _, _, c, r, _ in rows if c < r), None)
    if crossover is not None:
        log(f"chunked path WINS starting at L={crossover}")
    else:
        log("chunked path does NOT win at any tested L — transplant overhead "
            "dominates. Next-step optimization: on-device state copy (kill "
            "host round-trip).")


if __name__ == "__main__":
    main()
