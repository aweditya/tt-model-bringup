#!/usr/bin/env python3
"""S1a chunked-prefill correctness + TTFT gate.

Compares `forward_prefill_chunked_tp` (whole-prompt parallel: chunked-DN Neumann +
one causal SDPA + batched MLP) against `forward_prefill_tp_inner` (the proven
single-token stub), per-position logit cosine, each on a freshly-reset state.
Then times both for a TTFT speedup number. Default --max-pos 32 exercises the
chunked-DN Neumann path (dispatched for seq_len in {4,8,16,32}); >32 falls back to
per-position DN.

Run on qb1 (from repo root):
  make run PY=experiments/cb/validate/prefill.py
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


def _cos(a, b):
    a = a.astype(np.float64).reshape(-1); b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="8,16,29,32",
                    help="comma list of prompt token counts to sweep; powers of 2 hit "
                         "the chunked-DN Neumann path, others the per-position fallback")
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]

    log("bootstrap production 27B server (server_tp)…")
    state, base = bootstrap_27b_cb()
    tok = state.tok
    # qb1 baseline; the chunked path uses Neumann. Mode defaulted to "manual" by bootstrap_27b_cb.

    prompt = ("The capital of France is the city of Paris, which has long been a "
              "center of art, science, philosophy, and political history in Europe, "
              "drawing scholars and travelers from every corner of the wider world for "
              "many centuries of recorded human civilization and culture.")
    all_ids = tok.encode(prompt)
    log(f"prompt has {len(all_ids)} tokens; sweeping lens={lens}")

    rows = []
    for L in lens:
        if L > len(all_ids):
            log(f"  L={L}: skip (prompt too short)")
            continue
        ids = all_ids[:L]
        base._reset_state_buffers(state)
        ref = base.forward_prefill_tp_inner(state, ids, capture_logits=True)
        base._reset_state_buffers(state)
        chk = base.forward_prefill_chunked_tp(state, ids, capture_logits=True)
        ref_ids = ref.argmax(axis=-1).tolist(); chk_ids = chk.argmax(axis=-1).tolist()
        coss = [_cos(ref[p], chk[p]) for p in range(L)]
        worst = min(coss); wpos = int(np.argmin(coss))
        nmatch = sum(int(a == b) for a, b in zip(ref_ids, chk_ids))
        path = "Neumann" if L in (4, 8, 16, 32) else "per-pos fallback"
        last = "=" if ref_ids[-1] == chk_ids[-1] else "DIFF"
        log(f"  L={L:3d} [{path:16s}]: worst_cos={worst:.4f} @pos{wpos}  "
            f"argmax {nmatch}/{L}  last-pos {last}")
        rows.append((L, path, worst, nmatch))

    log("=== summary (gate: worst_cos >= 0.99) ===")
    for L, path, worst, nmatch in rows:
        log(f"  L={L:3d} [{path:16s}]: {'PASS' if worst >= 0.99 else 'FAIL'}  "
            f"worst_cos={worst:.4f}  argmax {nmatch}/{L}")


if __name__ == "__main__":
    main()
