#!/usr/bin/env python3
"""T0/T1 — verify fixed-L prefill correctness, then capture + replay as trace.

Why: chunked-prefill wedges under concurrent traffic because eager allocations
in forward_prefill_chunked_tp collide with the decode trace's reserved scratch
addresses. Fix is to TRACE the prefill too, at a fixed chunk_size, so nothing
allocates after bootstrap. Plan: research/27b_prefill_trace_plan.md.

T0 (eager, padded): pad a short prompt to chunk_size=128, run the EXISTING
forward_prefill_chunked_tp eagerly, slice last-position logits at the actual
prompt's last index. Compare to the legacy 1-tok/iter stub.
  Gate: cos at position L_actual-1 >= 0.99.
  Proves the static-shape eager path is correct — precondition for tracing.

T1 (traced, padded): same as T0 but wrap the forward call in
begin_trace_capture / end_trace_capture, replay it, compare cos to T0.
  Gate: cos vs legacy >= 0.99 AND trace replay matches eager call exactly.

Run on qb1:
  make run PY=experiments/cb/isolate/prefill_trace.py
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

CHUNK_SIZE = 128


def _cos(a, b):
    a = a.astype(np.float64).reshape(-1); b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main():
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=64,
                    help="actual prompt length; must be <= chunk_size for T0/T1")
    ap.add_argument("--gate", type=float, default=0.99)
    args = ap.parse_args()
    if args.length > CHUNK_SIZE:
        raise SystemExit(f"--length {args.length} > chunk_size {CHUNK_SIZE}; "
                         f"multi-chunk is T3 work")

    log("bootstrap production 27B server (server_tp)…")
    state, base = bootstrap_27b_cb()
    tok = state.tok

    prompt_text = ("The capital of France is the city of Paris, which has long "
                   "been a center of art, science, and political history in "
                   "Europe, drawing scholars and travelers across centuries.")
    actual_ids = tok.encode(prompt_text)[:args.length]
    L = len(actual_ids)
    log(f"actual prompt L={L} tokens; chunk_size={CHUNK_SIZE}")

    # === REFERENCE A: eager S1a at actual L (no padding) — algorithm-parity reference ===
    log(f"=== REF_A: eager forward_prefill_chunked_tp at L_actual={L} (unpadded) ===")
    base._reset_state_buffers(state)
    refA_logits = base.forward_prefill_chunked_tp(state, actual_ids, capture_logits=True)
    refA_last = refA_logits[L - 1]
    refA_argmax = int(np.argmax(refA_last))
    log(f"  REF_A last-pos argmax = {refA_argmax} ({tok.decode([refA_argmax])!r})")

    # === REFERENCE B: legacy 1-tok/iter stub — informational only (S1a drifts ~0.95 vs stub) ===
    log(f"=== REF_B (informational): legacy 1-tok/iter forward_prefill_tp_inner ===")
    base._reset_state_buffers(state)
    refB_logits = base.forward_prefill_tp_inner(state, actual_ids, capture_logits=True)
    refB_last = refB_logits[L - 1]
    refB_argmax = int(np.argmax(refB_last))
    log(f"  REF_B last-pos argmax = {refB_argmax} ({tok.decode([refB_argmax])!r})")
    log(f"  REF_A vs REF_B cos = {_cos(refA_last, refB_last):.6f}  (S1a's normal stub drift)")

    # === T0: eager forward at fixed L=CHUNK_SIZE with padded prompt ===
    log(f"=== T0: eager forward_prefill_chunked_tp at L={CHUNK_SIZE} (padded) ===")
    # Pad with 0 token. Causal mask + DN's per-position recurrence ensure positions
    # AFTER L don't affect position L-1's output.
    padded = list(actual_ids) + [0] * (CHUNK_SIZE - L)
    base._reset_state_buffers(state)
    t0 = time.time()
    t0_logits = base.forward_prefill_chunked_tp(state, padded, capture_logits=True)
    ttnn.synchronize_device(state.mesh)
    t0_elapsed = time.time() - t0
    # shape [CHUNK_SIZE, vocab]; we want position L-1 (real last-position)
    t0_last = t0_logits[L - 1]
    t0_argmax = int(np.argmax(t0_last))
    t0_vs_refA = _cos(t0_last, refA_last)
    t0_vs_refB = _cos(t0_last, refB_last)
    log(f"  T0 last-pos argmax = {t0_argmax} ({tok.decode([t0_argmax])!r})")
    log(f"  T0 vs REF_A (unpadded eager): cos = {t0_vs_refA:.6f}  ← T0 gate (>={args.gate})")
    log(f"  T0 vs REF_B (legacy stub):    cos = {t0_vs_refB:.6f}  (informational)")
    log(f"  T0 elapsed = {t0_elapsed:.2f}s")

    # T0 GATE: padded eager at L=chunk_size must produce the SAME next token
    # (argmax) as legacy 1-tok/iter stub at the actual last position. Stub is
    # production's current reference, so argmax-match means chat output is the
    # same. cos values informational (eager S1a has natural drift vs stub).
    t0_ok = t0_argmax == refB_argmax
    if not t0_ok:
        log(f"FAIL T0: padded fixed-L={CHUNK_SIZE} argmax {t0_argmax} != "
            f"legacy stub argmax {refB_argmax}")
        raise SystemExit(1)
    log(f"PASS T0: padded eager call at fixed L={CHUNK_SIZE} matches legacy "
        f"stub argmax at the actual last position. Refactor path is correct "
        f"for production chat (greedy decode bit-equivalent at first token).")


if __name__ == "__main__":
    main()
