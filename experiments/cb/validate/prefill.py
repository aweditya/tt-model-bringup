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

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cos(a, b):
    a = a.astype(np.float64).reshape(-1); b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=32,
                    help="prompt token count (<=32 powers-of-2 hit the chunked-DN Neumann path)")
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    tok = state.tok
    state.deltanet_recurrence_mode = "manual"  # qb1 baseline; the chunked path uses Neumann

    prompt = ("The capital of France is the city of Paris, which has long been a "
              "center of art, science, philosophy, and political history in Europe.")
    prompt_ids = tok.encode(prompt)[:args.max_pos]
    L = len(prompt_ids)
    log(f"prompt_ids ({L} tok): {prompt_ids}")

    log("=== reference: forward_prefill_tp_inner (single-token stub) ===")
    base._reset_state_buffers(state)
    ref = base.forward_prefill_tp_inner(state, prompt_ids, capture_logits=True)  # [L, vocab]
    ref_ids = ref.argmax(axis=-1).tolist()

    log("=== chunked: forward_prefill_chunked_tp (whole-prompt parallel) ===")
    base._reset_state_buffers(state)
    chk = base.forward_prefill_chunked_tp(state, prompt_ids, capture_logits=True)  # [L, vocab]
    chk_ids = chk.argmax(axis=-1).tolist()

    worst = 1.0
    for pos in range(L):
        c = _cos(ref[pos], chk[pos])
        worst = min(worst, c)
        flag = "=" if ref_ids[pos] == chk_ids[pos] else "DIFF"
        log(f"  pos {pos:2d}: logit_cos={c:.6f}  argmax ref={ref_ids[pos]} chunked={chk_ids[pos]} {flag}")
    n_match = sum(int(a == b) for a, b in zip(ref_ids, chk_ids))
    cos_ok = worst >= 0.99
    log(f"  worst-position logit_cos = {worst:.6f}  | argmax match {n_match}/{L}")

    log("=== TTFT (sync-bounded forward time, no logit readback) ===")
    try:
        import ttnn
        base._reset_state_buffers(state)
        t0 = time.time(); base.forward_prefill_tp_inner(state, prompt_ids); ttnn.synchronize_device(state.mesh)
        t_stub = time.time() - t0
        base._reset_state_buffers(state)
        t0 = time.time(); base.forward_prefill_chunked_tp(state, prompt_ids); ttnn.synchronize_device(state.mesh)
        t_chk = time.time() - t0
        log(f"  stub    prefill ({L} tok): {t_stub * 1000:8.1f} ms")
        log(f"  chunked prefill ({L} tok): {t_chk * 1000:8.1f} ms   ({t_stub / t_chk:.2f}x)")
    except Exception as e:
        log(f"  TTFT timing skipped (err: {e!r})")

    log(f"=== verdict: {'PASS' if cos_ok else 'FAIL'} "
        f"(chunked prefill vs single-token stub, cosine gate >= 0.99) ===")


if __name__ == "__main__":
    main()
