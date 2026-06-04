#!/usr/bin/env python3
"""S1a functional gate: chunked prefill -> decode -> coherent text?

Prefills a prompt two ways — the single-token stub (production-equivalent) and
forward_prefill_chunked_tp (whole-prompt parallel) — then greedily decodes N
tokens from each prefilled state (same decode path for both, so only the prefill
differs) and prints both continuations. Chunked prefill is "good enough to ship"
if its continuation is coherent + sensible (the usable-models bar), even though it
won't bit-match the bf16-noisy stub (see research/27b_chunked_prefill_plan.md).

Run on qb1 (from repo root):
  make run PY=experiments/cb/validate/prefill_generate.py
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
import server_tp as base                     # noqa: E402  (_decode uses base.* at module scope)


def _chip0_logits(state, rm):
    import ttnn
    t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return t.float().numpy()[0][:state.vocab_size]


def _decode(state, first_tid, start_pos, n):
    """Greedy-decode n tokens (incl. first_tid) from start_pos via the proven
    decode forward. Returns the generated token ids."""
    import ttnn
    gen = [int(first_tid)]
    tid, pos = int(first_tid), start_pos
    for _ in range(n - 1):
        base.update_input_buffers(state, tid, pos)
        rm = base.forward_token_tp_inner(state, return_logits=True)
        logits = _chip0_logits(state, rm)
        ttnn.deallocate(rm)
        tid = int(np.argmax(logits))
        gen.append(tid)
        pos += 1
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=32, help="prompt token count (32 hits chunked-DN Neumann)")
    ap.add_argument("--decode", type=int, default=40, help="tokens to generate from the prefilled state")
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, _ = bootstrap_27b_cb()
    tokenizer = state.tok

    prompt = ("The capital of France is the city of Paris, which has long been a "
              "center of art, science, philosophy, and political history in Europe, "
              "drawing scholars and travelers from every corner of the wider world.")
    ids = tokenizer.encode(prompt)[:args.max_pos]
    L = len(ids)
    log(f"prompt ({L} tok): {tokenizer.decode(ids)!r}")

    log("=== stub prefill (single-token) -> decode ===")
    base._reset_state_buffers(state)
    stub = base.forward_prefill_tp_inner(state, ids, capture_logits=True)
    stub_gen = _decode(state, int(np.argmax(stub[-1])), L, args.decode)
    log(f"  {tokenizer.decode(stub_gen)!r}")

    log("=== chunked prefill (whole-prompt parallel) -> decode ===")
    base._reset_state_buffers(state)
    chk = base.forward_prefill_chunked_tp(state, ids, capture_logits=True)
    chk_gen = _decode(state, int(np.argmax(chk[-1])), L, args.decode)
    log(f"  {tokenizer.decode(chk_gen)!r}")

    same = sum(int(a == b) for a, b in zip(stub_gen, chk_gen))
    log(f"=== token overlap {same}/{min(len(stub_gen), len(chk_gen))} "
        f"(greedy branches after first divergence — judge coherence, not exact match) ===")


if __name__ == "__main__":
    main()
