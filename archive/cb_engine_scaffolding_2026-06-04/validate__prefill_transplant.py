#!/usr/bin/env python3
"""S2.4 end-to-end gate — prefill→transplant→CB decode matches reference.

The actual win we're after: avoid 1-tok-per-iter CB prefill by running the
production S1a/S1b chunked prefill on a temp state, transplanting the
post-prefill state into a CB slot, then continuing decode through the existing
CB engine.

Both paths use S1a (forward_prefill_chunked_tp) as the prefill. We're not
gating S1a vs the 1-tok-per-iter stub here (S1a has known bf16 per-position
cos≈0.95-0.99 vs that stub; that's not the transplant's fault). What we ARE
gating: AFTER an S1a prefill, does a transplant + N CB decode steps produce
the same tokens as N production decode steps in-place?

Reference path: S1a prefill on production state → N decode steps via
forward_token_tp_inner on the SAME production state. State stays where it was
written; no transplant.

Test path: S1a prefill on production state → cb_prefill_transplant into slot
s → N CB decode steps via forward_batch_tp_inner on cb_dn/cb_kv slot s.

Gate: argmax of the N+1 generated tokens IDENTICAL between reference and
test. Drift here = transplant bug OR CB-step-vs-prod-step divergence (which
should be zero per CB1 "bit-identical to prod").

Run on qb1 (from repo root):
  make run PY=experiments/cb/validate/prefill_transplant.py
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
import server_tp_cb as cb                    # noqa: E402


def _chip0_logits(state, rm):
    import ttnn
    t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return t.float().numpy()[0][:state.vocab_size]


def _reference_decode(state, base, prompt_ids, n_decode):
    """Reference: S1a chunked prefill on production state + n_decode greedy
    decode steps via forward_token_tp_inner (production decode, NO transplant).
    Returns the list of n_decode+1 generated token ids."""
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


def _test_path_decode(state, base, prompt_ids, n_decode, slot_s=0):
    """Test: chunked prefill on production state → transplant into CB slot →
    n_decode CB step()s on that slot. Returns the list of n_decode+1 tokens.
    """
    import ttnn
    base._reset_state_buffers(state)
    # Capture the prefill's last-position argmax — same token the reference
    # decodes first.
    cap = base.forward_prefill_chunked_tp(state, prompt_ids, capture_logits=True)
    first_tok = int(np.argmax(cap[-1]))

    cb.cb_reset_states(state)
    cb.cb_prefill_transplant(state, slot_s, len(prompt_ids))
    ttnn.synchronize_device(state.mesh)

    gen = [first_tok]
    B = state.cb_B
    tids = [0] * B
    pos = [-1] * B
    tids[slot_s] = first_tok
    pos[slot_s] = len(prompt_ids)
    tid = first_tok
    for _ in range(n_decode):
        tids[slot_s] = tid
        pos[slot_s] = pos[slot_s]  # current; forward_batch reads pos and writes KV at pos
        cb.update_input_buffers_batched(state, tids, pos)
        am = cb.forward_batch_tp_inner(state)
        am_np = ttnn.to_torch(am,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).int().numpy()
        # am shape per chip [1, B]; ConcatMeshToTensor(dim=0) → [4, B] (chip 0 has
        # the canonical answer since lm_head is replicated / argmax is the same).
        tid = int(am_np[0, slot_s])
        gen.append(tid)
        pos[slot_s] = pos[slot_s] + 1
        ttnn.deallocate(am)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=64, help="prompt token count")
    ap.add_argument("--decode", type=int, default=3, help="decode steps AFTER first prefill tok")
    ap.add_argument("--slot", type=int, default=0)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, base = bootstrap_27b_cb()
    tok = state.tok

    prompt_text = ("The capital of France is the city of Paris, which has long been a "
                   "center of art, science, philosophy, and political history in Europe, "
                   "drawing scholars and travelers from every corner of the wider world "
                   "for many centuries of recorded human civilization and culture, "
                   "blending tradition and reinvention across countless generations.")
    ids = tok.encode(prompt_text)[:args.length]
    L = len(ids)
    log(f"prompt L={L} tokens; decoding {args.decode} steps AFTER prefill (total {args.decode+1} tokens)")

    log("=== reference: 1-tok-per-iter prefill + decode ===")
    ref_tokens = _reference_decode(state, base, ids, args.decode)
    log(f"  reference tokens: {ref_tokens}")
    log(f"  reference text:   {tok.decode(ref_tokens)!r}")

    log("=== CB setup ===")
    B = 4
    cb.setup_cb_state(state, B)
    # bit-identical to production: CB defaults to shiftacc conv (fast, drifts).
    # For correctness validation use kdim (sum-reduce — same math as prod).
    state.cb_conv_mode = 'kdim'
    state.cb_dn_recurrence_mode = 'manual'

    log("=== test: chunked prefill + transplant + CB decode ===")
    test_tokens = _test_path_decode(state, base, ids, args.decode, slot_s=args.slot)
    log(f"  test tokens:      {test_tokens}")
    log(f"  test text:        {tok.decode(test_tokens)!r}")

    match = ref_tokens == test_tokens
    if match:
        log(f"PASS: all {len(ref_tokens)} tokens identical. S2.4 gate green; "
            f"S2.5 (alternating scheduler) unblocked.")
    else:
        first_diff = next((i for i, (a, b) in enumerate(zip(ref_tokens, test_tokens))
                           if a != b), -1)
        log(f"FAIL: tokens diverge at position {first_diff} — "
            f"ref={ref_tokens[first_diff]!r} test={test_tokens[first_diff]!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
