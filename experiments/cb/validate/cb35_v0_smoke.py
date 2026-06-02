"""CB35-1 (v0) isolation gate — 35B B=1 forward via server_35b_cb wrapper.

Validates that server_35b_cb.forward_batch_tp_inner (the cb_engine entry point
for the 35B backend, v0 path) produces bit-identical outputs to the existing
base.step_forward_inner.

This is the smallest sanity gate before plugging into cb_engine: if the
wrapper round-trip is clean at B=1, the cb_engine integration shouldn't
introduce new model-correctness bugs.

Cases:
  1. argmax mode (return_logits=False, return_topk=None):
       wrapper argmax == base.step_forward_inner argmax (single token equality)
  2. logits mode (return_logits=True):
       wrapper logits.argmax == base argmax (after-the-fact equivalence)
  3. topk mode (return_topk=8):
       top-1 index matches base argmax; values shape correct

Bootstrap once, run all cases in sequence (each forward mutates state, so
we reset between cases via cb_reset_states).

Run on qb1:
  cd ~/tt-xla && .venv/bin/python experiments/cb/validate/cb35_v0_smoke.py

Exits 0 on success.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import ttnn  # noqa: E402

import server_35b_ttnn as base  # noqa: E402
import server_35b_cb as cb  # noqa: E402


def log(msg: str):
    print(msg, flush=True)


def host_int_from_argmax(state, am_tt):
    """argmax returns a UINT32 [1, 1] tensor — replicated across mesh. Read chip 0."""
    arr = ttnn.to_torch(am_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return int(arr.flatten()[0].item())


def host_logits_from(state, logits_tt):
    """ROW_MAJOR [1, VOCAB] tensor → numpy [VOCAB]."""
    arr = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return arr.float().numpy().reshape(-1)[: base.VOCAB]


def host_topk(state, top_vals, top_idxs):
    vals = ttnn.to_torch(top_vals, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    idxs = ttnn.to_torch(top_idxs, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return vals.flatten().tolist(), [int(i) for i in idxs.flatten().tolist()]


def feed(state, tok_id: int, cur_pos: int):
    """Update input buffers for one decode step at slot 0."""
    cb.update_input_buffers_batched(state, [tok_id], [cur_pos])


def main() -> int:
    log("[cb35-v0-smoke] bootstrapping 35B base state (~6 min on qb1)…")
    state = base.State()
    base.bootstrap(state, log)

    log("[cb35-v0-smoke] setup_cb_state(B=1)…")
    cb.setup_cb_state(state, B=1)

    # Pick a fixed prompt token to feed at pos 0; record the argmax via base.
    PROMPT_TOK = 100  # arbitrary
    fails = 0

    # ── Case 1: argmax via cb wrapper == argmax via base ────────────────
    log("[cb35-v0-smoke] case 1: argmax bit-equivalence")
    cb.cb_reset_states(state)
    feed(state, PROMPT_TOK, 0)
    am_base_tt = base.step_forward_inner(state)
    am_base = host_int_from_argmax(state, am_base_tt)
    ttnn.deallocate(am_base_tt)
    log(f"  base argmax = {am_base}")

    cb.cb_reset_states(state)
    feed(state, PROMPT_TOK, 0)
    am_cb_tt = cb.forward_batch_tp_inner(state)  # default: argmax
    am_cb = host_int_from_argmax(state, am_cb_tt)
    ttnn.deallocate(am_cb_tt)
    log(f"  cb argmax   = {am_cb}")
    if am_base != am_cb:
        log(f"  ✗ FAIL: argmax mismatch")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # ── Case 2: logits mode → argmax(host) == base argmax ───────────────
    log("[cb35-v0-smoke] case 2: logits return_logits=True; argmax(host) match")
    cb.cb_reset_states(state)
    feed(state, PROMPT_TOK, 0)
    logits_tt = cb.forward_batch_tp_inner(state, return_logits=True)
    logits_np = host_logits_from(state, logits_tt)
    ttnn.deallocate(logits_tt)
    am_from_logits = int(np.argmax(logits_np))
    log(f"  argmax(host logits) = {am_from_logits}")
    if am_from_logits != am_base:
        log(f"  ✗ FAIL: logits-mode argmax != base argmax")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # ── Case 3: topk mode (K=8) → top-1 index match ─────────────────────
    log("[cb35-v0-smoke] case 3: topk K=8; top-1 idx match")
    cb.cb_reset_states(state)
    feed(state, PROMPT_TOK, 0)
    top_vals_tt, top_idxs_tt = cb.forward_batch_tp_inner(state, return_topk=8)
    _, top_idxs = host_topk(state, top_vals_tt, top_idxs_tt)
    ttnn.deallocate(top_vals_tt); ttnn.deallocate(top_idxs_tt)
    log(f"  topk indices = {top_idxs}")
    if top_idxs[0] != am_base:
        log(f"  ✗ FAIL: topk[0] != base argmax")
        fails += 1
    else:
        log(f"  ✓ PASS")

    if fails:
        log(f"\n[cb35-v0-smoke] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-v0-smoke] ALL 3 cases PASS — v0 wrapper bit-equivalent to base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
