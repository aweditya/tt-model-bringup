"""CB35-prod — exercise the cb_scheduler-style topk path through the
unified forward_batch_tp_inner entry at B=1 and B=2.

This is the production wire-up gate: cb_scheduler calls
`cb.forward_batch_tp_inner(state, return_topk=K)` expecting
(top_vals, top_idxs) tuples. We test both shapes and per-slot semantics.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/prod_topk'
"""
from __future__ import annotations

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


def topk_chip0(vals_h, idxs_h, mesh, B):
    """Mirror cb_scheduler.step_sampled_topk:
      composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
      vals_t / idxs_t = ttnn.to_torch(..., composer)
      vals[:B], idxs[:B] = chip 0's view of B slots.
    """
    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
    vals_t = ttnn.to_torch(vals_h, mesh_composer=composer)
    idxs_t = ttnn.to_torch(idxs_h, mesh_composer=composer)
    vals = vals_t[:B].float().numpy()
    idxs = idxs_t[:B].long().numpy()
    return vals, idxs


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    mesh = state.mesh
    fails = 0
    K = 8

    # ── Case 1: B=1 topk via unified entry ────────────────────────────
    log("[cb35-prod-topk] case 1: B=1 forward_batch_tp_inner(return_topk=K)")
    cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [100], [0])
    vals_h, idxs_h = cb.forward_batch_tp_inner(state, return_topk=K)
    vals, idxs = topk_chip0(vals_h, idxs_h, mesh, B=1)
    ttnn.deallocate(vals_h); ttnn.deallocate(idxs_h)
    log(f"  vals.shape={vals.shape}, idxs.shape={idxs.shape}")
    log(f"  slot 0 top-{K} idxs = {idxs[0].flatten().tolist()}")
    # Sanity: top-1 should be in vocab range
    top1 = int(idxs[0].flatten()[0])
    if not (0 <= top1 < cb.VOCAB):
        log(f"  ✗ FAIL: top-1 {top1} out of vocab range")
        fails += 1
    else:
        log(f"  ✓ PASS — top-1 = {top1}")

    # ── Case 2: B=2 topk via unified entry ────────────────────────────
    log("[cb35-prod-topk] case 2: B=2 forward_batch_tp_inner(return_topk=K)")
    cb.setup_cb_state(state, B=2)
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [100, 200], [0, 0])
    vals_h, idxs_h = cb.forward_batch_tp_inner(state, return_topk=K)
    vals, idxs = topk_chip0(vals_h, idxs_h, mesh, B=2)
    ttnn.deallocate(vals_h); ttnn.deallocate(idxs_h)
    log(f"  vals.shape={vals.shape}, idxs.shape={idxs.shape}")
    log(f"  slot 0 top-{K} idxs = {idxs[0].flatten().tolist()}")
    log(f"  slot 1 top-{K} idxs = {idxs[1].flatten().tolist()}")

    s0_top1 = int(idxs[0].flatten()[0])
    s1_top1 = int(idxs[1].flatten()[0])
    if not (0 <= s0_top1 < cb.VOCAB and 0 <= s1_top1 < cb.VOCAB):
        log(f"  ✗ FAIL: top-1 out of vocab range")
        fails += 1
    elif s0_top1 == s1_top1:
        log(f"  ✗ FAIL: distinct prompts produced same top-1 ({s0_top1})")
        fails += 1
    else:
        log(f"  ✓ PASS — slot 0 top-1 = {s0_top1}, slot 1 top-1 = {s1_top1} (distinct)")

    # ── Case 3: B=1 default argmax via unified entry ──────────────────
    log("[cb35-prod-topk] case 3: B=1 forward_batch_tp_inner() default (argmax)")
    cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [100], [0])
    am_tt = cb.forward_batch_tp_inner(state)
    am = int(ttnn.to_torch(am_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).flatten()[0])
    ttnn.deallocate(am_tt)
    log(f"  B=1 argmax = {am}")
    if not (0 <= am < cb.VOCAB):
        log(f"  ✗ FAIL: argmax {am} out of vocab range")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # ── Case 4: B=2 default argmax via unified entry ──────────────────
    log("[cb35-prod-topk] case 4: B=2 forward_batch_tp_inner() default (argmax)")
    cb.setup_cb_state(state, B=2)
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [100, 200], [0, 0])
    am_tt = cb.forward_batch_tp_inner(state)
    am_arr = ttnn.to_torch(am_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).flatten().numpy()
    ttnn.deallocate(am_tt)
    log(f"  B=2 argmax slot 0 = {int(am_arr[0])}, slot 1 = {int(am_arr[1])}")
    if int(am_arr[0]) == int(am_arr[1]):
        log(f"  ✗ FAIL: B=2 distinct prompts produced same argmax")
        fails += 1
    else:
        log(f"  ✓ PASS — slots distinct")

    if fails:
        log(f"\n[cb35-prod-topk] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-prod-topk] ALL cases PASS — production wire-up gate clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
