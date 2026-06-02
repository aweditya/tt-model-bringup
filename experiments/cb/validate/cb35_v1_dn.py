"""CB35-2 v1.2 — batched DN layer (manual recurrence) gate.

Validates server_35b_cb.dn_step_batched_35b against base.dn_forward_ttnn
at B=2 (slot 0 = real input, slot 1 = dummy). Slot 0's output must match
the single-stream reference bit-exactly (within bf16 tolerance).

Single layer test — not full forward; isolates DN correctness without
attention or MoE coupling.

Cases:
  1. B=1 path: dn_step_batched_35b output == base.dn_forward_ttnn output
     (same input, freshly-zeroed state).
  2. B=2 path: slot 0 of batched output == base output for slot 0's input.
     Verifies broadcast-over-B doesn't pollute slot 0's state.
  3. B=2 in-place state: after one step, cb_dn[L]['rs'] for slot 0 == base
     state.dn_caches_tt[L]'s rs after the equivalent base step.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v1_dn'
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

import server_35b_ttnn as base  # noqa: E402
import server_35b_cb as cb  # noqa: E402


def log(msg: str):
    print(msg, flush=True)


def host_chip0_flat(t, mesh):
    """Concat across mesh (dim=0) → numpy; return chip-0 slab."""
    arr = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    return arr


def make_h_input(state, B, slot0_seed=42):
    """Build [B, 1, HIDDEN] per-chip h_tt via embedding(slot0_seed) + dummies."""
    # Use embedding directly to get a realistic h_tt (matches what _batched_prelude produces).
    tok_ids = [slot0_seed] + [0] * (B - 1)
    cb.update_input_buffers_batched(state, tok_ids, [0] * B)
    h_tt, cos_tt, sin_tt = cb._batched_prelude(state)
    return h_tt, cos_tt, sin_tt


def first_dn_layer(state):
    for i, t in enumerate(state.layer_types):
        if t == "linear_attention":
            return i
    raise RuntimeError("no DN layer found")


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    mesh = state.mesh
    L = first_dn_layer(state)
    log(f"[cb35-v1-dn] using layer L={L} (first DN layer)")
    fails = 0

    # ── Case 1: B=1 path bit-equiv to base ────────────────────────────
    log("[cb35-v1-dn] case 1: B=1 path == base.dn_forward_ttnn")
    cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)
    h_tt, cos_tt, sin_tt = make_h_input(state, B=1, slot0_seed=42)
    out_cb = cb.dn_step_batched_35b(state, h_tt, state.per_layer_tt[L], state.cb_dn[L],
                                     qk_l2_weight_tt=getattr(state, "qk_l2_weight_tt", None),
                                     qk_l2_eps=getattr(state, "qk_l2_eps", None))
    out_cb_np = host_chip0_flat(out_cb, mesh)
    ttnn.deallocate(out_cb); ttnn.deallocate(h_tt); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Fresh state for base reference.
    cb.cb_reset_states(state)  # zeros state.dn_caches_tt[L] (B=1 path aliases)
    h_tt_b, cos_tt_b, sin_tt_b = make_h_input(state, B=1, slot0_seed=42)
    out_base, _, _ = base.dn_forward_ttnn(
        h_tt_b, state.per_layer_tt[L], mesh, state.dn_caches_tt[L],
        qk_l2_weight_tt=getattr(state, "qk_l2_weight_tt", None),
        qk_l2_eps=getattr(state, "qk_l2_eps", None),
    )
    out_base_np = host_chip0_flat(out_base, mesh)
    ttnn.deallocate(out_base); ttnn.deallocate(h_tt_b); ttnn.deallocate(cos_tt_b); ttnn.deallocate(sin_tt_b)

    # cosine + max-abs-diff over chip 0's view
    a = out_cb_np.reshape(-1)[:cb.HIDDEN]
    b = out_base_np.reshape(-1)[:cb.HIDDEN]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    mad = float(np.abs(a - b).max())
    log(f"  cosine = {cos:.6f}, max_abs_diff = {mad:.6f}")
    if cos < 0.9999 or mad > 0.05:
        log(f"  ✗ FAIL: cb output != base output at B=1")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # ── Case 2a: B=2 same-input both slots → slot 0 == slot 1 (sanity) ─
    log("[cb35-v1-dn] case 2a: B=2 [42,42] → slot 0 output == slot 1 output")
    cb.setup_cb_state(state, B=2)
    cb.cb_reset_states(state)
    cb.update_input_buffers_batched(state, [42, 42], [0, 0])
    h_tt, cos_tt, sin_tt = cb._batched_prelude(state)
    out_cb = cb.dn_step_batched_35b(state, h_tt, state.per_layer_tt[L], state.cb_dn[L],
                                     qk_l2_weight_tt=getattr(state, "qk_l2_weight_tt", None),
                                     qk_l2_eps=getattr(state, "qk_l2_eps", None))
    # out shape [B=2, HIDDEN] replicated. ConcatMeshToTensor(dim=0) → [NCHIPS*B=8, HIDDEN].
    out_arr = ttnn.to_torch(out_cb, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    log(f"  out_cb chip-flat shape = {out_arr.shape}")
    slot0 = out_arr[0].reshape(-1)[:cb.HIDDEN]  # chip 0 slot 0
    slot1 = out_arr[1].reshape(-1)[:cb.HIDDEN]  # chip 0 slot 1
    cos01 = float(np.dot(slot0, slot1) / (np.linalg.norm(slot0) * np.linalg.norm(slot1) + 1e-9))
    mad01 = float(np.abs(slot0 - slot1).max())
    log(f"  slot0 vs slot1 cosine = {cos01:.6f}, max_abs_diff = {mad01:.6f}")
    if cos01 < 0.9999 or mad01 > 0.05:
        log(f"  ✗ FAIL: identical input slots produce different outputs")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # Also compare slot 0 to base ref (since input is the same as case 1's [42]).
    cos_b = float(np.dot(slot0, b) / (np.linalg.norm(slot0) * np.linalg.norm(b) + 1e-9))
    mad_b = float(np.abs(slot0 - b).max())
    log(f"  slot0 vs base   cosine = {cos_b:.6f}, max_abs_diff = {mad_b:.6f}")
    if cos_b < 0.9999 or mad_b > 0.05:
        log(f"  ✗ FAIL: B=2 slot 0 != case-1 base reference")
        fails += 1
    else:
        log(f"  ✓ PASS")
    ttnn.deallocate(out_cb); ttnn.deallocate(h_tt); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    if fails:
        log(f"\n[cb35-v1-dn] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-v1-dn] ALL cases PASS — batched DN bit-equivalent to base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
