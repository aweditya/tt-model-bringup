"""CB35-2 v1.3 — batched GatedAttention gate.

Validates server_35b_cb.attn_step_batched_35b against
base.attn_forward_ttnn_sdpa for a single attention layer at B=1 (bit-equiv)
and B=2 (slot 0 == base ref + slot 0 == slot 1 with same input).

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v1_attn'
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


def first_attn_layer(state):
    for i, t in enumerate(state.layer_types):
        if t != "linear_attention":
            return i
    raise RuntimeError("no attn layer found")


def make_inputs(state, B, slot0_tok=42, slot0_pos=0, slot1_tok=42, slot1_pos=0):
    if B == 1:
        cb.update_input_buffers_batched(state, [slot0_tok], [slot0_pos])
    else:
        cb.update_input_buffers_batched(state, [slot0_tok, slot1_tok], [slot0_pos, slot1_pos])
    return cb._batched_prelude(state)


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    mesh = state.mesh
    L = first_attn_layer(state)
    log(f"[cb35-v1-attn] using layer L={L} (first attn layer)")
    fails = 0

    # ── Case 1: B=1 path bit-equiv to base.attn_forward_ttnn_sdpa ─────
    log("[cb35-v1-attn] case 1: B=1 attn_step_batched == base.attn_forward_ttnn_sdpa")
    cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)
    h_tt, cos_tt, sin_tt = make_inputs(state, B=1, slot0_tok=42, slot0_pos=0)
    out_cb = cb.attn_step_batched_35b(state, h_tt, state.per_layer_tt[L],
                                       state.cb_kv[L], cos_tt, sin_tt)
    out_cb_np = ttnn.to_torch(out_cb, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    ttnn.deallocate(out_cb); ttnn.deallocate(h_tt); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Fresh KV for base reference.
    cb.cb_reset_states(state)
    h_tt_b, cos_tt_b, sin_tt_b = make_inputs(state, B=1, slot0_tok=42, slot0_pos=0)
    # base.attn_forward_ttnn_sdpa needs (kc, vc) via state._current_kv_cache.
    state._current_kv_cache = state.kv_caches_tt[L]
    out_base, _ = base.attn_forward_ttnn_sdpa(
        h_tt_b, state.per_layer_tt[L], mesh, cos_tt_b, sin_tt_b, state,
    )
    out_base_np = ttnn.to_torch(out_base, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    ttnn.deallocate(out_base); ttnn.deallocate(h_tt_b); ttnn.deallocate(cos_tt_b); ttnn.deallocate(sin_tt_b)

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

    # ── Case 2: B=2 same input → slot 0 == slot 1 + slot 0 == base ref ─
    log("[cb35-v1-attn] case 2: B=2 [42,42]@[0,0] → slot 0 == slot 1 + slot 0 == base ref")
    cb.setup_cb_state(state, B=2)
    cb.cb_reset_states(state)
    h_tt, cos_tt, sin_tt = make_inputs(state, B=2, slot0_tok=42, slot0_pos=0,
                                        slot1_tok=42, slot1_pos=0)
    out_cb = cb.attn_step_batched_35b(state, h_tt, state.per_layer_tt[L],
                                       state.cb_kv[L], cos_tt, sin_tt)
    out_arr = ttnn.to_torch(out_cb, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    log(f"  out_cb chip-flat shape = {out_arr.shape}")
    slot0 = out_arr[0].reshape(-1)[:cb.HIDDEN]
    slot1 = out_arr[1].reshape(-1)[:cb.HIDDEN]
    cos01 = float(np.dot(slot0, slot1) / (np.linalg.norm(slot0) * np.linalg.norm(slot1) + 1e-9))
    mad01 = float(np.abs(slot0 - slot1).max())
    log(f"  slot0 vs slot1 cosine = {cos01:.6f}, max_abs_diff = {mad01:.6f}")
    if cos01 < 0.9999 or mad01 > 0.05:
        log(f"  ✗ FAIL: identical-input slots differ")
        fails += 1
    else:
        log(f"  ✓ PASS")

    cos_b = float(np.dot(slot0, b) / (np.linalg.norm(slot0) * np.linalg.norm(b) + 1e-9))
    mad_b = float(np.abs(slot0 - b).max())
    log(f"  slot0 vs base cosine = {cos_b:.6f}, max_abs_diff = {mad_b:.6f}")
    if cos_b < 0.9999 or mad_b > 0.05:
        log(f"  ✗ FAIL: B=2 slot 0 != base reference")
        fails += 1
    else:
        log(f"  ✓ PASS")
    ttnn.deallocate(out_cb); ttnn.deallocate(h_tt); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    if fails:
        log(f"\n[cb35-v1-attn] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-v1-attn] ALL cases PASS — batched attention bit-equivalent to base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
