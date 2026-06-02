"""CB35-2 v1.1 — batched embed + RoPE prelude gate.

Validates server_35b_cb._batched_prelude at B=1 (bit-equiv to base
prelude) and B=2 (per-slot output diverges with per-slot input).

Cases:
  1. B=1: prelude output bit-identical to base.step_forward_inner's
     embed+RoPE prelude (compute base's by inlining the same ops).
  2. B=2 same-token sanity: feed [42, 42] @ pos [0, 0]; h_tt[0] should
     equal h_tt[1] element-wise (deterministic embed of same row).
  3. B=2 distinct-token sanity: feed [42, 99] @ pos [0, 0]; h_tt[0] != h_tt[1]
     by some host-side norm.
  4. B=2 distinct-position RoPE: feed [42, 42] @ pos [0, 5]; cos_tt[0] != cos_tt[1].

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v1_embed'
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


def host_chip0(t, mesh):
    """Read tensor as numpy, chip-0 view."""
    arr = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    # arr shape varies — concat replicates along dim 0 across chips, so chip 0 is first slab.
    return arr


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    mesh = state.mesh
    fails = 0

    # ── Case 1: B=1 prelude == base prelude ────────────────────────────
    log("[cb35-v1-embed] case 1: B=1 prelude bit-equiv to base")
    cb.setup_cb_state(state, B=1)
    cb.update_input_buffers_batched(state, [42], [0])
    h_cb, cos_cb, sin_cb = cb._batched_prelude(state)

    # Compute base prelude inline (same ops).
    eb_out = ttnn.embedding(state.tok_buf, state.embed_tt)
    h_base = ttnn.to_layout(eb_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(eb_out)

    h_cb_np = host_chip0(h_cb, mesh).reshape(-1)
    h_base_np = host_chip0(h_base, mesh).reshape(-1)
    diff = np.abs(h_cb_np[:cb.HIDDEN] - h_base_np[:cb.HIDDEN]).max()
    log(f"  max |h_cb - h_base| = {diff:.6f}")
    if diff > 0:
        log(f"  ✗ FAIL: B=1 prelude diverges from base prelude")
        fails += 1
    else:
        log(f"  ✓ PASS: bit-identical")
    ttnn.deallocate(h_cb); ttnn.deallocate(cos_cb); ttnn.deallocate(sin_cb)
    ttnn.deallocate(h_base)

    # ── Case 2: B=2 same-token sanity ─────────────────────────────────
    log("[cb35-v1-embed] case 2: B=2 same-token [42,42] → h_tt[0]==h_tt[1]")
    cb.setup_cb_state(state, B=2)
    cb.update_input_buffers_batched(state, [42, 42], [0, 0])
    h_cb, cos_cb, sin_cb = cb._batched_prelude(state)
    arr = host_chip0(h_cb, mesh)
    # Layout: ConcatMeshToTensor(dim=0) gives [NCHIPS*B, 1, HIDDEN] for [B,1,HIDDEN] replicated.
    # Take chip 0 → first B rows.
    chip0 = arr[:cb.cb_B if False else 2]  # first 2 rows = slot 0 + slot 1 on chip 0
    h_slot0 = chip0[0].reshape(-1)[:cb.HIDDEN]
    h_slot1 = chip0[1].reshape(-1)[:cb.HIDDEN]
    diff = np.abs(h_slot0 - h_slot1).max()
    log(f"  max |slot0 - slot1| = {diff:.6f}")
    if diff != 0:
        log(f"  ✗ FAIL: same-token slots should be identical")
        fails += 1
    else:
        log(f"  ✓ PASS")
    ttnn.deallocate(h_cb); ttnn.deallocate(cos_cb); ttnn.deallocate(sin_cb)

    # ── Case 3: B=2 distinct-token sanity ─────────────────────────────
    log("[cb35-v1-embed] case 3: B=2 distinct-token [42,99] → h_tt[0]!=h_tt[1]")
    cb.update_input_buffers_batched(state, [42, 99], [0, 0])
    h_cb, cos_cb, sin_cb = cb._batched_prelude(state)
    arr = host_chip0(h_cb, mesh)
    chip0 = arr[:2]
    h_slot0 = chip0[0].reshape(-1)[:cb.HIDDEN]
    h_slot1 = chip0[1].reshape(-1)[:cb.HIDDEN]
    diff = np.abs(h_slot0 - h_slot1).max()
    log(f"  max |slot0 - slot1| = {diff:.6f}")
    if diff == 0:
        log(f"  ✗ FAIL: distinct-token slots should differ")
        fails += 1
    else:
        log(f"  ✓ PASS")
    ttnn.deallocate(h_cb); ttnn.deallocate(cos_cb); ttnn.deallocate(sin_cb)

    # ── Case 4: B=2 distinct-position RoPE ────────────────────────────
    log("[cb35-v1-embed] case 4: B=2 distinct-pos [0,5] → cos[0]!=cos[1]")
    cb.update_input_buffers_batched(state, [42, 42], [0, 5])
    h_cb, cos_cb, sin_cb = cb._batched_prelude(state)
    arr = host_chip0(cos_cb, mesh)
    # cos shape after reshape is [B, ROTARY_DIM]; ConcatMeshToTensor(dim=0) gives [NCHIPS*B, ROTARY_DIM]
    chip0 = arr[:2]
    c0 = chip0[0].reshape(-1)[:cb.ROTARY_DIM]
    c1 = chip0[1].reshape(-1)[:cb.ROTARY_DIM]
    diff = np.abs(c0 - c1).max()
    log(f"  max |cos[0] - cos[1]| = {diff:.6f}")
    if diff == 0:
        log(f"  ✗ FAIL: distinct-pos RoPE rows should differ")
        fails += 1
    else:
        log(f"  ✓ PASS")
    ttnn.deallocate(h_cb); ttnn.deallocate(cos_cb); ttnn.deallocate(sin_cb)

    if fails:
        log(f"\n[cb35-v1-embed] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-v1-embed] ALL cases PASS — batched embed+RoPE prelude works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
