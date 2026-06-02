"""CB35-2 v1.0 — B-leading cache allocator gate.

Validates that setup_cb_state(B>1) allocates per-slot caches with the
correct shapes and zero contents. No forward yet — this is shape-only.

Cases:
  1. B=1 fast path: cb_dn[li] aliases state.dn_caches_tt[li] (no extra alloc).
     Verify by object identity.
  2. B=4: cb_dn[li]["rs"] has logical shape (4, B=4, NV_PER_CHIP, K, V);
     cb_dn[li]["cs"] (4, B=4, CONV_DIM_CHIP, KERNEL); cb_kv[li] sized
     B * sdpa_num_blocks; cb_page_table_tt has shape (B, sdpa_num_blocks)
     and slot s row equals arange(s*nb, (s+1)*nb).
  3. cb_reset_states(B=4): doubles back that all entries are zero on host
     (read 1 entry per layer-type-class via ttnn.to_torch).

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v1_alloc'
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


def _hostshape(t, mesh):
    """Concat-mesh-to-tensor then return numpy shape."""
    return tuple(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).shape)


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    fails = 0
    cfg = state.text_cfg
    mesh = state.mesh

    # ── Case 1: B=1 alias path ─────────────────────────────────────────
    log("[cb35-v1-alloc] case 1: B=1 alias path")
    cb.setup_cb_state(state, B=1)
    # Pick the first DN layer + first attn layer and check object identity.
    dn_li = next(i for i in range(cfg.num_hidden_layers) if state.layer_types[i] == "linear_attention")
    attn_li = next(i for i in range(cfg.num_hidden_layers) if state.layer_types[i] != "linear_attention")
    cs_alias = state.cb_dn[dn_li]["cs"] is state.dn_caches_tt[dn_li][0]
    rs_alias = state.cb_dn[dn_li]["rs"] is state.dn_caches_tt[dn_li][1]
    kc_alias = state.cb_kv[attn_li]["kc"] is state.kv_caches_tt[attn_li][0]
    vc_alias = state.cb_kv[attn_li]["vc"] is state.kv_caches_tt[attn_li][1]
    log(f"  alias cs={cs_alias} rs={rs_alias} kc={kc_alias} vc={vc_alias}")
    log(f"  cb_page_table_tt is None: {state.cb_page_table_tt is None}")
    if not (cs_alias and rs_alias and kc_alias and vc_alias and state.cb_page_table_tt is None):
        log(f"  ✗ FAIL: B=1 alias contract broken")
        fails += 1
    else:
        log(f"  ✓ PASS")

    # ── Case 2: B=4 separate-alloc shapes ──────────────────────────────
    log("[cb35-v1-alloc] case 2: B=4 alloc + shapes")
    B = 4
    cb.setup_cb_state(state, B=B)
    rs_shape = _hostshape(state.cb_dn[dn_li]["rs"], mesh)
    cs_shape = _hostshape(state.cb_dn[dn_li]["cs"], mesh)
    log(f"  cb_dn[{dn_li}].rs host shape = {rs_shape}")
    log(f"  cb_dn[{dn_li}].cs host shape = {cs_shape}")
    expect_rs = (cb.NCHIPS, B, cb.NV_PER_CHIP, cb.HEAD_K_DIM, cb.HEAD_V_DIM)
    expect_cs = (cb.NCHIPS, B, cb.CONV_DIM_CHIP, cb.CONV_KERNEL)
    if rs_shape != expect_rs:
        log(f"  ✗ FAIL: rs shape {rs_shape} != {expect_rs}")
        fails += 1
    if cs_shape != expect_cs:
        log(f"  ✗ FAIL: cs shape {cs_shape} != {expect_cs}")
        fails += 1

    kc_shape = _hostshape(state.cb_kv[attn_li]["kc"], mesh)
    log(f"  cb_kv[{attn_li}].kc host shape = {kc_shape}")
    expect_kc = (B * state.sdpa_num_blocks, cb.NCHIPS, state.sdpa_block_size, cb.HEAD_DIM_ATTN)
    if kc_shape != expect_kc:
        log(f"  ✗ FAIL: kc shape {kc_shape} != {expect_kc}")
        fails += 1

    pt_arr = ttnn.to_torch(state.cb_page_table_tt,
                           mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).numpy()
    # Replicated across chips → take chip 0 view (shape will be B*NCHIPS, sdpa_num_blocks after concat)
    pt_chip0 = pt_arr.reshape(cb.NCHIPS, B, state.sdpa_num_blocks)[0]
    log(f"  cb_page_table[0] = {pt_chip0[0].tolist()[:8]}…")
    nb = state.sdpa_num_blocks
    expected_pt = np.stack([np.arange(s*nb, (s+1)*nb) for s in range(B)], axis=0)
    if not np.array_equal(pt_chip0, expected_pt):
        log(f"  ✗ FAIL: page_table doesn't match expected per-slot block ranges")
        fails += 1
    else:
        log(f"  ✓ page_table matches per-slot block ranges")

    # ── Case 3: cb_reset_states zeros the B>1 caches ───────────────────
    log("[cb35-v1-alloc] case 3: cb_reset_states zero contract at B=4")
    # Write nonzero into cb_dn[dn_li]['rs'] to verify reset clears it.
    rs = state.cb_dn[dn_li]["rs"]
    one = ttnn.ones_like(rs)
    ttnn.copy(one, rs)
    ttnn.deallocate(one)
    cb.cb_reset_states(state)
    rs_arr = ttnn.to_torch(state.cb_dn[dn_li]["rs"],
                           mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
    if rs_arr.max() != 0 or rs_arr.min() != 0:
        log(f"  ✗ FAIL: rs not zero after reset (max={rs_arr.max():.4f} min={rs_arr.min():.4f})")
        fails += 1
    else:
        log(f"  ✓ PASS: rs all zero")

    if fails:
        log(f"\n[cb35-v1-alloc] {fails} case(s) FAILED")
        return 1
    log(f"\n[cb35-v1-alloc] ALL cases PASS — B-leading allocator works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
