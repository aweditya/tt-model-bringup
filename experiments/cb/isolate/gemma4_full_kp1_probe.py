#!/usr/bin/env python3
"""Phase 2.B.1.6 — full 48-layer B=K+1 verify forward isolation gate.

Validates `forward_token_gm4_inner_kp1` end-to-end:
1. PLUMBING: full forward returns argmax tensor shape [K+1, 1] non-NaN.
2. INVARIANCE: K+1 IDENTICAL candidate tokens → K+1 IDENTICAL argmaxes.
   Proves the per-row computation is independent (no cross-row leakage)
   across the full 48-layer chain.
3. PER-ROW B=1 EQUIVALENCE: each kp1 row's argmax must equal the
   argmax of an independent B=1 forward fed the same candidate token at
   the same position. THIS IS THE STRONG GATE.

Cache-write ordering: B=1 `step_forward_v031` writes to slot 5 (corrupting
the prior K/V). kp1 SKIPS the cache write. So we run the B=1 reference
FIRST (corrupts slot 5 to K_X/V_X), then run kp1 with candidate=[X]*Bv;
kp1 rows read slot 5 = K_X/V_X (same as B=1), so each row argmax must
match B=1.

Trigger:  touch ~/tt-xla/.cache/gm4_runtime/trig/gemma4_full_kp1_probe
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"

K = 5
Bv = K + 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def readback_argmax(t):
    """Read argmax tensor; return as a flat int numpy array.

    argmax_tt is shape [Bv, 1] uint32. Post-all_gather it's replicated
    across the mesh; ConcatMeshToTensor(dim=0) gives [NCHIPS, Bv, 1] →
    we take chip 0.
    """
    arr = ttnn.to_torch(
        t,
        mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0),
    ).int().numpy()
    if arr.ndim == 3:
        return arr[0].flatten()  # [Bv]
    return arr.flatten()


def main(state=None):
    cold_start = state is None
    if cold_start:
        log("cold-start path: bootstrap")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("dev-harness path: using pre-bootstrapped state")

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    L_prefill = int(prompt_ids.shape[0])
    cur_pos_val = L_prefill - 1
    if not getattr(state, "_kp1_probe_prefilled", False):
        log(f"prefill {L_prefill}-token canonical prompt (cold)")
        t = time.time()
        for pos in range(L_prefill):
            tok = int(prompt_ids[pos])
            srv.step_forward_v031(state, tok_id=tok, pos=pos)
        log(f"  prefill wall: {(time.time()-t)*1000:.1f} ms")
        state._kp1_probe_prefilled = True
    else:
        log(f"re-using cached prefill from prior probe run")
        srv._set_pos(state, cur_pos_val)
    log(f"  cur_pos = {cur_pos_val}")

    srv.setup_verify_kp1_state(state, K=K, log=log)

    # ── STEP A: B=1 reference forward at next position (cur_pos+1=6) ──
    # Choose a candidate token. Use prompt_ids[0] = BOS as a deterministic
    # default. Run a B=1 step which writes K_X/V_X to cache slot 6 (the
    # decode position we'll verify at). Capture the argmax for cross-check.
    decode_pos = cur_pos_val + 1
    cand_tok = int(prompt_ids[0])
    log("─" * 64)
    log(f"STEP A: B=1 reference at pos={decode_pos} tok={cand_tok}")
    log("─" * 64)
    t = time.time()
    argmax_b1 = srv.step_forward_v031(state, tok_id=cand_tok, pos=decode_pos)
    log(f"  B=1 wall: {(time.time()-t)*1000:.1f} ms")
    log(f"  B=1 argmax = {argmax_b1}  (type={type(argmax_b1).__name__})")

    # ── STEP B: kp1 forward with K+1 identical candidate tokens ──
    # All K+1 rows = same tok at the same pos. kp1 SKIPS cache write, so
    # all Bv rows read the cache state left by STEP A (K_X/V_X at slot
    # decode_pos). Each kp1 row's Q is from `cand_tok` (same as B=1), so
    # all rows should produce argmax == argmax_b1.
    log("─" * 64)
    log(f"STEP B: kp1 forward at pos={decode_pos} with {Bv} identical "
        f"candidate tokens = [{cand_tok}]*{Bv}")
    log("─" * 64)
    srv.update_verify_inputs(state, current_pos=decode_pos,
                             candidate_token_ids=[cand_tok] * Bv)
    t = time.time()
    argmax_kp1_tt = srv.forward_token_gm4_inner_kp1(state)
    ttnn.synchronize_device(state.mesh)
    log(f"  kp1 forward wall: {(time.time()-t)*1000:.1f} ms")
    argmax_kp1 = readback_argmax(argmax_kp1_tt)
    ttnn.deallocate(argmax_kp1_tt)
    log(f"  kp1 argmax (shape={argmax_kp1.shape}): {argmax_kp1.tolist()}")

    # ── GATE 1: PLUMBING ──
    rc = 0
    log("─" * 64)
    log("GATE 1: PLUMBING — argmax shape + non-NaN")
    log("─" * 64)
    if argmax_kp1.shape != (Bv,):
        log(f"  ✗ wrong shape: got {argmax_kp1.shape}, expected ({Bv},)")
        rc = 1
    else:
        log(f"  ✓ shape OK ({Bv},)")

    # ── GATE 2: INVARIANCE — all rows identical ──
    log("─" * 64)
    log("GATE 2: INVARIANCE — K+1 identical inputs → K+1 identical argmaxes")
    log("─" * 64)
    unique_args = np.unique(argmax_kp1)
    log(f"  unique argmax values: {unique_args.tolist()}")
    if len(unique_args) != 1:
        log(f"  ✗ row argmaxes differ ({len(unique_args)} unique values) — "
            f"forward has cross-row leakage")
        rc = 1
    else:
        log(f"  ✓ all {Bv} row argmaxes are identical ({unique_args[0]})")

    # ── GATE 3: PER-ROW B=1 EQUIVALENCE (THE STRONG GATE) ──
    log("─" * 64)
    log("GATE 3: PER-ROW B=1 EQUIVALENCE — each kp1 row argmax = B=1 argmax")
    log("─" * 64)
    log(f"  B=1 argmax = {argmax_b1}")
    log(f"  kp1 row argmaxes = {argmax_kp1.tolist()}")
    mismatches = [i for i in range(Bv) if int(argmax_kp1[i]) != int(argmax_b1)]
    if mismatches:
        log(f"  ✗ {len(mismatches)}/{Bv} rows mismatch B=1: rows {mismatches}")
        rc = 1
    else:
        log(f"  ✓ all {Bv} rows match B=1 argmax = {argmax_b1}")

    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — full 48-layer kp1 forward plumbing + invariance + "
            "per-row B=1 equivalence all gate-clean")
    else:
        log("VERDICT: FAIL — see gate diagnostics above")
    log("=" * 64)
    if cold_start:
        ttnn.close_mesh_device(state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main(state=None))
