#!/usr/bin/env python3
"""Phase 2.B.1.7 + 2.B.1.8 — end-to-end K+1 verify trace smoke.

Captures the B=K+1 verify trace (two-phase warmup), then validates that
trace replay produces argmaxes equivalent to:
1. Independent eager K+1 kp1 forward (proves trace == eager)
2. Independent B=1 forwards (proves verify rows == B=1 at same input)

Gates:
1. TRACE CAPTURES: ensure_verify_trace_kp1 returns a valid trace_id.
2. TRACE REPLAY: verify_step_traced returns shape (Bv,) non-NaN.
3. TRACE == EAGER: traced argmaxes match eager forward_token_gm4_inner_kp1
   argmaxes (bit-equivalent expected).
4. PER-ROW == B=1: each traced argmax matches an independent B=1 step
   argmax at the same cur_pos with the same input token.

Trigger:  touch ~/tt-xla/.cache/gm4_runtime/trig/gemma4_kp1_trace_e2e_smoke
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


def readback_argmax(t, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    ).int().numpy()
    if arr.ndim == 3:
        return arr[0].flatten()
    return arr.flatten()[:Bv]


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
        log(f"re-using cached prefill")
        srv._set_pos(state, cur_pos_val)
    log(f"  cur_pos = {cur_pos_val}")

    srv.setup_verify_kp1_state(state, K=K, log=log)

    # STEP A: B=1 reference at next decode position (writes K_X/V_X at slot
    # decode_pos so kp1's subsequent read sees the same cache state).
    decode_pos = cur_pos_val + 1
    cand_tok = int(prompt_ids[0])  # BOS
    log("─" * 64)
    log(f"STEP A: B=1 reference at pos={decode_pos} tok={cand_tok}")
    log("─" * 64)
    argmax_b1 = srv.step_forward_v031(state, tok_id=cand_tok, pos=decode_pos)
    log(f"  B=1 argmax = {argmax_b1}")

    # STEP B: Eager K+1 kp1 (canonical reference for trace).
    log("─" * 64)
    log("STEP B: EAGER kp1 forward at pos=%d with [%d]*%d" % (decode_pos, cand_tok, Bv))
    log("─" * 64)
    srv.update_verify_inputs(state, current_pos=decode_pos,
                             candidate_token_ids=[cand_tok] * Bv)
    t = time.time()
    argmax_eager_tt = srv.forward_token_gm4_inner_kp1(state)
    ttnn.synchronize_device(state.mesh)
    eager_wall_ms = (time.time() - t) * 1000
    argmax_eager = readback_argmax(argmax_eager_tt, state.mesh)
    ttnn.deallocate(argmax_eager_tt)
    log(f"  eager wall: {eager_wall_ms:.1f} ms")
    log(f"  eager argmaxes: {argmax_eager.tolist()}")

    # STEP C: Capture verify trace.
    log("─" * 64)
    log("STEP C: capture verify trace (two-phase warmup)")
    log("─" * 64)
    rc = 0
    t = time.time()
    try:
        srv.ensure_verify_trace_kp1(state, log=log)
    except Exception as e:
        log(f"  ✗ GATE 1 TRACE CAPTURE FAIL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1
    capture_wall_ms = (time.time() - t) * 1000
    log(f"  ✓ GATE 1 PASS — trace captured (id={state.verify_trace_id}) "
        f"in {capture_wall_ms:.0f} ms")

    # STEP D: Replay trace, gate-check.
    log("─" * 64)
    log("STEP D: replay trace — equivalence gates")
    log("─" * 64)
    # Re-set verify inputs (the trace's two warmup forwards consumed them,
    # but verify buffers are NOT corrupted by the trace itself — just refresh
    # to be defensive).
    srv.update_verify_inputs(state, current_pos=decode_pos,
                             candidate_token_ids=[cand_tok] * Bv)
    t = time.time()
    argmax_traced = srv.verify_step_traced(state)
    replay_wall_ms = (time.time() - t) * 1000
    log(f"  replay wall: {replay_wall_ms:.1f} ms")
    log(f"  traced argmaxes: {argmax_traced.tolist()}")

    # GATE 2: shape + non-NaN
    if argmax_traced.shape != (Bv,):
        log(f"  ✗ GATE 2 SHAPE FAIL: got {argmax_traced.shape}, expected ({Bv},)")
        rc = 1
    else:
        log(f"  ✓ GATE 2 PASS — replay shape OK ({Bv},)")

    # GATE 3: trace == eager (bit-equivalent)
    mismatches_te = [i for i in range(Bv) if int(argmax_traced[i]) != int(argmax_eager[i])]
    if mismatches_te:
        log(f"  ✗ GATE 3 TRACE==EAGER FAIL: rows {mismatches_te} differ")
        log(f"     traced = {argmax_traced.tolist()}")
        log(f"     eager  = {argmax_eager.tolist()}")
        rc = 1
    else:
        log(f"  ✓ GATE 3 PASS — traced argmaxes bit-equivalent to eager kp1")

    # GATE 4: trace == B=1 (per-row equivalence)
    mismatches_b1 = [i for i in range(Bv) if int(argmax_traced[i]) != int(argmax_b1)]
    if mismatches_b1:
        log(f"  ✗ GATE 4 TRACE==B=1 FAIL: rows {mismatches_b1} don't match B=1={argmax_b1}")
        rc = 1
    else:
        log(f"  ✓ GATE 4 PASS — all {Bv} traced rows match B=1 argmax = {argmax_b1}")

    # STEP E: measure traced perf (3 replays after JIT).
    log("─" * 64)
    log("STEP E: measure traced replay perf (3 warm replays)")
    log("─" * 64)
    times_ms = []
    for i in range(3):
        srv.update_verify_inputs(state, current_pos=decode_pos,
                                 candidate_token_ids=[cand_tok] * Bv)
        t = time.time()
        _ = srv.verify_step_traced(state)
        times_ms.append((time.time() - t) * 1000)
    log(f"  per-replay wall (incl. host+readback): {[f'{x:.1f}' for x in times_ms]} ms")
    log(f"  mean = {sum(times_ms)/len(times_ms):.1f} ms")

    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — Phase 2.B.1 COMPLETE. Verify trace captured + "
            "bit-equivalent to eager + per-row matches B=1.")
    else:
        log("VERDICT: FAIL — see gate diagnostics above")
    log("=" * 64)
    if cold_start:
        ttnn.close_mesh_device(state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main(state=None))
