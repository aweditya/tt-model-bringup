#!/usr/bin/env python3
"""CB2 — ragged per-slot positions + mid-batch admission validation.

CB1 proved the batched forward at equal per-slot positions (lockstep). Real
continuous batching is RAGGED: slots sit at different absolute positions, and a
finishing sequence is replaced by a new one in the same slot (Mamba-style state
slot reuse). The new primitive is `cb_reset_slots` — clear ONLY the admitted
slot's DN recurrent state (KV self-overwrites, bounded by cur_pos).

Test (avoids the untested cur_pos=-1 write path — both slots always active):
  1. ref_A = B=1 run of sequence A (6 positions).
  2. ref_B = B=1 run of sequence B (first 3 positions).
  3. Batched B=2:
     - steps 0..2: slot0=A, slot1=C (throwaway), both at pos=step (lockstep).
     - step 3: ADMIT B into slot1 → cb_reset_slots([1]).
     - steps 3..5: slot0 continues A at pos 3,4,5; slot1 runs B at pos 0,1,2.
       → the two slots are at DIFFERENT positions in the same forward.
  4. Assert slot0's full 0..5 argmax == ref_A (unaffected by slot1's churn +
     reset), and slot1's step-3..5 argmax == ref_B[0..2] (fresh state, own KV).

PASS ⇒ per-slot positions, admission reset, and slot isolation all work — the
device-level foundation an Orca scheduler (CB3) sits on top of.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_validate_ragged.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cb_ref(state, prompt_ids, n):
    """B=1 fresh CB run; returns argmax for the first n positions."""
    import ttnn
    cb.setup_cb_state(state, 1)
    cb.cb_reset_states(state)
    out = []
    for pos in range(n):
        cb.update_input_buffers_batched(state, [int(prompt_ids[pos])], [pos])
        am = cb.forward_batch_tp_inner(state)
        out.append(int(ttnn.to_torch(
            am, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).flatten()[0]))
        ttnn.deallocate(am)
    return out


def batched_step(state, toks, curs):
    import ttnn
    cb.update_input_buffers_batched(state, toks, curs)
    am = cb.forward_batch_tp_inner(state)
    vals = ttnn.to_torch(
        am, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).flatten().tolist()
    ttnn.deallocate(am)
    return [int(v) for v in vals[:len(toks)]]


def main():
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    tok = state.tok

    A = tok.encode("The capital of France is the city of")[:6]
    B = tok.encode("Once upon a time there lived a young")[:6]
    C = tok.encode("The largest planet in our solar system is")[:6]
    log(f"A={A}\n            B={B}\n            C={C}")

    log("=== references (B=1 fresh) ===")
    ref_A = cb_ref(state, A, 6)
    ref_B = cb_ref(state, B, 3)
    log(f"  ref_A (6): {ref_A}")
    log(f"  ref_B (3): {ref_B}")

    log("=== ragged B=2: admit B into slot1 at step3 (slot0 keeps running A) ===")
    cb.setup_cb_state(state, 2)
    cb.cb_reset_states(state)
    slot0_out, slot1_out = [], []
    # phase 1 — lockstep, slot0=A slot1=C, positions 0..2
    for s in range(3):
        o = batched_step(state, [A[s], C[s]], [s, s])
        slot0_out.append(o[0])
    # admission — new sequence B takes slot1; clear only slot1's DN state
    cb.cb_reset_slots(state, [1])
    log("  [admitted B into slot1 via cb_reset_slots([1])]")
    # phase 2 — slot0 continues A at pos 3..5; slot1 runs B at pos 0..2
    for s in range(3, 6):
        o = batched_step(state, [A[s], B[s - 3]], [s, s - 3])
        slot0_out.append(o[0])
        slot1_out.append(o[1])

    log(f"  slot0 argmax (A, pos0..5): {slot0_out}")
    log(f"  slot1 argmax (B, pos0..2): {slot1_out}")
    ok0 = (slot0_out == ref_A)
    ok1 = (slot1_out == ref_B)
    log(f"  slot0 vs ref_A: {'OK' if ok0 else 'MISMATCH'}")
    log(f"  slot1 vs ref_B: {'OK' if ok1 else 'MISMATCH'}")

    ok = ok0 and ok1
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log(f"  ragged different-position slots: {'OK' if ok0 else 'FAIL'} (slot0 unaffected)")
    log(f"  admission DN-state reset: {'OK' if ok1 else 'FAIL'} (slot1 fresh)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
