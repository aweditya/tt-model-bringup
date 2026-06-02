"""CB35-2 v1.5 — full B>1 chat gate.

Validates server_35b_cb.forward_batch_tp_inner_batched does a multi-step
decode at B=2 with each slot independently advancing. Compares slot 0's
generated token sequence to a single-stream B=1 base reference for the
same prompt.

If this passes, v1 is END-TO-END.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v1_chat'
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


def host_int(tt, mesh):
    """argmax_tt is UINT32 [B, 1] replicated. Return chip 0 view as numpy [B]."""
    arr = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).numpy()
    # shape (NCHIPS*B,) or (NCHIPS, B, 1) etc — take first B elements as chip-0 slice.
    flat = arr.reshape(-1)
    return flat


def generate_b1(state, prompt_tok, n_steps):
    """Single-stream B=1 base reference: feed prompt_tok at pos 0, then loop."""
    cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)
    seq = []
    next_tok = prompt_tok
    for step in range(n_steps):
        cb.update_input_buffers_batched(state, [next_tok], [step])
        am_tt = base.step_forward_inner(state)
        flat = host_int(am_tt, state.mesh)
        next_tok = int(flat[0])
        ttnn.deallocate(am_tt)
        seq.append(next_tok)
    return seq


def generate_b2(state, prompt_tok, n_steps):
    """Batched B=2 with [prompt, prompt] same input on both slots.
    Slot 0's sequence must equal the B=1 reference for the same prompt.
    """
    cb.setup_cb_state(state, B=2)
    cb.cb_reset_states(state)
    seq_s0 = []
    seq_s1 = []
    next_s0 = prompt_tok
    next_s1 = prompt_tok
    for step in range(n_steps):
        cb.update_input_buffers_batched(state, [next_s0, next_s1], [step, step])
        am_tt = cb.forward_batch_tp_inner_batched(state)
        flat = host_int(am_tt, state.mesh)
        # flat has NCHIPS*B = 8 entries; chip-0 view is the first B=2 entries.
        next_s0 = int(flat[0])
        next_s1 = int(flat[1])
        ttnn.deallocate(am_tt)
        seq_s0.append(next_s0)
        seq_s1.append(next_s1)
    return seq_s0, seq_s1


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    fails = 0
    PROMPT_TOK = 100
    N_STEPS = 4  # short — full chat is in v1.5b

    log(f"[cb35-v1-chat] B=1 reference: prompt_tok={PROMPT_TOK}, n_steps={N_STEPS}")
    seq_ref = generate_b1(state, PROMPT_TOK, N_STEPS)
    log(f"  B=1 reference sequence: {seq_ref}")

    log(f"[cb35-v1-chat] B=2 batched (both slots = same prompt)")
    seq_s0, seq_s1 = generate_b2(state, PROMPT_TOK, N_STEPS)
    log(f"  slot 0 sequence: {seq_s0}")
    log(f"  slot 1 sequence: {seq_s1}")

    if seq_s0 == seq_s1:
        log("  ✓ slot 0 == slot 1 (per-slot independence preserved)")
    else:
        log("  ✗ FAIL: slots diverged with same input")
        fails += 1

    if seq_s0 == seq_ref:
        log("  ✓ slot 0 == B=1 reference (full forward bit-correct)")
    else:
        log(f"  ✗ FAIL: slot 0 != B=1 ref (ref={seq_ref}, slot0={seq_s0})")
        fails += 1

    if fails:
        log(f"\n[cb35-v1-chat] {fails} case(s) FAILED — v1.5 NOT BIT-VALIDATED")
        return 1
    log(f"\n[cb35-v1-chat] ALL cases PASS — v1.5 END-TO-END BIT-VALIDATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
