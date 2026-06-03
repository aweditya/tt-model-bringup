"""CB35-1 v0 chat-shaped multi-step decode test.

Validates that cb.forward_batch_tp_inner (B=1) produces the SAME token
sequence as base.step_forward_inner over multiple decode steps. This
catches state-mutation bugs (DN recurrence, KV cache writes) that a
single-token argmax test would miss.

Cheap to run via cb35_dev_harness (no re-bootstrap; reloads server_35b_cb).

Triggers via:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v0_chat'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402

import server_35b_ttnn as base  # noqa: E402
import server_35b_cb as cb  # noqa: E402


def log(msg: str):
    print(msg, flush=True)


def host_int_from_argmax(state, am_tt):
    arr = ttnn.to_torch(am_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return int(arr.flatten()[0].item())


def feed(state, tok_id: int, cur_pos: int):
    cb.update_input_buffers_batched(state, [tok_id], [cur_pos])


def gen_via_base(state, prompt_ids: list[int], max_new: int) -> list[int]:
    """Generate max_new tokens via base.step_forward_inner. Reset first."""
    cb.cb_reset_states(state)
    # Prefill: feed each prompt token, last forward gives the first generated.
    last_am = None
    for i, t in enumerate(prompt_ids):
        feed(state, t, i)
        am = base.step_forward_inner(state)
        last_am = host_int_from_argmax(state, am)
        ttnn.deallocate(am)
    # Decode: feed previous argmax, generate max_new tokens.
    gen: list[int] = []
    pos = len(prompt_ids)
    for _ in range(max_new):
        gen.append(last_am)
        feed(state, last_am, pos)
        am = base.step_forward_inner(state)
        last_am = host_int_from_argmax(state, am)
        ttnn.deallocate(am)
        pos += 1
    return gen


def gen_via_cb(state, prompt_ids: list[int], max_new: int) -> list[int]:
    """Generate max_new tokens via cb.forward_batch_tp_inner argmax mode."""
    cb.cb_reset_states(state)
    last_am = None
    for i, t in enumerate(prompt_ids):
        feed(state, t, i)
        am = cb.forward_batch_tp_inner(state)  # default = argmax
        last_am = host_int_from_argmax(state, am)
        ttnn.deallocate(am)
    gen: list[int] = []
    pos = len(prompt_ids)
    for _ in range(max_new):
        gen.append(last_am)
        feed(state, last_am, pos)
        am = cb.forward_batch_tp_inner(state)
        last_am = host_int_from_argmax(state, am)
        ttnn.deallocate(am)
        pos += 1
    return gen


def main(state=None) -> int:
    if state is None:
        log("[cb35-v0-chat] bootstrapping 35B (~14 min)…")
        state = base.State()
        base.bootstrap(state, log)
    else:
        log("[cb35-v0-chat] using pre-bootstrapped state from harness")
    log(f"[cb35-v0-chat] setup_cb_state(B=1) (was B={getattr(state, 'cb_B', None)})")
    cb.setup_cb_state(state, B=1)

    # Simple fixed prompt — picked to stress decode rather than test prompts.
    # A few-token prompt gives us prefill + multi-step decode in one run.
    PROMPT_IDS = [100, 271, 271]   # arbitrary 3-token "prompt"
    MAX_NEW = 8                    # 8 decode steps after prefill

    log(f"[cb35-v0-chat] generating {MAX_NEW} tokens (prompt={PROMPT_IDS})…")

    log("  → via base.step_forward_inner…")
    base_seq = gen_via_base(state, PROMPT_IDS, MAX_NEW)
    log(f"    base_seq = {base_seq}")

    log("  → via cb.forward_batch_tp_inner (argmax mode)…")
    cb_seq = gen_via_cb(state, PROMPT_IDS, MAX_NEW)
    log(f"    cb_seq   = {cb_seq}")

    if base_seq == cb_seq:
        log(f"[cb35-v0-chat] ✓ PASS — both paths produced identical "
            f"{MAX_NEW}-token sequence")
        return 0

    # Find first divergence
    div = next((i for i, (a, b) in enumerate(zip(base_seq, cb_seq)) if a != b),
               min(len(base_seq), len(cb_seq)))
    log(f"[cb35-v0-chat] ✗ FAIL — diverged at position {div}:")
    log(f"    base[{div}] = {base_seq[div] if div < len(base_seq) else 'OOB'}")
    log(f"    cb  [{div}] = {cb_seq[div] if div < len(cb_seq) else 'OOB'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
