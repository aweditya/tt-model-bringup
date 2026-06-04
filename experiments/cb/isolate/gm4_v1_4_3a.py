#!/usr/bin/env python3
"""v1.4 / 3a — B=1 CB batched forward bit-identical to single-slot v0.4.

Validation: run the batched CB forward at B=1 on the canonical
"The capital of France is" prompt and compare argmax against the
single-slot `step_forward_v031`. PASS if all 6 argmaxes match.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v1_4_3a'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import server_gemma4_unified_cb as cb      # noqa: E402

PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    owned_state = state is None
    if owned_state:
        log("bootstrapping…")
        state = base.State()
        base.bootstrap(state, log=log)
    else:
        log("using harness state")

    log("single-slot reference (step_forward_v031)…")
    single = []
    for pos, tid in enumerate(PROMPT_IDS):
        single.append(int(base.step_forward_v031(state, tid, pos)))
    log(f"  single: {single}")

    log("setup_cb_state(B=1)…")
    if not hasattr(state, "cb_B"):
        cb.setup_cb_state(state, B=1)
    else:
        # Re-allocate if dimensions differ; for B=1 with existing same-B state,
        # just reuse.
        if state.cb_B != 1:
            log(f"  re-allocating cb_state from B={state.cb_B} to B=1")
            cb.setup_cb_state(state, B=1)
    cb.cb_reset_states(state)

    log("CB batched forward at B=1…")
    batched = []
    for pos, tid in enumerate(PROMPT_IDS):
        out = cb.step_forward_cb(state, token_ids=[tid], cur_positions=[pos])
        batched.append(int(out[0]))
    log(f"  batched: {batched}")

    n_match = sum(1 for a, b in zip(single, batched) if a == b)
    log("=" * 78)
    log(f"3a gate (B=1 CB == single-slot): {n_match}/{len(PROMPT_IDS)} match")
    log("=" * 78)
    verdict = "PASS" if n_match == len(PROMPT_IDS) else "FAIL"
    log(f"VERDICT: {verdict}")

    if owned_state:
        import ttnn
        ttnn.close_device(state.mesh)
    return 0 if n_match == len(PROMPT_IDS) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
