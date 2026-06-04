#!/usr/bin/env python3
"""v1.6 — 3a + 3b + 3c gates at B=4 (final v1 acceptance gate).

Combines 3a/3b/3c against B=4 of distinct prompts. PASS gates the
v1 CB ship for cb_api/cb_scheduler integration (v2 HTTP wire-up).

3a-style: each slot's argmax sequence must equal the B=1 single-slot
          forward for its own prompt.
3b-style: two slots with the same prompt produce identical sequences.
3c-style: distinct prompts produce distinct sequences AND each matches
          its B=1 reference.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v1_6_b4'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import server_gemma4_unified_cb as cb      # noqa: E402

# Four prompts: two identical (3b coverage), two distinct (3c coverage).
PROMPT_A = [2, 818, 5279, 529, 7001, 563]
PROMPT_B = [2, 818, 12544, 529, 17328, 563]
PROMPT_C = [2, 818, 5279, 529, 7001, 563]   # identical to A
PROMPT_D = [2, 5279, 529, 7001, 563, 818]   # permuted distinct
PROMPTS = [PROMPT_A, PROMPT_B, PROMPT_C, PROMPT_D]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_seq(state, prompts):
    B = len(prompts)
    N = len(prompts[0])
    out = [[] for _ in range(B)]
    for pos in range(N):
        toks = [p[pos] for p in prompts]
        positions = [pos] * B
        argmax = cb.step_forward_cb(state, token_ids=toks, cur_positions=positions)
        for b in range(B):
            out[b].append(int(argmax[b]))
    return out


def _ensure_B(state, B):
    if not hasattr(state, "cb_B") or state.cb_B != B:
        cb.setup_cb_state(state, B=B)
    cb.cb_reset_states(state)


def main(state=None):
    owned_state = state is None
    if owned_state:
        log("bootstrapping…"); state = base.State(); base.bootstrap(state, log=log)
    else:
        log("using harness state")

    log("--- v1.6 B=4 acceptance gate ---")
    _ensure_B(state, 4)
    t0 = time.time()
    out = _run_seq(state, PROMPTS)
    log(f"  B=4 forward took {time.time()-t0:.1f}s")
    for b, p in enumerate(PROMPTS):
        log(f"  slot {b}: prompt={p}  argmax={out[b]}")

    # B=1 references per prompt.
    log("computing B=1 references…")
    refs = []
    for p in PROMPTS:
        _ensure_B(state, 1)
        refs.append(_run_seq(state, [p])[0])
        log(f"  ref prompt={p[:3]}… → {refs[-1]}")

    matches = [out[b] == refs[b] for b in range(len(PROMPTS))]
    log(f"per-slot matches B=1 ref: {matches}")

    # 3b: slot 0 (A) == slot 2 (C) since A == C.
    b3b = (out[0] == out[2])
    log(f"3b (slot 0 == slot 2, identical prompts): {b3b}")
    # 3c: at least one pair of distinct prompts produces distinct outputs.
    b3c_distinct = (out[0] != out[1]) or (out[0] != out[3])
    log(f"3c (some distinct outputs across distinct prompts): {b3c_distinct}")

    log("=" * 78)
    all_pass = all(matches) and b3b and b3c_distinct
    log(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")

    if owned_state:
        import ttnn; ttnn.close_device(state.mesh)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
