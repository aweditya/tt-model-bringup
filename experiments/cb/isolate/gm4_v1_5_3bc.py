#!/usr/bin/env python3
"""v1.5 — 3b (identical-slot) + 3c (distinct-slot) gates at B=2.

3b: feed the SAME prompt to two slots; argmax sequences should be
    identical step-by-step. Catches per-slot determinism breaks.
3c: feed DIFFERENT prompts to two slots; argmax sequences should be
    DIFFERENT in general AND each slot's argmax should equal what that
    slot's prompt would produce in isolation. Catches cross-slot
    contamination via shared KV / page-table / RoPE bugs.

Run via harness:
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v1_5_3bc'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import server_gemma4_unified_cb as cb      # noqa: E402

# Two distinct 6-token prompt prefixes.
PROMPT_A = [2, 818, 5279, 529, 7001, 563]    # "The capital of France is"
PROMPT_B = [2, 818, 12544, 529, 17328, 563]  # synthetic distinct prefix


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_cb_seq(state, prompts):
    """Run len(prompts[0]) steps at B=len(prompts), each slot fed
    prompts[slot][pos]. Return [B, n] argmax matrix as a list-of-lists.
    """
    B = len(prompts)
    N = len(prompts[0])
    assert all(len(p) == N for p in prompts)
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

    # 3b: identical inputs at two slots.
    log("--- 3b: identical-slot ---")
    _ensure_B(state, 2)
    out = _run_cb_seq(state, [PROMPT_A, PROMPT_A])
    log(f"  slot 0: {out[0]}")
    log(f"  slot 1: {out[1]}")
    b3b = (out[0] == out[1])
    log(f"  identical: {b3b}")

    # 3c: distinct inputs at two slots, each slot's argmax should equal what
    # that slot's prompt produces in isolation (i.e., B=1 reference).
    log("--- 3c: distinct-slot ---")
    _ensure_B(state, 2)  # also resets cur_pos to -1
    out_dist = _run_cb_seq(state, [PROMPT_A, PROMPT_B])
    log(f"  slot A (prompt A): {out_dist[0]}")
    log(f"  slot B (prompt B): {out_dist[1]}")

    # B=1 isolation refs for each prompt.
    log("  computing B=1 single-slot refs for each prompt…")
    _ensure_B(state, 1)
    ref_a = _run_cb_seq(state, [PROMPT_A])[0]
    _ensure_B(state, 1)
    ref_b = _run_cb_seq(state, [PROMPT_B])[0]
    log(f"  ref A (B=1): {ref_a}")
    log(f"  ref B (B=1): {ref_b}")
    b3c_a = (out_dist[0] == ref_a)
    b3c_b = (out_dist[1] == ref_b)
    log(f"  3c slot A matches its B=1 ref: {b3c_a}")
    log(f"  3c slot B matches its B=1 ref: {b3c_b}")

    log("=" * 78)
    log(f"3b PASS: {b3b}")
    log(f"3c PASS: {b3c_a and b3c_b}")
    log("=" * 78)
    verdict = b3b and b3c_a and b3c_b
    log(f"VERDICT: {'PASS' if verdict else 'FAIL'}")

    if owned_state:
        import ttnn; ttnn.close_device(state.mesh)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
