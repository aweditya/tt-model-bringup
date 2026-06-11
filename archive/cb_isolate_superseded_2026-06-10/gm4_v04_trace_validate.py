#!/usr/bin/env python3
"""v0.4 — traced decode validator.

Gate from plan:
  100 traced steps == 100 eager steps, argmax token-for-token identical.

Flow:
  1. Bootstrap (or reuse harness state).
  2. Run N=100 EAGER steps starting from a BOS prompt, free-run via
     argmax. Record argmax sequence E.
  3. Capture decode trace (`ensure_decode_trace`).
  4. Run N=100 TRACED steps with the same starting token, free-run via
     argmax. Record argmax sequence T.
  5. Compare T vs E token-for-token. PASS iff all N match.

Cache handling: state.kv_caches_tt is shared between runs. Each step
writes K/V at the current cur_pos slot, overwriting the previous run's
value. Step k of run 2 only reads slots [0..k], all freshly written by
run 2. Slots > current cur_pos from run 1 are ignored by SDPA's cur_pos
mask. So no explicit reset is needed.

Run via the dev harness:
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v04_trace_validate'
  ssh qb1 'cat tt-xla/.cache/gm4_runtime/trig/last.log'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

N_STEPS = 100
# Use the v0.3.1 prompt for the first 6 steps (teacher-forced from
# "The capital of France is"), then free-run via argmax. BOS-only
# free-run degenerates to a constant <image|> loop in ~1 step (v0.3.2),
# which makes the eager-vs-traced match trivial — a 6-token prompt
# warms up real residual-stream dynamics so the gate has teeth.
PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]  # BOS + "The capital of France is"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_n_steps(state, fn, prompt_ids, n):
    """Run `fn(state, tok_id, pos)` n times. For pos < len(prompt_ids):
    teacher-force the prompt token. For pos >= len(prompt_ids): feed the
    previous step's argmax (free-run). Returns the list of n argmax tokens.
    """
    out = []
    prev_argmax = None
    for pos in range(n):
        if pos < len(prompt_ids):
            tok = prompt_ids[pos]
        else:
            tok = prev_argmax
        argmax = fn(state, int(tok), pos)
        out.append(int(argmax))
        prev_argmax = int(argmax)
    return out


def main(state=None):
    owned_state = state is None
    if owned_state:
        log("bootstrapping Gemma 4 12B server (~80s)…")
        t0 = time.time()
        state = srv.State()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("using pre-bootstrapped state from harness")

    log(f"running {N_STEPS} EAGER steps "
        f"(teacher-forced pos 0..{len(PROMPT_IDS)-1}, free-run after)…")
    t0 = time.time()
    eager_argmax = _run_n_steps(state, srv.step_forward_v031, PROMPT_IDS, N_STEPS)
    log(f"  eager done in {time.time()-t0:.1f}s "
        f"({(time.time()-t0)*1000/N_STEPS:.1f} ms/tok)")
    log(f"  eager first 10: {eager_argmax[:10]}")
    log(f"  eager last  10: {eager_argmax[-10:]}")

    log("capturing decode trace…")
    t0 = time.time()
    srv.ensure_decode_trace(state, log=log)
    log(f"  trace capture took {time.time()-t0:.1f}s")

    log(f"running {N_STEPS} TRACED steps (same prompt)…")
    t0 = time.time()
    traced_argmax = _run_n_steps(state, srv.step_forward_traced, PROMPT_IDS, N_STEPS)
    log(f"  traced done in {time.time()-t0:.1f}s "
        f"({(time.time()-t0)*1000/N_STEPS:.1f} ms/tok)")
    log(f"  traced first 10: {traced_argmax[:10]}")
    log(f"  traced last  10: {traced_argmax[-10:]}")

    n_match = sum(1 for a, b in zip(eager_argmax, traced_argmax) if a == b)
    log("=" * 78)
    log(f"v0.4 traced vs eager token-for-token match: {n_match}/{N_STEPS}")
    first_div = next((i for i, (a, b) in enumerate(zip(eager_argmax, traced_argmax)) if a != b),
                     None)
    if first_div is not None:
        log(f"  first divergence at pos={first_div}: "
            f"eager={eager_argmax[first_div]} traced={traced_argmax[first_div]}")
    log("=" * 78)
    verdict = "PASS" if n_match == N_STEPS else "FAIL"
    log(f"VERDICT: {verdict}")

    if owned_state:
        ttnn = sys.modules.get("ttnn")
        if ttnn is not None:
            ttnn.close_device(state.mesh)
    return 0 if n_match == N_STEPS else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
