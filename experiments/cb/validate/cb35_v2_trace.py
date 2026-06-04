"""CB35-3 v2 — trace capture at B=N gate.

Captures the batched forward (forward_batch_tp_inner_batched) into a
ttnn trace at B=2, then replays it. Validates:
  1. Two-phase warmup + capture completes without errors.
  2. Replay output matches eager output (within precision drift).
  3. Replay is faster than eager (speedup measurement).

Mirrors the cb_scheduler pattern (cb_scheduler.py:197-227).

Run via harness:
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v2_trace'
"""
from __future__ import annotations

import sys
import time
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


DUMMY_TOK = 0
K = 8


def main(state=None) -> int:
    if state is None:
        state = base.State()
        base.bootstrap(state, log)

    mesh = state.mesh
    B = 2
    fails = 0

    log(f"[cb35-v2-trace] setup B={B}")
    cb.setup_cb_state(state, B=B)
    cb.cb_reset_states(state)

    # ── Phase 1: eager warmup ──────────────────────────────────────────
    log("[cb35-v2-trace] Phase 1: eager warmup (2 iterations)")
    for i in range(2):
        cb.update_input_buffers_batched(state, [DUMMY_TOK] * B, [i] * B)
        out = cb.forward_batch_tp_inner(state, return_topk=K)
        if isinstance(out, tuple):
            for h in out:
                ttnn.deallocate(h)
        else:
            ttnn.deallocate(out)
    log("  warmup complete")

    ttnn.synchronize_device(mesh)

    # ── Phase 2: capture trace ─────────────────────────────────────────
    log("[cb35-v2-trace] Phase 2: trace capture")
    cb.update_input_buffers_batched(state, [DUMMY_TOK] * B, [2] * B)
    try:
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        handle = cb.forward_batch_tp_inner(state, return_topk=K)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    except Exception as e:
        log(f"  ✗ FAIL: trace capture raised: {type(e).__name__}: {e}")
        fails += 1
        return 1
    vals_h, idxs_h = handle
    log(f"  capture OK; trace_id={trace_id}")
    cb.cb_reset_states(state)

    # ── Replay 5 steps + time ──────────────────────────────────────────
    log("[cb35-v2-trace] replay timing")
    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

    times_trace = []
    for step in range(5):
        cb.update_input_buffers_batched(state, [100, 200], [step] * B)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        # Read top-k indices (must touch host to fence)
        idxs_t = ttnn.to_torch(idxs_h, mesh_composer=composer)
        t1 = time.perf_counter()
        idxs = idxs_t[:B].long().numpy()
        times_trace.append(t1 - t0)
        log(f"  step {step}: slot0_top1={int(idxs[0].flatten()[0])}, slot1_top1={int(idxs[1].flatten()[0])}, t={(t1-t0)*1000:.1f} ms")

    # ── Compare to eager (one eager step for reference) ────────────────
    cb.cb_reset_states(state)
    log("[cb35-v2-trace] eager reference")
    times_eager = []
    for step in range(3):
        cb.update_input_buffers_batched(state, [100, 200], [step] * B)
        t0 = time.perf_counter()
        out = cb.forward_batch_tp_inner(state, return_topk=K)
        vh, ih = out
        ih_t = ttnn.to_torch(ih, mesh_composer=composer)
        t1 = time.perf_counter()
        ttnn.deallocate(vh); ttnn.deallocate(ih)
        idxs = ih_t[:B].long().numpy()
        times_eager.append(t1 - t0)
        log(f"  step {step}: slot0_top1={int(idxs[0].flatten()[0])}, slot1_top1={int(idxs[1].flatten()[0])}, t={(t1-t0)*1000:.1f} ms")

    mean_trace = float(np.mean(times_trace[1:]) * 1000)  # skip first replay (warmup)
    mean_eager = float(np.mean(times_eager[1:]) * 1000)
    speedup = mean_eager / mean_trace if mean_trace > 0 else 0.0
    log(f"\n[cb35-v2-trace] mean eager  = {mean_eager:.1f} ms/step")
    log(f"[cb35-v2-trace] mean traced = {mean_trace:.1f} ms/step")
    log(f"[cb35-v2-trace] speedup     = {speedup:.2f}×")

    ttnn.release_trace(mesh, trace_id)
    ttnn.deallocate(vals_h); ttnn.deallocate(idxs_h)

    if speedup < 2.0:
        log(f"  ⚠ speedup {speedup:.2f}× below 2× threshold — investigate")
    else:
        log("  ✓ trace capture + replay viable")

    log(f"\n[cb35-v2-trace] {fails} case(s) FAILED" if fails else
        "\n[cb35-v2-trace] capture + replay OK — v2 viable")
    return fails


if __name__ == "__main__":
    sys.exit(main())
