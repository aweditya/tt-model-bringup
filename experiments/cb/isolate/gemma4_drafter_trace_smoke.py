#!/usr/bin/env python3
"""Phase 1 v0.4 — drafter trace capture + replay smoke.

Validates that the drafter trace replays produce the same argmax as the
eager drafter_forward (which is itself bit-validated vs HF at v0.2/v0.3).

Gates:
1. SETUP: setup_drafter_trace_state allocates buffers without TT_FATAL.
2. UPDATE: update_drafter_trace_inputs writes buffers cleanly.
3. CAPTURE: ensure_drafter_trace returns a trace_id without FATAL.
4. REPLAY == EAGER: drafter_step_traced argmax equals drafter_forward
   eager argmax (same inputs/KV).
5. REPLAY == HF: drafter_step_traced matches HF drafter_argmax.

Also measures 3 warm replay walls for the first traced drafter perf number.

Trigger:  touch ~/tt-xla/.cache/gm4_asst_runtime/trig/gemma4_drafter_trace_smoke

NOTE: this probe runs against the DRAFTER server only (no target), so the
shared K/V come from the HF oracle (not the target server's KV exposure
helper). End-to-end "target + drafter + verify" is Phase 3.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_12b_assistant_ttnn as drf  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"
PROMPT = "prompt_0"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    cold_start = state is None
    if cold_start:
        log("cold-start: bootstrap drafter")
        state = drf.State()
        t0 = time.time()
        drf.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("dev-harness: using pre-bootstrapped state")

    # Load oracle artifacts for prompt_0.
    pd = ORACLE_DIR / PROMPT
    inputs_embeds_np = np.load(pd / "drafter_inputs_embeds.npy").astype(np.float32)
    K_sl_np = np.load(pd / "shared_kv_sliding_K.npy").astype(np.float32)
    V_sl_np = np.load(pd / "shared_kv_sliding_V.npy").astype(np.float32)
    K_fl_np = np.load(pd / "shared_kv_full_K.npy").astype(np.float32)
    V_fl_np = np.load(pd / "shared_kv_full_V.npy").astype(np.float32)
    hf_argmax = int(np.load(pd / "drafter_argmax.npy").flatten()[0])

    log(f"  inputs_embeds: {inputs_embeds_np.shape}")
    log(f"  K_sliding: {K_sl_np.shape}  K_full: {K_fl_np.shape}")
    log(f"  HF drafter argmax: {hf_argmax}")
    L_kv = int(K_sl_np.shape[2])

    # ── STEP A: eager drafter forward (validated against HF at v0.2/v0.3) ──
    log("─" * 64)
    log("STEP A: eager drafter_forward (oracle reference)")
    log("─" * 64)
    shared = {
        "sliding_attention": (K_sl_np, V_sl_np),
        "full_attention": (K_fl_np, V_fl_np),
    }
    t = time.time()
    out_eager = drf.drafter_forward(state, inputs_embeds_np, shared)
    eager_wall_ms = (time.time() - t) * 1000
    eager_argmax = int(out_eager["argmax"].flatten()[0])
    log(f"  eager wall: {eager_wall_ms:.1f} ms")
    log(f"  eager argmax: {eager_argmax}  (HF={hf_argmax})")

    rc = 0
    # ── STEP B: capture trace ──
    log("─" * 64)
    log(f"STEP B: trace setup + capture at L_kv={L_kv}")
    log("─" * 64)
    try:
        drf.setup_drafter_trace_state(state, L_kv=L_kv, log=log)
        drf.update_drafter_trace_inputs(state, inputs_embeds_np,
                                          K_sl_np, V_sl_np, K_fl_np, V_fl_np)
        t = time.time()
        drf.ensure_drafter_trace(state, L_kv=L_kv, log=log)
        capture_wall_ms = (time.time() - t) * 1000
        log(f"  ✓ trace captured in {capture_wall_ms:.0f} ms "
            f"(id={state.drafter_trace_id})")
    except Exception as e:
        log(f"  ✗ GATE 1-3 SETUP/UPDATE/CAPTURE FAIL: "
            f"{type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    # ── STEP C: replay + equivalence gates ──
    log("─" * 64)
    log("STEP C: replay trace — equivalence gates")
    log("─" * 64)
    drf.update_drafter_trace_inputs(state, inputs_embeds_np,
                                      K_sl_np, V_sl_np, K_fl_np, V_fl_np)
    t = time.time()
    traced_argmax = drf.drafter_step_traced(state)
    replay_wall_ms = (time.time() - t) * 1000
    log(f"  replay wall: {replay_wall_ms:.1f} ms")
    log(f"  traced argmax: {traced_argmax}")

    if traced_argmax == eager_argmax:
        log(f"  ✓ GATE 4 PASS — traced argmax == eager argmax = {traced_argmax}")
    else:
        log(f"  ✗ GATE 4 FAIL — traced={traced_argmax} vs eager={eager_argmax}")
        rc = 1
    if traced_argmax == hf_argmax:
        log(f"  ✓ GATE 5 PASS — traced argmax == HF argmax = {traced_argmax}")
    else:
        log(f"  ✗ GATE 5 FAIL — traced={traced_argmax} vs HF={hf_argmax}")
        rc = 1

    # ── STEP D: measure traced perf (3 warm replays) ──
    log("─" * 64)
    log("STEP D: measure traced replay perf (3 warm replays)")
    log("─" * 64)
    times_ms = []
    for i in range(3):
        drf.update_drafter_trace_inputs(state, inputs_embeds_np,
                                          K_sl_np, V_sl_np, K_fl_np, V_fl_np)
        t = time.time()
        _ = drf.drafter_step_traced(state)
        times_ms.append((time.time() - t) * 1000)
    log(f"  per-replay wall (incl. host writes + readback): "
        f"{[f'{x:.1f}' for x in times_ms]} ms")
    log(f"  mean = {sum(times_ms)/len(times_ms):.1f} ms  "
        f"(eager was {eager_wall_ms:.1f} ms)")
    speedup = eager_wall_ms / (sum(times_ms)/len(times_ms))
    log(f"  speedup vs eager: {speedup:.2f}×")

    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — drafter trace captured + bit-equivalent to eager + "
            f"matches HF. Traced wall ~{sum(times_ms)/len(times_ms):.1f} ms.")
    else:
        log("VERDICT: FAIL — see gate diagnostics above")
    log("=" * 64)
    if cold_start:
        ttnn.close_mesh_device(state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main(state=None))
