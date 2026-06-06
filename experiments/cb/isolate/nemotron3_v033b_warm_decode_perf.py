#!/usr/bin/env python3
"""MM7 v0.3.3.b — JIT-warm decode step perf characterization.

Tells us whether eager+warm decode is already demo-tractable, or
whether v0.4 trace capture is needed for the perf gate.

Pipeline:
  1. Reset state
  2. Prefill 5-token prompt (1× cold JIT)
  3. Run 5 decode steps back-to-back; report time per step.
     • step 0: JIT cold (new S=1 shape, expect ~15s)
     • steps 1..4: JIT warm (cached); ms-scale if compute dominates

Outputs:
  • Per-step time
  • Eager-warm tok/s
  • Recommendation: trace vs eager-warm shipping

REUSE: forks `nemotron3_v033_nstep_chain_smoke.py`; drops HF comparison
since we're characterizing perf, not correctness.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v033b_warm_decode_perf'
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
N_LAYERS = 52
N_STEPS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _forward_layers(state, h_tt, srv, ttnn, *, attn_fn_name: str):
    attn_fn = getattr(srv, attn_fn_name)
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next_tt = attn_fn(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        ttnn.deallocate(h_tt)
        h_tt = h_next_tt
    return h_tt


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    prompt_ids = np.asarray(meta["prompt_ids"], dtype=np.int64)
    log(f"prompt ({len(prompt_ids)}): {prompt_ids.tolist()}")

    import server_nemotron3_nano_ttnn as srv
    import ttnn
    t_boot = 0.0
    if state is None:
        log("bootstrap…")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log)
        t_boot = time.time() - t0
        log(f"  bootstrap in {t_boot:.1f}s")
    else:
        log("[harness] reusing live state ✓")

    try:
        log("reset_decode_state…")
        srv.reset_decode_state(state, B=1, log=log)

        # ── PREFILL ────────────────────────────────────────────────
        log("PREFILL…")
        t0 = time.time()
        h_np = srv.embed_lookup(state, prompt_ids[None, :])
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt = _forward_layers(state, h_tt, srv, ttnn,
                               attn_fn_name="attn_prefill_tt")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        prefill_t = time.time() - t0
        prev_token = int(argmax_np.flatten()[-1])
        log(f"  prefill in {prefill_t:.2f}s  argmax={prev_token}")

        state.cur_pos = len(prompt_ids)

        # ── WARM DECODE LOOP ───────────────────────────────────────
        log(f"DECODE perf loop ({N_STEPS} steps):")
        step_times = []
        for step in range(N_STEPS):
            t0 = time.time()
            h_np_dec = srv.embed_lookup(
                state, np.asarray([[prev_token]], dtype=np.int64),
            )
            h_tt = ttnn.from_torch(
                torch.from_numpy(h_np_dec.astype(np.float32)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            h_tt = _forward_layers(state, h_tt, srv, ttnn,
                                   attn_fn_name="attn_decode_step_tt")
            h_np = ttnn.to_torch(
                h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )[:1].float().numpy()
            ttnn.deallocate(h_tt)
            h_final = srv.apply_final_norm(state, h_np)
            _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
            tt_next = int(argmax_np.flatten()[-1])
            elapsed = time.time() - t0
            step_times.append(elapsed)
            log(f"  step {step}  TT={tt_next:>6d}  "
                f"time={elapsed:.3f}s  "
                f"tok/s={1.0/elapsed:.3f}  "
                f"{'COLD' if step == 0 else 'WARM'}")
            state.cur_pos += 1
            prev_token = tt_next

        # ── REPORT ────────────────────────────────────────────────
        log("")
        log("=" * 60)
        log("PERF REPORT")
        log("=" * 60)
        cold = step_times[0]
        warm_avg = sum(step_times[1:]) / max(1, len(step_times) - 1)
        log(f"  cold step 0:    {cold:.2f}s ({1.0/cold:.3f} tok/s)")
        log(f"  warm mean:      {warm_avg:.3f}s ({1.0/warm_avg:.3f} tok/s)")
        log(f"  JIT overhead:   cold - warm = {cold - warm_avg:.2f}s")
        log(f"  cold/warm:      {cold/warm_avg:.1f}× slowdown")
        log("")
        if warm_avg < 1.0:
            log(f"  → WARM EAGER < 1s/step ({1.0/warm_avg:.1f} tok/s).")
            log(f"    Trace may give another ~3× but eager is already demo-tractable.")
        elif warm_avg < 5.0:
            log(f"  → WARM EAGER {warm_avg:.1f}s/step. Trace would unlock big perf win.")
        else:
            log(f"  → WARM EAGER still {warm_avg:.1f}s/step. Trace is REQUIRED for any demo.")
        return 0
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
