#!/usr/bin/env python3
"""MM7 v0.4.1.a — trace capture diagnostic probe.

Goal: tell us EXACTLY which numpy roundtrips block trace capture on
the current Nemotron-3 decode path. Don't try to be smart — just
attempt the capture, observe failures, document each.

Pipeline:
  1. Bootstrap state (reuse harness's live mesh).
  2. Prefill prompt eagerly.
  3. Run 2 eager decode steps (warmup all JIT kernels for the inner
     step shape, per [[ttnn-multi-trace-two-phase-warmup]]).
  4. Record the eager argmax sequence (baseline).
  5. Attempt: capture trace of one decode step + replay it 4 times.
  6. Compare traced argmax sequence to eager.

Three possible outcomes:
  A. Trace CRASH during capture (kernel asserts on host op) → we know
     which op fails. Read the backtrace.
  B. Trace SUCCEEDS but replay tokens DIFFER from eager → host ops
     silently skipped (the trace captures only device ops; numpy
     bridges become no-ops on replay). Mismatch positions = which
     numpy ops mattered.
  C. Trace SUCCEEDS and tokens MATCH → no host ops in the captured
     path, we're already traceable. Measure replay time + ship trace.

REUSE:
  - 27B trace pattern at server_tp.py:2143-2185 (_ensure_decode_trace
    + _traced_forward).
  - Two-phase warmup rule [[ttnn-multi-trace-two-phase-warmup]].

NOTE on what we KNOW about the current path's numpy roundtrips
(audited 2026-06-05 post-v0.4.0h.a):
  * mamba2 layers: PURE TT after v0.4.0g.b — no roundtrips.
  * moe layers: 3 host ops:
      - ttnn.to_torch(scores_tt) → host argpartition topk + bias
      - ttnn.to_torch(h_input_tt) → re-upload sharded for dispatch
      - ttnn.from_torch(topk_indices) → upload as device tensor
  * apply_final_norm: takes h_np, returns h_np (full host roundtrip).
  * apply_lm_head_and_argmax: takes h_np, returns argmax_int (host op).
  * embed_lookup: takes ids_np, returns h_np (host roundtrip).

So the probe will likely show outcome B with significant drift, or A
if ttnn objects to a from_torch call inside a captured region.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v041a_trace_probe'
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
N_LAYERS = 52
N_TRACED_STEPS = 4
N_WARMUP_STEPS = 2


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

    if state is None:
        log("bootstrap…")
        state = srv.State()
        srv.bootstrap(state, log)
    else:
        log("[harness] reusing live state ✓")

    srv.reset_decode_state(state, B=1, log=log)

    # ── PREFILL ──────────────────────────────────────────────────────
    log("PREFILL…")
    t0 = time.time()
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_tt = _forward_layers(state, h_tt, srv, ttnn, attn_fn_name="attn_prefill_tt")
    h_np = ttnn.to_torch(
        h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    ttnn.deallocate(h_tt)
    h_final = srv.apply_final_norm(state, h_np)
    _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
    prev_token = int(argmax_np.flatten()[-1])
    log(f"  prefill in {time.time() - t0:.1f}s  prev_token={prev_token}")
    state.cur_pos = len(prompt_ids)

    # ── 1) RUN 2+N_TRACED EAGER STEPS, RECORD TOKENS ─────────────────
    log("EAGER baseline + warmup (2 + 4 steps)…")
    eager_tokens = []
    eager_token = prev_token
    for s in range(N_WARMUP_STEPS + N_TRACED_STEPS):
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[eager_token]], dtype=np.int64),
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
        eager_token = int(argmax_np.flatten()[-1])
        eager_tokens.append(eager_token)
        state.cur_pos += 1
        log(f"  eager step {s}  TT={eager_token}")

    eager_baseline = eager_tokens[N_WARMUP_STEPS:]  # last N_TRACED_STEPS
    log(f"eager baseline (post-warmup): {eager_baseline}")

    # ── 2) ATTEMPT TRACE CAPTURE OF ONE DECODE STEP ──────────────────
    # Reset state to re-run the same N steps but traced. Otherwise the
    # 2 warmup eager steps already advanced state.
    log("Resetting state for traced run…")
    srv.reset_decode_state(state, B=1, log=log)
    state.cur_pos = 0
    # Re-prefill to get back to the same starting position.
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_tt = _forward_layers(state, h_tt, srv, ttnn, attn_fn_name="attn_prefill_tt")
    h_np = ttnn.to_torch(
        h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    ttnn.deallocate(h_tt)
    h_final = srv.apply_final_norm(state, h_np)
    _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
    pre_traced_token = int(argmax_np.flatten()[-1])
    log(f"  re-prefill prev_token={pre_traced_token} "
        f"(expected {prev_token}: {'OK' if pre_traced_token == prev_token else 'MISMATCH'})")
    state.cur_pos = len(prompt_ids)

    # ── Warmup 2 eager steps (different start pos than baseline; we ──
    # ── just need JIT compiled for THIS shape). ──
    log("Re-warming up 2 eager steps…")
    traced_token = pre_traced_token
    for w in range(N_WARMUP_STEPS):
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[traced_token]], dtype=np.int64),
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
        traced_token = int(argmax_np.flatten()[-1])
        state.cur_pos += 1
    log(f"  post-warmup pre-trace prev_token={traced_token}")

    # ── 3) CAPTURE TRACE ─────────────────────────────────────────────
    log("CAPTURE TRACE of one decode step…")
    log("  NOTE: the existing decode forward has host bridges. We expect")
    log("        one of: (A) ttnn TT_FATAL during capture; (B) trace OK")
    log("        but replay tokens drift; (C) all match.")
    try:
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        # Decode forward — this is what we want to trace.
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[traced_token]], dtype=np.int64),
        )
        h_tt_in = ttnn.from_torch(
            torch.from_numpy(h_np_dec.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt_out = _forward_layers(
            state, h_tt_in, srv, ttnn, attn_fn_name="attn_decode_step_tt",
        )
        # KEEP h_tt_out as the traced output tensor handle; do NOT deallocate.
        state.traced_h_tt = h_tt_out
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        log(f"  ✓ trace captured (id={trace_id})")
    except Exception as e:
        log(f"  ✗ trace CAPTURE FAILED: {type(e).__name__}: {e}")
        log("  → outcome A: ttnn objects to a host op inside the captured")
        log("    region. Backtrace identifies the offender:")
        traceback.print_exc()
        return 1

    # ── 4) REPLAY N TIMES + COMPARE TO EAGER ─────────────────────────
    log(f"REPLAY trace {N_TRACED_STEPS} times…")
    traced_tokens = []
    cur_pos_at_trace = state.cur_pos
    try:
        for s in range(N_TRACED_STEPS):
            # NOTE: we'd normally update tok_buf/cur_pos_buf here. But the
            # current path takes tok via ttnn.from_torch INSIDE the
            # captured region. The replay will re-run that captured
            # from_torch with whatever data was there at capture time —
            # which is `traced_token` from capture. Wrong but informative.
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=True)
            # Readback the traced output tensor to get argmax.
            h_np_traced = ttnn.to_torch(
                state.traced_h_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )[:1].float().numpy()
            h_final = srv.apply_final_norm(state, h_np_traced)
            _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
            tok = int(argmax_np.flatten()[-1])
            traced_tokens.append(tok)
            state.cur_pos += 1
            log(f"  traced step {s}  TT={tok}  "
                f"eager={eager_baseline[s]}  "
                f"{'PASS' if tok == eager_baseline[s] else 'FAIL'}")
    except Exception as e:
        log(f"  ✗ trace REPLAY FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        ttnn.release_trace(state.mesh, trace_id)
        return 2

    ttnn.release_trace(state.mesh, trace_id)

    # ── REPORT ───────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("REPORT")
    log("=" * 60)
    matches = sum(1 for t, e in zip(traced_tokens, eager_baseline) if t == e)
    log(f"  eager baseline:  {eager_baseline}")
    log(f"  traced replay:   {traced_tokens}")
    log(f"  matches:         {matches}/{len(eager_baseline)}")
    log("")
    if matches == len(eager_baseline):
        log("  → OUTCOME C: trace captured cleanly. No host bridges in")
        log("    the captured region. We're ready for v0.4.1.b multi-trace.")
        return 0
    else:
        log("  → OUTCOME B: trace captured but replay drifts. Host bridges")
        log("    are silently no-op on replay. Each replay re-uses the data")
        log("    that was on-device at CAPTURE time, so cur_pos/token never")
        log("    advance correctly.")
        log("")
        log("  REMAINING NUMPY ROUNDTRIPS TO ELIMINATE (audited):")
        log("    - moe: scores readback for host topk + topk_indices upload")
        log("    - moe: h_input readback for sharded re-upload")
        log("    - embed_lookup: ids → np → ttnn.from_torch")
        log("    - apply_final_norm: takes np, returns np")
        log("    - apply_lm_head_and_argmax: takes np, returns argmax_int")
        log("")
        log("  Each must be replaced with pre-allocated buffers + on-device")
        log("  ops. See research/nemotron3_trace_plan_2026-06-05.md for the")
        log("  full plan.")
        return 0  # Diagnostic success; integration not yet ready.


if __name__ == "__main__":
    sys.exit(main())
