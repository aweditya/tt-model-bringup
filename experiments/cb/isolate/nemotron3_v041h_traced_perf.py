#!/usr/bin/env python3
"""MM7 v0.4.1.h — traced decode perf measurement (router on-device).

Goal: measure the steady-state traced decode ms/tok with
NM3_ROUTER_ON_DEVICE=1. User accepts the tie-break drift for chatbot /
coding use cases (sampling-temperature production matches DeepSeek-V3
demo). This probe only reports speed — not correctness.

Pipeline (forks v0.4.1.a structure):
  1. Bootstrap state (reuse harness's live mesh).
  2. Prefill the oracle prompt eagerly.
  3. Run 2 eager warmup decode steps (JIT-prime inner shapes,
     per [[ttnn-multi-trace-two-phase-warmup]]).
  4. Capture trace of one decode step via pure-ttnn embed/forward/
     final_norm/lm_head/argmax variants (all 4 blockers cleared
     when router is on-device).
  5. Replay 30 times, measure per-step wall time, report
     mean / median / p95 ms/tok.

REUSE:
  - v0.4.1.a probe (mechanics + trace blockers documented there)
  - 27B trace pattern at server_tp.py:2143-2185
  - Pure-ttnn helpers from server_nemotron3_nano_ttnn.py (commit f45a710)

Run via the nm3 dev harness:
  NM3_ROUTER_ON_DEVICE=1 set on launch
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v041h_traced_perf'
"""
from __future__ import annotations

import json
import os
import statistics
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
N_WARMUP_STEPS = 2
N_TRACED_STEPS = 30  # enough samples for mean / median / p95


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
    os.environ.setdefault("NM3_ROUTER_ON_DEVICE", "1")  # REQUIRED for trace

    log(f"NM3_ROUTER_ON_DEVICE={os.environ.get('NM3_ROUTER_ON_DEVICE')}")
    log(f"N_WARMUP_STEPS={N_WARMUP_STEPS}  N_TRACED_STEPS={N_TRACED_STEPS}")

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

    # ── 2 WARMUP EAGER STEPS (JIT-prime) ─────────────────────────────
    log("WARMUP eager (2 steps, JIT-prime inner shapes)…")
    warm_token = prev_token
    for w in range(N_WARMUP_STEPS):
        t_w = time.time()
        # update cur_pos_buf BEFORE attn_decode_step_tt (host write moved
        # out of the function for trace compatibility — fork of 35B pattern)
        srv.update_cur_pos_buf(state, int(state.cur_pos))
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[warm_token]], dtype=np.int64),
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
        warm_token = int(argmax_np.flatten()[-1])
        state.cur_pos += 1
        log(f"  warmup eager step {w}  TT={warm_token}  ({(time.time()-t_w)*1e3:.0f} ms)")

    # ── CAPTURE TRACE ────────────────────────────────────────────────
    log("CAPTURE TRACE of one decode step (pure-ttnn path)…")
    # Both host buffers updated OUTSIDE capture; trace reads them.
    srv.update_tok_buf(state, warm_token)
    srv.update_cur_pos_buf(state, int(state.cur_pos))
    t_cap = time.time()
    try:
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        h_tt_in = srv.embed_lookup_tt(state)
        h_tt_out = _forward_layers(
            state, h_tt_in, srv, ttnn, attn_fn_name="attn_decode_step_tt",
        )
        h_norm_tt = srv.apply_final_norm_tt(state, h_tt_out)
        ttnn.deallocate(h_tt_out)
        logits_tt, argmax_tt = srv.apply_lm_head_argmax_tt(state, h_norm_tt)
        ttnn.deallocate(h_norm_tt)
        ttnn.deallocate(logits_tt)
        state.traced_argmax_tt = argmax_tt
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        log(f"  ✓ trace captured (id={trace_id}) in {time.time()-t_cap:.1f}s")
    except Exception as e:
        log(f"  ✗ trace CAPTURE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        except Exception:
            pass
        return 1

    # ── REPLAY N TIMES MEASURING TIME ────────────────────────────────
    log(f"REPLAY trace {N_TRACED_STEPS} times (measuring per-step time)…")
    step_ms = []
    traced_tokens = []
    cur_token = warm_token
    try:
        for s in range(N_TRACED_STEPS):
            srv.update_tok_buf(state, cur_token)
            srv.update_cur_pos_buf(state, int(state.cur_pos))
            t_s = time.time()
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=True)
            argmax_torch = ttnn.to_torch(
                state.traced_argmax_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )
            tok = int(argmax_torch[0].flatten()[-1])
            dt_ms = (time.time() - t_s) * 1e3
            step_ms.append(dt_ms)
            traced_tokens.append(tok)
            state.cur_pos += 1
            cur_token = tok
            if s < 5 or s % 5 == 0:
                log(f"  traced step {s:2d}  TT={tok}  ({dt_ms:.1f} ms)")
    except Exception as e:
        log(f"  ✗ trace REPLAY FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        ttnn.release_trace(state.mesh, trace_id)
        return 2

    ttnn.release_trace(state.mesh, trace_id)

    # ── REPORT ───────────────────────────────────────────────────────
    # Drop first 3 as cold-cache (trace replay also warms allocator).
    steady = step_ms[3:] if len(step_ms) > 3 else step_ms
    mean_ms = statistics.mean(steady)
    median_ms = statistics.median(steady)
    p95_ms = sorted(steady)[int(0.95 * len(steady))]
    tok_per_s = 1000.0 / mean_ms

    log("")
    log("=" * 60)
    log("REPORT")
    log("=" * 60)
    log(f"  all step times (ms): {[f'{x:.1f}' for x in step_ms]}")
    log(f"  steady (drop first 3): n={len(steady)}")
    log(f"  mean   = {mean_ms:.1f} ms/tok")
    log(f"  median = {median_ms:.1f} ms/tok")
    log(f"  p95    = {p95_ms:.1f} ms/tok")
    log(f"  → throughput = {tok_per_s:.2f} tok/s")
    log(f"  eager baseline (no trace) = 260 ms/tok ≈ 3.8 tok/s")
    log(f"  speedup vs eager: {260.0 / mean_ms:.2f}×")
    log("")
    log(f"  decoded tokens: {traced_tokens}")
    log("  NOTE: tokens may be gibberish (router on-device tie-break")
    log("        drift). User accepts this for sampling-temperature")
    log("        chatbot use case — production matches DeepSeek-V3 demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
