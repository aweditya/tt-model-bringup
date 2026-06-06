#!/usr/bin/env python3
"""MM7 v0.4.0f — per-layer-kind decode timing on the post-fold path.

v0.4.0e shrank decode from 15.5s → 0.6-0.7s warm (matmul-fold for
ttnn.conv1d). Where does the remaining 0.7s live?

This probe runs a small chain of warm decode steps and times each of
the 52 layers with ttnn.synchronize_device in between (so we measure
on-device work, not host-side latency). Aggregates by layer kind:

  • 23 mamba2 layers (post-fold)
  • 23 moe layers (Pattern-A or EP)
  • 6 attention layers (paged decode)

Reports:
  • Mean per-step kind time across N warm steps
  • % of total step
  • Predicted next-best lever

REUSE: forks v0.3.3.b warm_decode_perf; adds per-layer ttnn.synchronize
+ per-kind aggregation. Drops HF comparison (perf only).

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v040f_layer_kind_profile'
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
N_WARM_STEPS = 4  # decode-step samples after warmup


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _forward_layers_timed(state, h_tt, srv, ttnn, *, attn_fn_name: str,
                          per_layer_ms: list[float]):
    attn_fn = getattr(srv, attn_fn_name)
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        ttnn.synchronize_device(state.mesh)
        t0 = time.time()
        if kind == "attention":
            h_next_tt = attn_fn(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        ttnn.synchronize_device(state.mesh)
        per_layer_ms[L] += (time.time() - t0) * 1000.0
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

    own_state = state is None
    t_boot = 0.0
    if state is None:
        log("bootstrap…")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log)
        t_boot = time.time() - t0
        log(f"  bootstrap done in {t_boot:.1f}s")
    else:
        log("[harness] reusing live State ✓")

    def _embed_to_tt(ids_2d):
        h_np = srv.embed_lookup(state, ids_2d)
        return ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    def _step_to_tok(h_tt):
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        return int(argmax_np.flatten()[-1])

    srv.reset_decode_state(state, B=1, log=log)

    # ── PREFILL prompt ────────────────────────────────────────────────
    log("PREFILL…")
    t_pre = time.time()
    h_tt = _embed_to_tt(prompt_ids[None, :])
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next_tt = srv.attn_prefill_tt(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        ttnn.deallocate(h_tt)
        h_tt = h_next_tt
    prev_tok = _step_to_tok(h_tt)
    state.cur_pos = len(prompt_ids)
    log(f"PREFILL done in {time.time() - t_pre:.1f}s  argmax={prev_tok}")

    # ── WARMUP DECODE (1 throw-away step to trigger JIT) ──────────────
    log("WARMUP step (JIT cold)…")
    t0 = time.time()
    # cur_pos_buf now caller-updated (moved out of attn_decode_step_tt
    # for trace compatibility).
    srv.update_cur_pos_buf(state, int(state.cur_pos))
    h_tt = _embed_to_tt(np.asarray([[prev_tok]], dtype=np.int64))
    dummy_ms = [0.0] * N_LAYERS
    h_tt = _forward_layers_timed(
        state, h_tt, srv, ttnn,
        attn_fn_name="attn_decode_step_tt", per_layer_ms=dummy_ms,
    )
    prev_tok = _step_to_tok(h_tt)
    state.cur_pos += 1
    log(f"  warmup step in {time.time() - t0:.1f}s  TT={prev_tok}")

    # ── WARM DECODE PROFILE ───────────────────────────────────────────
    log(f"profiling {N_WARM_STEPS} warm decode steps…")
    per_layer_ms_acc = [0.0] * N_LAYERS
    step_times = []
    for s in range(N_WARM_STEPS):
        t0 = time.time()
        srv.update_cur_pos_buf(state, int(state.cur_pos))
        h_tt = _embed_to_tt(np.asarray([[prev_tok]], dtype=np.int64))
        h_tt = _forward_layers_timed(
            state, h_tt, srv, ttnn,
            attn_fn_name="attn_decode_step_tt",
            per_layer_ms=per_layer_ms_acc,
        )
        prev_tok = _step_to_tok(h_tt)
        state.cur_pos += 1
        step_times.append(time.time() - t0)
        log(f"  step {s}: {step_times[-1]*1000:.0f} ms  TT={prev_tok}")

    # ── AGGREGATE BY KIND ─────────────────────────────────────────────
    kind_sum_ms = {"attention": 0.0, "mamba2": 0.0, "moe": 0.0}
    kind_count = {"attention": 0, "mamba2": 0, "moe": 0}
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        kind_sum_ms[kind] += per_layer_ms_acc[L]
        kind_count[kind] += 1
    # Convert to per-step (divide by N_WARM_STEPS), then per-layer-mean.
    log("")
    log("=" * 60)
    log("REPORT — per-step time by layer kind (mean across N warm)")
    log("=" * 60)
    mean_step_ms = sum(step_times) / len(step_times) * 1000.0
    log(f"  total step (wall clock): {mean_step_ms:.0f} ms")
    summed_layer_ms = 0.0
    for kind in ("attention", "mamba2", "moe"):
        per_step_kind_ms = kind_sum_ms[kind] / N_WARM_STEPS
        per_layer_mean = per_step_kind_ms / max(1, kind_count[kind])
        pct = per_step_kind_ms / mean_step_ms * 100
        log(f"  {kind:<10s}  {kind_count[kind]:>2d} layers  "
            f"total={per_step_kind_ms:>6.1f} ms  "
            f"per-layer={per_layer_mean:>5.1f} ms  "
            f"share={pct:>4.1f}%")
        summed_layer_ms += per_step_kind_ms
    overhead = mean_step_ms - summed_layer_ms
    log(f"  embed+lm_head+sample overhead: {overhead:.1f} ms "
        f"({overhead / mean_step_ms * 100:.1f}%)")
    log("")
    log("NEXT LEVER — biggest share is the next thing to attack.")
    log("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
