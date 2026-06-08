#!/usr/bin/env python3
"""Phase 2.B.1.4 — global-attention B=K+1 verify variant isolation gate.

Mirror of `gemma4_sliding_kp1_probe.py` but exercises the global-attention
layer (NKV=1, single SDPA call, head_dim=512, p-RoPE).

Same gates:
1. PLUMBING: kp1 forward returns [Bv, HIDDEN] non-NaN.
2. INVARIANCE: K+1 IDENTICAL h_norm rows → K+1 IDENTICAL output rows.
3. SENSITIVITY: K+1 DISTINCT h_norm rows → outputs that differ row-to-row.

Trigger:  touch ~/tt-xla/.cache/gm4_runtime/trig/gemma4_global_kp1_probe
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"

K = 5
Bv = K + 1
GATE_INVARIANCE_COS = 0.9999
GATE_SENSITIVITY_MAX_COS = 0.999


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def upload_h_replicated(h_np, mesh):
    return ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def readback_h_out(t):
    arr = ttnn.to_torch(
        t,
        mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0),
    ).float().numpy()
    if arr.ndim == 3:
        return arr[0]  # [Bv, HIDDEN]
    return arr


def main(state=None):
    cold_start = state is None
    if cold_start:
        log("cold-start path: bootstrap")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("dev-harness path: using pre-bootstrapped state")

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    L_prefill = int(prompt_ids.shape[0])
    cur_pos_val = L_prefill - 1
    if not getattr(state, "_kp1_probe_prefilled", False):
        log(f"prefill {L_prefill}-token canonical prompt (cold)")
        t = time.time()
        for pos in range(L_prefill):
            tok = int(prompt_ids[pos])
            srv.step_forward_v031(state, tok_id=tok, pos=pos)
        log(f"  prefill wall: {(time.time()-t)*1000:.1f} ms")
        state._kp1_probe_prefilled = True
    else:
        log(f"re-using cached prefill from prior probe run")
        srv._set_pos(state, cur_pos_val)
    log(f"  cur_pos = {cur_pos_val}")

    srv.setup_verify_kp1_state(state, K=K, log=log)
    srv.update_verify_inputs(state, current_pos=cur_pos_val,
                             candidate_token_ids=[0] * Bv)

    # Pick the FIRST global-attention layer.
    layer_idx = next(
        i for i in range(srv.NUM_LAYERS)
        if state.layer_types[i] == "full_attention"
    )
    log(f"  layer_idx = {layer_idx} (first global layer)")
    w = state.per_layer_tt[layer_idx]

    rope_cache = srv._compute_rope_for_forward(state)
    rope_gl = (rope_cache[2], rope_cache[3])

    rc = 0
    # GATE 1 — invariance
    log("─" * 64)
    log("GATE 1: B=Kp1 with K+1 IDENTICAL h_norm rows → all outputs equal")
    log("─" * 64)
    rng = np.random.default_rng(seed=42)
    h_one = rng.normal(0, 0.5, (1, srv.HIDDEN)).astype(np.float32)
    h_identical_np = np.tile(h_one, (Bv, 1))

    h_identical_tt = upload_h_replicated(h_identical_np, state.mesh)
    t = time.time()
    out_identical_tt = srv._layer_pos0_global_paged_kp1(
        state, h_identical_tt, w, layer_idx, rope=rope_gl)
    ttnn.synchronize_device(state.mesh)
    log(f"  kp1 forward (identical inputs) wall: {(time.time()-t)*1000:.1f} ms")

    out_identical_np = readback_h_out(out_identical_tt)
    ttnn.deallocate(out_identical_tt); ttnn.deallocate(h_identical_tt)

    log(f"  output shape = {out_identical_np.shape}")
    if np.isnan(out_identical_np).any():
        log("  ✗ output contains NaN — FAIL"); return 1
    log(f"  output mean={out_identical_np.mean():.4f} "
        f"std={out_identical_np.std():.4f} "
        f"max|x|={np.abs(out_identical_np).max():.4f}")

    log("  pairwise cos between output rows (expect ≥ "
        f"{GATE_INVARIANCE_COS:.4f}):")
    bad_pairs = []
    for i in range(Bv):
        for j in range(i + 1, Bv):
            c = cos(out_identical_np[i], out_identical_np[j])
            sym = "✓" if c >= GATE_INVARIANCE_COS else "✗"
            log(f"    {sym} row {i} vs row {j}: cos = {c:.6f}")
            if c < GATE_INVARIANCE_COS:
                bad_pairs.append((i, j, c))
    if bad_pairs:
        log(f"  ✗ GATE 1 FAIL — {len(bad_pairs)} row-pairs below {GATE_INVARIANCE_COS}")
        rc = 1
    else:
        log(f"  ✓ GATE 1 PASS — all {Bv*(Bv-1)//2} row-pairs identical")

    # GATE 2 — sensitivity
    log("─" * 64)
    log("GATE 2: B=Kp1 with K+1 DISTINCT h_norm rows → outputs DIFFER")
    log("─" * 64)
    h_distinct_np = rng.normal(0, 0.5, (Bv, srv.HIDDEN)).astype(np.float32)
    h_distinct_tt = upload_h_replicated(h_distinct_np, state.mesh)
    t = time.time()
    out_distinct_tt = srv._layer_pos0_global_paged_kp1(
        state, h_distinct_tt, w, layer_idx, rope=rope_gl)
    ttnn.synchronize_device(state.mesh)
    log(f"  kp1 forward (distinct inputs) wall: {(time.time()-t)*1000:.1f} ms")

    out_distinct_np = readback_h_out(out_distinct_tt)
    ttnn.deallocate(out_distinct_tt); ttnn.deallocate(h_distinct_tt)

    if np.isnan(out_distinct_np).any():
        log("  ✗ output contains NaN — FAIL"); return 1
    log(f"  output mean={out_distinct_np.mean():.4f} "
        f"std={out_distinct_np.std():.4f} "
        f"max|x|={np.abs(out_distinct_np).max():.4f}")

    log("  pairwise cos between output rows (expect at LEAST ONE pair < "
        f"{GATE_SENSITIVITY_MAX_COS}):")
    max_cos = -1.0
    distinct_pair_count = 0
    for i in range(Bv):
        for j in range(i + 1, Bv):
            c = cos(out_distinct_np[i], out_distinct_np[j])
            log(f"    row {i} vs row {j}: cos = {c:.6f}")
            max_cos = max(max_cos, c)
            if c < GATE_SENSITIVITY_MAX_COS:
                distinct_pair_count += 1
    log(f"  max pairwise cos = {max_cos:.6f}  distinct pairs = "
        f"{distinct_pair_count}/{Bv*(Bv-1)//2}")
    if distinct_pair_count == 0:
        log(f"  ✗ GATE 2 FAIL — all row-pairs cos ≥ {GATE_SENSITIVITY_MAX_COS}")
        rc = 1
    else:
        log(f"  ✓ GATE 2 PASS — distinct inputs produce distinct outputs")

    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — global kp1 fork plumbing + invariance + sensitivity all gate-clean")
    else:
        log("VERDICT: FAIL — see gate diagnostics above")
    log("=" * 64)
    if cold_start:
        ttnn.close_mesh_device(state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main(state=None))
