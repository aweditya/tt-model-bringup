#!/usr/bin/env python3
"""Phase 2.B.1.3 — sliding-attention B=K+1 verify variant isolation gate.

Validates `_layer_pos0_sliding_paged_kp1` in isolation:
1. PLUMBING: forward at Bv=K+1 returns shape [Bv, HIDDEN] non-NaN.
2. INVARIANCE: K+1 IDENTICAL h_norm rows → K+1 IDENTICAL output rows
   (cos ≥ 0.9999 between any two rows). Proves per-row computation is
   independent (no row-to-row leakage).
3. SENSITIVITY: K+1 DISTINCT h_norm rows → outputs that DIFFER row to row
   (row-pair cos < 0.999 between distinct rows). Proves the forward is
   functional in h_norm, not a constant.

The strong "per-row equals independent B=1 forward" gate runs at
Step #267 (end-to-end smoke after the full forward is built), where we can
do a clean fresh-bootstrap A/B without cache-mutation conflicts.

Run on qb1 via dev harness:
  touch ~/tt-xla/.cache/gm4_runtime/trig/gemma4_sliding_kp1_probe
or direct:
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \
      experiments/cb/isolate/gemma4_sliding_kp1_probe.py'
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
GATE_SENSITIVITY_MAX_COS = 0.999  # row pairs must NOT all be > this


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def upload_h_replicated(h_np, mesh):
    """Upload [Bv, HIDDEN] float numpy as bf16 replicated across mesh.

    Production h_norm is replicated across chips (each chip sees the full
    HIDDEN-wide activation; weights are column-sharded). Mirror that here.
    """
    return ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def readback_h_out(t):
    """Read [Bv, HIDDEN] output back to numpy fp32. The matmul + all_reduce
    output is replicated across mesh after all_reduce, so we read chip 0.
    """
    return ttnn.to_torch(
        t,
        mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0),
    ).float().numpy()[0:Bv]  # take first chip's view (all are identical post-AR)


def run(state, log=log):
    """Probe entrypoint. Accepts pre-bootstrapped state (dev harness) or
    None (cold bootstrap + prefill).
    """
    cold_start = state is None
    if cold_start:
        log("cold-start path: bootstrap + prefill 6-token canonical prompt")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
        # Prefill canonical 6-token prompt to populate cache slots 0..5.
        prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
        L_prefill = int(prompt_ids.shape[0])
        log(f"prefill {L_prefill} tokens")
        for pos in range(L_prefill):
            tok = int(prompt_ids[pos])
            srv.step_forward_v031(state, tok_id=tok, pos=pos)
    else:
        log("dev-harness path: assuming state has been bootstrapped + prefilled")

    # The probe needs cur_pos in [0, MAX_KV). Reading the current
    # cur_pos_buf value to know what position the cache reflects.
    cur_pos_val = int(ttnn.to_torch(
        state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(state.cur_pos_buf.device(), dim=0),
    ).flatten()[0].item())
    log(f"  cur_pos (post-prefill) = {cur_pos_val}")

    # Ensure verify-state is set up at K=5.
    srv.setup_verify_kp1_state(state, K=K, log=log)

    # All K+1 verify candidates verify AT cur_pos_val (the position the
    # cache reflects). The candidate_token_ids are irrelevant for the
    # sliding-layer probe (tok_buf is only read by embed at the start of
    # the full forward — not by the per-layer kp1 fork itself).
    srv.update_verify_inputs(state, current_pos=cur_pos_val,
                             candidate_token_ids=[0] * Bv)

    # Pick the FIRST sliding-attention layer.
    layer_idx = next(
        i for i in range(srv.NUM_LAYERS)
        if state.layer_types[i] == "sliding_attention"
    )
    log(f"  layer_idx = {layer_idx} (first sliding layer)")
    w = state.per_layer_tt[layer_idx]

    # Compute rope cache once (forward-scoped optimization; matches production).
    rope_cache = srv._compute_rope_for_forward(state)
    rope_sl = (rope_cache[0], rope_cache[1])

    # ───────────── GATE 1: PLUMBING + INVARIANCE ─────────────
    log("─" * 64)
    log("GATE 1: B=Kp1 with K+1 IDENTICAL h_norm rows → all outputs equal")
    log("─" * 64)
    rng = np.random.default_rng(seed=42)
    # Single random h_norm row, tiled K+1 times. Use HIDDEN (not HIDDEN_PER_CHIP)
    # because h_norm is REPLICATED across chips (each chip sees full HIDDEN).
    h_one = rng.normal(0, 0.5, (1, srv.HIDDEN)).astype(np.float32)
    h_identical_np = np.tile(h_one, (Bv, 1))  # [Bv, HIDDEN]
    log(f"  h_identical shape = {h_identical_np.shape} (all {Bv} rows == row 0)")

    h_identical_tt = upload_h_replicated(h_identical_np, state.mesh)
    t = time.time()
    out_identical_tt = srv._layer_pos0_sliding_paged_kp1(
        state, h_identical_tt, w, layer_idx, rope=rope_sl)
    ttnn.synchronize_device(state.mesh)
    log(f"  kp1 forward (identical inputs) wall: {(time.time()-t)*1000:.1f} ms")

    out_identical_np = readback_h_out(out_identical_tt)  # [Bv, HIDDEN]
    ttnn.deallocate(out_identical_tt); ttnn.deallocate(h_identical_tt)

    log(f"  output shape = {out_identical_np.shape}")
    if np.isnan(out_identical_np).any():
        log("  ✗ output contains NaN — FAIL")
        return 1
    log(f"  output mean={out_identical_np.mean():.4f} "
        f"std={out_identical_np.std():.4f} "
        f"max|x|={np.abs(out_identical_np).max():.4f}")

    # All Bv rows must be identical (or near-identical post-bf16).
    rc = 0
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
        log(f"  ✗ GATE 1 FAIL — {len(bad_pairs)} row-pairs below "
            f"{GATE_INVARIANCE_COS:.4f}")
        rc = 1
    else:
        log(f"  ✓ GATE 1 PASS — all {Bv*(Bv-1)//2} row-pairs identical")

    # ───────────── GATE 2: SENSITIVITY (different Q → different out) ─────────────
    log("─" * 64)
    log("GATE 2: B=Kp1 with K+1 DISTINCT h_norm rows → outputs DIFFER")
    log("─" * 64)
    h_distinct_np = rng.normal(0, 0.5, (Bv, srv.HIDDEN)).astype(np.float32)
    log(f"  h_distinct shape = {h_distinct_np.shape} (all rows independent)")

    h_distinct_tt = upload_h_replicated(h_distinct_np, state.mesh)
    t = time.time()
    out_distinct_tt = srv._layer_pos0_sliding_paged_kp1(
        state, h_distinct_tt, w, layer_idx, rope=rope_sl)
    ttnn.synchronize_device(state.mesh)
    log(f"  kp1 forward (distinct inputs) wall: {(time.time()-t)*1000:.1f} ms")

    out_distinct_np = readback_h_out(out_distinct_tt)
    ttnn.deallocate(out_distinct_tt); ttnn.deallocate(h_distinct_tt)

    if np.isnan(out_distinct_np).any():
        log("  ✗ output contains NaN — FAIL"); rc = 1; return rc
    log(f"  output mean={out_distinct_np.mean():.4f} "
        f"std={out_distinct_np.std():.4f} "
        f"max|x|={np.abs(out_distinct_np).max():.4f}")

    log("  pairwise cos between output rows (expect at LEAST ONE pair < "
        f"{GATE_SENSITIVITY_MAX_COS:.3f}):")
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
        log(f"  ✗ GATE 2 FAIL — all row-pairs cos ≥ {GATE_SENSITIVITY_MAX_COS}; "
            f"forward may be ignoring h_norm inputs")
        rc = 1
    else:
        log(f"  ✓ GATE 2 PASS — distinct inputs produce distinct outputs")

    # ───────────── VERDICT ─────────────
    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — sliding kp1 fork plumbing + invariance + "
            "sensitivity all gate-clean")
    else:
        log("VERDICT: FAIL — see gate diagnostics above")
    log("=" * 64)
    if cold_start:
        ttnn.close_mesh_device(state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(run(state=None))
