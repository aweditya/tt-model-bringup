#!/usr/bin/env python3
"""MM7 v0.4.1.e — reduce_scatter with CORRECT signature for replicate→shard.

v0.4.1.c probe used wrong kwargs (`math_op=ttnn.ReduceType.Sum`) — not
in ttnn.reduce_scatter's signature, so it hit TypeError. The actual
signature per the docstring (verified 2026-06-05) is:

  ttnn.reduce_scatter(input_tensor, dim, *, cluster_axis=None, ...)

Semantics:
  Reduce (sum across devices) along the cluster_axis, then scatter
  the reduced tensor back to devices along `dim`. Output shape
  [..., dim/num_devices, ...].

For our replicate→shard pattern:
  1. Take h_repl_tt with REPLICATED layout (all chips identical).
  2. Pre-scale by 1/NCHIPS — h_scaled = h/4 on each chip.
  3. reduce_scatter(h_scaled, dim=2, cluster_axis=1):
     - Reduce: sum across 4 chips = 4·(h/4) = h (back to original values)
     - Scatter: chip i gets h[..., i*S_per_chip:(i+1)*S_per_chip, ...]
  4. Result: SHARDED layout, original values. Exactly what we need.

This probe:
  1. Builds h_repl_tt [B=1, 1, S=4, HIDDEN=2688] REPLICATED bf16 TILE.
  2. Builds reference h_sharded_ref_tt via the host bridge
     (ShardTensorToMesh(dim=2)).
  3. Runs the on-device path: mul(1/N) + reduce_scatter.
  4. Compares the two reconstructed (each via ConcatMeshToTensor) at cos ≥ 0.999.

If PASS: trace blocker #3 is GONE without any refactor. Drop in
`ttnn.reduce_scatter` to convert h_input_tt → h_sharded for the
all_to_all_dispatch call.

If FAIL: investigate why the math doesn't hold (TILE layout vs the
scatter-breaks-tiles fallback path mentioned in the docstring).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
S = 4               # padded seq length for decode in MoE (S_orig=1, NCHIPS=4)
HIDDEN = 2688
NCHIPS = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None) -> int:
    import ttnn
    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh ✓")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=50_000_000,
        )

    try:
        rng = np.random.default_rng(seed=37)
        h_np = rng.standard_normal((B, 1, S, HIDDEN), dtype=np.float32) * 0.1

        # ── Reference: host-bridged sharded ──────────────────────────
        log("building reference (host-bridged SHARDED)…")
        h_sharded_ref_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(h_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        # Pull each chip's slice back via ConcatMeshToTensor(dim=2);
        # reconstructed tensor = original [B, 1, S, HIDDEN].
        ref_full_np = ttnn.to_torch(
            h_sharded_ref_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2),
        )[:B, :1].float().numpy()
        log(f"  reference reconstructed shape: {list(ref_full_np.shape)}")

        # ── Candidate: on-device reduce_scatter from REPLICATED ──────
        log("building candidate (REPLICATED -> mul(1/N) -> reduce_scatter)…")
        h_repl_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(h_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Pre-scale so the reduce-sum across NCHIPS chips gives back h.
        h_scaled_tt = ttnn.multiply(h_repl_tt, 1.0 / NCHIPS)
        try:
            log("  calling ttnn.reduce_scatter(h_scaled, dim=2, cluster_axis=1)")
            h_cand_tt = ttnn.reduce_scatter(
                h_scaled_tt, dim=2, cluster_axis=1,
            )
            log(f"  candidate per-chip shape: {list(h_cand_tt.shape)}")
            cand_full_np = ttnn.to_torch(
                h_cand_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2),
            )[:B, :1].float().numpy()
            log(f"  candidate reconstructed shape: {list(cand_full_np.shape)}")
            ttnn.deallocate(h_cand_tt)
            cand_ok = True
        except Exception as e:
            log(f"  ✗ reduce_scatter FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            cand_ok = False
            cand_full_np = None

        ttnn.deallocate(h_repl_tt)
        ttnn.deallocate(h_scaled_tt)
        ttnn.deallocate(h_sharded_ref_tt)

        log("")
        log("=" * 60)
        log("REPORT")
        log("=" * 60)
        if cand_ok:
            cos = float(
                np.dot(ref_full_np.reshape(-1), cand_full_np.reshape(-1))
                / (np.linalg.norm(ref_full_np)
                   * np.linalg.norm(cand_full_np) + 1e-12)
            )
            mad = float(np.mean(np.abs(ref_full_np - cand_full_np)))
            log(f"  ref shape:  {list(ref_full_np.shape)}")
            log(f"  cand shape: {list(cand_full_np.shape)}")
            log(f"  cos: {cos:.6f}  mad: {mad:.4e}")
            ok = cos >= 0.999
            log("")
            if ok:
                log("  → reduce_scatter+scale PRODUCES the equivalent of")
                log("    host-bridged SHARDED. Trace blocker #3 unblocks via:")
                log("      h_shard = ttnn.reduce_scatter(")
                log("        ttnn.multiply(h_repl, 1.0/NCHIPS),")
                log("        dim=2, cluster_axis=1)")
            else:
                log("  → cos below gate — investigate before integrating.")
            return 0 if ok else 1
        else:
            log("  candidate failed; see traceback. Likely needs different")
            log("  layout (ROW_MAJOR? specific memory_config?) per the")
            log("  docstring note: \"When the layout is row-major or the")
            log("  scatter breaks apart tiles, we use the composite\".")
            return 2
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
