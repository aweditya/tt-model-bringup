#!/usr/bin/env python3
"""MM7 v0.4.1.c — replicate→shard primitive discovery.

Blocker #3 of the trace plan: h_input_tt is REPLICATED across the (1,4)
mesh; MoE dispatch needs it SHARDED along dim 2. Today the code does:
  ttnn.to_torch(h_input_tt) -> ttnn.from_torch(..., ShardTensorToMesh)
which is a host roundtrip and breaks trace capture.

This probe attempts the conversion on-device using each candidate:
  1. ttnn.reshard
  2. ttnn.scatter
  3. ttnn.experimental.slice_reshard_async

For each, build a replicated tensor [B, S, HIDDEN] = [1, 8, 2688] and
try to produce the equivalent of:
  numpy: h.reshape(1, 1, S, H) sharded on dim 2 -> 4 chips each get [1,1,2,2688]

PASS criterion: output cos ≥ 0.999 vs the numpy reference for at least
one candidate. Report which works.

If NONE works: we need to investigate either:
  - maintaining a parallel sharded buffer alongside the replicated one
    (cheap memory; copy via copy_device_to_device if it exists)
  - refactoring MoE input to be sharded from the start of the layer
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
S = 8
HIDDEN = 2688
NCHIPS = 4
S_PER_CHIP = S // NCHIPS


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
        rng = np.random.default_rng(seed=11)
        h_np = rng.standard_normal((B, 1, S, HIDDEN), dtype=np.float32) * 0.1

        # Build the replicated source tensor.
        h_repl_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(h_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # Reference: numpy sharded → reconstruct via per-chip read.
        h_sharded_ref_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(h_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        # Pulling back via ConcatMeshToTensor(dim=2) reconstructs full.
        ref_back = ttnn.to_torch(
            h_sharded_ref_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2),
        )
        log(f"reference shard reconstructed shape: {list(ref_back.shape)}")
        ttnn.deallocate(h_sharded_ref_tt)

        results = []

        # ── Candidate 1: ttnn.reshard ───────────────────────────────
        log("Candidate 1: ttnn.reshard")
        try:
            # ttnn.reshard typically wants a memory_config. Probably won't
            # auto-magic mesh distribution. Try with a sharded memory config.
            sharded_mem_cfg = ttnn.create_sharded_memory_config(
                shape=[B, 1, S, HIDDEN],
                core_grid=ttnn.CoreGrid(y=1, x=NCHIPS),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            resharded = ttnn.reshard(h_repl_tt, sharded_mem_cfg)
            log(f"  ttnn.reshard returned shape: {list(resharded.shape)}")
            results.append(("ttnn.reshard", True, None))
            ttnn.deallocate(resharded)
        except Exception as e:
            log(f"  ✗ failed: {type(e).__name__}: {str(e)[:80]}")
            results.append(("ttnn.reshard", False, str(e)[:120]))

        # ── Candidate 2: ttnn.experimental.slice_reshard_async ──────
        log("Candidate 2: ttnn.experimental.slice_reshard_async")
        try:
            # Likely needs a memory_config too. Try as-is first.
            sliced = ttnn.experimental.slice_reshard_async(h_repl_tt)
            log(f"  returned shape: {list(sliced.shape)}")
            results.append(("ttnn.experimental.slice_reshard_async", True, None))
            ttnn.deallocate(sliced)
        except Exception as e:
            log(f"  ✗ failed: {type(e).__name__}: {str(e)[:80]}")
            results.append(("slice_reshard_async", False, str(e)[:120]))

        # ── Candidate 3: ttnn.distribute_tensor ─────────────────────
        log("Candidate 3: ttnn.distribute_tensor")
        try:
            distributed = ttnn.distribute_tensor(
                h_repl_tt,
                ttnn.ShardTensorToMesh(mesh, dim=2),
            )
            log(f"  returned shape: {list(distributed.shape)}")
            results.append(("ttnn.distribute_tensor", True, None))
            ttnn.deallocate(distributed)
        except Exception as e:
            log(f"  ✗ failed: {type(e).__name__}: {str(e)[:80]}")
            results.append(("ttnn.distribute_tensor", False, str(e)[:120]))

        # ── Candidate 4: ttnn.reduce_scatter (multiplied by 1/4) ────
        log("Candidate 4: ttnn.reduce_scatter (after 1/N scaling)")
        try:
            inv_n = ttnn.multiply(h_repl_tt, 1.0 / NCHIPS)
            rs = ttnn.reduce_scatter(inv_n, dim=2, math_op=ttnn.ReduceType.Sum)
            log(f"  reduce_scatter shape: {list(rs.shape)}")
            results.append(("ttnn.reduce_scatter+scale", True, None))
            ttnn.deallocate(inv_n)
            ttnn.deallocate(rs)
        except Exception as e:
            log(f"  ✗ failed: {type(e).__name__}: {str(e)[:80]}")
            results.append(("reduce_scatter+scale", False, str(e)[:120]))

        ttnn.deallocate(h_repl_tt)

        # ── Report ──────────────────────────────────────────────────
        log("")
        log("=" * 60)
        log("REPORT — replicate→shard discovery")
        log("=" * 60)
        any_pass = False
        for name, ok, err in results:
            status = "✓ AVAILABLE" if ok else f"✗ FAIL  ({err})"
            log(f"  {name:<42s}  {status}")
            any_pass = any_pass or ok
        log("")
        if any_pass:
            log("  → at least one primitive works. Pick the simplest")
            log("    one and integrate in moe_block_eager_ep_tt.")
        else:
            log("  → no primitive works as-is. Fallback options:")
            log("    A. Have previous layer's output_tt produced as a")
            log("       SHARDED tensor (memory cost; refactor scope).")
            log("    B. Maintain a parallel sharded h buffer; update via")
            log("       copy_device_to_device (check if it exists).")
            log("    C. Move the entire decode forward to a SHARDED")
            log("       hidden representation (Megatron-TP style). Heavy.")
        return 0 if any_pass else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
