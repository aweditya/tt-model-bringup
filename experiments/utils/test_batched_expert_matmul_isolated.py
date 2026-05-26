#!/usr/bin/env python3
"""Regression test for the batched expert matmul shape used in production.

Production layout (moe_forward_ttnn_pattern_a_batched):
  W_local: [E_LOCAL, HIDDEN, 2*MOE_INTER] bf16 TILE per chip
           (uploaded sharded dim 0 from [NCHIPS*E_LOCAL, H, 2I]).
  h_tt:    [1, HIDDEN] bf16 TILE replicated, then reshaped to [1, 1, HIDDEN]
           and broadcast across the expert dim via on-device ttnn.concat.

PASS = matmul runs without exception AND cos > 0.999 vs a per-expert loop.

The 12-variant matrix that exhausted other shape combos lives in
experiments/utils/archive/test_batched_expert_matmul_variants_2026_05_25.py.

Run (qb1): see HANDOFF.md for env-var bootstrap; then
  .venv/bin/python -u experiments/utils/test_batched_expert_matmul_isolated.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

HIDDEN = 2048
MOE_INTER = 512
E_LOCAL = 64
NCHIPS = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_np(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_ttnn_replicated(arr_np, mesh):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_ttnn_sharded(arr_np, mesh, shard_dim):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard_dim),
    )


def chip0_to_np(t, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()
    if arr.shape[0] >= NCHIPS:
        return arr[: arr.shape[0] // NCHIPS]
    return arr


def main():
    rng = np.random.default_rng(42)
    h_np = rng.normal(0, 1.0, size=(1, HIDDEN)).astype(np.float32)
    # All chips hold identical synthetic slabs — this is a shape regression,
    # not a cross-chip correctness check.
    W_local_np = rng.normal(0, 0.1, size=(E_LOCAL, HIDDEN, 2 * MOE_INTER)).astype(np.float32)
    W_stacked_np = np.broadcast_to(
        W_local_np, (NCHIPS, E_LOCAL, HIDDEN, 2 * MOE_INTER)
    ).reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER).copy()

    ref = np.stack([h_np @ W_local_np[e] for e in range(E_LOCAL)], axis=0)  # [E_LOCAL,1,2I]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        log(f"mesh open: {mesh}")
        W_tt = to_ttnn_sharded(W_stacked_np, mesh, shard_dim=0)
        h_tt = to_ttnn_replicated(h_np, mesh)
        h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
        h_repeated = ttnn.concat([h_3d] * E_LOCAL, dim=0)  # [E_LOCAL,1,HIDDEN]
        out_tt = ttnn.matmul(h_repeated, W_tt)             # [E_LOCAL,1,2I]
        out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
        c = cos_np(out_np, ref)
        log(f"cos vs per-expert loop = {c:.6f}")
        assert c > 0.999, f"FAIL: cos {c:.6f} <= 0.999"
        log("PASS  batched expert matmul matches per-expert reference.")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
