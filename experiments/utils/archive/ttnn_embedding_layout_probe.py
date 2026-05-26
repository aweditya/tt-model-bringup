#!/usr/bin/env python3
"""Probe — does ttnn.embedding support TILE-layout tables?

If yes, MoE on-device gather can keep TILE throughout and avoid the per-slice
ROW_MAJOR→TILE conversion that overflows L1 in the B17-B attempt.

Method:
  - Upload a small [16, 128] table in both ROW_MAJOR and TILE
  - Run ttnn.embedding(indices, table) with each
  - Compare outputs against numpy reference

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \\
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
    export ARCH_NAME=blackhole && \\
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
    .venv/bin/python -u experiments/utils/ttnn_embedding_layout_probe.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1); b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

    np.random.seed(42)
    V, D = 16, 128
    table_np = np.random.randn(V, D).astype(np.float32)
    idx_np = np.array([[3, 7, 1, 0]], dtype=np.int32)  # [B=1, K=4]
    expected = table_np[idx_np[0]]  # [4, 128]

    print("=== variant A: ROW_MAJOR table ===")
    table_rm = ttnn.from_torch(
        torch.from_numpy(table_np), dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    idx_tt = ttnn.from_torch(
        torch.from_numpy(idx_np), dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    try:
        out_rm = ttnn.embedding(idx_tt, table_rm)
        out_rm_np = ttnn.to_torch(
            out_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        out_rm_chip0 = np.split(out_rm_np, 4, axis=0)[0]
        # shape might be [1, 4, 128] or [4, 128]
        print(f"  output shape: {out_rm_chip0.shape}")
        flat = out_rm_chip0.reshape(-1)
        cos = cosine(flat, expected.reshape(-1))
        print(f"  cos vs numpy: {cos:.6f} ✓" if cos > 0.99 else f"  cos vs numpy: {cos:.6f} ✗")
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        ttnn.deallocate(table_rm); ttnn.deallocate(idx_tt)
        if 'out_rm' in dir():
            try:
                ttnn.deallocate(out_rm)
            except Exception:
                pass

    print("\n=== variant B: TILE table ===")
    table_tile = ttnn.from_torch(
        torch.from_numpy(table_np), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    idx_tt2 = ttnn.from_torch(
        torch.from_numpy(idx_np), dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    try:
        out_tile = ttnn.embedding(idx_tt2, table_tile)
        out_tile_np = ttnn.to_torch(
            out_tile, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        out_tile_chip0 = np.split(out_tile_np, 4, axis=0)[0]
        print(f"  output shape: {out_tile_chip0.shape}")
        flat = out_tile_chip0.reshape(-1)
        cos = cosine(flat, expected.reshape(-1))
        print(f"  cos vs numpy: {cos:.6f} ✓" if cos > 0.99 else f"  cos vs numpy: {cos:.6f} ✗")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
