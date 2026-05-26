#!/usr/bin/env python3
"""Micro-probe: run server_35b_ttnn._apply_partial_rope on (1,4) mesh with
fixed input, compare to numpy oracle. Numpy oracle is verified bit-exact
to HF apply_rotary_pos_emb (see rope_numpy_oracle.py).

If ttnn output cosine vs numpy < 0.999, my ttnn op composition has a bug.

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \
    export ARCH_NAME=blackhole && \
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \
    .venv/bin/python -u experiments/utils/rope_ttnn_micro_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))

import ttnn  # noqa: E402
import server_35b_ttnn as srv  # noqa: E402
from rope_numpy_oracle import (  # noqa: E402
    my_apply_partial_rope_numpy,
    compute_cos_sin_numpy,
)


HEAD_DIM_ATTN = 256
ROTARY_DIM = 64
NQ_PER_CHIP = 4
NCHIPS = 4


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"mesh: {mesh}")

    np.random.seed(42)
    q_np = np.random.randn(NQ_PER_CHIP, HEAD_DIM_ATTN).astype(np.float32)
    k_np = np.random.randn(1, HEAD_DIM_ATTN).astype(np.float32)

    print("=== Q [4, 256] ===")
    for pos in [0, 1, 5, 50]:
        cos_np, sin_np = compute_cos_sin_numpy(pos)
        q_expected = my_apply_partial_rope_numpy(q_np, cos_np, sin_np)

        # Upload q replicated; chip 0's view should match numpy expected
        q_tt = ttnn.from_torch(
            torch.from_numpy(q_np).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cos_tt = ttnn.from_torch(
            torch.from_numpy(cos_np),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sin_tt = ttnn.from_torch(
            torch.from_numpy(sin_np),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        q_out_tt = srv._apply_partial_rope(q_tt, cos_tt, sin_tt, NQ_PER_CHIP)
        q_out_all = ttnn.to_torch(
            q_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        # Take chip 0's slice [NQ_PER_CHIP, HEAD_DIM]
        q_out_chip0 = np.split(q_out_all, NCHIPS, axis=0)[0]
        ttnn.deallocate(q_out_tt); ttnn.deallocate(q_tt)
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

        # Quantize q_expected to bf16 for fair comparison
        q_exp_bf16 = torch.from_numpy(q_expected).to(torch.bfloat16).float().numpy()

        cos_val = cosine(q_out_chip0, q_exp_bf16)
        max_diff = np.abs(q_out_chip0 - q_exp_bf16).max()
        flag = "✓" if cos_val > 0.999 else "✗"
        print(f"  pos {pos:3d}: cos vs numpy={cos_val:.6f}  max|Δ|={max_diff:.6f}  {flag}")

        if cos_val < 0.999 and pos == 5:
            # Drill into first-bad position: print first row first 16 values
            print(f"    ttnn  : {q_out_chip0[0, :16]}")
            print(f"    numpy : {q_exp_bf16[0, :16]}")
            # Also rotary section comparison
            print(f"    ttnn rotary section [0, :16]:  {q_out_chip0[0, :16]}")
            print(f"    numpy rotary section [0, :16]: {q_exp_bf16[0, :16]}")
            print(f"    ttnn pass section [0, 64:80]:  {q_out_chip0[0, 64:80]}")
            print(f"    numpy pass section [0, 64:80]: {q_exp_bf16[0, 64:80]}")

    print("\n=== K [1, 256] via broadcast→rotate→slice workaround ===")
    for pos in [0, 1, 5, 50]:
        cos_np, sin_np = compute_cos_sin_numpy(pos)
        k_expected = my_apply_partial_rope_numpy(k_np, cos_np, sin_np)
        k_tt = ttnn.from_torch(
            torch.from_numpy(k_np).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cos_tt = ttnn.from_torch(torch.from_numpy(cos_np), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sin_tt = ttnn.from_torch(torch.from_numpy(sin_np), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Workaround: broadcast K [1, 256] → [NQ_PER_CHIP, 256] via concat, rotate, slice row 0
        k_broadcast = ttnn.concat([k_tt] * NQ_PER_CHIP, dim=0)
        k_rotated_4 = srv._apply_partial_rope(k_broadcast, cos_tt, sin_tt, NQ_PER_CHIP)
        k_out_tt = ttnn.slice(k_rotated_4, [0, 0], [1, HEAD_DIM_ATTN])
        k_out_all = ttnn.to_torch(
            k_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        k_out_chip0 = np.split(k_out_all, NCHIPS, axis=0)[0]
        ttnn.deallocate(k_out_tt); ttnn.deallocate(k_rotated_4); ttnn.deallocate(k_broadcast)
        ttnn.deallocate(k_tt); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)
        k_exp_bf16 = torch.from_numpy(k_expected).to(torch.bfloat16).float().numpy()
        cos_val = cosine(k_out_chip0, k_exp_bf16)
        max_diff = np.abs(k_out_chip0 - k_exp_bf16).max()
        flag = "✓" if cos_val > 0.999 else "✗"
        print(f"  pos {pos:3d}: cos vs numpy={cos_val:.6f}  max|Δ|={max_diff:.6f}  {flag}")

    print("\n=== K [1, 256] (single-head case, direct) ===")
    for pos in [0, 1, 5, 50]:
        cos_np, sin_np = compute_cos_sin_numpy(pos)
        k_expected = my_apply_partial_rope_numpy(k_np, cos_np, sin_np)
        k_tt = ttnn.from_torch(
            torch.from_numpy(k_np).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cos_tt = ttnn.from_torch(
            torch.from_numpy(cos_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sin_tt = ttnn.from_torch(
            torch.from_numpy(sin_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        k_out_tt = srv._apply_partial_rope(k_tt, cos_tt, sin_tt, 1)
        k_out_all = ttnn.to_torch(
            k_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        k_out_chip0 = np.split(k_out_all, NCHIPS, axis=0)[0]
        ttnn.deallocate(k_out_tt); ttnn.deallocate(k_tt)
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)
        k_exp_bf16 = torch.from_numpy(k_expected).to(torch.bfloat16).float().numpy()
        cos_val = cosine(k_out_chip0, k_exp_bf16)
        max_diff = np.abs(k_out_chip0 - k_exp_bf16).max()
        flag = "✓" if cos_val > 0.999 else "✗"
        print(f"  pos {pos:3d}: cos vs numpy={cos_val:.6f}  max|Δ|={max_diff:.6f}  {flag}")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
