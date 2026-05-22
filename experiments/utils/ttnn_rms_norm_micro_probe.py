#!/usr/bin/env python3
"""Micro-probe: ttnn.rms_norm vs numpy on a known fixed input.

Background: dn_norm cosine 0.9494 vs HF at pos 0, but inputs match cos 1.0,
and a numpy reimpl of the HF formula (both fp32 and bf16) matches HF at cos
1.0000. So the bug is in ttnn.rms_norm itself, not in math or precision.

This probe takes the actual L0 dn_core_attn_out from the HF oracle (one
position), passes it through ttnn.rms_norm with the actual norm.weight, and
compares vs numpy. Single-chip (replicated mesh), no TP complexity, no
embedding/projection/recurrence — pure ttnn.rms_norm isolation.

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \
    export ARCH_NAME=blackhole && \
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \
    .venv/bin/python -u experiments/utils/ttnn_rms_norm_micro_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_35b"
HEAD_V_DIM = 128
NV_HEADS = 32
EPS = 1e-6


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    # Load HF norm weight
    print("loading HF model for norm.weight…")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
    )
    weight = m.model.layers[0].linear_attn.norm.weight.detach().float().numpy()  # [128]
    eps_hf = float(m.model.layers[0].linear_attn.norm.variance_epsilon)
    print(f"weight shape={weight.shape} eps={eps_hf}")

    # Load HF core_attn_out from oracle (pos 0): shape [seq, NV_HEADS*HEAD_V_DIM]
    x_full = np.load(ORACLE_DIR / "L0_dn_core_attn_out.npy")  # [5, 4096]
    x_pos0 = x_full[0].reshape(NV_HEADS, HEAD_V_DIM)  # [32, 128]
    print(f"x_pos0 shape={x_pos0.shape} dtype={x_pos0.dtype} "
          f"range=[{x_pos0.min():.4f}, {x_pos0.max():.4f}]")

    # Numpy reference: weight * (x / rms(x))
    x32 = x_pos0.astype(np.float32)
    var = (x32 * x32).mean(axis=-1, keepdims=True)
    x_n = x32 / np.sqrt(var + eps_hf)
    expected = weight.astype(np.float32) * x_n
    print(f"numpy expected (fp32): range [{expected.min():.4f}, {expected.max():.4f}]")

    # FABRIC_1D requires multi-chip; use the same (1,4) mesh as the main server.
    # Replicate everything across chips; we only read chip 0's view.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh: {mesh}")

    # Upload x and weight to single chip
    x_bf = torch.from_numpy(x_pos0).to(torch.bfloat16)
    weight_bf = torch.from_numpy(weight).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x_bf, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    weight_tt = ttnn.from_torch(weight_bf, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Variant A: stock ttnn.rms_norm with weight=
    out_a = ttnn.rms_norm(x_tt, weight=weight_tt, epsilon=eps_hf)
    out_a_np = ttnn.to_torch(out_a, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out_a_np = np.split(out_a_np.float().numpy(), 4, axis=0)[0]  # take chip 0 view (32 rows)
    print(f"\nvariant A: ttnn.rms_norm(x, weight, eps)")
    print(f"  shape={out_a_np.shape} cos vs numpy: {cosine(out_a_np, expected):.6f}")
    print(f"  first row vs expected first row:")
    print(f"    ttnn   : {out_a_np[0, :8]}")
    print(f"    expect : {expected[0, :8]}")

    # Variant B: ttnn.rms_norm without weight, then explicit mul
    out_b_raw = ttnn.rms_norm(x_tt, epsilon=eps_hf)
    out_b = ttnn.mul(out_b_raw, weight_tt)
    out_b_np = ttnn.to_torch(out_b, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out_b_np = np.split(out_b_np.float().numpy(), 4, axis=0)[0]
    print(f"\nvariant B: ttnn.rms_norm(x, eps) * weight (explicit)")
    print(f"  shape={out_b_np.shape} cos vs numpy: {cosine(out_b_np, expected):.6f}")
    print(f"  first row: {out_b_np[0, :8]}")

    # Variant C: manual via sum + divide + rsqrt + mul
    x_sq = ttnn.mul(x_tt, x_tt)
    x_sumsq = ttnn.sum(x_sq, dim=-1, keepdim=True)
    mean_sq = ttnn.multiply(x_sumsq, 1.0 / HEAD_V_DIM)
    rsqrt_v = ttnn.rsqrt(ttnn.add(mean_sq, eps_hf))
    x_normed = ttnn.mul(x_tt, rsqrt_v)
    out_c = ttnn.mul(x_normed, weight_tt)
    out_c_np = ttnn.to_torch(out_c, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out_c_np = np.split(out_c_np.float().numpy(), 4, axis=0)[0]
    print(f"\nvariant C: manual sum→/D→+eps→rsqrt→*x→*weight")
    print(f"  shape={out_c_np.shape} cos vs numpy: {cosine(out_c_np, expected):.6f}")
    print(f"  first row: {out_c_np[0, :8]}")

    # Variant D: server-style — pass [32, 128] 2D sharded across chips (dim 0).
    # Each chip ends up with [8, 128] and runs rms_norm independently.
    NCHIPS = 4
    NV_PER_CHIP = NV_HEADS // NCHIPS  # 8
    x_2d_full = x_pos0  # already [32, 128]
    x_sharded = ttnn.from_torch(
        torch.from_numpy(x_2d_full).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    weight_replicated = ttnn.from_torch(
        weight_bf, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out_d = ttnn.rms_norm(x_sharded, weight=weight_replicated, epsilon=eps_hf)
    out_d_full = ttnn.to_torch(out_d, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out_d_full = out_d_full.float().numpy()  # expected [32, 128]
    print(f"\nvariant D: server-style 2D [32,128] sharded → per-chip [8,128] rms_norm")
    print(f"  shape={out_d_full.shape} cos vs numpy: "
          f"{cosine(out_d_full, expected):.6f}")
    print(f"  first row: {out_d_full[0, :8]}")
    print(f"  expect   : {expected[0, :8]}")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
