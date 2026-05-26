#!/usr/bin/env python3
"""Isolated probes for fused unary-on-input activations in ttnn.mul.

ttnn.mul supports input_tensor_a_activations=[UnaryOpType.X] (verified for
SILU per DeepSeek-V3 ref). This harness checks two patterns relevant to the
35B decode hot path:

  P1. silu(a) * b  (DN RMSNormGated tail; fires 30x/token at linear_attention
                    layers — `gated = mul(z_2d, normed, [SILU])`)
  P2. sigmoid(a) * b (shared expert gating; fires 40x/token —
                      `gated_shared = mul(gate_logit, shared_full, [SIGMOID])`)

Gate per probe: cos(fused, sequential) > 0.9999 AND fused ms/iter <= sequential.

Run (qb1): see HANDOFF.md for env-var bootstrap; then
  .venv/bin/python -u experiments/utils/test_fused_binary_activations_isolated.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

NCHIPS = 4
N_ITERS = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_np(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_ttnn(arr_np, mesh):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def time_op(label, fn, mesh):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(N_ITERS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.time() - t0) * 1000.0 / N_ITERS


def probe(name, mesh, shape, act_op_type):
    rng = np.random.default_rng(0)
    a_np = rng.normal(0, 1.0, size=shape).astype(np.float32)
    b_np = rng.normal(0, 1.0, size=shape).astype(np.float32)
    a_tt = to_ttnn(a_np, mesh)
    b_tt = to_ttnn(b_np, mesh)

    if act_op_type == "SILU":
        ref_a = a_np * (1.0 / (1.0 + np.exp(-a_np)))
        unary_kind = ttnn.UnaryOpType.SILU
        seq_fn = lambda: ttnn.mul(ttnn.silu(a_tt), b_tt)
    elif act_op_type == "SIGMOID":
        ref_a = 1.0 / (1.0 + np.exp(-a_np))
        unary_kind = ttnn.UnaryOpType.SIGMOID
        seq_fn = lambda: ttnn.mul(ttnn.sigmoid(a_tt), b_tt)
    else:
        raise ValueError(act_op_type)
    ref_np = ref_a * b_np

    fused_fn = lambda: ttnn.mul(a_tt, b_tt, input_tensor_a_activations=[unary_kind])

    log(f"--- {name}: shape={shape}, act={act_op_type} ---")
    try:
        out_seq = seq_fn()
        out_fused = fused_fn()
        ttnn.synchronize_device(mesh)
    except Exception as e:
        log(f"  EXCEPTION: {e!r}")
        ttnn.deallocate(a_tt); ttnn.deallocate(b_tt)
        return

    s_np = ttnn.to_torch(
        out_seq, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()[0]
    f_np = ttnn.to_torch(
        out_fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()[0]
    ttnn.deallocate(out_seq); ttnn.deallocate(out_fused)

    c_seq_ref = cos_np(s_np, ref_np[0] if s_np.ndim != ref_np.ndim else ref_np)
    c_fused_seq = cos_np(f_np, s_np)
    log(f"  cos(seq, numpy_ref) = {c_seq_ref:.8f}")
    log(f"  cos(fused, seq)     = {c_fused_seq:.8f}")
    if c_fused_seq < 0.9999:
        log(f"  FAIL: cos {c_fused_seq:.6f} < 0.9999")
        ttnn.deallocate(a_tt); ttnn.deallocate(b_tt)
        return

    t_seq = time_op("seq  ", seq_fn, mesh)
    t_fused = time_op("fused", fused_fn, mesh)
    speedup = t_seq / t_fused if t_fused > 0 else float("inf")
    log(f"  seq:   {t_seq:.3f} ms/iter")
    log(f"  fused: {t_fused:.3f} ms/iter")
    log(f"  speedup: {speedup:.2f}x")
    ttnn.deallocate(a_tt); ttnn.deallocate(b_tt)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        log(f"mesh open: {mesh}")
        # DN RMSNormGated tail: silu(z_2d) * normed at shape [NV_PER_CHIP=8, V_DIM=128]
        # per chip. Replicated synthetic for isolation.
        probe("DN_RMSNormGated", mesh, (8, 128), "SILU")
        # Shared expert gating: sigmoid(gate_logit) * shared_full at [1, HIDDEN=2048].
        probe("Shared_expert_gating", mesh, (1, 2048), "SIGMOID")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
