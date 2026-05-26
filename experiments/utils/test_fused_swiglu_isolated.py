#!/usr/bin/env python3
"""Isolated test for SwiGLU fusion: ttnn.silu + ttnn.mul → ttnn.mul(...silu activation).

The 35B MoE batched FFN currently runs sequential ops:
    silu_gate = ttnn.silu(gate_batched)            # UnaryNg dispatch
    mid_batched = ttnn.mul(silu_gate, up_batched)  # BinaryNg dispatch
That's 2 dispatches per MoE call × 40 layers = 80 dispatches/token.

Fused alternative (per tt-metal/models/demos/deepseek_v3/tt/experts.py:185, 261):
    mid_batched = ttnn.mul(
        gate_batched, up_batched,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    )
One dispatch, mathematically equivalent.

This probe measures:
  - cos(fused, sequential) on synthetic [E_LOCAL, 1, MOE_INTER] tensors
  - kernel-time delta over 100 iterations (sync-bounded)

Gate: cos > 0.9999 AND fused kernel-time ≤ sequential.

Run (qb1): see HANDOFF.md for env-var bootstrap; then
  .venv/bin/python -u experiments/utils/test_fused_swiglu_isolated.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

E_LOCAL = 64
MOE_INTER = 512
NCHIPS = 4
N_ITERS = 100


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


def time_op(label, fn, mesh):
    # warmup
    for _ in range(3):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(N_ITERS):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    elapsed_ms = (time.time() - t0) * 1000.0 / N_ITERS
    log(f"  {label}: {elapsed_ms:.3f} ms/iter")
    return elapsed_ms


def main():
    rng = np.random.default_rng(0)
    gate_np = rng.normal(0, 1.0, size=(E_LOCAL, 1, MOE_INTER)).astype(np.float32)
    up_np = rng.normal(0, 1.0, size=(E_LOCAL, 1, MOE_INTER)).astype(np.float32)

    # numpy reference: silu(gate) * up
    silu_np = gate_np * (1.0 / (1.0 + np.exp(-gate_np)))
    ref_np = silu_np * up_np

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        log(f"mesh open: {mesh}")
        gate_tt = to_ttnn_replicated(gate_np, mesh)
        up_tt = to_ttnn_replicated(up_np, mesh)

        # Path A: sequential silu + mul (current production)
        def seq_fn():
            silu_g = ttnn.silu(gate_tt)
            out = ttnn.mul(silu_g, up_tt)
            ttnn.deallocate(silu_g)
            return out

        # Path B: fused mul with SILU on input_tensor_a
        def fused_fn():
            return ttnn.mul(
                gate_tt, up_tt,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )

        # Correctness
        out_seq = seq_fn()
        out_fused = fused_fn()
        ttnn.synchronize_device(mesh)
        # take chip 0 (all chips identical for replicated input)
        out_seq_np = ttnn.to_torch(
            out_seq, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()[0]
        out_fused_np = ttnn.to_torch(
            out_fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()[0]
        ttnn.deallocate(out_seq); ttnn.deallocate(out_fused)

        c_seq_vs_ref = cos_np(out_seq_np, ref_np[0] if out_seq_np.ndim != ref_np.ndim else ref_np)
        c_fused_vs_seq = cos_np(out_fused_np, out_seq_np)
        log(f"cos(sequential, numpy_ref) = {c_seq_vs_ref:.8f}")
        log(f"cos(fused, sequential)     = {c_fused_vs_seq:.8f}")
        assert c_fused_vs_seq > 0.9999, f"FAIL: fused vs sequential cos {c_fused_vs_seq:.6f} < 0.9999"

        # Timing
        log("timing sequential silu + mul…")
        t_seq = time_op("seq", seq_fn, mesh)
        log("timing fused mul(silu)…")
        t_fused = time_op("fused", fused_fn, mesh)
        log(f"speedup: {t_seq / t_fused:.2f}x  (seq {t_seq:.3f} ms, fused {t_fused:.3f} ms)")
        if t_fused < t_seq:
            log("PASS  fused is faster.")
        else:
            log("WARN  fused not faster — investigate.")
        ttnn.deallocate(gate_tt); ttnn.deallocate(up_tt)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
