#!/usr/bin/env python3
"""Pre-flight probe: which ttnn GELU variant matches torch's `gelu_pytorch_tanh`?

Step 0.2 of `research/gemma4_12b_bringup_plan.md` §6.3. Gemma 4's MLP
uses `gelu_pytorch_tanh` (tanh approximation). ttnn exposes:
  - `ttnn.gelu(x, fast_and_approximate_mode=False)`  default
  - `ttnn.gelu(x, fast_and_approximate_mode=True)`   approximate variant
  - `ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.GELU])`
    fused path used by 35B SwiGLU — variant determined by kernel default

This probe compares each variant pointwise against
`torch.nn.functional.gelu(x, approximate="tanh")` AND `approximate="none"`
on x ∈ [-5, 5] and reports max-abs-diff + cosine. Tells us which call
shape to use in the Gemma 4 MLP forward, and whether the fused path
is safe.

Fork base: `experiments/utils/test_fused_swiglu_isolated.py` (mesh
boot + numpy/torch reference + cos helper).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

N_POINTS = 4096  # enough to tile cleanly (4096 = 128*32)
RANGE_LO, RANGE_HI = -5.0, 5.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_np(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_ttnn_replicated(arr_np, mesh):
    # [N] → [1, 1, 32, ceil(N/32)*32] padded for TILE_LAYOUT.
    t = torch.from_numpy(arr_np.astype(np.float32)).reshape(1, 1, 1, -1)
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def main():
    log(f"opening mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))  # single chip; pointwise probe
    try:
        x = np.linspace(RANGE_LO, RANGE_HI, N_POINTS).astype(np.float32)

        # torch references
        tx = torch.from_numpy(x)
        ref_tanh = torch.nn.functional.gelu(tx, approximate="tanh").numpy()
        ref_exact = torch.nn.functional.gelu(tx, approximate="none").numpy()

        x_tt = to_ttnn_replicated(x, mesh)

        # variant A — ttnn.gelu(fast_and_approximate_mode=False)
        out_a = ttnn.gelu(x_tt, fast_and_approximate_mode=False)
        a_np = ttnn.to_torch(out_a, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0, 0, 0, :N_POINTS].float().numpy()
        ttnn.deallocate(out_a)

        # variant B — ttnn.gelu(fast_and_approximate_mode=True)
        out_b = ttnn.gelu(x_tt, fast_and_approximate_mode=True)
        b_np = ttnn.to_torch(out_b, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0, 0, 0, :N_POINTS].float().numpy()
        ttnn.deallocate(out_b)

        # variant C — fused ttnn.mul(x, ones, activations=[UnaryOpType.GELU])
        ones = to_ttnn_replicated(np.ones_like(x), mesh)
        out_c = ttnn.mul(x_tt, ones, input_tensor_a_activations=[ttnn.UnaryOpType.GELU])
        c_np = ttnn.to_torch(out_c, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0, 0, 0, :N_POINTS].float().numpy()
        ttnn.deallocate(out_c)
        ttnn.deallocate(ones)
        ttnn.deallocate(x_tt)

        # Pointwise diffs vs each torch reference
        log("=" * 70)
        log(f"GELU variant probe — x ∈ [{RANGE_LO}, {RANGE_HI}], N={N_POINTS}, bf16 round-trip")
        log("=" * 70)
        for label, arr in [("A: ttnn.gelu(fast_and_approximate_mode=False)", a_np),
                           ("B: ttnn.gelu(fast_and_approximate_mode=True) ", b_np),
                           ("C: ttnn.mul(.., UnaryOpType.GELU)            ", c_np)]:
            d_tanh = np.abs(arr - ref_tanh)
            d_exact = np.abs(arr - ref_exact)
            log(f"  {label}")
            log(f"      vs torch tanh  : max_abs={d_tanh.max():.6e} mean_abs={d_tanh.mean():.6e} cos={cos_np(arr, ref_tanh):.8f}")
            log(f"      vs torch exact : max_abs={d_exact.max():.6e} mean_abs={d_exact.mean():.6e} cos={cos_np(arr, ref_exact):.8f}")
        log("=" * 70)
        # Verdict: which variant best matches the TANH reference (what Gemma 4 wants)?
        scores_tanh = {
            "A": float(np.abs(a_np - ref_tanh).max()),
            "B": float(np.abs(b_np - ref_tanh).max()),
            "C": float(np.abs(c_np - ref_tanh).max()),
        }
        best = min(scores_tanh, key=scores_tanh.get)
        log(f"VERDICT: variant {best} is closest to gelu_pytorch_tanh "
            f"(max_abs={scores_tanh[best]:.6e}). Use this in Gemma 4 MLP.")
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
