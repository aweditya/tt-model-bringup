"""Isolation probe: validate `ttnn.matmul(..., activation="gelu")` matches
the unfused `gelu(matmul(...), fast_and_approximate_mode=False)` pair.

Current chain in _layer_forward_pos0_paged
(server_gemma4_unified_ttnn.py:1409-1413):
    gate     = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4)
    up       = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ...
    gelu_gate = ttnn.gelu(gate, fast_and_approximate_mode=False)

Proposed: fold the gelu into the matmul itself via the `activation="gelu"`
fused-activation parameter. tt-metal's string-to-unary mapping
(`tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:833`)
shows `"gelu"` → UnaryOpType::GELU with fast_and_approximate=false — matches
our current ttnn.gelu(fast_and_approximate_mode=False) exactly.

Per forward: 48 fewer dispatches (1 per layer). Both kernel time and
dispatch time should drop.

Probe shapes (Gemma 4 12B IT per-chip on qb2 mesh):
- pre_ff: [1, 3840]    (hidden, REPLICATED — but we test as if per-chip)
- gate_proj: [3840, INTERMED_PER_CHIP=3840]  (FFN-shard slice)

Forks: experiments/cb/isolate/gm4_addcmul_rope_probe.py (probe scaffolding,
mesh-mapper to_tt/to_torch helpers).
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

HIDDEN = 3840
INTERMED_PER_CHIP = 3840  # qb2 has TP=4, so each chip holds 1/4 of 15360


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(state=None):
    if state is None:
        log("ERR: probe requires a harness mesh; run via gm4 dev harness.")
        return 1
    device = state.mesh

    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    torch.manual_seed(0)
    x_t = torch.randn(1, HIDDEN, dtype=torch.float32) * 0.5
    w_t = torch.randn(HIDDEN, INTERMED_PER_CHIP, dtype=torch.float32) * 0.05

    # Torch reference: gelu(x @ w) with exact gelu (matches
    # fast_and_approximate_mode=False).
    y_ref = torch.nn.functional.gelu(x_t @ w_t, approximate="none")

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        return arr[: expected_shape[0]].reshape(*expected_shape)

    x_tt = to_tt(x_t)
    w_tt = to_tt(w_t)

    # Baseline: separate matmul + gelu.
    gate = ttnn.matmul(x_tt, w_tt,
                       compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                           math_fidelity=ttnn.MathFidelity.HiFi4,
                           math_approx_mode=False,
                           fp32_dest_acc_en=True,
                           packer_l1_acc=True,
                       ))
    y_base_tt = ttnn.gelu(gate, fast_and_approximate_mode=False)
    y_base = to_torch(y_base_tt, (1, INTERMED_PER_CHIP))
    ttnn.deallocate(gate); ttnn.deallocate(y_base_tt)

    # Fused: matmul with activation="gelu" param.
    y_fused_tt = ttnn.matmul(
        x_tt, w_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        activation="gelu",
    )
    y_fused = to_torch(y_fused_tt, (1, INTERMED_PER_CHIP))
    ttnn.deallocate(y_fused_tt)

    ttnn.deallocate(x_tt); ttnn.deallocate(w_tt)

    cos_base_ref = _cos(y_base, y_ref)
    cos_fused_ref = _cos(y_fused, y_ref)
    cos_pair = _cos(y_base, y_fused)
    max_abs = float((y_base - y_fused).abs().max())
    mad = float((y_base - y_fused).abs().mean())

    log(f"shape: {y_base.shape}")
    log(f"cos(baseline, torch_ref) = {cos_base_ref:.7f}")
    log(f"cos(fused,    torch_ref) = {cos_fused_ref:.7f}")
    log(f"cos(baseline, fused)     = {cos_pair:.7f}")
    log(f"max|baseline - fused|    = {max_abs:.6f}")
    log(f"mean|baseline - fused|   = {mad:.6f}")

    ok = cos_pair > 0.99999 and abs(cos_base_ref - cos_fused_ref) < 1e-4
    log("")
    log(f"PASS: {'yes' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
