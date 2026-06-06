"""Isolation probe: validate `ttnn.geglu([up | gate]_concat) == gelu(gate) * up`.

Current chain (post round-4):
    gelu_gate = ttnn.matmul(pre_ff, gate_proj, activation="gelu")
    up        = ttnn.matmul(pre_ff, up_proj)
    mid       = ttnn.mul(gelu_gate, up)  # = gelu(gate) * up

Proposed chain (saves 1 op/layer × 48 = 48 ops/forward AND potentially a
larger matmul that hits BW more efficiently):
    gate_up   = ttnn.matmul(pre_ff, [up_proj | gate_proj]_concat)  # one bigger mm
    mid       = ttnn.geglu(gate_up, dim=-1)
                # geglu(x) = x[..., :half] * gelu(x[..., half:])
                #         = up * gelu(gate)    (since we concatenated [up | gate])

This probe just verifies geglu semantics + reference equality with a known
[up, gate] split. The actual production land would also need to concatenate
the weights at bootstrap.

Forks: experiments/cb/isolate/gm4_matmul_gelu_probe.py.
"""
from __future__ import annotations

import sys
import time

import torch

import ttnn

HIDDEN = 3840
INTERMED_PER_CHIP = 3840


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(state=None):
    if state is None:
        log("ERR: probe requires harness mesh.")
        return 1
    device = state.mesh
    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    torch.manual_seed(0)
    x_t = torch.randn(1, HIDDEN, dtype=torch.float32) * 0.5
    w_gate = torch.randn(HIDDEN, INTERMED_PER_CHIP, dtype=torch.float32) * 0.05
    w_up = torch.randn(HIDDEN, INTERMED_PER_CHIP, dtype=torch.float32) * 0.05

    # Reference: gelu(gate) * up.
    gate_ref = x_t @ w_gate
    up_ref = x_t @ w_up
    ref = torch.nn.functional.gelu(gate_ref, approximate="none") * up_ref

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        # arr is shape (NCHIPS * orig_shape[0], …rest). Take chip 0 then
        # flatten + slice to expected size.
        flat = arr.float().flatten()
        n = 1
        for d in expected_shape:
            n *= d
        # First chip's contiguous chunk: total / NCHIPS prefix.
        per_chip = flat.numel() // 4  # mesh = 4 chips
        return flat[:per_chip][:n].reshape(*expected_shape)

    HIFI4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # --- Baseline path: matmul activation=gelu, matmul, mul ---
    x_tt = to_tt(x_t)
    w_gate_tt = to_tt(w_gate)
    w_up_tt = to_tt(w_up)
    gelu_gate_tt = ttnn.matmul(x_tt, w_gate_tt, compute_kernel_config=HIFI4,
                               activation="gelu")
    up_tt = ttnn.matmul(x_tt, w_up_tt, compute_kernel_config=HIFI4)
    base_tt = ttnn.mul(gelu_gate_tt, up_tt)
    base = to_torch(base_tt, (1, INTERMED_PER_CHIP))
    ttnn.deallocate(gelu_gate_tt); ttnn.deallocate(up_tt); ttnn.deallocate(base_tt)
    ttnn.deallocate(w_gate_tt); ttnn.deallocate(w_up_tt)

    # --- Fused path: concat[up, gate] matmul + geglu ---
    # geglu requires rank-4 input — reshape [1, 2*INTERMED] → [1, 1, 1, 2*INTERMED].
    w_concat = torch.cat([w_up, w_gate], dim=-1)  # [HIDDEN, 2*INTERMED]
    w_concat_tt = to_tt(w_concat)
    gate_up_tt = ttnn.matmul(x_tt, w_concat_tt, compute_kernel_config=HIFI4)
    log(f"gate_up_tt shape: {gate_up_tt.shape}")
    # Reshape to rank-4 for geglu.
    gate_up_4d = ttnn.reshape(gate_up_tt, [1, 1, 1, 2 * INTERMED_PER_CHIP])
    fused_tt = ttnn.geglu(gate_up_4d, dim=-1)
    fused = to_torch(fused_tt, (1, INTERMED_PER_CHIP))
    ttnn.deallocate(gate_up_tt); ttnn.deallocate(fused_tt); ttnn.deallocate(w_concat_tt)
    ttnn.deallocate(x_tt)

    cos_base_ref = _cos(base, ref)
    cos_fused_ref = _cos(fused, ref)
    cos_pair = _cos(base, fused)
    max_abs = float((base - fused).abs().max())
    mad = float((base - fused).abs().mean())

    log(f"base shape: {base.shape}  fused shape: {fused.shape}")
    log(f"cos(baseline, torch_ref) = {cos_base_ref:.7f}")
    log(f"cos(fused,    torch_ref) = {cos_fused_ref:.7f}")
    log(f"cos(baseline, fused)     = {cos_pair:.7f}")
    log(f"max|baseline - fused|    = {max_abs:.6f}")
    log(f"mean|baseline - fused|   = {mad:.6f}")

    ok = cos_pair > 0.999  # geglu is bf16-noisy; relax slightly
    log("")
    log(f"PASS: {'yes' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
