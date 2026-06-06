"""Round 6 isolation probe: fuse `add(a, b)` + `multiply(., scalar)` into
a single `add(a, b, post_activations=[MUL_UNARY_SFPU, scalar])`.

Math currently in `_layer_forward_pos0_paged` (server_gemma4_unified_ttnn.py:1454-1456):
    h_residual_2 = ttnn.add(h_after_attn, post_ff)         # 1 op (BinaryNg)
    h_out = ttnn.multiply(h_residual_2, w["layer_scalar"]) # 1 op (BinaryNg)
# = 2 device ops per layer × 48 = 96 ops/forward attackable

Proposed: fuse the trailing scalar multiply via the `post_activations`
parameter on `ttnn.add`, using `UnaryOpType.MUL_UNARY_SFPU` (which the
LLK exposes as `mul_unary_tile(idst, scalar)` — see
tt-metal `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:340`).

    h_out = ttnn.add(h_after_attn, post_ff,
                     post_activations=[(UnaryOpType.MUL_UNARY_SFPU, layer_scalar)])
# = 1 device op per layer × 48 = 48 ops/forward saved

Why bit-equivalent (or very close):
    (a + b) * s == a * s + b * s (distributivity)
At the SFPU level, the fused path computes `(tile_a + tile_b) * scalar`
as a single SFPU operation chain in one kernel pass, vs the legacy path
which does add → store → load → mul. Mathematically identical to within
SFPU floating-point rounding (same op order: add THEN mul).

Gate: cos(baseline, fused) > 0.99999 + max|delta| < 0.05 (bf16 round-off).

Forks: experiments/cb/isolate/gm4_roll_rope_probe.py (single-device,
torch sanity + ttnn cos check).
"""
from __future__ import annotations

import sys
import time

import torch

import ttnn

# Hidden state on a single chip: per-mesh-chip is [1, 3840 / 4 = 960] but we
# use [1, 960] flat here for the eltwise op. Use a bigger shape to stress
# the kernel: [1, 960] (single chip)
M = 1
N = 960
LAYER_SCALAR = 0.054  # Gemma 4 12B L0 layer_scalar value (real layer 0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _add_mul_baseline(a, b, scalar):
    """Current 2-op chain: add + multiply (scalar)."""
    h_residual = ttnn.add(a, b)
    h_out = ttnn.multiply(h_residual, scalar)
    ttnn.deallocate(h_residual)
    return h_out


def _add_mul_fused(a, b, scalar):
    """Proposed 1-op fused: add with post-activation MUL_UNARY_SFPU."""
    post_acts = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, float(scalar))]
    h_out = ttnn.add(a, b, activations=post_acts)
    return h_out


def main():
    log("opening single-device mesh…")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        log("seeding test tensors…")
        torch.manual_seed(0)
        a_torch = torch.randn(M, N, dtype=torch.bfloat16)
        b_torch = torch.randn(M, N, dtype=torch.bfloat16)

        # Reference (torch)
        ref_torch = (a_torch.float() + b_torch.float()) * LAYER_SCALAR
        log(f"torch ref [{M}, {N}] @ scalar={LAYER_SCALAR}: "
            f"mean={ref_torch.mean().item():.6f} max|.|={ref_torch.abs().max().item():.6f}")

        # Baseline path
        a_tt_1 = ttnn.from_torch(a_torch, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 dtype=ttnn.bfloat16)
        b_tt_1 = ttnn.from_torch(b_torch, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 dtype=ttnn.bfloat16)
        out_baseline_tt = _add_mul_baseline(a_tt_1, b_tt_1, LAYER_SCALAR)
        out_baseline = ttnn.to_torch(out_baseline_tt).float()
        ttnn.deallocate(out_baseline_tt)
        ttnn.deallocate(a_tt_1)
        ttnn.deallocate(b_tt_1)
        log(f"baseline shape={list(out_baseline.shape)} "
            f"mean={out_baseline.mean().item():.6f} "
            f"max|.|={out_baseline.abs().max().item():.6f}")

        # Fused path
        a_tt_2 = ttnn.from_torch(a_torch, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 dtype=ttnn.bfloat16)
        b_tt_2 = ttnn.from_torch(b_torch, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 dtype=ttnn.bfloat16)
        out_fused_tt = _add_mul_fused(a_tt_2, b_tt_2, LAYER_SCALAR)
        out_fused = ttnn.to_torch(out_fused_tt).float()
        ttnn.deallocate(out_fused_tt)
        ttnn.deallocate(a_tt_2)
        ttnn.deallocate(b_tt_2)
        log(f"fused    shape={list(out_fused.shape)} "
            f"mean={out_fused.mean().item():.6f} "
            f"max|.|={out_fused.abs().max().item():.6f}")

        # Compare
        flat_b = out_baseline.flatten()
        flat_f = out_fused.flatten()
        cos = torch.nn.functional.cosine_similarity(
            flat_b.unsqueeze(0), flat_f.unsqueeze(0)).item()
        max_delta = (out_baseline - out_fused).abs().max().item()
        log(f"cos(baseline, fused) = {cos:.7f}")
        log(f"max|baseline - fused| = {max_delta:.6f}")

        cos_t1 = torch.nn.functional.cosine_similarity(
            flat_b.unsqueeze(0), ref_torch.flatten().unsqueeze(0)).item()
        cos_t2 = torch.nn.functional.cosine_similarity(
            flat_f.unsqueeze(0), ref_torch.flatten().unsqueeze(0)).item()
        log(f"cos(baseline, torch_ref) = {cos_t1:.7f}")
        log(f"cos(fused,    torch_ref) = {cos_t2:.7f}")

        if cos >= 0.99999 and max_delta < 0.05:
            log("VERDICT: PASS (cos >= 0.99999, max|delta| < 0.05)")
            return 0
        else:
            log(f"VERDICT: FAIL (cos={cos:.7f}, max|delta|={max_delta:.6f})")
            return 1
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main() or 0)
