#!/usr/bin/env python3
"""A008 correctness: bf8_b vs bf16 weights for MoE batched matmul.

Production weights currently uploaded as bf16 (server_35b_ttnn.py:327,332).
Memory says bf8 MLP weights are 27B-production and bit-safe at >0.999 cos
on 24 layers. But 35B has different expert dims (HIDDEN=2048, MOE_INTER=512,
E_LOCAL=64) so we re-verify before integrating.

Three weight configs at production shape per chip:
  bf16/bf16: gate_up + down BOTH bf16 (today's reference)
  bf8/bf16:  gate_up bf8, down bf16
  bf8/bf8:   BOTH bf8 (full proposed change — halves W DRAM read)

For each: run the full mini-MoE FFN inner chain (matmul + silu*up + matmul)
and compare against numpy fp32 ground truth. Report:
  - PCC of each weight-dtype variant vs fp32 numpy
  - PCC of bf8 variants vs bf16 reference (the "did we regress" check)
  - max abs diff at each stage
  - magnitude stats (to spot a global scale shift from quantization)

Gate: bf8/bf8 PCC vs bf16 must be >= 0.9995 (the "no regression" threshold).
27B reports >0.999 across 24 layers; we want similar on the single-call
shape before integration.

Run on qb1 (1,4) mesh:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/test_moe_bf8_weights_correctness.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


HIDDEN = 2048
MOE_INTER = 512
E_LOCAL = 64
TWO_I = 2 * MOE_INTER
NCHIPS = 4

# Correctness threshold: bf8 must hold this cos vs bf16 reference.
PCC_GATE_VS_BF16 = 0.9995  # tighter than the workflow's 0.999 because
                            # this is the kernel-op level, not e2e


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc(a, b):
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    af -= af.mean(); bf -= bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / denom) if denom > 0 else 1.0


def cos_full(a, b):
    """Full cosine (not mean-centered)."""
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    return float((af * bf).sum() / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def numpy_mini_moe(h, W_gate_up, W_down):
    """Full per-expert chain in fp64 (ground truth).
    h: [E, 1, H]; W_gate_up: [E, H, 2I]; W_down: [E, I, H]
    Returns expert_out: [E, 1, H]
    """
    gate_up = h @ W_gate_up  # [E, 1, 2I]
    gate = gate_up[..., :MOE_INTER]
    up   = gate_up[..., MOE_INTER:]
    mid  = silu_np(gate) * up  # [E, 1, I]
    out  = mid @ W_down  # [E, 1, H]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="Number of random seeds to test (reduces single-seed luck).")
    args = ap.parse_args()

    import ttnn
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except Exception as e:
        log(f"fabric warning: {e}")

    log("opening (1,4) mesh on qb1")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        HIFI4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        CG = ttnn.CoreGrid(y=10, x=11)

        def to_replicated(arr, dtype):
            return ttnn.from_torch(
                torch.from_numpy(arr), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def run_path(h_np, wgu_np, wd_np, gu_dtype, d_dtype, label):
            """Run gate_up matmul + silu*up + down matmul on device."""
            h_tt   = to_replicated(h_np.astype(np.float32), ttnn.bfloat16)
            wgu_tt = to_replicated(wgu_np.astype(np.float32), gu_dtype)
            wd_tt  = to_replicated(wd_np.astype(np.float32), d_dtype)
            gate_up = ttnn.matmul(h_tt, wgu_tt, compute_kernel_config=HIFI4, core_grid=CG)
            gate = ttnn.slice(gate_up, [0, 0, 0], [E_LOCAL, 1, MOE_INTER])
            up   = ttnn.slice(gate_up, [0, 0, MOE_INTER], [E_LOCAL, 1, TWO_I])
            ttnn.deallocate(gate_up)
            mid = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(gate); ttnn.deallocate(up)
            out = ttnn.matmul(mid, wd_tt, compute_kernel_config=HIFI4, core_grid=CG)
            ttnn.deallocate(mid)
            ttnn.synchronize_device(mesh)
            # Per-chip out (chip 0).
            out_np = ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().numpy()[:E_LOCAL]   # first chip's E_LOCAL slice
            ttnn.deallocate(out); ttnn.deallocate(h_tt)
            ttnn.deallocate(wgu_tt); ttnn.deallocate(wd_tt)
            return out_np

        results = []
        for seed in range(args.seed, args.seed + args.n_seeds):
            log(f"\n=== seed {seed} ===")
            rng = np.random.default_rng(seed)
            # Production-realistic scales:
            # h post-rmsnorm ~ N(0, 5.0) (per the earlier profile bench)
            # W initialization ~ N(0, 0.02) (standard transformer init)
            h_np   = rng.normal(0.0, 5.0,  size=(E_LOCAL, 1, HIDDEN)).astype(np.float32)
            wgu_np = rng.normal(0.0, 0.02, size=(E_LOCAL, HIDDEN, TWO_I)).astype(np.float32)
            wd_np  = rng.normal(0.0, 0.02, size=(E_LOCAL, MOE_INTER, HIDDEN)).astype(np.float32)

            # fp64 numpy reference
            out_ref = numpy_mini_moe(h_np.astype(np.float64),
                                     wgu_np.astype(np.float64),
                                     wd_np.astype(np.float64))  # [E, 1, H]
            log(f"  ref magnitude: mean abs {np.mean(np.abs(out_ref)):.4f}  "
                f"max abs {np.max(np.abs(out_ref)):.4f}")

            # bf16/bf16 (today's production)
            out_bf16 = run_path(h_np, wgu_np, wd_np,
                                ttnn.bfloat16, ttnn.bfloat16, "bf16/bf16")
            pcc_bf16_vs_ref = pcc(out_bf16, out_ref)
            cos_bf16_vs_ref = cos_full(out_bf16, out_ref)
            log(f"  bf16/bf16 vs fp64 ref:  pcc={pcc_bf16_vs_ref:.6f}  "
                f"cos={cos_bf16_vs_ref:.6f}  "
                f"max_abs_diff={np.max(np.abs(out_bf16-out_ref)):.4e}")

            # bf8/bf16 (gate_up bf8 only)
            out_bf8_gu = run_path(h_np, wgu_np, wd_np,
                                   ttnn.bfloat8_b, ttnn.bfloat16, "bf8/bf16")
            pcc_g_vs_ref  = pcc(out_bf8_gu, out_ref)
            pcc_g_vs_bf16 = pcc(out_bf8_gu, out_bf16)
            log(f"  bf8/bf16  vs fp64 ref:  pcc={pcc_g_vs_ref:.6f}  "
                f"max_abs_diff={np.max(np.abs(out_bf8_gu-out_ref)):.4e}")
            log(f"  bf8/bf16  vs bf16 prod: pcc={pcc_g_vs_bf16:.6f}  "
                f"max_abs_diff={np.max(np.abs(out_bf8_gu-out_bf16)):.4e}")

            # bf8/bf8 (full proposed change)
            out_bf8_both = run_path(h_np, wgu_np, wd_np,
                                     ttnn.bfloat8_b, ttnn.bfloat8_b, "bf8/bf8")
            pcc_b_vs_ref  = pcc(out_bf8_both, out_ref)
            pcc_b_vs_bf16 = pcc(out_bf8_both, out_bf16)
            log(f"  bf8/bf8   vs fp64 ref:  pcc={pcc_b_vs_ref:.6f}  "
                f"max_abs_diff={np.max(np.abs(out_bf8_both-out_ref)):.4e}")
            log(f"  bf8/bf8   vs bf16 prod: pcc={pcc_b_vs_bf16:.6f}  "
                f"max_abs_diff={np.max(np.abs(out_bf8_both-out_bf16)):.4e}")

            # Magnitude check: bf8 should not shift the scale.
            ratio = np.mean(np.abs(out_bf8_both)) / (np.mean(np.abs(out_bf16)) + 1e-12)
            log(f"  bf8/bf16 magnitude ratio (should be ~1.0): {ratio:.4f}")

            results.append({
                "seed": seed,
                "pcc_bf16_vs_ref":  pcc_bf16_vs_ref,
                "pcc_bf8gu_vs_bf16": pcc_g_vs_bf16,
                "pcc_bf8both_vs_bf16": pcc_b_vs_bf16,
                "magnitude_ratio": ratio,
            })

        log("\n=== summary across seeds ===")
        log(f"  PCC (bf8/bf8 vs bf16 prod) min={min(r['pcc_bf8both_vs_bf16'] for r in results):.6f}  "
            f"mean={np.mean([r['pcc_bf8both_vs_bf16'] for r in results]):.6f}")
        log(f"  PCC (bf8/bf16 vs bf16 prod) min={min(r['pcc_bf8gu_vs_bf16'] for r in results):.6f}  "
            f"mean={np.mean([r['pcc_bf8gu_vs_bf16'] for r in results]):.6f}")
        log(f"  Magnitude ratio min={min(r['magnitude_ratio'] for r in results):.4f}  "
            f"max={max(r['magnitude_ratio'] for r in results):.4f}")

        worst = min(r['pcc_bf8both_vs_bf16'] for r in results)
        if worst < PCC_GATE_VS_BF16:
            log(f"\nFAIL: worst-case bf8/bf8 vs bf16 pcc = {worst:.6f} < gate {PCC_GATE_VS_BF16}")
            log("Do NOT integrate bf8 into production until this passes.")
            raise SystemExit(1)
        log(f"\nPASS: bf8/bf8 holds pcc >= {PCC_GATE_VS_BF16} vs bf16 prod across "
            f"{args.n_seeds} seeds. Safe to integrate (next: e2e single-layer cos).")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
