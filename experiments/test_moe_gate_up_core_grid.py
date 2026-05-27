#!/usr/bin/env python3
"""A004 isolation: sweep core grid for the batched MoE gate_up matmul.

Production profile (tt-perf-report on tracy_one_moe):
  MatmulDeviceOperation b={64} x 32 x 2048 x 1024 — 1,838 us, 11/110 cores,
  29.9% of 404 GB/s peak. 62.1% of MoE step. ×40 layers = ~72 ms/tok.

Hypothesis: ttnn's default program config picks 11 cores. Forcing more
cores via core_grid/program_config could reduce per-call kernel time
substantially. Roofline: 64 * 2048 * 1024 * 2 bytes = 256 MB DRAM read
per call. At 404 GB/s = 634 us. We're at 1,838 us = 35% of roofline.
Room to ~3x if we hit BW peak.

This isolates the matmul: sets up the exact production shape on (1,4)
mesh with the production sharded weight, then runs the matmul with
several different core-grid configs (default + explicit larger grids)
and reports per-config kernel time.

Run on qb1 (1,4) mesh:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/test_moe_gate_up_core_grid.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# Production-realistic per-chip shapes for MoE gate_up (35B-A3B, (1,4) TP):
#   h_3d_repeat: [E_LOCAL=64, 1, HIDDEN=2048]
#   W:           [E_LOCAL=64, HIDDEN=2048, 2*MOE_INTER=1024] sharded across mesh
#   out:         [E_LOCAL=64, 1, 2*MOE_INTER=1024]
HIDDEN = 2048
MOE_INTER = 512
E_LOCAL = 64
TWO_I = 2 * MOE_INTER  # 1024
NCHIPS = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pcc(a, b):
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    af -= af.mean(); bf -= bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / denom) if denom > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=30)
    args = ap.parse_args()

    import ttnn
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except Exception as e:
        log(f"fabric warning: {e}")

    log("opening (1,4) mesh on qb1")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        # Build per-chip weights (replicated for isolation — only per-chip work matters).
        rng = np.random.default_rng(0)
        h_3d_np  = rng.normal(0, 0.5, size=(E_LOCAL, 1, HIDDEN)).astype(np.float32)
        w_np     = rng.normal(0, 0.02, size=(E_LOCAL, HIDDEN, TWO_I)).astype(np.float32)
        log(f"shapes: h_3d_repeat={h_3d_np.shape}  W={w_np.shape}")

        def to_replicated(arr):
            return ttnn.from_torch(
                torch.from_numpy(arr), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        h_tt = to_replicated(h_3d_np)
        w_tt = to_replicated(w_np)

        HIFI4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def time_matmul(name, fn):
            # warmup
            for _ in range(args.n_warmup):
                out = fn()
                ttnn.synchronize_device(mesh)
                ttnn.deallocate(out)
            ts = []
            for _ in range(args.n_iters):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                out = fn()
                ttnn.synchronize_device(mesh)
                ts.append((time.perf_counter() - t0) * 1000.0)
                ttnn.deallocate(out)
            log(f"  {name:30s} mean {np.mean(ts):7.3f} ms  median {np.median(ts):7.3f}  "
                f"min {np.min(ts):.3f}  max {np.max(ts):.3f}")
            return np.array(ts)

        log("\n=== variant: default (no core_grid) ===")
        ts_default = time_matmul(
            "default",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4),
        )

        log("\n=== variant: core_grid=10x11 (110 cores) ===")
        cg_full = ttnn.CoreGrid(y=10, x=11)
        ts_full = time_matmul(
            "core_grid 10x11",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4, core_grid=cg_full),
        )

        log("\n=== variant: core_grid=8x8 (64 cores) ===")
        cg_64 = ttnn.CoreGrid(y=8, x=8)
        ts_64 = time_matmul(
            "core_grid 8x8",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4, core_grid=cg_64),
        )

        log("\n=== variant: core_grid=4x11 (44 cores) ===")
        cg_44 = ttnn.CoreGrid(y=4, x=11)
        ts_44 = time_matmul(
            "core_grid 4x11",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4, core_grid=cg_44),
        )

        # ---- Correctness across variants ----
        log("\n=== correctness ===")
        out_default = ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4)
        out_full    = ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4, core_grid=cg_full)
        ttnn.synchronize_device(mesh)
        def get_chip0(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()[0]
        nd = get_chip0(out_default)
        nf = get_chip0(out_full)
        log(f"  pcc(default, 10x11) = {pcc(nd, nf):.6f}  "
            f"max_abs_diff = {np.max(np.abs(nd-nf)):.4e}")
        ttnn.deallocate(out_default); ttnn.deallocate(out_full)

        log("\n=== summary ===")
        baseline = ts_default.mean()
        for name, ts in [("default", ts_default), ("10x11 (110)", ts_full),
                         ("8x8 (64)", ts_64), ("4x11 (44)", ts_44)]:
            speedup = baseline / ts.mean()
            log(f"  {name:15s} mean {ts.mean():7.3f} ms  speedup {speedup:.2f}x")
        per_token_baseline = baseline * 40
        per_token_best = min(ts_default.mean(), ts_full.mean(), ts_64.mean(), ts_44.mean()) * 40
        log(f"\n  per-token (x40 layers):  default {per_token_baseline:.1f} ms/tok  "
            f"best {per_token_best:.1f} ms/tok  delta {per_token_baseline - per_token_best:.1f} ms/tok")

        ttnn.deallocate(h_tt); ttnn.deallocate(w_tt)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
