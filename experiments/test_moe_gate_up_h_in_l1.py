#!/usr/bin/env python3
"""A007 iso: place input 0 (h_3d_repeat) in L1 for the batched MoE gate_up.

tt-perf-report advice (post-A004 profile):
  MatmulDeviceOperation b={64} x 32 x 2048 x 1024 — at 110 cores
  "If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)"

h_3d_repeat shape: [E_LOCAL=64, 1, HIDDEN=2048] bf16 = 256 KB per chip.
Fits trivially in L1 (1.39 MiB per core × 110 cores).

This isolates the L1 placement effect: same A004 core_grid=10x11, but
sweep input-0 memory config:
  default       — h_3d_repeat in DRAM_INTERLEAVED (today)
  L1 interleave — h_3d_repeat in L1_INTERLEAVED
  L1 sharded    — h_3d_repeat in L1 height/width sharded (best case)

Run on qb1 (1,4) mesh.
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
        rng = np.random.default_rng(0)
        h_3d_np = rng.normal(0, 0.5, size=(E_LOCAL, 1, HIDDEN)).astype(np.float32)
        w_np    = rng.normal(0, 0.02, size=(E_LOCAL, HIDDEN, TWO_I)).astype(np.float32)
        log(f"shapes: h={h_3d_np.shape}  W={w_np.shape}")
        log(f"h memory: {E_LOCAL * 1 * HIDDEN * 2 / 1024:.1f} KB per chip")
        log(f"W memory: {E_LOCAL * HIDDEN * TWO_I * 2 / 1024 / 1024:.1f} MB per chip")

        def to_replicated(arr, mem_cfg=None):
            return ttnn.from_torch(
                torch.from_numpy(arr), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=mem_cfg,
            )

        # h_dram and h_l1: same data, different memory configs.
        DRAM_INTERLEAVED = ttnn.DRAM_MEMORY_CONFIG
        L1_INTERLEAVED   = ttnn.L1_MEMORY_CONFIG
        h_dram = to_replicated(h_3d_np, DRAM_INTERLEAVED)
        h_l1   = to_replicated(h_3d_np, L1_INTERLEAVED)
        w_tt   = to_replicated(w_np, DRAM_INTERLEAVED)
        log(f"h_dram memory: {h_dram.memory_config()}")
        log(f"h_l1   memory: {h_l1.memory_config()}")
        log(f"w_tt   memory: {w_tt.memory_config()}")

        HIFI4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        CG = ttnn.CoreGrid(y=10, x=11)

        def time_matmul(name, fn):
            for _ in range(args.n_warmup):
                out = fn(); ttnn.synchronize_device(mesh); ttnn.deallocate(out)
            ts = []
            for _ in range(args.n_iters):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                out = fn()
                ttnn.synchronize_device(mesh)
                ts.append((time.perf_counter() - t0) * 1000.0)
                ttnn.deallocate(out)
            log(f"  {name:30s} mean {np.mean(ts):7.3f} ms  median {np.median(ts):7.3f}  "
                f"min {np.min(ts):.3f}  std {np.std(ts):.3f}")
            return np.array(ts)

        log("\n=== variant: h in DRAM (A004 baseline) ===")
        ts_dram = time_matmul(
            "h DRAM, core_grid 10x11",
            lambda: ttnn.matmul(h_dram, w_tt, compute_kernel_config=HIFI4, core_grid=CG),
        )

        log("\n=== variant: h in L1 interleaved ===")
        ts_l1 = time_matmul(
            "h L1, core_grid 10x11",
            lambda: ttnn.matmul(h_l1, w_tt, compute_kernel_config=HIFI4, core_grid=CG),
        )

        # Also: try matmul with output going to L1 (sometimes ttnn picks
        # different program configs when output is L1).
        log("\n=== variant: h in L1 + output L1 ===")
        ts_l1_out_l1 = time_matmul(
            "h L1, out L1",
            lambda: ttnn.matmul(h_l1, w_tt, compute_kernel_config=HIFI4, core_grid=CG,
                                memory_config=L1_INTERLEAVED),
        )

        # Also: production-realistic — h starts in DRAM, we'd insert a
        # to_memory_config(L1) before the matmul. Measure with that cast
        # included in the per-call cost.
        log("\n=== variant: cast h DRAM->L1 each call + matmul ===")
        def cast_then_matmul():
            h_cast = ttnn.to_memory_config(h_dram, L1_INTERLEAVED)
            out = ttnn.matmul(h_cast, w_tt, compute_kernel_config=HIFI4, core_grid=CG)
            ttnn.deallocate(h_cast)
            return out
        ts_cast = time_matmul("cast h then matmul", cast_then_matmul)

        # ---- Correctness ----
        log("\n=== correctness ===")
        out_dram = ttnn.matmul(h_dram, w_tt, compute_kernel_config=HIFI4, core_grid=CG)
        out_l1   = ttnn.matmul(h_l1,   w_tt, compute_kernel_config=HIFI4, core_grid=CG)
        ttnn.synchronize_device(mesh)
        def chip0(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()[0]
        n_d = chip0(out_dram); n_l = chip0(out_l1)
        log(f"  pcc(DRAM, L1) = {pcc(n_d, n_l):.6f}  "
            f"max_abs_diff = {np.max(np.abs(n_d-n_l)):.4e}")
        ttnn.deallocate(out_dram); ttnn.deallocate(out_l1)

        log("\n=== summary ===")
        base = ts_dram.mean()
        for name, ts in [("DRAM (A004)", ts_dram), ("L1 in", ts_l1),
                         ("L1 in + L1 out", ts_l1_out_l1),
                         ("cast+matmul", ts_cast)]:
            sp = base / ts.mean()
            log(f"  {name:18s} mean {ts.mean():7.4f} ms  speedup {sp:.2f}x  "
                f"delta {base-ts.mean():+.4f}")
        log(f"\n  per-token (x40 layers × 2 matmuls each = 80):")
        best = min(ts_l1.mean(), ts_l1_out_l1.mean(), ts_cast.mean())
        log(f"    DRAM baseline {base*80:.1f} ms/tok (-) — at 110 cores")
        log(f"    best L1       {best*80:.1f} ms/tok  delta {(base-best)*80:+.1f}")

        ttnn.deallocate(h_dram); ttnn.deallocate(h_l1); ttnn.deallocate(w_tt)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
