#!/usr/bin/env python3
"""A006 iso: sweep core_grid for lm_head matmul.

Production: lm_head matmul is [1, HIDDEN=2048] @ [2048, VOCAB=152064]
(actually VOCAB padded to 152096 = TILE-aligned). Called once per step,
replicated across mesh.

Per memory: vocab-sharded lm_head saved 5.1% on 27B; for 35B trace
baseline 110 ms/tok this could be ~5 ms/tok if similar. Easier first
attempt: just try explicit core_grid=10x11 like A004.

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
VOCAB = 152064  # padded to multiple of TILE=32 internally


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=20)
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
        h_np = rng.normal(0, 0.5, size=(1, HIDDEN)).astype(np.float32)
        w_np = rng.normal(0, 0.02, size=(HIDDEN, VOCAB)).astype(np.float32)
        log(f"shapes: h={h_np.shape}  W={w_np.shape}")

        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        w_tt = ttnn.from_torch(
            torch.from_numpy(w_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        HIFI4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

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
            log(f"  {name:25s} mean {np.mean(ts):7.3f} ms  median {np.median(ts):7.3f}  "
                f"min {np.min(ts):.3f}")
            return np.array(ts)

        ts_default = time_matmul("default",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4))

        ts_10x11 = time_matmul("core_grid 10x11",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4,
                                core_grid=ttnn.CoreGrid(y=10, x=11)))

        ts_8x8 = time_matmul("core_grid 8x8",
            lambda: ttnn.matmul(h_tt, w_tt, compute_kernel_config=HIFI4,
                                core_grid=ttnn.CoreGrid(y=8, x=8)))

        log(f"\nspeedup 10x11: {ts_default.mean()/ts_10x11.mean():.2f}x  "
            f"delta {ts_default.mean()-ts_10x11.mean():.4f} ms")
        log(f"speedup 8x8:   {ts_default.mean()/ts_8x8.mean():.2f}x  "
            f"delta {ts_default.mean()-ts_8x8.mean():.4f} ms")
        log(f"per-token (lm_head called 1x): same as per-call delta")

        ttnn.deallocate(h_tt); ttnn.deallocate(w_tt)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
