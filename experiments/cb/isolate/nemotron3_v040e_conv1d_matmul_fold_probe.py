#!/usr/bin/env python3
"""MM7 v0.4.0e — matmul-fold for the depthwise k=4 conv1d step.

ttnn.conv1d alone is 651ms on our decode shape (probe v0.4.0d). But
the math for our case is trivial:
  output[c] = sum_{k=0..3} w[c,k] * x[k,c] + b[c]
            = w0[c]*x0[c] + w1[c]*x1[c] + w2[c]*x2[c] + w3[c]*x3[c] + b[c]
where wK[c] is per-channel weight at kernel position K (shape [C]),
and xK[c] is the input value at position K (shape [B, C]).

That's 4 element-wise muls + 3 adds + 1 bias add = 8 ttnn ops on
shape [B=1, 1, 1, C=6144] each. Trivial work for the device.

This probe:
  • Builds the same random conv1d input + weight + bias.
  • Computes output via ttnn.conv1d (baseline).
  • Computes output via 4 muls + 3 adds + 1 bias add (new path).
  • Compares cos ≥ 0.999.
  • Times each (warm-mean across N calls).

Gate: cos ≥ 0.999 AND new mean ≤ 100ms.

If PASS: integrate as the decode-path depthwise conv replacement.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
CONV_DIM_M = 6144
CONV_KERNEL = 4
S = 1
INPUT_LEN = CONV_KERNEL - 1 + S  # = 4
N_CALLS = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main(state=None) -> int:
    import ttnn

    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh ✓")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=400_000_000,
        )

    try:
        rng = np.random.default_rng(seed=0)
        w_np = rng.standard_normal((CONV_DIM_M, 1, 1, CONV_KERNEL),
                                    dtype=np.float32) * 0.05
        b_np = rng.standard_normal((1, 1, 1, CONV_DIM_M), dtype=np.float32) * 0.01

        # ── BASELINE: ttnn.conv1d weights (host-side) ─────────────────
        w_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(w_np)),
            dtype=ttnn.bfloat16,
        )
        b_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(b_np)),
            dtype=ttnn.bfloat16,
        )

        # ── MATMUL-FOLD weights: split weight by kernel position ──────
        # w_np shape: [C, 1, 1, K]. We want 4 tensors each [1, 1, 1, C] for
        # broadcast-multiply against [B, 1, 1, C] input slices.
        w_per_pos_np = [
            w_np[:, 0, 0, k].reshape(1, 1, 1, CONV_DIM_M) for k in range(CONV_KERNEL)
        ]
        w_per_pos_tt = [
            ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for arr in w_per_pos_np
        ]
        b_tt_tile = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(b_np.astype(np.float32))),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # ── Build input + pre-slice the 4 positions for matmul-fold ──
        x_np = rng.standard_normal((B, 1, INPUT_LEN, CONV_DIM_M),
                                    dtype=np.float32) * 0.1
        x_rm = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(x_np)),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Same input as TILE for the fold path (one upload per smoke OK)
        x_tile = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(x_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        log(f"input shape: {list(x_rm.shape)}")

        # ── BASELINE TIMING ───────────────────────────────────────────
        log(f"timing ttnn.conv1d (baseline) over {N_CALLS} calls…")
        baseline_times = []
        for call in range(N_CALLS):
            t0 = time.time()
            out_tt = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=w_tt,
                device=mesh,
                in_channels=CONV_DIM_M, out_channels=CONV_DIM_M,
                batch_size=B, input_length=INPUT_LEN,
                kernel_size=CONV_KERNEL, stride=1,
                padding=0, dilation=1, groups=CONV_DIM_M,
                bias_tensor=b_tt,
            )
            elapsed = time.time() - t0
            baseline_times.append(elapsed)
            if call == 0:
                baseline_out_np = ttnn.to_torch(
                    out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
                )[:1].float().numpy().reshape(-1)
            ttnn.deallocate(out_tt)
            log(f"  call {call:>2}  conv1d={elapsed*1000:>6.1f} ms")

        # ── MATMUL-FOLD TIMING + CORRECTNESS ──────────────────────────
        log(f"timing matmul-fold (4 muls + 3 adds + 1 bias) "
            f"over {N_CALLS} calls…")
        fold_times = []
        for call in range(N_CALLS):
            t0 = time.time()
            # Slice each input position [B, 1, k:k+1, C] (TILE layout)
            position_slices = []
            for k in range(CONV_KERNEL):
                pos_view = ttnn.slice(
                    x_tile, [0, 0, k, 0], [B, 1, k + 1, CONV_DIM_M],
                )
                position_slices.append(pos_view)
            # Multiply each by its per-position weight
            products = []
            for k in range(CONV_KERNEL):
                prod = ttnn.mul(position_slices[k], w_per_pos_tt[k])
                products.append(prod)
            # Sum across kernel positions
            accum = products[0]
            for k in range(1, CONV_KERNEL):
                new_accum = ttnn.add(accum, products[k])
                ttnn.deallocate(accum)
                ttnn.deallocate(products[k])
                accum = new_accum
            # Add bias
            fold_out = ttnn.add(accum, b_tt_tile)
            ttnn.deallocate(accum)
            elapsed = time.time() - t0
            fold_times.append(elapsed)
            if call == 0:
                fold_out_np = ttnn.to_torch(
                    fold_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
                )[:1].float().numpy().reshape(-1)
            ttnn.deallocate(fold_out)
            log(f"  call {call:>2}  matmul-fold={elapsed*1000:>6.1f} ms")

        ttnn.deallocate(x_rm)
        ttnn.deallocate(x_tile)
        for t in w_per_pos_tt:
            ttnn.deallocate(t)
        ttnn.deallocate(b_tt_tile)

        # ── REPORT ────────────────────────────────────────────────────
        baseline_warm = sum(baseline_times[1:]) / max(1, len(baseline_times) - 1)
        fold_warm = sum(fold_times[1:]) / max(1, len(fold_times) - 1)
        cos, mad = cos_and_mad(baseline_out_np, fold_out_np)

        log("")
        log("=" * 60)
        log("REPORT")
        log("=" * 60)
        log(f"  ttnn.conv1d warm:   {baseline_warm*1000:.1f} ms")
        log(f"  matmul-fold warm:   {fold_warm*1000:.1f} ms")
        log(f"  speedup:            {baseline_warm/fold_warm:.1f}×")
        log(f"  correctness cos:    {cos:.6f}  mad: {mad:.4e}")
        log("")
        cos_ok = cos >= 0.999
        speed_ok = fold_warm * 1000 <= 100  # target
        if cos_ok and speed_ok:
            log(f"v0.4.0e PASS ✓  (cos ≥ 0.999 + fold ≤ 100ms)")
            log(f"  Projected per-step: 23 layers × {fold_warm*1000:.0f} ms = "
                f"{23 * fold_warm:.2f}s "
                f"(vs current 15.5s → {15.5/(23*fold_warm):.1f}× speedup)")
        else:
            log(f"v0.4.0e {'cos OK' if cos_ok else 'COS FAIL'}; "
                f"{'speed OK' if speed_ok else 'speed marginal'}")
        return 0 if cos_ok else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
