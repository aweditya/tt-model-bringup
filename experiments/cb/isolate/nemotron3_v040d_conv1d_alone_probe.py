#!/usr/bin/env python3
"""MM7 v0.4.0d — time ttnn.conv1d ALONE on the decode shape.

v0.4.0c eliminated all host bridges around the conv1d call but per-step
time stayed at 15.5s. Either the kernel itself takes ~600ms (no fix
from trace) or per-call dispatch overhead is huge (trace will help).
This probe tells us which.

Timed in isolation (no surrounding mamba2 ops):
  N=10 ttnn.conv1d calls with the exact decode shape:
    input  : [B=1, 1, CONV_KERNEL-1+S=4, CONV_DIM_M=6144] ROW_MAJOR bf16
    weight : [CONV_DIM_M, 1, 1, CONV_KERNEL=4] bf16
    bias   : [1, 1, 1, CONV_DIM_M] bf16
    output : [B=1, 1, S=1, CONV_DIM_M] ROW_MAJOR bf16

Reports:
  • cold time (call 0)
  • warm mean (calls 1..N-1)
  • If warm < 50ms: kernel is fast, the 600ms must include
    per-call dispatch — trace would compress this big chunk.
  • If warm > 400ms: kernel is genuinely slow — need to find a
    matmul-equivalent or kernel rewrite.

REUSE: harness-aware (state=None ok).
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

        # Weights + bias (held constant across calls)
        w_np = rng.standard_normal((CONV_DIM_M, 1, 1, CONV_KERNEL),
                                    dtype=np.float32) * 0.05
        b_np = rng.standard_normal((1, 1, 1, CONV_DIM_M), dtype=np.float32) * 0.01
        w_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(w_np)),
            dtype=ttnn.bfloat16,
        )  # weight stays host; ttnn.conv1d takes it that way
        b_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(b_np)),
            dtype=ttnn.bfloat16,
        )

        # Pre-upload the input ONCE so we don't measure upload time.
        # (For warm calls, only the conv1d call is timed.)
        x_np = rng.standard_normal((B, 1, INPUT_LEN, CONV_DIM_M),
                                    dtype=np.float32) * 0.1
        x_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(x_np)),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        log(f"input shape: {list(x_tt.shape)}; "
            f"weight shape: {list(w_tt.shape)}; "
            f"timing {N_CALLS} ttnn.conv1d calls…")

        times = []
        outputs = []  # keep references so we measure pure conv1d cost
        for call in range(N_CALLS):
            t0 = time.time()
            out_tt = ttnn.conv1d(
                input_tensor=x_tt,
                weight_tensor=w_tt,
                device=mesh,
                in_channels=CONV_DIM_M, out_channels=CONV_DIM_M,
                batch_size=B, input_length=INPUT_LEN,
                kernel_size=CONV_KERNEL, stride=1,
                padding=0, dilation=1, groups=CONV_DIM_M,
                bias_tensor=b_tt,
            )
            elapsed = time.time() - t0
            times.append(elapsed)
            outputs.append(out_tt)
            log(f"  call {call:>2}  conv1d={elapsed*1000:>6.1f} ms  "
                f"{'COLD' if call == 0 else 'WARM'}")

        for o in outputs:
            try:
                ttnn.deallocate(o)
            except Exception:
                pass
        ttnn.deallocate(x_tt)

        warm_mean = sum(times[1:]) / max(1, len(times) - 1)
        log("")
        log("=" * 60)
        log("VERDICT")
        log("=" * 60)
        log(f"  cold call 0:  {times[0]*1000:.1f} ms")
        log(f"  warm mean:    {warm_mean*1000:.1f} ms")
        log(f"  cold/warm:    {times[0] / warm_mean:.1f}×")
        log("")
        if warm_mean * 1000 < 50:
            log(f"  → conv1d kernel is FAST ({warm_mean*1000:.0f} ms).")
            log(f"    The 600ms in production must include per-call DISPATCH.")
            log(f"    v0.4.1 TRACE is the right path — will compress dispatch.")
        elif warm_mean * 1000 < 400:
            log(f"  → conv1d kernel is moderate ({warm_mean*1000:.0f} ms).")
            log(f"    Both trace AND a faster conv would help.")
        else:
            log(f"  → conv1d kernel is GENUINELY SLOW ({warm_mean*1000:.0f} ms).")
            log(f"    v0.4.0d MATMUL-FOLD or kernel rewrite needed.")
        log(f"  Projected mamba2 kernel-only time per step: "
            f"23 layers × {warm_mean*1000:.0f} ms = {23 * warm_mean:.2f}s")
        return 0
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
