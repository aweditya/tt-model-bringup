#!/usr/bin/env python3
"""MM7 v0.4.0a — isolation probe for pre-uploaded SSD constants.

Compares two wrapper variants over N=10 random-input decode calls:
  • OLD: `mamba2_decode_step_ttnn` — pads + uploads dt_bias/A_log/D every call
  • NEW: `mamba2_decode_step_ttnn_with_const_tt` — constants pre-uploaded
    via `prepare_mamba2_constants` ONCE

Gates:
  • Output (state, y) cos ≥ 0.999 between OLD and NEW for each call
  • NEW per-call mean time < OLD per-call mean time (any improvement)
  • Report observed speedup

REUSE: forks G2 multi-head wrapper smoke; no full server bootstrap needed.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v040a_const_wrapper_smoke'

OR standalone (creates its own mesh):
  TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
      TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
      PYTHONPATH=$TT_METAL_HOME/ttnn \\
      LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
      .venv/bin/python -u experiments/cb/isolate/nemotron3_v040a_const_wrapper_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE = 128
B = 1
N_CALLS = 10
COS_GATE = 0.999


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main(state=None) -> int:
    import importlib
    import ttnn
    import nemotron3_mamba2_step as step_mod
    # Harness caches the module; force-reload to pick up v0.4.0a additions.
    step_mod = importlib.reload(step_mod)

    # Reuse harness mesh if available, else open one.
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
        # ── Build random per-layer constants (would normally come from
        #    safetensors weights). ─────────────────────────────────────
        rng = np.random.default_rng(seed=42)
        dt_bias_np = rng.standard_normal(NUM_HEADS, dtype=np.float32) * 0.01
        A_log_np   = rng.standard_normal(NUM_HEADS, dtype=np.float32) * 0.1
        D_np       = rng.standard_normal(NUM_HEADS, dtype=np.float32) * 0.05

        log("prepare_mamba2_constants (one-shot pad + upload)…")
        t0 = time.time()
        const = step_mod.prepare_mamba2_constants(
            dt_bias_np, A_log_np, D_np,
            B=B, num_heads=NUM_HEADS, device=mesh,
        )
        log(f"  const prep in {time.time() - t0:.2f}s")

        # ── Per-call random inputs (variable across calls)─────────────
        old_times = []
        new_times = []
        cos_state_list = []
        cos_y_list = []
        ssm_state_np = np.zeros((B, NUM_HEADS, HEAD_DIM, SSM_STATE), dtype=np.float32)
        # NB: we use the SAME ssm_state input for OLD and NEW (don't thread)
        # so we compare the same compute for both. Threading the state would
        # introduce an ordering dependency that confuses the comparison.

        for call in range(N_CALLS):
            x_np = rng.standard_normal((B, NUM_HEADS, HEAD_DIM), dtype=np.float32) * 0.1
            z_np = rng.standard_normal((B, NUM_HEADS, HEAD_DIM), dtype=np.float32) * 0.1
            dt_np = rng.standard_normal((B, NUM_HEADS), dtype=np.float32) * 0.01
            B_in_np = rng.standard_normal((B, N_GROUPS, SSM_STATE), dtype=np.float32) * 0.05
            C_in_np = rng.standard_normal((B, N_GROUPS, SSM_STATE), dtype=np.float32) * 0.05

            # OLD wrapper
            t0 = time.time()
            state_old, y_old = step_mod.mamba2_decode_step_ttnn(
                x=x_np, z=z_np, dt=dt_np,
                dt_bias=dt_bias_np, A_log=A_log_np, D=D_np,
                B_in=B_in_np, C_in=C_in_np,
                ssm_state=ssm_state_np,
                device=mesh, debug_mode=5,
            )
            t_old = time.time() - t0
            old_times.append(t_old)

            # NEW wrapper
            t0 = time.time()
            state_new, y_new = step_mod.mamba2_decode_step_ttnn_with_const_tt(
                x=x_np, z=z_np, dt=dt_np,
                dt_bias_tt=const["dt_bias_tt"],
                A_log_tt=const["A_log_tt"],
                D_tt=const["D_tt"],
                B_in=B_in_np, C_in=C_in_np,
                ssm_state=ssm_state_np,
                device=mesh, debug_mode=5,
            )
            t_new = time.time() - t0
            new_times.append(t_new)

            cos_s, mad_s = cos_and_mad(state_new, state_old)
            cos_y, mad_y = cos_and_mad(y_new, y_old)
            cos_state_list.append(cos_s)
            cos_y_list.append(cos_y)
            log(f"call {call:>2}  OLD={t_old:.3f}s  NEW={t_new:.3f}s  "
                f"cos(state)={cos_s:.6f}  cos(y)={cos_y:.6f}  "
                f"{'OK' if cos_s >= COS_GATE and cos_y >= COS_GATE else 'FAIL'}")

        # Skip call 0 (cold JIT) when summarising mean.
        warm_old = sum(old_times[1:]) / max(1, len(old_times) - 1)
        warm_new = sum(new_times[1:]) / max(1, len(new_times) - 1)
        speedup = warm_old / warm_new if warm_new > 0 else float("inf")
        min_cos = min(cos_state_list + cos_y_list)
        all_ok = all(c >= COS_GATE for c in cos_state_list + cos_y_list)

        log("")
        log("=" * 60)
        log("SUMMARY")
        log("=" * 60)
        log(f"  cold (call 0):  OLD={old_times[0]:.3f}s  NEW={new_times[0]:.3f}s")
        log(f"  warm mean:      OLD={warm_old:.3f}s  NEW={warm_new:.3f}s")
        log(f"  warm speedup:   {speedup:.2f}×  (NEW saves {warm_old - warm_new:.3f}s/call)")
        log(f"  min cos across all calls: {min_cos:.6f}")
        log("")
        log(f"v0.4.0a {'PASS ✓' if all_ok and speedup >= 1.0 else 'FAIL ✗'} "
            f"(cos ≥ {COS_GATE} on all {N_CALLS} calls + speedup ≥ 1×)")
        # Project per-step impact: 23 mamba2 layers × delta-per-call
        projected = 23 * (warm_old - warm_new)
        log(f"  Projected per-decode-step saving: "
            f"23 layers × {warm_old - warm_new:.3f}s = {projected:.2f}s")
        return 0 if all_ok else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
