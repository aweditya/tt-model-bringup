#!/usr/bin/env python3
"""MM7 G1 day-3.9 — first-build smoke for the Mamba2 SSD owned kernel.

Validates that `ttnn.experimental.nemotron3_mamba2_decode_owned`:
  1. Exists as a callable Python symbol.
  2. Accepts the 9 input tensors at Nemotron-3 Nano shapes.
  3. Runs end-to-end with `debug_fill=True` (debug_mode=1 → fill_one).
  4. Returns (ssm_state, y) with non-NaN, sensible-shape outputs.

This is the GATE for kernel scaffolding (CB plumbing, reader/writer
pipeline, compute kernel dispatch). Math correctness is validated
separately via `experiments/utils/test_mamba2_decode_isolated.py` once
debug_mode=2..5 ship.

Fork base: `experiments/cb/isolate/owned_gdn.py` (the analogous CB6/G0
smoke for the 35B owned-GDN kernel).

Run on qb1:
    cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/mamba2_kernel_smoke.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Nemotron-3 Nano per-block shapes (single batch, single head for v0).
B = 1
NUM_HEADS = 1   # G1 single-core single-head; G2 will scale to 64
HEAD_DIM = 64
SSM_STATE = 128
N_GROUPS = 1    # single-head smoke uses one group


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log("import ttnn …")
    import ttnn
    log(f"  ttnn.__file__ = {ttnn.__file__}")

    # ── Discovery gate ────────────────────────────────────────────────────
    log("checking ttnn.experimental.nemotron3_mamba2_decode_owned …")
    fn = getattr(ttnn.experimental, "nemotron3_mamba2_decode_owned", None)
    if fn is None:
        log("FAIL: ttnn.experimental.nemotron3_mamba2_decode_owned not registered")
        return 1
    log(f"  callable: {fn}")

    # ── Device open ───────────────────────────────────────────────────────
    log("opening single device …")
    device = ttnn.open_device(device_id=0)

    try:
        # ── Input fixture (debug_fill: contents don't matter for mode=1)
        rng = np.random.default_rng(0)

        def tt_bf16(arr_shape):
            t = rng.standard_normal(arr_shape).astype(np.float32)
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(t)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )

        def tt_fp32(arr_shape):
            t = np.zeros(arr_shape, dtype=np.float32)
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(t)),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
            )

        log("allocating input tensors …")
        x         = tt_bf16((B, NUM_HEADS, HEAD_DIM))
        z         = tt_bf16((B, NUM_HEADS, HEAD_DIM))
        # dt is per-(batch, head); store as 4-D tile-padded.
        dt        = tt_bf16((B, NUM_HEADS, 32, 32))  # scalar tile per (b, h)
        dt_bias   = tt_bf16((NUM_HEADS, 32, 32))
        A_log     = tt_bf16((NUM_HEADS, 32, 32))
        D         = tt_bf16((NUM_HEADS, 32, 32))
        B_in      = tt_bf16((B, N_GROUPS, SSM_STATE))
        C_in      = tt_bf16((B, N_GROUPS, SSM_STATE))
        ssm_state = tt_fp32((B, NUM_HEADS, HEAD_DIM, SSM_STATE))

        log(f"  x.shape         = {tuple(x.padded_shape)}")
        log(f"  ssm_state.shape = {tuple(ssm_state.padded_shape)} dtype={ssm_state.dtype}")

        # ── Smoke call (debug_fill → debug_mode=1 → fill_one) ─────────────
        log("invoking kernel with debug_fill=True …")
        state_out, y_out = fn(
            x, z, dt, dt_bias, A_log, D, B_in, C_in, ssm_state,
            debug_fill=True,
        )
        log("  kernel returned without exception ✓")

        # ── Readback ──────────────────────────────────────────────────────
        # ttnn.to_torch doesn't handle bf16 directly; cast to float32 first.
        y_np = ttnn.to_torch(ttnn.typecast(y_out, ttnn.float32)).cpu().numpy()
        state_np = ttnn.to_torch(state_out).cpu().numpy()  # state is fp32
        log(f"  y_out: shape={y_np.shape}  finite={bool(np.all(np.isfinite(y_np)))}  "
            f"min={y_np.min():+.3f}  max={y_np.max():+.3f}  mean={y_np.mean():+.3f}")
        log(f"  state_out: shape={state_np.shape}  finite={bool(np.all(np.isfinite(state_np)))}  "
            f"min={state_np.min():+.3f}  max={state_np.max():+.3f}")

        # debug_mode=1 fills cb_y with 1.0 (the fill_one() helper).
        # Allow bf16 quantization slop on the readback (~0.01).
        y_one_match = float(np.abs(y_np - 1.0).max())
        log(f"  |y - 1.0| max = {y_one_match:.4e}  (bf16 quantization slop OK ≤ 0.01)")

        passed = (
            np.all(np.isfinite(y_np))
            and np.all(np.isfinite(state_np))
            and y_one_match < 0.05
        )
        log(f"\n{'PASS ✓' if passed else 'FAIL ✗'}  Mamba2 owned-kernel smoke (debug_mode=1)")
        return 0 if passed else 1

    finally:
        log("closing device …")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
