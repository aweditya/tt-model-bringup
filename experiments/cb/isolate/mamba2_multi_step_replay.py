#!/usr/bin/env python3
"""MM7 G1 day-5 — multi-step decode replay (recurrence validation).

Drives the owned `nemotron3_mamba2_decode_owned` kernel through N
sequential decode steps with the ssm_state THREADED through each
step (state_out[t] → state_in[t+1]). Compares each step's y + final
state against `mamba2_decode_step` (numpy oracle).

The single-step smoke (`mamba2_kernel_mode3_smoke.py`) already proves
the kernel computes `state_out` and `y` correctly for one step. This
probe additionally exercises the RECURRENCE: bf16 / fp32-acc precision
drift accumulates across steps, and `pack_reconfig_data_format` /
LLK-state errors that don't surface in step 0 sometimes appear at
step 2+.

Gate: per-step cos ≥ 0.999 against the oracle. No drift past pos 8.

REUSE: pulls `multistep_replay` + `compare_outputs` from
`experiments/utils/test_mamba2_decode_isolated.py` so the comparison
logic is the same one the rest of the G-ladder uses.

Differences from the canonical G0a harness:
- Uses a SMALL fixture (NUM_HEADS=1) matching the kernel's per-block
  shape. G2 multi-core will support the full 64-head fixture directly.
- Per-step we drive the kernel via ttnn (single program per step), then
  read back to numpy for the cos check.

Run on qb1:
    cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/mamba2_multi_step_replay.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))
from mamba2_numpy_oracle import mamba2_decode_step  # noqa: E402
from test_mamba2_decode_isolated import compare_outputs, multistep_replay  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Single-head shapes (kernel's native unit).
B = 1
NUM_HEADS = 1
HEAD_DIM = 64
SSM_STATE = 128
N_GROUPS = 1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_small_fixture(seed: int = 0) -> dict:
    """Build a (B=1, NUM_HEADS=1, HEAD_DIM=64, SSM_STATE=128) fixture
    matching the kernel's per-block shape. Inputs sized for dt_eff
    inside the [floor, max] clamp window (same magnitudes as the
    single-step smoke)."""
    rng = np.random.default_rng(seed)
    # dt: scalar per (B, head). Set so softplus(dt+dt_bias) ~ 0.030.
    dt = np.full((B, NUM_HEADS), -3.5, dtype=np.float32)
    dt_bias = np.zeros((NUM_HEADS,), dtype=np.float32)
    # A_log: scalar per head. Set so decay ~ 0.996.
    A_log = np.full((NUM_HEADS,), -2.0, dtype=np.float32)
    D = np.full((NUM_HEADS,), 0.5, dtype=np.float32)
    return dict(
        x=rng.standard_normal((B, NUM_HEADS, HEAD_DIM)).astype(np.float32) * 0.5,
        z=rng.standard_normal((B, NUM_HEADS, HEAD_DIM)).astype(np.float32),
        dt=dt,
        dt_bias=dt_bias,
        A_log=A_log,
        D=D,
        B_in=rng.standard_normal((B, N_GROUPS, SSM_STATE)).astype(np.float32) * 0.5,
        C_in=rng.standard_normal((B, N_GROUPS, SSM_STATE)).astype(np.float32) * 0.5,
        ssm_state=np.zeros((B, NUM_HEADS, HEAD_DIM, SSM_STATE), dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=8,
                        help="number of decode steps to replay")
    parser.add_argument("--seed", type=int, default=42,
                        help="rng seed for fixture and per-step input draws")
    parser.add_argument("--cos-gate", type=float, default=0.999,
                        help="per-step cosine pass threshold")
    args = parser.parse_args()

    log("import ttnn …")
    import ttnn
    log(f"  ttnn.__file__ = {ttnn.__file__}")

    fn = getattr(ttnn.experimental, "nemotron3_mamba2_decode_owned", None)
    if fn is None:
        log("FAIL: ttnn.experimental.nemotron3_mamba2_decode_owned not registered")
        return 1

    log(f"opening single device …")
    device = ttnn.open_device(device_id=0)

    try:
        def tt_bf16(arr):
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(arr)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )

        def tt_fp32(arr):
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(arr)),
                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
            )

        # ── kernel step_fn: matches mamba2_decode_step's signature ──
        def kernel_step_fn(*, x, z, dt, dt_bias, A_log, D, B_in, C_in, ssm_state):
            """Drive the kernel for one decode step. Mutates ssm_state in
            place to thread it across steps (the harness reads
            fixture["ssm_state"] after each step). Returns y as a numpy
            array matching the oracle's signature.
            """
            # Pad scalar-like inputs to (NUM_HEADS, 32, 32) / (B, NUM_HEADS, 32, 32).
            # The kernel reader expects tile-padded scalars.
            def pad_scalar_per_head(arr, leading=()):
                # arr shape: (NUM_HEADS,) or (B, NUM_HEADS) → (leading..., NUM_HEADS, 32, 32)
                shape = (*leading, NUM_HEADS, 32, 32)
                out = np.zeros(shape, dtype=np.float32)
                if arr.ndim == 1:  # (NUM_HEADS,)
                    out[..., 0, 0] = arr
                else:  # (B, NUM_HEADS)
                    out[..., 0, 0] = arr
                return out

            x_tt        = tt_bf16(x)
            z_tt        = tt_bf16(z)
            dt_tt       = tt_bf16(pad_scalar_per_head(dt, leading=(B,)))
            dt_bias_tt  = tt_bf16(pad_scalar_per_head(dt_bias))
            A_log_tt    = tt_bf16(pad_scalar_per_head(A_log))
            D_tt        = tt_bf16(pad_scalar_per_head(D))
            B_in_tt     = tt_bf16(B_in)
            C_in_tt     = tt_bf16(C_in)
            ssm_state_tt = tt_fp32(ssm_state)

            state_out, y_out = fn(
                x_tt, z_tt, dt_tt, dt_bias_tt, A_log_tt, D_tt,
                B_in_tt, C_in_tt, ssm_state_tt,
                debug_mode=5,  # production
            )

            y_np = ttnn.to_torch(ttnn.typecast(y_out, ttnn.float32)).cpu().numpy()
            state_np = ttnn.to_torch(state_out).cpu().numpy()  # fp32

            # Slice back to (B, NUM_HEADS, HEAD_DIM) and (B, NUM_HEADS, HEAD_DIM, SSM_STATE)
            y = y_np[0, 0, :HEAD_DIM].reshape(B, NUM_HEADS, HEAD_DIM)
            new_state = state_np[0, 0, :HEAD_DIM, :SSM_STATE].reshape(
                B, NUM_HEADS, HEAD_DIM, SSM_STATE)

            # Mutate fixture's ssm_state in place (matches oracle's contract).
            ssm_state[...] = new_state
            return y

        # ── Build fixture, replay both paths ─────────────────────────────
        log(f"making small fixture (B={B}, NUM_HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}, "
            f"SSM_STATE={SSM_STATE}, N_GROUPS={N_GROUPS}) …")
        fix_oracle = make_small_fixture(seed=args.seed)
        fix_kernel = make_small_fixture(seed=args.seed)  # IDENTICAL inputs

        log(f"running oracle replay ({args.n_steps} steps) …")
        hist_oracle = multistep_replay(mamba2_decode_step, fix_oracle,
                                        n_steps=args.n_steps,
                                        randomise_inputs=True, seed=args.seed + 1)

        log(f"running kernel replay ({args.n_steps} steps) …")
        hist_kernel = multistep_replay(kernel_step_fn, fix_kernel,
                                        n_steps=args.n_steps,
                                        randomise_inputs=True, seed=args.seed + 1)

        # ── Per-step comparison ──────────────────────────────────────────
        log("\nper-step comparison (kernel vs oracle):")
        all_passed = True
        for step in range(args.n_steps):
            print(f"\n── step {step} ──")
            y_oracle = hist_oracle[step]["y"]
            y_kernel = hist_kernel[step]["y"]
            rep_y = compare_outputs(y_kernel, y_oracle,
                                     label=f"y[{step}]",
                                     cos_threshold=args.cos_gate)
            state_oracle = hist_oracle[step]["ssm_state"]
            state_kernel = hist_kernel[step]["ssm_state"]
            # State has shape [B, num_heads, head_dim, ssm_state]; treat head_axis=1.
            rep_state = compare_outputs(state_kernel, state_oracle,
                                         label=f"state[{step}]",
                                         cos_threshold=args.cos_gate)
            if not (rep_y["passed"] and rep_state["passed"]):
                all_passed = False

        log(f"\n{'PASS ✓' if all_passed else 'FAIL ✗'}  Mamba2 multi-step replay "
            f"({args.n_steps} steps, gate cos ≥ {args.cos_gate})")
        return 0 if all_passed else 1

    finally:
        log("closing device …")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
