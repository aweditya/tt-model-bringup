#!/usr/bin/env python3
"""MM7 G1 day-4 — debug_mode=2/3 smoke for the Mamba2 SSD owned kernel.

Validates the SSM state-update math against a numpy oracle. Mode is
selectable via CLI (defaults to 3, the full state update):

  mode=2: state_out = decay * state_in  (no input contribution)
  mode=3: state_out = decay * state_in + dt_eff * x[d] * B[s]

In both modes `y` is sentinel-1.0 (the C·state reduce + D·x skip land at
day-4.5 modes 4/5). Pass criteria: per-tile cos ≥ 0.999, rel err < 5e-2
(bf16 inputs + fp32 accumulator), |y - 1.0| < 0.05.

Fork base: `experiments/cb/isolate/mamba2_kernel_smoke.py` (the day-3.9
mode=1 scaffolding smoke).

Run on qb1:
    cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/mamba2_kernel_mode3_smoke.py [--mode 2|3]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Nemotron-3 Nano per-block shapes (single batch, single head for v0).
B = 1
NUM_HEADS = 1
HEAD_DIM = 64
SSM_STATE = 128
N_GROUPS = 1

# Constants — must match the hardcoded bit patterns in
# device/kernels/compute/nemotron3_mamba2_decode_owned.cpp (file header §:
# SOFTPLUS_*_BITS / TIME_STEP_*_BITS).
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
TIME_STEP_FLOOR = 1e-4
TIME_STEP_MAX = 0.1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def softplus(x: np.ndarray) -> np.ndarray:
    # Standard softplus with the same threshold (x > 20 → linear) the kernel uses.
    out = np.where(x > SOFTPLUS_THRESHOLD, x, np.log1p(np.exp(np.minimum(x, SOFTPLUS_THRESHOLD))))
    return out.astype(np.float32)


def numpy_oracle_state_out(
    state_in: np.ndarray,     # (1, 1, HEAD_DIM, SSM_STATE) fp32
    x: np.ndarray,            # (1, 1, 32, 32) bf16-rounded fp32 (tile-padded)
    dt: np.ndarray,           # (1, 1, 32, 32) bf16-rounded fp32 — scalar in [0,0]
    dt_bias: np.ndarray,      # (1, 32, 32)    bf16-rounded fp32 — scalar in [0,0]
    A_log: np.ndarray,        # (1, 32, 32)    bf16-rounded fp32 — scalar in [0,0]
    B_in: np.ndarray,         # (1, 1, 32, 32) bf16-rounded fp32 — vector in row 0
) -> np.ndarray:
    """Replicates the kernel's mode=3 math in fp32 (kernel accumulates in fp32 dest).

    All input tiles are 32×32 tile-padded; meaningful values live in:
      - dt, dt_bias, A_log: cell [0, 0] (broadcast scalar)
      - x: row 0 (head_dim vector, cols 0..31 in tile 0, cols 32..63 in tile 1)
      - B: row 0 (ssm_state vector, cols 0..31 .. 96..127 in tiles 0..3)

    The kernel reads `dt`, `dt_bias`, `A_log` via `add_tiles + softplus +
    clamp` which is full-tile element-wise — every position computes the
    same scalar because the inputs are constant. For the oracle, we use
    cell [0,0] semantics.
    """
    # Extract scalars from the tile-padded inputs.
    dt_scalar      = float(dt[0, 0, 0, 0])
    dt_bias_scalar = float(dt_bias[0, 0, 0])
    A_log_scalar   = float(A_log[0, 0, 0])

    # dt_eff = clamp(softplus(dt + dt_bias), floor, max)
    dt_eff = np.clip(
        softplus(np.array([dt_scalar + dt_bias_scalar], dtype=np.float32)),
        TIME_STEP_FLOOR, TIME_STEP_MAX,
    )[0]
    # decay = exp(-exp(A_log) * dt_eff)
    A = -np.exp(A_log_scalar)
    decay = float(np.exp(A * dt_eff))

    # Extract x as a (HEAD_DIM,) vector and B as a (SSM_STATE,) vector.
    # x: tile r=0 contains x[0..31]; we passed it as (1, 1, 64) so the
    # tile layout puts row 0 = head_dim vec, packed across head_dim_tiles.
    # The smoke uses ttnn.from_torch with shape (B, num_heads, head_dim)
    # which lays out head_dim along the W axis → row 0 of each tile holds
    # 32 contiguous values.
    x_vec = x.reshape(B, NUM_HEADS, -1)[0, 0, :HEAD_DIM]  # (HEAD_DIM,)
    B_vec = B_in.reshape(B, N_GROUPS, -1)[0, 0, :SSM_STATE]  # (SSM_STATE,)

    # state_out[d, s] = decay * state_in[d, s] + dt_eff * x[d] * B[s]
    # state_in shape: (1, 1, HEAD_DIM, SSM_STATE)
    state_in_2d = state_in[0, 0]  # (HEAD_DIM, SSM_STATE)
    outer = np.outer(x_vec.astype(np.float32), B_vec.astype(np.float32)) * dt_eff
    state_out = decay * state_in_2d.astype(np.float32) + outer
    return state_out[None, None, :, :].astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=3, choices=[2, 3],
                        help="debug_mode to test (2=decay only, 3=full state update)")
    args = parser.parse_args()
    mode = args.mode
    log(f"smoke target: debug_mode={mode}")

    log("import ttnn …")
    import ttnn
    log(f"  ttnn.__file__ = {ttnn.__file__}")

    fn = getattr(ttnn.experimental, "nemotron3_mamba2_decode_owned", None)
    if fn is None:
        log("FAIL: ttnn.experimental.nemotron3_mamba2_decode_owned not registered")
        return 1
    log(f"  callable: {fn}")

    log("opening single device …")
    device = ttnn.open_device(device_id=0)

    try:
        rng = np.random.default_rng(0)

        # Build numpy inputs with controlled magnitudes so dt_eff sits well
        # inside [floor, max] (i.e. softplus doesn't saturate, clamp doesn't trim).
        # Targeting dt_eff ≈ 0.03 → log(softplus(0.03)) ≈ -3.5, well below threshold.
        dt_np      = np.zeros((B, NUM_HEADS, 32, 32), dtype=np.float32)
        dt_np[..., 0, 0] = -3.5  # softplus(-3.5) ≈ 0.030; clamp leaves it.
        dt_bias_np = np.zeros((NUM_HEADS, 32, 32), dtype=np.float32)
        # A_log = -2 → A = -exp(-2) ≈ -0.135 → decay = exp(-0.135 * 0.030) ≈ 0.996
        A_log_np   = np.zeros((NUM_HEADS, 32, 32), dtype=np.float32)
        A_log_np[..., 0, 0] = -2.0
        # D, C, z: present (reader drains them) but unused at mode=3.
        D_np       = np.zeros((NUM_HEADS, 32, 32), dtype=np.float32)
        D_np[..., 0, 0] = 0.5  # value doesn't affect mode=3 output

        # x[d]: 64 head_dim values, row 0 only.
        x_np_flat = rng.standard_normal((B, NUM_HEADS, HEAD_DIM)).astype(np.float32) * 0.5
        # B[s]: 128 ssm_state values, row 0 only.
        B_np_flat = rng.standard_normal((B, N_GROUPS, SSM_STATE)).astype(np.float32) * 0.5
        C_np_flat = rng.standard_normal((B, N_GROUPS, SSM_STATE)).astype(np.float32) * 0.5
        z_np_flat = rng.standard_normal((B, NUM_HEADS, HEAD_DIM)).astype(np.float32)

        # state_in: (HEAD_DIM, SSM_STATE) fp32 with moderate magnitude.
        state_np = rng.standard_normal((B, NUM_HEADS, HEAD_DIM, SSM_STATE)).astype(np.float32) * 0.3

        # ── Upload to ttnn ────────────────────────────────────────────────
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

        log("allocating input tensors …")
        x         = tt_bf16(x_np_flat)
        z         = tt_bf16(z_np_flat)
        dt        = tt_bf16(dt_np)
        dt_bias   = tt_bf16(dt_bias_np)
        A_log     = tt_bf16(A_log_np)
        D         = tt_bf16(D_np)
        B_in      = tt_bf16(B_np_flat)
        C_in      = tt_bf16(C_np_flat)
        ssm_state = tt_fp32(state_np)

        log(f"invoking kernel with debug_mode={mode} …")
        state_out, y_out = fn(
            x, z, dt, dt_bias, A_log, D, B_in, C_in, ssm_state,
            debug_mode=mode,
        )
        log("  kernel returned without exception ✓")

        # ── Readback ──────────────────────────────────────────────────────
        y_tt = ttnn.to_torch(ttnn.typecast(y_out, ttnn.float32)).cpu().numpy()
        state_tt = ttnn.to_torch(state_out).cpu().numpy()  # fp32 readback
        log(f"  y_out:     shape={y_tt.shape}     finite={bool(np.all(np.isfinite(y_tt)))}  "
            f"min={y_tt.min():+.3f}  max={y_tt.max():+.3f}")
        log(f"  state_out: shape={state_tt.shape}  finite={bool(np.all(np.isfinite(state_tt)))}  "
            f"min={state_tt.min():+.3f}  max={state_tt.max():+.3f}")

        # ── Oracle (round inputs to bf16 to match what the kernel actually saw)
        def bf16_roundtrip(a: np.ndarray) -> np.ndarray:
            return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(torch.float32).numpy()

        oracle = numpy_oracle_state_out(
            state_in=state_np,
            # For mode=2 (decay×state only) we zero out x so the input
            # contribution drops out and the oracle reduces to decay*state.
            x=bf16_roundtrip(_pad_x_to_tile(x_np_flat if mode == 3
                                             else np.zeros_like(x_np_flat))),
            dt=bf16_roundtrip(dt_np),
            dt_bias=bf16_roundtrip(dt_bias_np),
            A_log=bf16_roundtrip(A_log_np),
            B_in=bf16_roundtrip(_pad_B_to_tile(B_np_flat)),
        )

        # The kernel writes state_out into the same (B, num_heads, head_dim, ssm_state)
        # tile layout as state_in. Extract the meaningful (HEAD_DIM, SSM_STATE) slice.
        state_tt_2d = state_tt[0, 0, :HEAD_DIM, :SSM_STATE]
        oracle_2d = oracle[0, 0]

        cos = cosine_sim(state_tt_2d, oracle_2d)
        mad = float(np.max(np.abs(state_tt_2d - oracle_2d)))
        rel = float(mad / (np.max(np.abs(oracle_2d)) + 1e-30))
        log(f"\n  state_out vs oracle:  cos = {cos:.6f}   max|err| = {mad:.4e}   "
            f"rel = {rel:.4e}   max|oracle| = {np.max(np.abs(oracle_2d)):.4f}")

        # y at mode=3 must still be sentinel 1.0 (write path proven by mode=1).
        y_one_match = float(np.abs(y_tt - 1.0).max())
        log(f"  |y - 1.0| max = {y_one_match:.4e}   (mode=3 keeps y sentinel; mode=4 wires D·x)")

        passed = (
            np.all(np.isfinite(y_tt))
            and np.all(np.isfinite(state_tt))
            and cos >= 0.999
            and rel < 5e-2          # bf16 inputs + fp32 acc → ~few % rel err is OK
            and y_one_match < 0.05
        )
        log(f"\n{'PASS ✓' if passed else 'FAIL ✗'}  Mamba2 owned-kernel smoke (debug_mode=3)")
        return 0 if passed else 1

    finally:
        log("closing device …")
        ttnn.close_device(device)


def _pad_x_to_tile(x_flat: np.ndarray) -> np.ndarray:
    """Reshape (B, NUM_HEADS, HEAD_DIM=64) into the (B, NUM_HEADS, 32, 32) tile
    layout used by the oracle (which expects row 0 to contain the head_dim
    values, packed across head_dim_tiles).

    The ttnn ttiler does this implicitly when from_torch sees a final-axis
    length of 64 — it produces (32, 64) padded to two 32×32 tiles, each tile
    holding 32 contiguous values along the W axis on row 0. The oracle
    extracts via `reshape(B, NUM_HEADS, -1)` and slices [:HEAD_DIM], so we
    just need to keep the values dense along the last axis.
    """
    out = np.zeros((x_flat.shape[0], x_flat.shape[1], 32, 64), dtype=np.float32)
    out[:, :, 0, :] = x_flat
    return out


def _pad_B_to_tile(B_flat: np.ndarray) -> np.ndarray:
    out = np.zeros((B_flat.shape[0], B_flat.shape[1], 32, 128), dtype=np.float32)
    out[:, :, 0, :] = B_flat
    return out


if __name__ == "__main__":
    sys.exit(main())
