#!/usr/bin/env python3
"""MM7 G2 — Mamba2 SSD decode kernel @ full 64-head Nemotron shapes.

Drives the kernel at NUM_HEADS=64 (production) instead of NUM_HEADS=1
(G1 single-head). The program_factory already routes work as
`total_blocks = batch * num_heads` via `split_work_to_cores`, so this
test exercises the multi-core SPMD partition: 64 blocks distributed
across ~64 Tensix cores (one head per core).

Compares against `mamba2_decode_step` (the canonical numpy oracle which
handles arbitrary num_heads / n_groups). PASS gate:
  - state cos ≥ 0.999 overall, per-head cos ≥ 0.999 (min)
  - y cos ≥ 0.999 overall, per-head cos ≥ 0.999 (min)

Single-step only — multi-step recurrence at full shape is a follow-up.

Reads:
  - experiments/utils/mamba2_numpy_oracle.py (oracle)
  - experiments/utils/test_mamba2_decode_isolated.py (compare_outputs)

Run on QuietBox:
  ssh $TT_HOST 'cd ~/tt-xla && \\
      TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
      TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
      PYTHONPATH=$TT_METAL_HOME/ttnn \\
      LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
      .venv/bin/python -u experiments/cb/isolate/mamba2_g2_multihead_smoke.py'
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))
from mamba2_numpy_oracle import mamba2_decode_step  # noqa: E402
from test_mamba2_decode_isolated import compare_outputs  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_fixture(B: int, num_heads: int, head_dim: int, ssm_state: int,
                  n_groups: int, *, seed: int = 0) -> dict:
    """Build a multi-head fixture. Inputs sized so dt_eff ~ 0.030 (well
    inside the [floor, max] clamp window — same magnitudes as the G1
    smoke and multi-step probe).
    """
    rng = np.random.default_rng(seed)
    dt = np.full((B, num_heads), -3.5, dtype=np.float32)
    dt_bias = np.zeros((num_heads,), dtype=np.float32)
    A_log = np.full((num_heads,), -2.0, dtype=np.float32)
    D = np.full((num_heads,), 0.5, dtype=np.float32)
    return dict(
        x=rng.standard_normal((B, num_heads, head_dim)).astype(np.float32) * 0.5,
        z=rng.standard_normal((B, num_heads, head_dim)).astype(np.float32),
        dt=dt,
        dt_bias=dt_bias,
        A_log=A_log,
        D=D,
        B_in=rng.standard_normal((B, n_groups, ssm_state)).astype(np.float32) * 0.5,
        C_in=rng.standard_normal((B, n_groups, ssm_state)).astype(np.float32) * 0.5,
        ssm_state=(
            rng.standard_normal((B, num_heads, head_dim, ssm_state)).astype(np.float32)
            * 0.3
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1,
                        help="batch size (G2 default 1; set to 2+ to exercise G3 batched)")
    parser.add_argument("--num-heads", type=int, default=64,
                        help="number of heads (default: 64 = full Nemotron)")
    parser.add_argument("--n-groups", type=int, default=8,
                        help="number of B/C groups (default: 8 = Nemotron config)")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ssm-state", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-gate", type=float, default=0.999)
    args = parser.parse_args()

    B = args.batch
    log(f"G2 multi-head smoke: B={B} num_heads={args.num_heads} "
        f"head_dim={args.head_dim} ssm_state={args.ssm_state} "
        f"n_groups={args.n_groups}")

    log("import ttnn …")
    import ttnn
    fn = getattr(ttnn.experimental, "nemotron3_mamba2_decode_owned", None)
    if fn is None:
        log("FAIL: ttnn.experimental.nemotron3_mamba2_decode_owned not registered")
        return 1
    log(f"  callable: {fn}")

    log("opening single device …")
    device = ttnn.open_device(device_id=0)

    try:
        fixture = make_fixture(B, args.num_heads, args.head_dim, args.ssm_state,
                                args.n_groups, seed=args.seed)
        # Deep-copy so the kernel and oracle each get a fresh state buffer
        # (the oracle mutates ssm_state in place).
        fix_oracle = copy.deepcopy(fixture)
        fix_kernel = copy.deepcopy(fixture)

        # ── Oracle ──────────────────────────────────────────────────────
        log("running numpy oracle …")
        y_oracle = mamba2_decode_step(**fix_oracle)
        state_oracle = fix_oracle["ssm_state"]
        log(f"  y_oracle: shape={y_oracle.shape} "
            f"min={y_oracle.min():+.3f} max={y_oracle.max():+.3f}")
        log(f"  state_oracle: shape={state_oracle.shape} "
            f"min={state_oracle.min():+.3f} max={state_oracle.max():+.3f}")

        # ── Kernel ──────────────────────────────────────────────────────
        log("uploading kernel inputs …")

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

        def pad_scalar_per_head(arr, leading=()):
            """Pad a (...,num_heads) array to (...,num_heads,32,32) — each
            scalar lands in tile cell [0,0]. The kernel reader uses block
            index = (batch * num_heads + head) to fetch tile-per-head."""
            shape = (*leading, args.num_heads, 32, 32)
            out = np.zeros(shape, dtype=np.float32)
            out[..., 0, 0] = arr
            return out

        def pad_scalar_per_head_per_batch(arr, B_):
            """G3 helper: tile a per-head (NUM_HEADS,) array across batches
            to (B, NUM_HEADS, 32, 32). The reader uses `global_block` (which
            includes batch) as the tile index for per-head weights, so for
            B>1 we must replicate so each (batch, head) block reads the
            correct head's weight tile. With B=1 this collapses to the
            single-batch path above (no replication needed)."""
            shape = (B_, args.num_heads, 32, 32)
            out = np.zeros(shape, dtype=np.float32)
            # arr is (NUM_HEADS,); broadcast over batch axis.
            out[..., 0, 0] = arr[None, :]
            return out

        def pad_per_head_vector(arr, last_dim):
            """Pad shape (B, NUM_HEADS, last_dim) → (B, NUM_HEADS, 32, last_dim)
            with values in row 0 of each (32, last_dim) plane. This puts each
            head's `last_dim` values in a separate set of tile-rows so the
            reader's `global_block * tiles_per_block` indexing maps to the
            correct head."""
            B_, H, D = arr.shape
            out = np.zeros((B_, H, 32, last_dim), dtype=np.float32)
            out[:, :, 0, :] = arr
            return out

        def replicate_per_group_to_per_head(group_arr, num_heads):
            """Take (B, n_groups, last_dim) → (B, num_heads, 32, last_dim)
            replicating each group's values to all heads in that group. Used
            for B and C, which are per-group in the model but the reader
            treats as per-head."""
            B_, G, D = group_arr.shape
            heads_per_group = num_heads // G
            assert num_heads % G == 0, (
                f"num_heads {num_heads} not divisible by n_groups {G}")
            out = np.zeros((B_, num_heads, 32, D), dtype=np.float32)
            for h in range(num_heads):
                g = h // heads_per_group
                out[:, h, 0, :] = group_arr[:, g, :]
            return out

        x_padded = pad_per_head_vector(fix_kernel["x"], args.head_dim)
        z_padded = pad_per_head_vector(fix_kernel["z"], args.head_dim)
        B_padded = replicate_per_group_to_per_head(fix_kernel["B_in"], args.num_heads)
        C_padded = replicate_per_group_to_per_head(fix_kernel["C_in"], args.num_heads)

        # Per-head weights (dt_bias, A_log, D) need batch replication for B>1
        # so the reader's global_block tile-index points at the right head
        # for every (batch, head) block.
        x_tt        = tt_bf16(x_padded)
        z_tt        = tt_bf16(z_padded)
        dt_tt       = tt_bf16(pad_scalar_per_head(fix_kernel["dt"], leading=(B,)))
        dt_bias_tt  = tt_bf16(pad_scalar_per_head_per_batch(fix_kernel["dt_bias"], B))
        A_log_tt    = tt_bf16(pad_scalar_per_head_per_batch(fix_kernel["A_log"], B))
        D_tt        = tt_bf16(pad_scalar_per_head_per_batch(fix_kernel["D"], B))
        B_in_tt     = tt_bf16(B_padded)
        C_in_tt     = tt_bf16(C_padded)
        ssm_state_tt = tt_fp32(fix_kernel["ssm_state"])

        log("invoking kernel (debug_mode=5) …")
        t0 = time.time()
        state_out, y_out = fn(
            x_tt, z_tt, dt_tt, dt_bias_tt, A_log_tt, D_tt,
            B_in_tt, C_in_tt, ssm_state_tt,
            debug_mode=5,
        )
        log(f"  kernel returned in {time.time() - t0:.3f}s")

        y_kernel = ttnn.to_torch(ttnn.typecast(y_out, ttnn.float32)).cpu().numpy()
        state_kernel = ttnn.to_torch(state_out).cpu().numpy()
        log(f"  y_kernel raw shape: {y_kernel.shape}")
        log(f"  state_kernel raw shape: {state_kernel.shape}")

        # y was uploaded as (B, NUM_HEADS, 32, HEAD_DIM) so output should be
        # same shape; extract row 0 of each (32, HEAD_DIM) plane.
        if y_kernel.ndim == 4 and y_kernel.shape[-2] == 32:
            y_kernel = y_kernel[:, :, 0, :args.head_dim]
        else:
            y_kernel = y_kernel[:, :, :args.head_dim].reshape(
                B, args.num_heads, args.head_dim)
        # State has logical shape (B, NUM_HEADS, HEAD_DIM, SSM_STATE); no
        # row-0 extraction needed.
        state_kernel = state_kernel[:, :, :args.head_dim, :args.ssm_state].reshape(
            B, args.num_heads, args.head_dim, args.ssm_state)
        log(f"  y_kernel logical shape: {y_kernel.shape}")
        log(f"  state_kernel logical shape: {state_kernel.shape}")

        # ── Compare ─────────────────────────────────────────────────────
        log("comparing kernel vs oracle …")
        rep_y = compare_outputs(y_kernel, y_oracle,
                                 label="y", cos_threshold=args.cos_gate)
        rep_state = compare_outputs(state_kernel, state_oracle,
                                     label="state", cos_threshold=args.cos_gate)

        passed = rep_y["passed"] and rep_state["passed"]
        log(f"\n{'PASS ✓' if passed else 'FAIL ✗'}  G2 multi-head smoke "
            f"(num_heads={args.num_heads}, n_groups={args.n_groups}, "
            f"gate cos ≥ {args.cos_gate})")
        return 0 if passed else 1

    finally:
        log("closing device …")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
