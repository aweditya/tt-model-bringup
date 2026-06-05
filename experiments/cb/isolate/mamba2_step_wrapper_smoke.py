#!/usr/bin/env python3
"""MM7 G4 — smoke for the `mamba2_decode_step_ttnn` Python wrapper.

Sanity-checks `experiments/serve/nemotron3_mamba2_step.py` against the
numpy oracle at full Nemotron-3 Nano shapes (B=1, NUM_HEADS=64,
HEAD_DIM=64, SSM_STATE=128, N_GROUPS=8). PASS gate: state cos ≥ 0.999,
y cos ≥ 0.999.

This is essentially the G2 multi-head smoke (`mamba2_g2_multihead_smoke.py`)
re-routed through the wrapper module instead of inlining the padding +
upload logic, so we confirm the wrapper preserves correctness BEFORE
the server scaffold imports it.

Run on the QuietBox:
    cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/mamba2_step_wrapper_smoke.py
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
from mamba2_numpy_oracle import mamba2_decode_step  # noqa: E402
from nemotron3_mamba2_step import mamba2_decode_step_ttnn  # noqa: E402
from test_mamba2_decode_isolated import compare_outputs  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_fixture(B, num_heads, head_dim, ssm_state, n_groups, *, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        x=rng.standard_normal((B, num_heads, head_dim)).astype(np.float32) * 0.5,
        z=rng.standard_normal((B, num_heads, head_dim)).astype(np.float32),
        dt=np.full((B, num_heads), -3.5, dtype=np.float32),
        dt_bias=np.zeros((num_heads,), dtype=np.float32),
        A_log=np.full((num_heads,), -2.0, dtype=np.float32),
        D=np.full((num_heads,), 0.5, dtype=np.float32),
        B_in=rng.standard_normal((B, n_groups, ssm_state)).astype(np.float32) * 0.5,
        C_in=rng.standard_normal((B, n_groups, ssm_state)).astype(np.float32) * 0.5,
        ssm_state=rng.standard_normal(
            (B, num_heads, head_dim, ssm_state)).astype(np.float32) * 0.3,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ssm-state", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-gate", type=float, default=0.999)
    args = parser.parse_args()

    log(f"wrapper smoke: B={args.batch} NUM_HEADS={args.num_heads} "
        f"N_GROUPS={args.n_groups}")

    log("import ttnn …")
    import ttnn
    log("opening single device …")
    device = ttnn.open_device(device_id=0)

    try:
        fix = make_fixture(args.batch, args.num_heads, args.head_dim,
                            args.ssm_state, args.n_groups, seed=args.seed)

        # Oracle (mutates ssm_state in place)
        fix_oracle = copy.deepcopy(fix)
        log("running numpy oracle …")
        y_oracle = mamba2_decode_step(**fix_oracle)
        state_oracle = fix_oracle["ssm_state"]

        # Wrapper (returns new_state separately)
        log("running ttnn wrapper …")
        state_kernel, y_kernel = mamba2_decode_step_ttnn(
            **fix, device=device, debug_mode=5,
        )

        log("comparing …")
        rep_y = compare_outputs(y_kernel, y_oracle,
                                 label="y", cos_threshold=args.cos_gate)
        rep_state = compare_outputs(state_kernel, state_oracle,
                                     label="state", cos_threshold=args.cos_gate)
        passed = rep_y["passed"] and rep_state["passed"]

        log(f"\n{'PASS ✓' if passed else 'FAIL ✗'}  wrapper smoke "
            f"(B={args.batch}, NUM_HEADS={args.num_heads}, "
            f"gate cos ≥ {args.cos_gate})")
        return 0 if passed else 1
    finally:
        log("closing device …")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
