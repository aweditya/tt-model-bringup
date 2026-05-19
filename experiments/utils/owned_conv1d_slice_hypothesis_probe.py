#!/usr/bin/env python3
"""G1 probe: localize the owned conv1d wire-in bug.

Background: commit `df1cccc` documented that owned_conv1d G3 fails at
7.8% top-1 disagreement with immediate divergence, while G0 isolated test
passed at PCC 0.99999. Hypothesis: ttnn.slice(conv_st, [0, k], [D, k+1])
for k > 0 leaves the data at column k of the new tile rather than rebasing
it to column 0 where the kernel reads.

This probe runs three checks on a single-device fresh tensor:

1. Slice-readback equivalence: slice each column of a synthetic [D=2560, 3]
   conv_st and a [D, 4] w_conv into single-column tensors, read each slice
   back via ttnn.to_torch, and compare to the numpy equivalent. If readback
   matches numpy → logical slice extraction is correct.

2. Manual chain vs numpy oracle: run the production manual conv1d body
   (concat + mul + sum + silu) on the un-sliced tensors and compare to
   numpy.

3. Owned chain vs numpy oracle: run ttnn.experimental.qwen36_conv1d_decode
   _owned on the sliced tensors and compare to numpy + to the manual
   chain output.

Outcome interpretation:
- If check 1 passes AND owned matches manual: bug is elsewhere (e.g., the
  wire-in interacts with deltanet_step_tp state in a way this probe
  doesn't reproduce).
- If check 1 passes but owned doesn't match manual: bug is in how the
  kernel reads its sliced inputs (tile-layout assumption diverges from
  what slice produces internally).
- If check 1 fails: ttnn.slice has data-placement semantics we
  misunderstood; need to use a different extraction method (e.g.,
  pre-split at construction time, the G4 design).

Run on qb2 after stopping the resident server (chip contention):

    ssh qb2 'cd ~/tt-xla && bash experiments/serve/scripts/serve_tp.sh stop'
    ssh qb2 'cd ~/tt-xla && TT_METAL_HOME=$HOME/tenstorrent/tt-metal ARCH_NAME=blackhole \
        PYTHONPATH=$TT_METAL_HOME/ttnn LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_METAL_HOME/build_tracy_gcc12_nodist/ttnn:$TT_METAL_HOME/build_tracy_gcc12_nodist/lib \
        .venv/bin/python experiments/utils/owned_conv1d_slice_hypothesis_probe.py --d 2560 --device-id 0 \
        --output-json ~/tt-xla/.cache/qb2_tp_deltanet/owned_conv1d_slice_hypothesis_$(date +%Y%m%d_%H%M).json'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def numpy_oracle(conv_st, mixed, w_conv):
    """Reference: matches server_tp.py:572-580 manual production chain.
    conv_st: [D, 3]  (state0/1/2 across the K dim)
    mixed:   [D]
    w_conv:  [D, 4]  (4 taps)
    """
    pre_act = (conv_st[:, 0] * w_conv[:, 0] +
               conv_st[:, 1] * w_conv[:, 1] +
               conv_st[:, 2] * w_conv[:, 2] +
               mixed         * w_conv[:, 3])
    out = pre_act / (1.0 + np.exp(-pre_act))  # silu
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=2560,
                    help="CONV_DIM_CHIP. Default 2560 = qb2 TP4 production per-chip.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    D = args.d
    assert D > 0 and D % 32 == 0

    # Distinguishing magnitudes per column so slice readbacks are
    # visually obvious. col 0 = ~+1, col 1 = ~+2, col 2 = ~+3 for conv_st;
    # w_conv columns and mixed similar.
    np_conv_st = np.stack([
        rng.uniform(0.5, 1.5, D),
        rng.uniform(1.5, 2.5, D),
        rng.uniform(2.5, 3.5, D),
    ], axis=1).astype(np.float32)
    np_w_conv = np.stack([
        rng.uniform(0.05, 0.15, D),
        rng.uniform(0.15, 0.25, D),
        rng.uniform(0.25, 0.35, D),
        rng.uniform(0.35, 0.45, D),
    ], axis=1).astype(np.float32)
    np_mixed = rng.uniform(3.5, 4.5, D).astype(np.float32)

    oracle = numpy_oracle(np_conv_st, np_mixed, np_w_conv)

    import torch  # noqa: E402
    import ttnn   # noqa: E402

    dtype_map = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    ttnn_dtype = dtype_map[args.dtype]
    device = ttnn.open_device(device_id=args.device_id)

    def to_tt_2d(np_arr_2d):
        return ttnn.from_torch(torch.from_numpy(np_arr_2d.astype(np.float32)),
                                dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def to_tt_1d_as_2d(np_arr_1d):
        return ttnn.from_torch(torch.from_numpy(np_arr_1d.astype(np.float32)),
                                dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    report = {"d": D, "dtype": args.dtype, "seed": args.seed}

    try:
        conv_st_tt = to_tt_2d(np_conv_st)            # [D, 3]
        w_conv_tt = to_tt_2d(np_w_conv)              # [D, 4]
        mixed_tt = to_tt_1d_as_2d(np_mixed)          # [D]
        ttnn.synchronize_device(device)

        # ============================================================
        # CHECK 1 — slice readback equivalence
        # ============================================================
        print("=== CHECK 1 — slice readback vs numpy ===")
        slice_check = {}
        for k in range(3):
            sl = ttnn.slice(conv_st_tt, [0, k], [D, k + 1])
            sl_back = ttnn.to_torch(sl).float().cpu().numpy()
            expected = np_conv_st[:, k:k + 1]
            max_diff = float(np.abs(sl_back - expected).max()) if sl_back.shape == expected.shape else float("inf")
            entry = {
                "ttnn_shape_str": str(sl.shape),
                "readback_shape": list(sl_back.shape),
                "first_3_values": sl_back[:3].flatten().tolist(),
                "expected_first_3": expected[:3].flatten().tolist(),
                "max_abs_diff_vs_numpy_slice": max_diff,
            }
            slice_check[f"conv_st_col_{k}"] = entry
            print(f"  conv_st[:, {k}:{k+1}]: shape={sl.shape} "
                  f"readback={sl_back.shape} first[0]={sl_back[0].tolist()} "
                  f"expected[0]={expected[0].tolist()} max_diff={max_diff:.6e}")

        for k in range(4):
            sl = ttnn.slice(w_conv_tt, [0, k], [D, k + 1])
            sl_back = ttnn.to_torch(sl).float().cpu().numpy()
            expected = np_w_conv[:, k:k + 1]
            max_diff = float(np.abs(sl_back - expected).max()) if sl_back.shape == expected.shape else float("inf")
            slice_check[f"w_conv_col_{k}"] = {
                "max_abs_diff_vs_numpy_slice": max_diff,
                "readback_shape": list(sl_back.shape),
                "first_3_values": sl_back[:3].flatten().tolist(),
                "expected_first_3": expected[:3].flatten().tolist(),
            }
            print(f"  w_conv[:, {k}:{k+1}]: readback={sl_back.shape} first[0]={sl_back[0].tolist()} "
                  f"expected[0]={expected[0].tolist()} max_diff={max_diff:.6e}")

        report["check1_slice_readback"] = slice_check

        # ============================================================
        # CHECK 2 — manual chain on un-sliced vs numpy oracle
        # ============================================================
        print("\n=== CHECK 2 — manual chain (un-sliced) vs numpy oracle ===")
        mixed_col_manual = ttnn.reshape(mixed_tt, [D, 1])
        conv_input = ttnn.concat([conv_st_tt, mixed_col_manual], dim=-1)
        conv_prod = ttnn.mul(conv_input, w_conv_tt)
        conv_out_manual = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
        manual_back = ttnn.to_torch(conv_out_manual).float().cpu().numpy()
        if manual_back.ndim == 2 and manual_back.shape[1] == 1:
            manual_back = manual_back[:, 0]
        elif manual_back.ndim != 1:
            manual_back = manual_back.reshape(-1)
        manual_max_diff = float(np.abs(manual_back[:D] - oracle).max())
        report["check2_manual_vs_oracle"] = {
            "max_abs_diff": manual_max_diff,
            "manual_first_5": manual_back[:5].tolist(),
            "oracle_first_5": oracle[:5].tolist(),
        }
        print(f"  manual vs oracle: max_diff={manual_max_diff:.6f} (≤ 0.01 expected for bf16)")

        # ============================================================
        # CHECK 3 — owned chain (sliced inputs) vs numpy oracle + vs manual
        # ============================================================
        print("\n=== CHECK 3 — owned kernel (sliced inputs) vs oracle + manual ===")
        state0 = ttnn.slice(conv_st_tt, [0, 0], [D, 1])
        state1 = ttnn.slice(conv_st_tt, [0, 1], [D, 2])
        state2 = ttnn.slice(conv_st_tt, [0, 2], [D, 3])
        w0 = ttnn.slice(w_conv_tt, [0, 0], [D, 1])
        w1 = ttnn.slice(w_conv_tt, [0, 1], [D, 2])
        w2 = ttnn.slice(w_conv_tt, [0, 2], [D, 3])
        w3 = ttnn.slice(w_conv_tt, [0, 3], [D, 4])
        mixed_col_owned = ttnn.reshape(mixed_tt, [D, 1])

        s0_out, s1_out, s2_out, conv_out_owned_2d = ttnn.experimental.qwen36_conv1d_decode_owned(
            mixed_col_owned, state0, state1, state2, w0, w1, w2, w3)
        owned_back_2d = ttnn.to_torch(conv_out_owned_2d).float().cpu().numpy()
        if owned_back_2d.ndim == 2 and owned_back_2d.shape[1] == 1:
            owned_back = owned_back_2d[:, 0]
        else:
            owned_back = owned_back_2d.reshape(-1)

        owned_vs_oracle = float(np.abs(owned_back[:D] - oracle).max())
        owned_vs_manual = float(np.abs(owned_back[:D] - manual_back[:D]).max())
        report["check3_owned_vs_oracle_and_manual"] = {
            "owned_vs_oracle_max_diff": owned_vs_oracle,
            "owned_vs_manual_max_diff": owned_vs_manual,
            "owned_first_5": owned_back[:5].tolist(),
            "manual_first_5": manual_back[:5].tolist(),
            "oracle_first_5": oracle[:5].tolist(),
        }
        print(f"  owned vs oracle:  max_diff={owned_vs_oracle:.6f}")
        print(f"  owned vs manual:  max_diff={owned_vs_manual:.6f}")
        print(f"  owned[0:5]:  {[f'{x:.4f}' for x in owned_back[:5]]}")
        print(f"  manual[0:5]: {[f'{x:.4f}' for x in manual_back[:5]]}")
        print(f"  oracle[0:5]: {[f'{x:.4f}' for x in oracle[:5]]}")

        # ============================================================
        # DIAGNOSIS
        # ============================================================
        slice_clean = all(
            v["max_abs_diff_vs_numpy_slice"] <= 0.05  # accept up to half a BF16 ULP at magnitude 4
            for v in slice_check.values()
        )
        manual_clean = manual_max_diff <= 0.05
        owned_oracle_clean = owned_vs_oracle <= 0.05
        owned_manual_clean = owned_vs_manual <= 0.05

        diagnosis = {
            "slice_readback_clean": slice_clean,
            "manual_chain_correct": manual_clean,
            "owned_matches_oracle": owned_oracle_clean,
            "owned_matches_manual": owned_manual_clean,
        }

        if slice_clean and owned_oracle_clean and owned_manual_clean:
            diagnosis["conclusion"] = (
                "BUG NOT REPRODUCED in this probe. Single-step single-layer slice + "
                "owned kernel produces correct output. The G3 7.8% disagreement must "
                "come from somewhere else — maybe the state-shift writes interact with "
                "subsequent forward calls in a way this single-shot probe doesn't see."
            )
        elif not slice_clean:
            diagnosis["conclusion"] = (
                "SLICE BUG CONFIRMED. ttnn.slice's per-column extraction does NOT match "
                "numpy column slicing at the logical level. The kernel is being fed wrong "
                "data. Fix: G4 pre-split at bootstrap (each tap stored as its own [D, 1] "
                "tensor from the start)."
            )
        elif not owned_manual_clean:
            diagnosis["conclusion"] = (
                "KERNEL READS WRONG TILE POSITION. Slices match numpy at logical level, "
                "and the manual chain matches the oracle, but the owned kernel doesn't "
                "match the manual chain on the same inputs. The kernel must be reading "
                "from a tile position that ttnn.slice doesn't put the data at. Fix: G4 "
                "pre-split eliminates the slice entirely."
            )
        else:
            diagnosis["conclusion"] = "INCONCLUSIVE — manual chain itself disagrees with the oracle by more than 0.05."

        report["diagnosis"] = diagnosis
        print(f"\n=== DIAGNOSIS ===\n  {diagnosis['conclusion']}")

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2))
            print(f"\n[save] {args.output_json}")

        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    raise SystemExit(main())
