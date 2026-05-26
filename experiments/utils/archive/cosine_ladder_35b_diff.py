#!/usr/bin/env python3
"""Diff two cosine_ladder_35b JSON outputs at per-position, per-layer granularity.

Use to confirm whether two ablation runs produced different numerics (or
whether a flag didn't take effect).

Usage:
  python3 experiments/utils/cosine_ladder_35b_diff.py <a.json> <b.json>
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("usage: cosine_ladder_35b_diff.py <a.json> <b.json>", file=sys.stderr)
        sys.exit(2)
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())

    print(f"A: {sys.argv[1]}")
    print(f"B: {sys.argv[2]}")
    print()

    n = min(len(a["per_pos"]), len(b["per_pos"]))
    diff_pos = 0
    diff_argmax = 0
    max_layer_delta_overall = 0.0
    max_layer_delta_pos = -1
    max_layer_delta_layer = -1

    for i in range(n):
        pa, pb = a["per_pos"][i], b["per_pos"][i]
        a_l = pa["cos_per_layer"]
        b_l = pb["cos_per_layer"]
        per_layer_deltas = [abs(x - y) for x, y in zip(a_l, b_l)]
        max_delta = max(per_layer_deltas)
        max_layer = per_layer_deltas.index(max_delta)
        if max_delta > max_layer_delta_overall:
            max_layer_delta_overall = max_delta
            max_layer_delta_pos = pa["pos"]
            max_layer_delta_layer = max_layer
        any_diff = max_delta > 0
        if any_diff:
            diff_pos += 1
        if pa["tt_argmax"] != pb["tt_argmax"]:
            diff_argmax += 1
            print(f"  pos {pa['pos']:>3} argmax differs: A={pa['tt_argmax']} B={pb['tt_argmax']}  "
                  f"(max_layer_delta={max_delta:.6f} at layer_idx={max_layer})")
        elif any_diff:
            name = "embed" if max_layer == 0 else f"L{max_layer-1:02d}"
            print(f"  pos {pa['pos']:>3} argmax SAME, cos differs: max_layer_delta={max_delta:.6f} at {name}")

    print()
    print(f"positions with any cos_per_layer difference:  {diff_pos}/{n}")
    print(f"positions with different tt_argmax:           {diff_argmax}/{n}")
    print(f"overall max layer cosine delta:               {max_layer_delta_overall:.6f}")
    if max_layer_delta_overall > 0:
        layer_name = "embed" if max_layer_delta_layer == 0 else f"L{max_layer_delta_layer-1:02d}"
        print(f"  at pos {max_layer_delta_pos}, layer {layer_name}")
    print()
    print(f"median cos_final  A={a['median_cos_final_norm']:.6f}  B={b['median_cos_final_norm']:.6f}  "
          f"(delta={a['median_cos_final_norm'] - b['median_cos_final_norm']:+.6f})")
    print(f"top1 match        A={a['top1_match_count']}/{a['n_positions_tested']}  "
          f"B={b['top1_match_count']}/{b['n_positions_tested']}")


if __name__ == "__main__":
    main()
