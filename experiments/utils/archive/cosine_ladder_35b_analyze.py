#!/usr/bin/env python3
"""Summarize cosine_ladder_35b JSON output.

Prints:
  - per-position table (final, logits, selected layer cosines)
  - per-layer aggregate (min/mean across all positions)
  - first-divergence point + top-1 match rate

Usage:
  python3 experiments/utils/cosine_ladder_35b_analyze.py \\
    .cache/sanity_2026_05_22/cosine_ladder_35b_smoke10.json
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: cosine_ladder_35b_analyze.py <json>", file=sys.stderr)
        sys.exit(2)
    d = json.loads(Path(sys.argv[1]).read_text())
    n_pos = d["n_positions_tested"]
    n_layers = d["n_layers"]
    pp = d["per_pos"]
    n_layers_plus_1 = len(pp[0]["cos_per_layer"])  # 41

    print(f"positions tested: {n_pos}, layers: {n_layers}")
    print(f"top1 match: {d['top1_match_count']}/{n_pos} ({100*d['top1_match_rate']:.1f}%)")
    print(f"median cos_final: {d['median_cos_final_norm']:.4f}  cos_logits: {d['median_cos_logits']:.4f}")
    print(f"first divergence (cos<{d['drift_cos_threshold']}): pos={d['first_divergence_pos']} layer={d['first_divergence_layer']}")
    print()

    # Per-position table at selected layers
    pick = [0, 1, 11, 21, 31, 33, 34, 40]  # embed, L0, L10, L20, L30, L32, L33, L39
    pick_names = ["emb"] + [f"L{i-1:02d}" for i in pick[1:]]
    head = f"{'pos':>3} {'tt':>6} {'hf':>6} m " + " ".join(f"{n:>6}" for n in pick_names) + f" {'fnorm':>6} {'logit':>6}"
    print(head)
    for p in pp:
        cpl = p["cos_per_layer"]
        m = "Y" if p["top1_match"] else "N"
        line = f"{p['pos']:>3} {p['tt_argmax']:>6} {p['hf_argmax']:>6} {m} "
        line += " ".join(f"{cpl[i]:>6.4f}" for i in pick)
        line += f" {p['cos_final_norm']:>6.4f} {p['cos_logits']:>6.4f}"
        print(line)
    print()

    # Per-layer aggregate
    print(f"{'layer':>6}  {'min':>6}  {'mean':>6}  {'min@pos':>7}  flag")
    worst_layers = []
    for li in range(n_layers_plus_1):
        vals = [p["cos_per_layer"][li] for p in pp]
        mn = min(vals)
        mean = sum(vals) / len(vals)
        argmin = vals.index(mn)
        flag = ""
        if mn < 0.95:
            flag = "** <0.95 **"
        elif mn < 0.97:
            flag = "* <0.97 *"
        elif mn < 0.99:
            flag = "<0.99"
        name = "embed" if li == 0 else f"L{li-1:02d}"
        if mn < 0.99:
            worst_layers.append((name, mn, mean, argmin))
        print(f"{name:>6}  {mn:.4f}  {mean:.4f}  {argmin:>7}  {flag}")
    print()
    print(f"layers with min < 0.99: {len(worst_layers)}")
    if worst_layers:
        print("top 5 worst by min:")
        for name, mn, mean, argmin in sorted(worst_layers, key=lambda x: x[1])[:5]:
            print(f"  {name}: min={mn:.4f} (pos {argmin}) mean={mean:.4f}")


if __name__ == "__main__":
    main()
