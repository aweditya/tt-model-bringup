#!/usr/bin/env python3
"""Quick dump of per-layer cos at a few positions from a cosine_ladder JSON."""
import json
import sys

path = sys.argv[1]
positions = [int(p) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 5, 50, 96]
layers = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [29, 30, 31, 32, 38, 39]

d = json.load(open(path))
results = None
for k, v in d.items():
    if isinstance(v, list) and v and isinstance(v[0], dict) and "cos_per_layer" in v[0]:
        results = v
        print(f"results key={k}, len={len(v)}")
        break

print(f"{'pos':>4}  " + "  ".join(f"L{L:>2}" for L in layers) + "  cos_final_norm")
for p in positions:
    if p < len(results):
        e = results[p]
        cpl = e["cos_per_layer"]
        cells = "  ".join(f"{cpl[L]:.4f}" for L in layers)
        cfn = e.get("cos_final_norm", float("nan"))
        print(f"{p:>4}  {cells}  {cfn:.4f}")
