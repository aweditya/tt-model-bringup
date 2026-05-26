#!/usr/bin/env python3
"""Quick inspection of HF hidden-state magnitudes layer-by-layer to see if
L31 / L39 outputs are unusual relative to their neighbors.

For each decoder layer, compute:
  - L2 norm of HF hidden state at pos 0 (post-layer output)
  - L2 norm of CONTRIBUTION (post_layer - prev_layer)  — the layer's net change to residual
  - max |abs value| of HF hidden state at pos 0
  - max |abs value| of CONTRIBUTION

Helps answer: do L31 and L39 contribute more / less than their neighbors at
pos 0? Wide-magnitude contributions get bf16-quantized worse.

Usage:
  python3 experiments/utils/hf_layer_magnitude_inspect.py \\
      .cache/hf_oracle_35b_needle100/hidden_states.npy \\
      [--pos 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hidden_states_npy")
    ap.add_argument("--pos", type=int, default=0)
    args = ap.parse_args()

    hs = np.load(args.hidden_states_npy)  # [n_layers+1, seq, HIDDEN]
    n_layers_plus_1, seq, hidden = hs.shape
    n_layers = n_layers_plus_1 - 1
    p = args.pos
    print(f"hidden_states shape: {hs.shape}, inspecting pos={p}")
    print()

    # Per-layer: norm of output, max abs value, norm of contribution (vs prev)
    print(f"{'layer':>7}  {'type?':>5}  {'L2(out)':>10}  {'max|out|':>10}  "
          f"{'L2(delta)':>10}  {'max|delta|':>11}")
    print("-" * 70)
    for li in range(n_layers + 1):
        out = hs[li, p]
        l2_out = float(np.linalg.norm(out))
        max_out = float(np.max(np.abs(out)))
        if li == 0:
            l2_delta = 0.0
            max_delta = 0.0
            name = "embed"
            type_marker = ""
        else:
            delta = out - hs[li - 1, p]
            l2_delta = float(np.linalg.norm(delta))
            max_delta = float(np.max(np.abs(delta)))
            name = f"L{li-1:02d}"
            # 35B AT pattern: L03, L07, L11, ... (every 4th)
            type_marker = "AT" if (li - 1) % 4 == 3 else "DN"
        print(f"{name:>7}  {type_marker:>5}  {l2_out:>10.2f}  {max_out:>10.4f}  "
              f"{l2_delta:>10.2f}  {max_delta:>11.4f}")


if __name__ == "__main__":
    main()
