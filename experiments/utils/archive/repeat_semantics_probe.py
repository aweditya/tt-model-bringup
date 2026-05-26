#!/usr/bin/env python3
"""
Permanent utility — probe ttnn.repeat semantics.

PyTorch has TWO different broadcast operations:

  torch.tensor.repeat([N, 1]):   tile-style
    input  [h0, h1, h2]      →  [h0, h1, h2, h0, h1, h2]
  torch.repeat_interleave(t, N): interleave-style
    input  [h0, h1, h2]      →  [h0, h0, h1, h1, h2, h2]

For GQA broadcasting (n_kv heads → n_q heads), we need interleave so that
adjacent q-heads share a k/v-head. Getting this wrong corrupts the
attention computation silently — same shapes, wrong values.

This script tests ttnn.repeat with a tiny tensor whose values reveal
which semantics ttnn uses. If tile-style (torch-like), we need a
workaround for interleave: reshape to [n, 1, d], repeat to [n, k, d],
reshape to [n*k, d].

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/repeat_semantics_probe.py
"""
import os, sys
import numpy as np
import torch
import ttnn


def main():
    print("=" * 64)
    print("Probe — ttnn.repeat semantics for GQA broadcast")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)

    # Tiny tensor where the values reveal the semantics
    # Each row is distinctive: row 0 = [1.0]*128, row 1 = [2.0]*128, etc.
    N_HEADS = 4
    HEAD_DIM = 128
    N_REP = 3

    base_np = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
    for i in range(N_HEADS):
        base_np[i] = float(i + 1)  # row i has all entries = (i+1)
    base = torch.from_numpy(base_np)
    base_tt = ttnn.from_torch(base.unsqueeze(0).unsqueeze(0), dtype=ttnn.float32,
                               device=device, layout=ttnn.TILE_LAYOUT)
    # Drop the leading 1s back off for the [N_HEADS, HEAD_DIM] shape
    base_tt = ttnn.reshape(base_tt, [N_HEADS, HEAD_DIM])
    print(f"input shape: {base_tt.shape}")
    print(f"input values: row i has all entries = (i+1) for i in [0, {N_HEADS-1}]")

    # Test 1: ttnn.repeat with [N_REP, 1]
    print(f"\n[Test 1] ttnn.repeat(input, ttnn.Shape([{N_REP}, 1]))")
    out = ttnn.repeat(base_tt, ttnn.Shape([N_REP, 1]))
    out_np = ttnn.to_torch(out).float().numpy()
    if out_np.ndim > 2:
        out_np = out_np.reshape(-1, HEAD_DIM)
    print(f"  output shape: {out_np.shape}")
    # Show first element of each row to identify the values
    head_pattern = [int(out_np[i, 0]) for i in range(out_np.shape[0])]
    print(f"  row-leading values: {head_pattern}")

    expected_tile = ([1, 2, 3, 4] * N_REP)
    expected_interleave = [v for v in [1, 2, 3, 4] for _ in range(N_REP)]
    print(f"  expected if TILE        (torch.repeat-like): {expected_tile}")
    print(f"  expected if INTERLEAVE  (torch.repeat_interleave-like): {expected_interleave}")

    if head_pattern == expected_tile:
        verdict = "TILE semantics (need workaround for GQA interleave)"
    elif head_pattern == expected_interleave:
        verdict = "INTERLEAVE semantics (works directly for GQA)"
    else:
        verdict = f"NEITHER — unexpected pattern: {head_pattern}"
    print(f"  ┌── VERDICT: {verdict}")

    # Test 2: workaround via unsqueeze + repeat + reshape
    # If ttnn.repeat is tile-style, this trick still gives interleave semantics
    # because the repeated dim is singleton.
    print(f"\n[Test 2] interleave-via-unsqueeze: "
          f"reshape→[{N_HEADS},1,{HEAD_DIM}] then repeat [1,{N_REP},1] then flatten")
    unsq = ttnn.reshape(base_tt, [N_HEADS, 1, HEAD_DIM])
    rep = ttnn.repeat(unsq, ttnn.Shape([1, N_REP, 1]))
    flat = ttnn.reshape(rep, [N_HEADS * N_REP, HEAD_DIM])
    flat_np = ttnn.to_torch(flat).float().numpy()
    if flat_np.ndim > 2:
        flat_np = flat_np.reshape(-1, HEAD_DIM)
    head_pattern2 = [int(flat_np[i, 0]) for i in range(flat_np.shape[0])]
    print(f"  output shape: {flat_np.shape}")
    print(f"  row-leading values: {head_pattern2}")
    if head_pattern2 == expected_interleave:
        print(f"  ✓ workaround produces correct interleave semantics")
    else:
        print(f"  ✗ workaround does NOT produce interleave; got {head_pattern2}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
