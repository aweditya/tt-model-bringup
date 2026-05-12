#!/usr/bin/env python3
"""
Experiment 81c — introspect ttnn.swiglu signature on Blackhole / 0.69.

A0 found `ttnn.swiglu` exists but doesn't accept (g, u). Probe variants:
single-tensor packed, with-dim, with-weights, etc.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/81c_swiglu_signature.py
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn


def to_dev(arr, dtype=ttnn.bfloat16, device=None):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def main():
    device = ttnn.open_device(device_id=0)
    print("ttnn.swiglu introspection:\n")
    print(f"  help-like: {ttnn.swiglu.__doc__ if hasattr(ttnn.swiglu, '__doc__') else '(no doc)'}")
    print()

    # Common variants we should try
    intermediate = 1408

    # Variant 1: single packed tensor of size 2*intermediate, split along last dim
    g_and_u_np = np.random.randn(1, 2 * intermediate).astype(np.float32) * 0.1
    packed = to_dev(g_and_u_np, device=device)

    print(f"Trying ttnn.swiglu on packed [1, {2*intermediate}] tensor:")
    for label, fn in [
        ("ttnn.swiglu(packed)", lambda: ttnn.swiglu(packed)),
        ("ttnn.swiglu(packed, dim=-1)", lambda: ttnn.swiglu(packed, dim=-1)),
        ("ttnn.swiglu(packed, dim=1)", lambda: ttnn.swiglu(packed, dim=1)),
    ]:
        try:
            out = fn()
            print(f"  OK  {label}    -> shape={list(out.shape)}")
        except Exception as e:
            print(f"  FAIL {label}: {str(e)[:120]}")

    # Variant 2: two separate tensors with various kwargs
    g = to_dev(np.random.randn(1, intermediate).astype(np.float32) * 0.1, device=device)
    u = to_dev(np.random.randn(1, intermediate).astype(np.float32) * 0.1, device=device)
    print(f"\nTrying ttnn.swiglu on two [1, {intermediate}] tensors:")
    for label, fn in [
        ("ttnn.swiglu(g, u, dim=-1)", lambda: ttnn.swiglu(g, u, dim=-1)),
        ("ttnn.swiglu(g, dim=-1)", lambda: ttnn.swiglu(g, dim=-1)),
        ("ttnn.swiglu(input_tensor=packed)", lambda: ttnn.swiglu(input_tensor=packed)),
    ]:
        try:
            out = fn()
            print(f"  OK  {label}    -> shape={list(out.shape)}")
        except Exception as e:
            print(f"  FAIL {label}: {str(e)[:120]}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
