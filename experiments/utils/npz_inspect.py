#!/usr/bin/env python3
"""Inspect a numpy .npz file: print keys, shapes, dtypes, mean/std.

Usage: python experiments/utils/npz_inspect.py <path.npz>
"""
import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("usage: npz_inspect.py <path.npz>")
        sys.exit(2)
    path = sys.argv[1]
    try:
        d = np.load(path)
    except FileNotFoundError:
        print(f"FAIL: not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"npz: {path}")
    print(f"{'key':<32s} {'shape':<24s} {'dtype':<10s} {'mean':>10s} {'std':>10s}")
    print("-" * 90)
    for k in d.files:
        a = d[k]
        try:
            m = float(a.mean()) if a.size > 0 else 0.0
            s = float(a.std()) if a.size > 0 else 0.0
        except Exception:
            m, s = float("nan"), float("nan")
        print(f"{k:<32s} {str(a.shape):<24s} {str(a.dtype):<10s} {m:>10.4f} {s:>10.4f}")


if __name__ == "__main__":
    main()
