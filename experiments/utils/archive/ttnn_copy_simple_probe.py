#!/usr/bin/env python3
"""
Probe: does plain ttnn.copy(src, dst) work? (no slice involved)

Previous probe hung when dst was a ttnn.slice of a cache. This isolates
whether ttnn.copy itself works on Blackhole, or only specific dst types
hang.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/ttnn_copy_simple_probe.py
"""
import sys
import time
import signal
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def timeout_handler(signum, frame):
    raise TimeoutError("op took > 30 s")


def main():
    print("=" * 64)
    print("Probe: ttnn.copy on simple same-shape tensors (no slice dst)")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)
    try:
        shape = (1, 4, 32, 32)
        src_np = (np.random.default_rng(0).standard_normal(shape) * 0.1).astype(np.float32)
        dst_np = np.zeros(shape, dtype=np.float32)

        src_tt = ttnn.from_torch(torch.from_numpy(src_np), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        dst_tt = ttnn.from_torch(torch.from_numpy(dst_np), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT)

        print(f"  src shape: {tuple(src_tt.shape)}, dst shape: {tuple(dst_tt.shape)}")
        print("  calling ttnn.copy(src, dst)...")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            t0 = time.time()
            ttnn.copy(src_tt, dst_tt)
            ttnn.synchronize_device(device)
            signal.alarm(0)
            print(f"  ✓ copy returned in {(time.time()-t0)*1000:.2f} ms")
        except TimeoutError as e:
            signal.alarm(0)
            print(f"  ✗ HANG: {e}")
            return
        except Exception as e:
            signal.alarm(0)
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:200]}")
            return

        # Read back and verify
        back = ttnn.to_torch(dst_tt).float().cpu().numpy()
        cos = float(np.dot(back.flatten().astype(np.float64),
                           src_np.flatten().astype(np.float64)) /
                    (np.linalg.norm(back) * np.linalg.norm(src_np) + 1e-12))
        max_diff = float(np.abs(back - src_np).max())
        print(f"  dst vs src: cos={cos:.6f}, max|Δ|={max_diff:.4e}")
        if cos > 0.99:
            print("  ✓ ttnn.copy works for plain same-shape tensors.")
            print("    The earlier hang was specific to slice-as-dst.")
        else:
            print("  ✗ ttnn.copy ran but produced wrong data.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
