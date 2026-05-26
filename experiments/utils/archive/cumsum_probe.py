#!/usr/bin/env python3
"""
Probe ttnn.cumsum availability and correctness on Blackhole.

Risk #1 from `research/c5_chunked_prefill_plan.md`: ttnn.cumsum is unverified
on Blackhole; never used in our codebase. The C'5 chunked-prefill plan needs
cumsum along the chunk-size axis (C=64) for the decay scalar g.

Questions answered:
1. Does ttnn.cumsum exist at all?
2. If so, does it accept fp32 and bf16?
3. Does it work at our chunk-sized shapes (1, 1, 64) or (32, 64)?
4. Numerical accuracy vs numpy reference?

Run on qb1 (qb2 is busy with C'2 gates):
    cd ~/tt-xla && .venv/bin/python experiments/utils/cumsum_probe.py
"""
import sys
import inspect
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe: ttnn.cumsum availability + correctness")
    print("=" * 64)

    fn = getattr(ttnn, "cumsum", None)
    if fn is None:
        print("\n✗ ttnn.cumsum NOT FOUND")
        print("  Fallback: prefix-sum tree via slice+add (log2(C)=6 levels)")
        # Look for alternatives
        for name in sorted(dir(ttnn)):
            if "sum" in name.lower() or "scan" in name.lower() or "cum" in name.lower():
                print(f"  related: ttnn.{name}")
        return

    print("\nttnn.cumsum FOUND")
    try:
        sig = inspect.signature(fn)
        print(f"  signature: {sig}")
    except (ValueError, TypeError) as e:
        print(f"  (signature unavailable: {e})")
    doc = inspect.getdoc(fn) or ""
    if doc:
        print(f"  doc:")
        for line in doc.split("\n")[:15]:
            print(f"    | {line}")

    # Test invocations
    device = ttnn.open_device(device_id=0)
    try:
        # Test 1: simple 1D fp32 cumsum
        print("\n" + "=" * 64)
        print("Test 1: fp32 vector of length 64, dim=-1")
        print("=" * 64)
        x_np = np.arange(64, dtype=np.float32) * 0.01 - 0.32  # values around 0
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np).unsqueeze(0),  # shape [1, 64]
            dtype=ttnn.float32,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        try:
            y_tt = ttnn.cumsum(x_tt, dim=-1)
            y_np = ttnn.to_torch(y_tt).float().cpu().numpy().flatten()
            ref = np.cumsum(x_np)
            cos = _cosine(y_np, ref)
            max_diff = float(np.abs(y_np[:len(ref)] - ref).max())
            print(f"  shape: {tuple(x_tt.shape)} → {tuple(y_tt.shape)}")
            print(f"  cosine vs numpy: {cos:.8f}")
            print(f"  max|Δ|: {max_diff:.6f}")
            print(f"  sample y[0:8]:  {y_np[:8]}")
            print(f"  ref y[0:8]:     {ref[:8]}")
            if cos < 0.9999 or max_diff > 1e-3:
                print(f"  ⚠ accuracy concern")
            else:
                print(f"  ✓ accurate")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

        # Test 2: bf16 — does it work?
        print("\n" + "=" * 64)
        print("Test 2: bf16 vector of length 64, dim=-1")
        print("=" * 64)
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        try:
            y_tt = ttnn.cumsum(x_tt, dim=-1)
            y_np = ttnn.to_torch(y_tt).float().cpu().numpy().flatten()
            ref = np.cumsum(x_np)
            cos = _cosine(y_np, ref)
            print(f"  cosine vs numpy: {cos:.8f}")
            print(f"  (bf16 expected: cosine ≥ 0.99)")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

        # Test 3: batched, our actual shape — [N_V=32, C=64]
        print("\n" + "=" * 64)
        print("Test 3: batched fp32 [N_V=32, C=64], dim=-1 (chunked-prefill shape)")
        print("=" * 64)
        x_np = np.random.randn(32, 64).astype(np.float32) * 0.1
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np), dtype=ttnn.float32,
            device=device, layout=ttnn.TILE_LAYOUT,
        )
        try:
            y_tt = ttnn.cumsum(x_tt, dim=-1)
            y_np = ttnn.to_torch(y_tt).float().cpu().numpy()
            ref = np.cumsum(x_np, axis=-1)
            cos = _cosine(y_np, ref)
            max_diff = float(np.abs(y_np - ref).max())
            print(f"  shape: {tuple(x_tt.shape)} → {tuple(y_tt.shape)}")
            print(f"  cosine vs numpy: {cos:.8f}")
            print(f"  max|Δ|: {max_diff:.6f}")
            if cos > 0.9999 and max_diff < 1e-3:
                print(f"  ✓ accurate at production shape")
            else:
                print(f"  ⚠ degraded accuracy at production shape")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

        # Test 4: dim=0 (along batch axis), just to know
        print("\n" + "=" * 64)
        print("Test 4: dim=0 on same [32, 64] tensor (sanity)")
        print("=" * 64)
        try:
            y_tt = ttnn.cumsum(x_tt, dim=0)
            y_np = ttnn.to_torch(y_tt).float().cpu().numpy()
            ref = np.cumsum(x_np, axis=0)
            cos = _cosine(y_np, ref)
            print(f"  cosine vs numpy: {cos:.8f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}")

        print("\n" + "=" * 64)
        print("Verdict")
        print("=" * 64)
        print("If test 3 passes cleanly, ttnn.cumsum is usable for C'5 directly.")
        print("Otherwise fallback: log2(64)=6-level prefix-sum tree via slice+add.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
