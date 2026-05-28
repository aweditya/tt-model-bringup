#!/usr/bin/env python3
"""
Experiment 39: Reproduce ttnn.mean memory over-allocation bug (tt-metal #32546)

The issue reports that ttnn.mean() allocates 256-1024x more L1 memory than needed
for tensors > ~1MB. This experiment:
  1. Tests ttnn.mean on progressively larger shapes
  2. Measures which shapes succeed/fail
  3. Compares against numpy reference for correctness on passing shapes
  4. Tests workaround: manual reduce via ttnn.sum + division

Reference: https://github.com/tenstorrent/tt-metal/issues/32546
"""

import numpy as np
import torch
import time
import ttnn

print("=" * 60)
print("Experiment 39: ttnn.mean Memory Over-Allocation Bug")
print("Reproducing tt-metal #32546")
print("=" * 60)

device = ttnn.open_device(device_id=0)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor):
    return ttnn.to_torch(tensor).float().numpy()

# ── Test shapes from the issue ──────────────────────────────
# Issue reports failing at (1000, 256, 1, 2) = 0.98 MB
# and passing at (100, 256, 1, 2) = 0.10 MB
test_shapes = [
    (32, 256, 1, 2),      # ~16 KB — should pass
    (100, 256, 1, 2),     # 100 KB — last known passing
    (256, 256, 1, 2),     # 256 KB — boundary?
    (512, 256, 1, 2),     # 512 KB — boundary?
    (1000, 256, 1, 2),    # ~1 MB — reported failing
    # Transformer-relevant shapes
    (1, 1, 32, 768),      # GPT-2 style: 1 batch, 32 tokens, hidden=768
    (1, 1, 128, 896),     # Qwen style: 1 batch, 128 tokens, hidden=896
    (1, 12, 32, 64),      # Attention scores: batch, heads, seq, head_dim
]

print("\n── Phase 1: Testing ttnn.mean on various shapes ──")
print(f"{'Shape':<30} {'Size (KB)':<12} {'Status':<10} {'Error'}")
print("-" * 80)

results = []
for shape in test_shapes:
    size_kb = np.prod(shape) * 2 / 1024  # bfloat16 = 2 bytes
    np_arr = np.random.randn(*shape).astype(np.float32)

    try:
        x = to_dev(np_arr)
        result = ttnn.mean(x, dim=-1)
        out = from_dev(result)

        # Check correctness vs numpy
        np_ref = np_arr.mean(axis=-1, keepdims=True)
        # Squeeze to match
        if out.shape != np_ref.shape:
            out = out.reshape(np_ref.shape) if np.prod(out.shape) == np.prod(np_ref.shape) else out

        cos_sim = np.dot(out.flatten(), np_ref.flatten()) / (
            np.linalg.norm(out.flatten()) * np.linalg.norm(np_ref.flatten()) + 1e-8)

        status = "PASS" if cos_sim > 0.99 else f"LOW({cos_sim:.4f})"
        print(f"{str(shape):<30} {size_kb:<12.1f} {status:<10}")
        results.append((shape, size_kb, "pass", cos_sim))

        ttnn.deallocate(result)
        ttnn.deallocate(x)

    except Exception as e:
        err_str = str(e)[:60]
        print(f"{str(shape):<30} {size_kb:<12.1f} {'FAIL':<10} {err_str}")
        results.append((shape, size_kb, "fail", err_str))

# ── Phase 2: Test workaround via ttnn.sum ───────────────────
print("\n── Phase 2: Workaround via ttnn.sum + division ──")
print(f"{'Shape':<30} {'Size (KB)':<12} {'Status':<10}")
print("-" * 60)

for shape in test_shapes:
    size_kb = np.prod(shape) * 2 / 1024
    np_arr = np.random.randn(*shape).astype(np.float32)

    try:
        x = to_dev(np_arr)
        # Manual mean: sum / count
        s = ttnn.sum(x, dim=-1)
        count = shape[-1]
        inv_count = to_dev(np.array([[1.0 / count]], dtype=np.float32))
        result = ttnn.multiply(s, inv_count)
        out = from_dev(result)

        np_ref = np_arr.mean(axis=-1, keepdims=True)
        if out.shape != np_ref.shape:
            try:
                out = out.reshape(np_ref.shape)
            except:
                pass

        cos_sim = np.dot(out.flatten(), np_ref.flatten()) / (
            np.linalg.norm(out.flatten()) * np.linalg.norm(np_ref.flatten()) + 1e-8)

        status = "PASS" if cos_sim > 0.99 else f"LOW({cos_sim:.4f})"
        print(f"{str(shape):<30} {size_kb:<12.1f} {status:<10}")

        ttnn.deallocate(result)
        ttnn.deallocate(s)
        ttnn.deallocate(x)

    except Exception as e:
        err_str = str(e)[:60]
        print(f"{str(shape):<30} {size_kb:<12.1f} {'FAIL':<10}")

# ── Phase 3: Find exact threshold ──────────────────────────
print("\n── Phase 3: Binary search for failure threshold ──")
# Search between 100 and 1000 in the first dim (with shape (N, 256, 1, 2))
lo, hi = 100, 1000
threshold = None

while lo < hi:
    mid = (lo + hi) // 2
    shape = (mid, 256, 1, 2)
    np_arr = np.random.randn(*shape).astype(np.float32)

    try:
        x = to_dev(np_arr)
        result = ttnn.mean(x, dim=-1)
        ttnn.deallocate(result)
        ttnn.deallocate(x)
        lo = mid + 1
    except:
        hi = mid
        threshold = mid

if threshold:
    size_kb = threshold * 256 * 1 * 2 * 2 / 1024
    print(f"  Failure threshold: N={threshold} (shape ({threshold}, 256, 1, 2), {size_kb:.0f} KB)")
else:
    print(f"  No failure found in range [100, 1000] — bug may be fixed in our version")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

n_pass = sum(1 for _, _, s, _ in results if s == "pass")
n_fail = sum(1 for _, _, s, _ in results if s == "fail")
print(f"  Shapes tested: {len(results)}")
print(f"  Passed: {n_pass}")
print(f"  Failed: {n_fail}")

if n_fail > 0:
    print(f"\n  BUG CONFIRMED: ttnn.mean fails on shapes > threshold")
    print(f"  Workaround: use ttnn.sum(x, dim) / count instead")
else:
    print(f"\n  Bug may be fixed in our tt-metal version")

ttnn.close_device(device)
print("\nDone!")
