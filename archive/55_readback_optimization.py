#!/usr/bin/env python3
"""
Experiment 55: Readback optimization — reducing the 3.6ms argmax overhead.

In 54b, trace exec is 7.6ms but total is 11.2ms due to readback + argmax.
That's 3.6ms wasted on CPU-side post-processing of 151K logits.

Tests:
  1. ttnn.argmax — can we do argmax on-device?
  2. Readback size profiling — how much of 3.6ms is PCIe transfer vs numpy?
  3. Partial readback — only read top-k tokens instead of full vocab?
  4. On-device argmax via matmul trick?
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

vocab_size = 151936

# ══════════════════════════════════════════════════════════════
# TEST 1: ttnn.argmax
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: ttnn.argmax")
print("=" * 60)

logits_np = np.random.randn(1, 1, vocab_size).astype(np.float32)
logits_np[0, 0, 42] = 100.0  # make token 42 the max

x = ttnn.from_torch(torch.from_numpy(logits_np), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
try:
    result = ttnn.argmax(x, dim=-1)
    r_np = ttnn.to_torch(result).numpy()
    print(f"  ttnn.argmax OK: {r_np.flatten()[:5]} (expected 42)")

    # Benchmark
    for _ in range(5):
        _ = ttnn.argmax(x, dim=-1)
    ttnn.synchronize_device(device)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        r = ttnn.argmax(x, dim=-1)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - t0)
    print(f"  Time: {np.mean(times)*1e3:.2f}ms +/- {np.std(times)*1e3:.2f}ms")

    # Time the readback of the argmax result (tiny tensor)
    times_read = []
    for _ in range(50):
        r = ttnn.argmax(x, dim=-1)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        val = ttnn.to_torch(r).numpy()
        times_read.append(time.perf_counter() - t0)
    print(f"  Argmax result readback: {np.mean(times_read)*1e3:.2f}ms")

except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# TEST 2: Readback size profiling
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Readback size profiling (PCIe transfer time)")
print("=" * 60)

sizes = [
    ("1x32 (argmax result)", (1, 32)),
    ("1x64", (1, 64)),
    ("1x896 (hidden)", (1, 896)),
    ("1x4864 (MLP intermediate)", (1, 4864)),
    ("1x32768", (1, 32768)),
    ("1x151936 (full vocab)", (1, 151936)),
]

for name, shape in sizes:
    # Pad to tile-compatible
    padded_shape = (max(32, shape[0]), max(32, ((shape[1] + 31) // 32) * 32))
    t = ttnn.from_torch(
        torch.randn(1, 1, padded_shape[0], padded_shape[1]),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Warmup
    for _ in range(5):
        _ = ttnn.to_torch(t)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        out = ttnn.to_torch(t)
        times.append(time.perf_counter() - t0)

    bytes_size = padded_shape[0] * padded_shape[1] * 2  # bfloat16 = 2 bytes
    bw = bytes_size / np.mean(times) / 1e9  # GB/s
    print(f"  {name:30s}: {np.mean(times)*1e3:.2f}ms ({bytes_size/1024:.0f}KB, {bw:.1f} GB/s)")
    t.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST 3: CPU argmax timing (isolated)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: CPU argmax timing (numpy)")
print("=" * 60)

logits_cpu = np.random.randn(vocab_size).astype(np.float32)

# np.argmax
times_argmax = []
for _ in range(100):
    t0 = time.perf_counter()
    idx = np.argmax(logits_cpu)
    times_argmax.append(time.perf_counter() - t0)
print(f"  np.argmax({vocab_size}): {np.mean(times_argmax)*1e3:.3f}ms")

# np.argpartition (for top-k)
times_topk = []
for _ in range(100):
    t0 = time.perf_counter()
    top_idx = np.argpartition(logits_cpu, -50)[-50:]
    top_logits = logits_cpu[top_idx]
    times_topk.append(time.perf_counter() - t0)
print(f"  np.argpartition top-50: {np.mean(times_topk)*1e3:.3f}ms")


# ══════════════════════════════════════════════════════════════
# TEST 4: Full pipeline timing breakdown
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Full pipeline timing breakdown")
print("=" * 60)

# Simulate what happens after trace exec
x = ttnn.from_torch(torch.from_numpy(logits_np), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Method A: Full readback + CPU argmax
times_a = []
for _ in range(50):
    t0 = time.perf_counter()
    out = ttnn.to_torch(x).float().numpy().flatten()[:vocab_size]
    idx = int(np.argmax(out))
    times_a.append(time.perf_counter() - t0)
print(f"  Method A (readback + argmax):    {np.mean(times_a)*1e3:.2f}ms -> token {idx}")

# Method B: Device argmax + tiny readback
try:
    times_b = []
    for _ in range(50):
        t0 = time.perf_counter()
        r = ttnn.argmax(x, dim=-1)
        ttnn.synchronize_device(device)
        idx = int(ttnn.to_torch(r).flatten()[0])
        times_b.append(time.perf_counter() - t0)
    print(f"  Method B (device argmax + read): {np.mean(times_b)*1e3:.2f}ms -> token {idx}")

    speedup = np.mean(times_a) / np.mean(times_b)
    print(f"  Speedup: {speedup:.2f}x")
except Exception as e:
    print(f"  Method B failed: {e}")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The readback optimization question:")
print(f"  Full readback of {vocab_size} logits: {np.mean(times_a)*1e3:.2f}ms")
print(f"  Target: reduce this to < 1ms")

ttnn.close_device(device)
print("\nDone!")
