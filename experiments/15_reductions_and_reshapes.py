"""
Experiment 15: Reduction Operations and Extending the Interpreter
=================================================================
The Jaxpr interpreter (experiment 14) failed on softmax because it needs
reduce_max and reduce_sum along an axis. This experiment systematically
tests which reduction, reshape, and composite ops TT-NN supports on
Blackhole, and benchmarks built-in softmax vs manual computation.
"""

import ttnn
import torch
import numpy as np
import time

device = ttnn.open_device(device_id=0)
print("Device: Blackhole p150a")
print()

# Helper: create a tile-aligned TT-NN tensor from numpy
def make_ttnn(arr, dev=device):
    t = torch.from_numpy(arr.copy()).float()
    while t.dim() < 2:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)

# Test data: 32x64 matrix (tile-aligned)
np.random.seed(42)
x_np = np.random.randn(32, 64).astype(np.float32)
x_torch = torch.from_numpy(x_np)


# ============================================================
# TEST 1: ttnn.max — axis reduction
# ============================================================
print("=" * 60)
print("TEST 1: ttnn.max (reduction along axis)")
print("=" * 60)

x_tt = make_ttnn(x_np)

# 1a: Global max (no axis)
print("\n  1a: ttnn.max(tensor) — global max")
try:
    result = ttnn.max(x_tt)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.max().item()
    val = result_torch.flatten()[0].item()
    print(f"      TT-NN: {val:.4f},  PyTorch ref: {ref:.4f},  diff: {abs(val - ref):.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 1b: Max along dim=-1 (reduce last axis)
print("\n  1b: ttnn.max(tensor, dim=-1) — reduce last axis")
try:
    x_tt2 = make_ttnn(x_np)
    result = ttnn.max(x_tt2, dim=-1)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.max(dim=-1).values
    # Compare first 32 elements (original height before padding)
    vals = result_torch.flatten()[:32]
    err = (vals - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}, Mean err: {err.mean():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 1c: Max along dim=0 (reduce first axis)
print("\n  1c: ttnn.max(tensor, dim=0) — reduce first axis (rows)")
try:
    x_tt3 = make_ttnn(x_np)
    result = ttnn.max(x_tt3, dim=0)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.max(dim=0).values
    vals = result_torch.flatten()[:64]
    err = (vals - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}, Mean err: {err.mean():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 1d: Max along dim=-1 with keepdim (needed for softmax broadcasting)
print("\n  1d: ttnn.max(tensor, dim=-1, keepdim=True) — keepdim")
try:
    x_tt4 = make_ttnn(x_np)
    result = ttnn.max(x_tt4, dim=-1, keepdim=True)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    print(f"      Shape: {result_torch.shape}")
    print(f"      OK — keepdim supported")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 2: ttnn.sum — axis reduction
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: ttnn.sum (reduction along axis)")
print("=" * 60)

# 2a: Global sum
print("\n  2a: ttnn.sum(tensor) — global sum")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.sum(x_tt)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.sum().item()
    val = result_torch.flatten()[0].item()
    print(f"      TT-NN: {val:.4f},  PyTorch ref: {ref:.4f},  diff: {abs(val - ref):.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 2b: Sum along dim=-1
print("\n  2b: ttnn.sum(tensor, dim=-1) — reduce last axis")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.sum(x_tt, dim=-1)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.sum(dim=-1)
    vals = result_torch.flatten()[:32]
    err = (vals - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}, Mean err: {err.mean():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 2c: Sum along dim=0
print("\n  2c: ttnn.sum(tensor, dim=0) — reduce first axis")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.sum(x_tt, dim=0)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.sum(dim=0)
    vals = result_torch.flatten()[:64]
    err = (vals - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}, Mean err: {err.mean():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 3: ttnn.mean — axis reduction
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: ttnn.mean (reduction along axis)")
print("=" * 60)

# 3a: Global mean
print("\n  3a: ttnn.mean(tensor) — global mean")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.mean(x_tt)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.mean().item()
    val = result_torch.flatten()[0].item()
    print(f"      TT-NN: {val:.4f},  PyTorch ref: {ref:.4f},  diff: {abs(val - ref):.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 3b: Mean along dim=-1
print("\n  3b: ttnn.mean(tensor, dim=-1) — reduce last axis")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.mean(x_tt, dim=-1)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.mean(dim=-1)
    vals = result_torch.flatten()[:32]
    err = (vals - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}, Mean err: {err.mean():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 4: ttnn.reshape
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: ttnn.reshape")
print("=" * 60)

# 4a: Reshape 32x64 -> 64x32
print("\n  4a: reshape(32x64 -> 64x32)")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.reshape(x_tt, (64, 32))
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.reshape(64, 32)
    err = (result_torch[:64, :32] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 4b: Reshape to 3D: 32x64 -> 1x32x64
print("\n  4b: reshape(32x64 -> 1x32x64)")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.reshape(x_tt, (1, 32, 64))
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    print(f"      Shape (before squeeze): {ttnn.to_torch(result).shape}")
    print(f"      Shape (after squeeze): {result_torch.shape}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 4c: Flatten: 32x64 -> 1x2048
print("\n  4c: reshape(32x64 -> 1x2048) — flatten")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.reshape(x_tt, (1, 2048))
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.reshape(-1)
    err = (result_torch.flatten()[:2048] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 5: ttnn.permute / ttnn.transpose
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 5: Transposition (ttnn.permute / ttnn.transpose)")
print("=" * 60)

# 5a: ttnn.permute
print("\n  5a: ttnn.permute(tensor, (1, 0)) — transpose 2D")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.permute(x_tt, (1, 0))
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.T
    err = (result_torch[:64, :32] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 5b: ttnn.transpose
print("\n  5b: ttnn.transpose(tensor, 0, 1) — transpose 2D")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.transpose(x_tt, 0, 1)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x_torch.T
    err = (result_torch[:64, :32] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 5c: 3D permute
print("\n  5c: 3D permute — reshape to (2, 32, 32) then permute (0, 2, 1)")
try:
    x3d_np = np.random.randn(2, 32, 32).astype(np.float32)
    x3d_torch = torch.from_numpy(x3d_np)
    x3d_tt = ttnn.from_torch(x3d_torch, dtype=ttnn.bfloat16, device=device,
                              layout=ttnn.TILE_LAYOUT)
    result = ttnn.permute(x3d_tt, (0, 2, 1))
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = x3d_torch.permute(0, 2, 1)
    err = (result_torch[:2, :32, :32] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.4f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 6: ttnn.softmax — built-in
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 6: ttnn.softmax — built-in composite op")
print("=" * 60)

# 6a: softmax along last dim
print("\n  6a: ttnn.softmax(tensor, dim=-1)")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.softmax(x_tt, dim=-1)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = torch.softmax(x_torch, dim=-1)
    err = (result_torch[:32, :64] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.6f}, Mean err: {err.mean():.6f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")

# 6b: softmax along dim=0
print("\n  6b: ttnn.softmax(tensor, dim=0)")
try:
    x_tt = make_ttnn(x_np)
    result = ttnn.softmax(x_tt, dim=0)
    ttnn.synchronize_device(device)
    result_torch = ttnn.to_torch(result).squeeze().float()
    ref = torch.softmax(x_torch, dim=0)
    err = (result_torch[:32, :64] - ref).abs()
    print(f"      Shape: {result_torch.shape}")
    print(f"      Max err: {err.max():.6f}, Mean err: {err.mean():.6f}")
    print(f"      OK")
except Exception as e:
    print(f"      FAIL: {e}")


# ============================================================
# TEST 7: Benchmark — built-in softmax vs manual
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 7: Benchmark — built-in softmax vs manual (exp/sum/div)")
print("=" * 60)

# Larger tensor for benchmarking
x_big_np = np.random.randn(128, 512).astype(np.float32)
REPS = 100

# 7a: Built-in softmax
print(f"\n  Benchmarking on 128x512 tensor, {REPS} reps each...")
try:
    times_builtin = []
    for i in range(REPS):
        x_tt = make_ttnn(x_big_np)
        start = time.perf_counter()
        result = ttnn.softmax(x_tt, dim=-1)
        ttnn.synchronize_device(device)
        times_builtin.append(time.perf_counter() - start)

    avg_builtin = np.mean(times_builtin[10:]) * 1000  # skip warmup
    print(f"  Built-in softmax: {avg_builtin:.3f} ms")
except Exception as e:
    print(f"  Built-in softmax FAIL: {e}")
    avg_builtin = None

# 7b: Manual softmax via exp/sum/div
try:
    times_manual = []
    for i in range(REPS):
        x_tt = make_ttnn(x_big_np)
        start = time.perf_counter()
        # Manual: max -> sub -> exp -> sum -> reciprocal -> mul
        x_max = ttnn.max(x_tt, dim=-1, keepdim=True)
        x_sub = ttnn.sub(x_tt, x_max)
        x_exp = ttnn.exp(x_sub)
        x_sum = ttnn.sum(x_exp, dim=-1, keepdim=True)
        x_recip = ttnn.reciprocal(x_sum)
        result = ttnn.mul(x_exp, x_recip)
        ttnn.synchronize_device(device)
        times_manual.append(time.perf_counter() - start)

    avg_manual = np.mean(times_manual[10:]) * 1000
    print(f"  Manual softmax:   {avg_manual:.3f} ms")
except Exception as e:
    print(f"  Manual softmax FAIL: {e}")
    avg_manual = None

# 7c: Manual softmax without max subtraction (numerically unstable but simpler)
try:
    times_simple = []
    for i in range(REPS):
        x_tt = make_ttnn(x_big_np)
        start = time.perf_counter()
        x_exp = ttnn.exp(x_tt)
        x_sum = ttnn.sum(x_exp, dim=-1, keepdim=True)
        x_recip = ttnn.reciprocal(x_sum)
        result = ttnn.mul(x_exp, x_recip)
        ttnn.synchronize_device(device)
        times_simple.append(time.perf_counter() - start)

    avg_simple = np.mean(times_simple[10:]) * 1000
    print(f"  Simple softmax (no max sub): {avg_simple:.3f} ms")
except Exception as e:
    print(f"  Simple softmax FAIL: {e}")
    avg_simple = None

if avg_builtin and avg_manual:
    print(f"\n  Speedup (builtin vs manual): {avg_manual/avg_builtin:.2f}x")
if avg_builtin and avg_simple:
    print(f"  Speedup (builtin vs simple): {avg_simple/avg_builtin:.2f}x")

# 7d: Verify manual softmax correctness
print(f"\n  Verifying manual softmax correctness...")
try:
    x_tt = make_ttnn(x_big_np)
    x_max = ttnn.max(x_tt, dim=-1, keepdim=True)
    x_sub = ttnn.sub(x_tt, x_max)
    x_exp = ttnn.exp(x_sub)
    x_sum = ttnn.sum(x_exp, dim=-1, keepdim=True)
    x_recip = ttnn.reciprocal(x_sum)
    manual_result = ttnn.mul(x_exp, x_recip)
    ttnn.synchronize_device(device)
    manual_torch = ttnn.to_torch(manual_result).squeeze().float()

    ref = torch.softmax(torch.from_numpy(x_big_np), dim=-1)
    err = (manual_torch[:128, :512] - ref).abs()
    print(f"  Manual softmax max err: {err.max():.6f}, mean err: {err.mean():.6f}")
except Exception as e:
    print(f"  Manual softmax verification FAIL: {e}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("Summary: TT-NN Reduction and Reshape Support on Blackhole")
print("=" * 60)
print("""
  Tests completed. Check results above for which ops work:

  Reductions:
    - ttnn.max (global, dim=-1, dim=0, keepdim)
    - ttnn.sum (global, dim=-1, dim=0)
    - ttnn.mean (global, dim=-1)

  Reshapes:
    - ttnn.reshape (2D->2D, 2D->3D, flatten)
    - ttnn.permute (2D transpose, 3D permute)
    - ttnn.transpose (2D swap)

  Composite ops:
    - ttnn.softmax (built-in, along different dims)

  Key question for interpreter: can we now implement
  reduce_sum and reduce_max Jaxpr ops using ttnn.sum/ttnn.max?
""")

ttnn.close_device(device)
print("Done!")
