"""
Experiment 10: Datatype Exploration on Blackhole
=================================================
Hypothesis: Blackhole's Tensix matrix engine operates on tiles of FP-based
formats. BFloat16 is the default, but TT-NN supports other dtypes (fp32,
bfp8_b, bfp4_b, etc). Lower precision should give higher throughput if
the matrix engine can exploit it. We test what works and measure throughput.

Key question: Does Blackhole's matrix engine have native fp32 ALUs, or does
fp32 matmul decompose into multiple bf16 operations? The spec says 372 TFLOPS
for bf16 — what's the fp32 number?
"""

import ttnn
import torch
import time
import traceback

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print()

def bench_matmul(M, K, N, dtype, label, reps=20):
    """Benchmark matmul at a given dtype. Returns TFLOPS or None on failure."""
    try:
        x = torch.randn(M, K)
        w = torch.randn(K, N)
        x_tt = ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        w_tt = ttnn.from_torch(w, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        # Warmup
        for _ in range(5):
            r = ttnn.matmul(x_tt, w_tt)
            ttnn.synchronize_device(device)
            r.deallocate()

        times = []
        for _ in range(reps):
            start = time.perf_counter()
            r = ttnn.matmul(x_tt, w_tt)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - start)
            r.deallocate()

        x_tt.deallocate()
        w_tt.deallocate()

        avg = sum(times) / len(times)
        flops = 2 * M * K * N
        tflops = flops / avg / 1e12
        return avg, tflops
    except Exception as e:
        return None, str(e)[:100]


# ============================================================
# TEST 1: Which dtypes work for matmul?
# ============================================================
print("=" * 60)
print("TEST 1: Which dtypes support matmul?")
print("=" * 60)
print()

dtypes_to_test = [
    (ttnn.bfloat16, "BFloat16"),
    (ttnn.float32, "Float32"),
    (ttnn.bfloat8_b, "BFloat8_b (block FP8)"),
    (ttnn.bfloat4_b, "BFloat4_b (block FP4)"),
]

M, K, N = 1024, 1024, 1024
print(f"  Matmul size: {M}x{K}x{N}")
print(f"  {'Dtype':<25} {'Status':<10} {'Time (ms)':<12} {'TFLOPS':<10} {'% peak bf16'}")
print(f"  {'-'*75}")

results = {}
for dtype, label in dtypes_to_test:
    avg, tflops = bench_matmul(M, K, N, dtype, label)
    if avg is not None:
        pct = tflops / 372 * 100
        print(f"  {label:<25} {'OK':<10} {avg*1000:<12.3f} {tflops:<10.1f} {pct:.1f}%")
        results[label] = tflops
    else:
        print(f"  {label:<25} {'FAIL':<10} {tflops}")

# ============================================================
# TEST 2: Throughput scaling by dtype at larger sizes
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Dtype throughput at 4096x4096x4096")
print("=" * 60)
print()

M, K, N = 4096, 4096, 4096
print(f"  Matmul size: {M}x{K}x{N}")
print(f"  {'Dtype':<25} {'Time (ms)':<12} {'TFLOPS':<10} {'% peak bf16'}")
print(f"  {'-'*60}")

for dtype, label in dtypes_to_test:
    avg, tflops = bench_matmul(M, K, N, dtype, label)
    if avg is not None:
        pct = tflops / 372 * 100
        print(f"  {label:<25} {avg*1000:<12.3f} {tflops:<10.1f} {pct:.1f}%")
    else:
        print(f"  {label:<25} FAIL: {tflops}")

# ============================================================
# TEST 3: Elementwise ops across dtypes
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Elementwise add across dtypes (2048x2048)")
print("=" * 60)
print()

M, N = 2048, 2048
elem_dtypes = [
    (ttnn.bfloat16, "BFloat16"),
    (ttnn.float32, "Float32"),
]

for dtype, label in elem_dtypes:
    try:
        a = torch.randn(1, 1, M, N)
        b = torch.randn(1, 1, M, N)
        a_tt = ttnn.from_torch(a, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

        for _ in range(5):
            r = ttnn.add(a_tt, b_tt)
            r.deallocate()

        REPS = 50
        times = []
        for _ in range(REPS):
            start = time.perf_counter()
            r = ttnn.add(a_tt, b_tt)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - start)
            r.deallocate()

        a_tt.deallocate()
        b_tt.deallocate()

        avg = sum(times) / len(times)
        bytes_per_elem = 4 if dtype == ttnn.float32 else 2
        total_bytes = M * N * bytes_per_elem * 3  # 2 reads + 1 write
        bw = total_bytes / avg / 1e9
        print(f"  {label}: {avg*1000:.3f} ms ({bw:.1f} GB/s effective)")
    except Exception as e:
        print(f"  {label}: FAIL ({str(e)[:80]})")

# ============================================================
# TEST 4: Accuracy comparison across dtypes
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Numerical accuracy by dtype (256x256 matmul)")
print("=" * 60)
print()

M, K, N = 256, 256, 256
x = torch.randn(M, K)
w = torch.randn(K, N)
ref = x @ w  # fp32 reference

for dtype, label in dtypes_to_test:
    try:
        x_tt = ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        w_tt = ttnn.from_torch(w, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        r_tt = ttnn.matmul(x_tt, w_tt)
        r_torch = ttnn.to_torch(r_tt).squeeze()

        # Compute error metrics
        abs_err = (r_torch.float() - ref).abs()
        rel_err = abs_err / (ref.abs() + 1e-8)

        print(f"  {label}:")
        print(f"    Max abs error:  {abs_err.max().item():.6f}")
        print(f"    Mean abs error: {abs_err.mean().item():.6f}")
        print(f"    Max rel error:  {rel_err.max().item():.6f}")
        print(f"    Mean rel error: {rel_err.mean().item():.6f}")

        x_tt.deallocate(); w_tt.deallocate(); r_tt.deallocate()
    except Exception as e:
        print(f"  {label}: FAIL ({str(e)[:80]})")

ttnn.close_device(device)
print("\nDone!")
