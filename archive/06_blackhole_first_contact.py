"""
Experiment 06: First Contact with Blackhole
============================================
Hypothesis: We can open the Blackhole device 0, query its properties,
and run a basic tensor operation (add, matmul) through TT-NN.

This is our first computation on Tenstorrent hardware.
"""

import ttnn
import torch
import time

print("=" * 60)
print("Opening Blackhole device 0...")
print("=" * 60)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"  Compute grid: {grid}")
print(f"  Grid total: {grid.x * grid.y} Tensix cores")
print(f"  Device type: {type(device)}")

# ============================================================
# TEST 1: Simple tensor addition on device
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 1: Tensor addition on Blackhole")
print("=" * 60)

# Create torch tensors, convert to TT-NN format
a_torch = torch.randn(1, 1, 32, 32)  # MUST be multiple of 32 (tile size!)
b_torch = torch.randn(1, 1, 32, 32)

# Move to device with tile layout
a_tt = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
b_tt = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

print(f"  a shape: {a_tt.shape}, dtype: {a_tt.dtype}, layout: {a_tt.layout}")
print(f"  b shape: {b_tt.shape}, dtype: {b_tt.dtype}, layout: {b_tt.layout}")

# Run addition on device
c_tt = ttnn.add(a_tt, b_tt)

# Move result back to CPU
c_torch = ttnn.to_torch(c_tt)

# Verify correctness
expected = a_torch + b_torch
max_diff = (c_torch - expected).abs().max().item()
print(f"  Max absolute error: {max_diff:.6f}")
print(f"  Correct: {max_diff < 0.01}")

# Cleanup
a_tt.deallocate()
b_tt.deallocate()
c_tt.deallocate()

# ============================================================
# TEST 2: Matrix multiplication on device
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Matrix multiplication on Blackhole")
print("=" * 60)

x_torch = torch.randn(1, 1, 64, 256)
w_torch = torch.randn(1, 1, 256, 128)

x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
w_tt = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

result_tt = ttnn.matmul(x_tt, w_tt)
result_torch = ttnn.to_torch(result_tt)

expected_mm = x_torch @ w_torch
max_diff_mm = (result_torch - expected_mm).abs().max().item()
mean_diff_mm = (result_torch - expected_mm).abs().mean().item()
print(f"  Input: ({x_torch.shape}) @ ({w_torch.shape}) = ({result_torch.shape})")
print(f"  Max absolute error:  {max_diff_mm:.4f}")
print(f"  Mean absolute error: {mean_diff_mm:.6f}")
print(f"  Correct (within bf16 tolerance): {max_diff_mm < 1.0}")

x_tt.deallocate()
w_tt.deallocate()
result_tt.deallocate()

# ============================================================
# TEST 3: MLP layer (matmul + relu) on device
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: MLP layer (matmul + relu) on Blackhole")
print("=" * 60)

x_t = torch.randn(1, 1, 64, 512)
w_t = torch.randn(1, 1, 512, 256)

x_tt = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
w_tt = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Matmul then ReLU — two separate TT-NN operations
h_tt = ttnn.matmul(x_tt, w_tt)
out_tt = ttnn.relu(h_tt)

out_torch = ttnn.to_torch(out_tt)
expected_mlp = torch.relu(x_t @ w_t)
max_diff_mlp = (out_torch - expected_mlp).abs().max().item()
print(f"  relu(x @ w) max error: {max_diff_mlp:.4f}")
print(f"  Correct: {max_diff_mlp < 1.0}")

h_tt.deallocate()
x_tt.deallocate()
w_tt.deallocate()
out_tt.deallocate()

# ============================================================
# TEST 4: Benchmark — matmul throughput
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Matmul benchmark on Blackhole")
print("=" * 60)

# Larger matmul to actually stress the hardware
x_big = torch.randn(1, 1, 512, 1024)
w_big = torch.randn(1, 1, 1024, 512)

x_tt = ttnn.from_torch(x_big, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
w_tt = ttnn.from_torch(w_big, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Warmup
for _ in range(5):
    r = ttnn.matmul(x_tt, w_tt)
    r.deallocate()

# Benchmark
N = 100
start = time.perf_counter()
for _ in range(N):
    r = ttnn.matmul(x_tt, w_tt)
    r.deallocate()
elapsed = time.perf_counter() - start

flops = 2 * 512 * 1024 * 512  # 2*M*K*N for matmul
tflops = (flops * N) / elapsed / 1e12
print(f"  Shape: (512, 1024) @ (1024, 512)")
print(f"  Time per matmul: {elapsed/N*1000:.3f} ms")
print(f"  Throughput: {tflops:.2f} TFLOPS (bf16)")
print(f"  (Peak spec: ~372 TFLOPS bf16)")
print(f"  Utilization: {tflops/372*100:.1f}%")

# Also try with L1 memory config
print(f"\n  With L1 memory config (SRAM instead of DRAM):")
for _ in range(5):
    r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    r.deallocate()

start = time.perf_counter()
for _ in range(N):
    r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    r.deallocate()
elapsed_l1 = time.perf_counter() - start
tflops_l1 = (flops * N) / elapsed_l1 / 1e12
print(f"  Time per matmul: {elapsed_l1/N*1000:.3f} ms")
print(f"  Throughput: {tflops_l1:.2f} TFLOPS (bf16)")
print(f"  Speedup over DRAM: {elapsed/elapsed_l1:.2f}x")

x_tt.deallocate()
w_tt.deallocate()

# ============================================================
# TEST 5: CPU torch vs Blackhole comparison
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 5: CPU vs Blackhole matmul speed")
print("=" * 60)

x_cpu = torch.randn(512, 1024)
w_cpu = torch.randn(1024, 512)

# CPU benchmark
for _ in range(10):
    _ = x_cpu @ w_cpu

start = time.perf_counter()
for _ in range(N):
    _ = x_cpu @ w_cpu
cpu_time = (time.perf_counter() - start) / N

print(f"  CPU (torch):   {cpu_time*1000:.3f} ms")
print(f"  Blackhole:     {elapsed/N*1000:.3f} ms")
print(f"  Speedup:       {cpu_time/(elapsed/N):.1f}x")

ttnn.close_device(device)
print(f"\nDevice closed. First contact complete!")
