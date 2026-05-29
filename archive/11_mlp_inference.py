"""
Experiment 11: End-to-End MLP Inference on Blackhole
=====================================================
Hypothesis: We can build a real multi-layer neural network using TT-NN
primitives and run inference on Blackhole. This tests whether TT-NN's
op-by-op dispatch is practical for real model inference, and quantifies
the dispatch overhead for a realistic workload.

We'll build:
  - A 4-layer MLP: 784→1024→512→256→10 (MNIST-style classifier)
  - Compare: TT-NN on Blackhole vs PyTorch on CPU
  - Measure: throughput (samples/sec), latency, dispatch overhead fraction
"""

import ttnn
import torch
import torch.nn as nn
import time

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print()

# ============================================================
# Model definition
# ============================================================
LAYERS = [
    (784, 1024),
    (1024, 512),
    (512, 256),
    (256, 10),
]

print("=" * 60)
print("MLP Architecture: 784→1024→512→256→10")
print("=" * 60)
total_params = sum(m*n + n for m, n in LAYERS)
print(f"  Total parameters: {total_params:,} ({total_params*2/1024:.1f} KB in bf16)")
print(f"  Layers: {len(LAYERS)} linear + ReLU (no ReLU on last)")
print()

# Initialize weights (same for both paths)
torch.manual_seed(42)
weights_torch = []
biases_torch = []
for in_f, out_f in LAYERS:
    w = torch.randn(in_f, out_f) * (2.0 / in_f) ** 0.5  # He init
    b = torch.zeros(1, out_f)
    weights_torch.append(w)
    biases_torch.append(b)

# ============================================================
# TEST 1: PyTorch CPU baseline
# ============================================================
print("=" * 60)
print("TEST 1: PyTorch CPU baseline")
print("=" * 60)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (in_f, out_f) in enumerate(LAYERS):
            self.layers.append(nn.Linear(in_f, out_f))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

model_cpu = MLP()
# Copy our weights into the PyTorch model
with torch.no_grad():
    for i, layer in enumerate(model_cpu.layers):
        layer.weight.copy_(weights_torch[i].T)
        layer.bias.copy_(biases_torch[i].squeeze())

batch_sizes = [1, 32, 128, 512]

print(f"\n  {'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15}")
print(f"  {'-'*40}")

for batch in batch_sizes:
    x = torch.randn(batch, 784)

    # Warmup
    for _ in range(10):
        _ = model_cpu(x)

    REPS = 100
    start = time.perf_counter()
    for _ in range(REPS):
        _ = model_cpu(x)
    elapsed = (time.perf_counter() - start) / REPS

    samples_per_sec = batch / elapsed
    print(f"  {batch:<8} {elapsed*1000:<15.3f} {samples_per_sec:<15,.0f} samples/s")

# ============================================================
# TEST 2: TT-NN on Blackhole (eager dispatch)
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: TT-NN on Blackhole (eager op-by-op dispatch)")
print("=" * 60)

# Upload weights to device
weights_tt = []
biases_tt = []
for i, (in_f, out_f) in enumerate(LAYERS):
    w_tt = ttnn.from_torch(weights_torch[i], dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(biases_torch[i], dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    weights_tt.append(w_tt)
    biases_tt.append(b_tt)

def mlp_forward_tt(x_tt):
    """Forward pass using TT-NN ops."""
    h = x_tt
    for i in range(len(LAYERS)):
        h = ttnn.matmul(h, weights_tt[i])
        h = ttnn.add(h, biases_tt[i])
        if i < len(LAYERS) - 1:
            h = ttnn.relu(h)
    return h

print(f"\n  {'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15} {'vs CPU'}")
print(f"  {'-'*55}")

cpu_throughputs = {}
tt_throughputs = {}

# Re-run CPU to store results
for batch in batch_sizes:
    x = torch.randn(batch, 784)
    for _ in range(10):
        _ = model_cpu(x)
    REPS = 100
    start = time.perf_counter()
    for _ in range(REPS):
        _ = model_cpu(x)
    elapsed = (time.perf_counter() - start) / REPS
    cpu_throughputs[batch] = batch / elapsed

for batch in batch_sizes:
    # Pad to tile-aligned (multiple of 32)
    padded_batch = ((batch + 31) // 32) * 32
    x = torch.randn(padded_batch, 784)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Warmup
    for _ in range(5):
        out = mlp_forward_tt(x_tt)
        ttnn.synchronize_device(device)
        out.deallocate()

    REPS = 50
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        out = mlp_forward_tt(x_tt)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - start)
        out.deallocate()

    x_tt.deallocate()

    avg = sum(times) / len(times)
    throughput = batch / avg  # use original batch, not padded
    tt_throughputs[batch] = throughput
    speedup = throughput / cpu_throughputs[batch]
    print(f"  {batch:<8} {avg*1000:<15.3f} {throughput:<15,.0f} samples/s  {speedup:.1f}x")

# ============================================================
# TEST 3: Dispatch overhead analysis
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Dispatch overhead breakdown")
print("=" * 60)
print("  Each forward pass = 4 matmul + 4 add + 3 relu = 11 ops")
print("  At ~21µs/op dispatch, expect ~0.231 ms dispatch overhead")
print()

batch = 512
padded_batch = ((batch + 31) // 32) * 32
x = torch.randn(padded_batch, 784)
x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Time with sync after every op (measures dispatch + compute per op)
for _ in range(5):
    out = mlp_forward_tt(x_tt)
    ttnn.synchronize_device(device)
    out.deallocate()

REPS = 20

# Method A: sync at end only (pipelined)
times_end = []
for _ in range(REPS):
    start = time.perf_counter()
    out = mlp_forward_tt(x_tt)
    ttnn.synchronize_device(device)
    times_end.append(time.perf_counter() - start)
    out.deallocate()

# Method B: sync after each op (serialized)
def mlp_forward_synced(x_tt):
    h = x_tt
    for i in range(len(LAYERS)):
        h = ttnn.matmul(h, weights_tt[i])
        ttnn.synchronize_device(device)
        h = ttnn.add(h, biases_tt[i])
        ttnn.synchronize_device(device)
        if i < len(LAYERS) - 1:
            h = ttnn.relu(h)
            ttnn.synchronize_device(device)
    return h

for _ in range(5):
    out = mlp_forward_synced(x_tt)
    out.deallocate()

times_synced = []
for _ in range(REPS):
    start = time.perf_counter()
    out = mlp_forward_synced(x_tt)
    times_synced.append(time.perf_counter() - start)
    out.deallocate()

x_tt.deallocate()

avg_end = sum(times_end) / len(times_end)
avg_synced = sum(times_synced) / len(times_synced)
overhead = avg_synced - avg_end

print(f"  Batch size: {batch}")
print(f"  Pipelined (sync at end):    {avg_end*1000:.3f} ms")
print(f"  Serialized (sync per op):   {avg_synced*1000:.3f} ms")
print(f"  Overhead from serializing:  {overhead*1000:.3f} ms ({overhead/avg_end*100:.0f}%)")
print(f"  Per-op sync cost:           {overhead/11*1000:.3f} ms")
print()
print(f"  → Pipelined dispatch is key: TT-NN's command queue lets")
print(f"    ops overlap, hiding dispatch behind compute.")

# ============================================================
# TEST 4: Correctness check
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Correctness — TT-NN vs PyTorch")
print("=" * 60)

x_test = torch.randn(32, 784)
# PyTorch reference
with torch.no_grad():
    ref = model_cpu(x_test)

# TT-NN
x_tt = ttnn.from_torch(x_test, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
out_tt = mlp_forward_tt(x_tt)
out_torch = ttnn.to_torch(out_tt).squeeze()[:32, :10]

abs_err = (out_torch.float() - ref).abs()
print(f"\n  Max abs error:  {abs_err.max().item():.4f}")
print(f"  Mean abs error: {abs_err.mean().item():.4f}")

# Check if same class predictions
pred_cpu = ref.argmax(dim=1)
pred_tt = out_torch.float().argmax(dim=1)
agreement = (pred_cpu == pred_tt).float().mean().item() * 100
print(f"  Class agreement: {agreement:.0f}% ({(pred_cpu == pred_tt).sum()}/{len(pred_cpu)})")

x_tt.deallocate(); out_tt.deallocate()

# Cleanup
for w, b in zip(weights_tt, biases_tt):
    w.deallocate(); b.deallocate()

ttnn.close_device(device)
print("\nDone!")
