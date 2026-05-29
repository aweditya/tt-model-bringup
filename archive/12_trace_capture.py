"""
Experiment 12: TT-NN Trace Capture — Eliminating Dispatch Overhead
===================================================================
Hypothesis: TT-NN's trace capture (begin_trace_capture / end_trace_capture /
execute_trace) records a sequence of ops into a replayable trace. This should
eliminate per-op Python dispatch overhead, similar to CUDA graphs or XLA
compilation. We expect this to significantly improve latency for our MLP
forward pass, especially at small batch sizes where dispatch dominates.

Key question: How much does trace capture help, and does it change the
crossover point vs CPU?
"""

import ttnn
import torch
import time

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print()

# ============================================================
# Model setup (same as experiment 11)
# ============================================================
LAYERS = [
    (784, 1024),
    (1024, 512),
    (512, 256),
    (256, 10),
]

torch.manual_seed(42)
weights_tt = []
biases_tt = []
for in_f, out_f in LAYERS:
    w = torch.randn(in_f, out_f) * (2.0 / in_f) ** 0.5
    b = torch.zeros(1, out_f)
    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    weights_tt.append(w_tt)
    biases_tt.append(b_tt)

def mlp_forward(x_tt):
    h = x_tt
    for i in range(len(LAYERS)):
        h = ttnn.matmul(h, weights_tt[i])
        h = ttnn.add(h, biases_tt[i])
        if i < len(LAYERS) - 1:
            h = ttnn.relu(h)
    return h

# ============================================================
# TEST 1: Trace capture for MLP forward pass
# ============================================================
print("=" * 60)
print("TEST 1: Eager vs Traced MLP forward pass")
print("=" * 60)
print()

batch_sizes = [32, 128, 512]

for batch in batch_sizes:
    padded = ((batch + 31) // 32) * 32

    # --- Eager baseline ---
    x_data = torch.randn(padded, 784)
    x_tt = ttnn.from_torch(x_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Warmup eager
    for _ in range(5):
        out = mlp_forward(x_tt)
        ttnn.synchronize_device(device)
        out.deallocate()

    REPS = 50
    times_eager = []
    for _ in range(REPS):
        start = time.perf_counter()
        out = mlp_forward(x_tt)
        ttnn.synchronize_device(device)
        times_eager.append(time.perf_counter() - start)
        out.deallocate()

    # --- Trace capture ---
    # Allocate output buffer for trace (need to pre-allocate)
    # First, do a dry run to get the output shape
    out_dry = mlp_forward(x_tt)
    ttnn.synchronize_device(device)
    out_dry.deallocate()

    # Capture trace
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out_traced = mlp_forward(x_tt)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Warmup trace
    for _ in range(5):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

    times_trace = []
    for _ in range(REPS):
        start = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        times_trace.append(time.perf_counter() - start)

    ttnn.release_trace(device, trace_id)
    x_tt.deallocate()
    out_traced.deallocate()

    avg_eager = sum(times_eager) / len(times_eager)
    avg_trace = sum(times_trace) / len(times_trace)
    speedup = avg_eager / avg_trace

    print(f"  Batch {batch} (padded to {padded}):")
    print(f"    Eager:  {avg_eager*1000:.3f} ms  ({batch/avg_eager:,.0f} samples/s)")
    print(f"    Traced: {avg_trace*1000:.3f} ms  ({batch/avg_trace:,.0f} samples/s)")
    print(f"    Speedup: {speedup:.2f}x")
    print()

# ============================================================
# TEST 2: Trace with different input data
# ============================================================
print("=" * 60)
print("TEST 2: Can we feed different data through a trace?")
print("=" * 60)
print("  Traces replay the exact same commands. To use new input")
print("  data, we write into the same input buffer before replay.")
print()

padded = 32
x_data1 = torch.randn(padded, 784)
x_data2 = torch.ones(padded, 784) * 0.5  # Different data

x_tt = ttnn.from_torch(x_data1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Eager with data1
out_eager1 = mlp_forward(x_tt)
result1_eager = ttnn.to_torch(out_eager1).squeeze()[:padded, :10]
out_eager1.deallocate()

# Capture trace
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
out_traced = mlp_forward(x_tt)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Execute trace with original data
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
result1_trace = ttnn.to_torch(out_traced).squeeze()[:padded, :10]

# Now overwrite the input buffer with new data and replay
ttnn.copy_host_to_device_tensor(
    ttnn.from_torch(x_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
    x_tt
)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
result2_trace = ttnn.to_torch(out_traced).squeeze()[:padded, :10]

# Also compute eager with data2 for reference
x_tt2 = ttnn.from_torch(x_data2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
out_eager2 = mlp_forward(x_tt2)
result2_eager = ttnn.to_torch(out_eager2).squeeze()[:padded, :10]
out_eager2.deallocate()
x_tt2.deallocate()

err1 = (result1_trace.float() - result1_eager.float()).abs().max().item()
err2 = (result2_trace.float() - result2_eager.float()).abs().max().item()
data_diff = (result1_trace.float() - result2_trace.float()).abs().max().item()

print(f"  Same data (trace vs eager):      max err = {err1:.6f}")
print(f"  New data (trace vs eager):        max err = {err2:.6f}")
print(f"  Different inputs give different output: {data_diff > 0.01}")
print(f"  Output difference magnitude:      {data_diff:.4f}")

ttnn.release_trace(device, trace_id)
x_tt.deallocate()
out_traced.deallocate()

# ============================================================
# TEST 3: Trace overhead — how fast is the trace dispatch itself?
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Trace dispatch overhead (tiny tensor)")
print("=" * 60)
print("  Using 32x32 tensors to isolate dispatch cost.")
print()

a_data = torch.randn(32, 32)
a_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
b_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Eager single add
for _ in range(10):
    r = ttnn.add(a_tt, b_tt)
    ttnn.synchronize_device(device)
    r.deallocate()

REPS = 200
times_eager = []
for _ in range(REPS):
    start = time.perf_counter()
    r = ttnn.add(a_tt, b_tt)
    ttnn.synchronize_device(device)
    times_eager.append(time.perf_counter() - start)
    r.deallocate()

# Trace single add
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
r_traced = ttnn.add(a_tt, b_tt)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

for _ in range(10):
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

times_trace = []
for _ in range(REPS):
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    times_trace.append(time.perf_counter() - start)

ttnn.release_trace(device, trace_id)
a_tt.deallocate(); b_tt.deallocate(); r_traced.deallocate()

avg_eager = sum(times_eager) / len(times_eager)
avg_trace = sum(times_trace) / len(times_trace)
print(f"  Single add (eager):  {avg_eager*1000:.3f} ms")
print(f"  Single add (trace):  {avg_trace*1000:.3f} ms")
print(f"  Speedup: {avg_eager/avg_trace:.2f}x")

# ============================================================
# TEST 4: Scaling — trace benefit for 10-op chain
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: 10-op chain — trace vs eager")
print("=" * 60)

a_data = torch.randn(32, 32)
a_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
b_tt = ttnn.from_torch(a_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def chain_10(x, y):
    h = x
    for _ in range(10):
        h = ttnn.add(h, y)
    return h

# Eager
for _ in range(5):
    r = chain_10(a_tt, b_tt)
    ttnn.synchronize_device(device)

times_eager = []
for _ in range(REPS):
    start = time.perf_counter()
    r = chain_10(a_tt, b_tt)
    ttnn.synchronize_device(device)
    times_eager.append(time.perf_counter() - start)

# Trace
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
r_traced = chain_10(a_tt, b_tt)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

for _ in range(5):
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

times_trace = []
for _ in range(REPS):
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    times_trace.append(time.perf_counter() - start)

ttnn.release_trace(device, trace_id)
a_tt.deallocate(); b_tt.deallocate(); r_traced.deallocate()

avg_eager = sum(times_eager) / len(times_eager)
avg_trace = sum(times_trace) / len(times_trace)
print(f"\n  10-add chain (eager):  {avg_eager*1000:.3f} ms  ({avg_eager/10*1000:.3f} ms/op)")
print(f"  10-add chain (trace):  {avg_trace*1000:.3f} ms  ({avg_trace/10*1000:.3f} ms/op)")
print(f"  Speedup: {avg_eager/avg_trace:.2f}x")
print(f"  Dispatch saved: {(avg_eager-avg_trace)*1000:.3f} ms")

# Cleanup
for w, b in zip(weights_tt, biases_tt):
    w.deallocate(); b.deallocate()

ttnn.close_device(device)
print("\nDone!")
