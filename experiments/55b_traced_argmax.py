#!/usr/bin/env python3
"""
Experiment 55b: Can we put argmax inside the trace?

If ttnn.argmax works inside trace capture, we can:
1. Do argmax on-device (5.18ms standalone, but free inside trace?)
2. Only read back 1 integer instead of 151K floats (0.04ms vs 3.47ms)
3. Total overhead: trace_exec + tiny_readback = 7.6ms + 0.04ms = 7.64ms

The question: does argmax inside a trace produce correct dynamic results,
or does it bake in the argmax value like Python scalars?
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
# TEST 1: argmax inside trace capture
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: argmax inside trace capture")
print("=" * 60)

# Create input buffer
logits_buf = ttnn.from_torch(
    torch.randn(1, 1, 32, ((vocab_size + 31) // 32) * 32),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# Warmup
for _ in range(3):
    r = ttnn.argmax(logits_buf, dim=-1)
ttnn.synchronize_device(device)

# Enable program cache
try:
    device.enable_program_cache()
except:
    pass

# Capture trace with argmax
print("  Capturing trace with argmax...")
try:
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    argmax_result = ttnn.argmax(logits_buf, dim=-1)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    print("  Trace captured!")

    # Test with different inputs
    results = []
    for trial in range(5):
        # Create logits with a known max at different positions
        target_token = 100 + trial * 1000
        logits_np = np.random.randn(1, 1, 32, ((vocab_size + 31) // 32) * 32).astype(np.float32)
        logits_np[0, 0, 0, target_token] = 100.0

        new_logits = ttnn.from_torch(torch.from_numpy(logits_np),
                                      dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(new_logits, logits_buf)

        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

        got = int(ttnn.to_torch(argmax_result).flatten()[0])
        results.append((target_token, got))
        print(f"    Trial {trial}: expected={target_token}, got={got}, {'OK' if got == target_token else 'WRONG'}")

    all_correct = all(exp == got for exp, got in results)
    print(f"  {'ALL CORRECT' if all_correct else 'SOME WRONG'}")

    if all_correct:
        # Benchmark: trace exec + tiny readback
        times = []
        for _ in range(50):
            logits_np = np.random.randn(1, 1, 32, ((vocab_size + 31) // 32) * 32).astype(np.float32)
            logits_np[0, 0, 0, np.random.randint(0, vocab_size)] = 100.0
            new_logits = ttnn.from_torch(torch.from_numpy(logits_np),
                                          dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.copy(new_logits, logits_buf)

            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            idx = int(ttnn.to_torch(argmax_result).flatten()[0])
            times.append(time.perf_counter() - t0)

        print(f"\n  Trace + argmax + readback: {np.mean(times)*1e3:.2f}ms")
        print(f"  vs previous (trace + full readback + CPU argmax): ~11.17ms")
        print(f"  Savings: {11.17 - np.mean(times)*1e3:.2f}ms per token")

    ttnn.release_trace(device, trace_id)

except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# TEST 2: Can we do argmax on a smaller slice? (logits[:, -1:])
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Argmax on device — timing breakdown")
print("=" * 60)

# How fast is just the argmax execution (without readback)?
x = ttnn.from_torch(
    torch.randn(1, 1, 32, ((vocab_size + 31) // 32) * 32),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

for _ in range(5):
    r = ttnn.argmax(x, dim=-1)
ttnn.synchronize_device(device)

# Time argmax alone (no readback)
times_exec = []
for _ in range(50):
    t0 = time.perf_counter()
    r = ttnn.argmax(x, dim=-1)
    ttnn.synchronize_device(device)
    times_exec.append(time.perf_counter() - t0)

# Time argmax + readback
times_full = []
for _ in range(50):
    t0 = time.perf_counter()
    r = ttnn.argmax(x, dim=-1)
    ttnn.synchronize_device(device)
    val = int(ttnn.to_torch(r).flatten()[0])
    times_full.append(time.perf_counter() - t0)

print(f"  Argmax exec only:     {np.mean(times_exec)*1e3:.2f}ms")
print(f"  Argmax + readback:    {np.mean(times_full)*1e3:.2f}ms")
print(f"  Readback overhead:    {(np.mean(times_full)-np.mean(times_exec))*1e3:.2f}ms")


# ══════════════════════════════════════════════════════════════
# TEST 3: What about using ttnn.argmax inside a FULL trace?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Matmul + argmax in a single trace")
print("=" * 60)

# Simulate: x (1,1,32,896) @ lm_head (1,1,896,151936) -> argmax
hidden = 896
x_buf = ttnn.from_torch(
    torch.randn(1, 1, 32, hidden),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
w = ttnn.from_torch(
    torch.randn(1, 1, hidden, ((vocab_size + 31) // 32) * 32),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# Warmup
for _ in range(3):
    logits = ttnn.matmul(x_buf, w, compute_kernel_config=hifi4)
    r = ttnn.argmax(logits, dim=-1)
    logits.deallocate()
ttnn.synchronize_device(device)

try:
    trace_id2 = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref2 = ttnn.matmul(x_buf, w, compute_kernel_config=hifi4)
    argmax_ref2 = ttnn.argmax(logits_ref2, dim=-1)
    ttnn.end_trace_capture(device, trace_id2, cq_id=0)
    print("  Trace captured (matmul + argmax)")

    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id2, cq_id=0, blocking=True)
        idx = int(ttnn.to_torch(argmax_ref2).flatten()[0])
        times.append(time.perf_counter() - t0)

    print(f"  Matmul + argmax (traced) + readback: {np.mean(times)*1e3:.2f}ms")

    # Compare: traced matmul only + full readback + CPU argmax
    trace_id3 = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref3 = ttnn.matmul(x_buf, w, compute_kernel_config=hifi4)
    ttnn.end_trace_capture(device, trace_id3, cq_id=0)

    times2 = []
    for _ in range(50):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id3, cq_id=0, blocking=True)
        out = ttnn.to_torch(logits_ref3).float().numpy().flatten()[:vocab_size]
        idx = int(np.argmax(out))
        times2.append(time.perf_counter() - t0)

    print(f"  Matmul (traced) + readback + CPU argmax: {np.mean(times2)*1e3:.2f}ms")
    print(f"  Savings from traced argmax: {(np.mean(times2)-np.mean(times))*1e3:.2f}ms")

    ttnn.release_trace(device, trace_id2)
    ttnn.release_trace(device, trace_id3)

except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Current bottleneck: 3.57ms readback of 151K logits (9.5MB at 2.8 GB/s)")
print("If traced argmax works: reduce to ~0.04ms readback of 1 int")
print("Expected total: 7.6ms trace + 0.04ms readback ≈ 7.64ms (131 tok/sec)")

ttnn.close_device(device)
print("\nDone!")
