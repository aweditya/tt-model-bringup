#!/usr/bin/env python3
"""
Experiment 86: Can flash_decode handle 8 KV heads directly?

Currently we split 8 KV heads into 2 groups of 4 because flash_decode
only compiled with power-of-2 KV heads (exp 64 finding). This doubles
kernel launches: 64 flash_decode calls instead of 32 per forward pass.

If 8 KV heads work directly:
  - 32 fewer flash_decode calls per forward pass
  - 128 fewer slice + to_memory_config ops per forward pass
  - 32 fewer concat ops
  - Could save 3-5ms of trace overhead

Test with a simple microbenchmark first, then full 8B if it works.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device grid: {grid.x}x{grid.y}")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

TILE = 32
batch_size = 1
head_dim = 128
MAX_SEQ = 512

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

print("\n" + "="*60)
print("TEST: flash_decode with different KV head counts")
print("="*60)

for n_kv_heads in [1, 2, 4, 8]:
    n_q_heads = 32  # Llama-8B has 32 Q heads
    gqa_ratio = n_q_heads // n_kv_heads

    # Create Q: [1, 1, n_q_heads, head_dim]
    q = to_dev_4d(np.random.randn(1, 1, n_q_heads, head_dim).astype(np.float32))

    # Create KV cache: [batch, n_kv_heads, MAX_SEQ, head_dim]
    kv_sh = ((n_kv_heads + TILE - 1) // TILE) * TILE
    kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    kv_cfg = ttnn.create_sharded_memory_config(
        shape=(kv_sh, head_dim), core_grid=kv_cg,
        strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

    k_cache = to_dev_4d(np.random.randn(batch_size, n_kv_heads, MAX_SEQ, head_dim).astype(np.float32))
    v_cache = to_dev_4d(np.random.randn(batch_size, n_kv_heads, MAX_SEQ, head_dim).astype(np.float32))
    pos_buf = ttnn.from_torch(torch.tensor([100], dtype=torch.int32), device=device)

    try:
        t0 = time.perf_counter()
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q, k_cache, v_cache, cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()

        out_shape = attn.shape
        print(f"  {n_kv_heads:2d} KV heads (GQA {gqa_ratio}:1): OK! {(t1-t0)*1000:.1f}ms, out={out_shape}")

        # Benchmark: 10 runs
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q, k_cache, v_cache, cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times[1:]) * 1000  # skip first
        print(f"           Avg: {avg:.2f}ms/call (over 9 runs)")

    except Exception as e:
        err_str = str(e)
        # Extract the key error message
        if "power of 2" in err_str.lower():
            print(f"  {n_kv_heads:2d} KV heads (GQA {gqa_ratio}:1): FAILED — requires power-of-2 KV heads")
        else:
            short = err_str[:200] if len(err_str) > 200 else err_str
            print(f"  {n_kv_heads:2d} KV heads (GQA {gqa_ratio}:1): FAILED — {short}")


# Also test the split approach (our current workaround) for comparison
print(f"\n  SPLIT approach (2 x 4 KV heads):")
n_kv_split = 4
n_q_split = 16
q_full = to_dev_4d(np.random.randn(1, 1, 32, head_dim).astype(np.float32))

kv_sh4 = ((n_kv_split + TILE - 1) // TILE) * TILE
kv_cg4 = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg4 = ttnn.create_sharded_memory_config(
    shape=(kv_sh4, head_dim), core_grid=kv_cg4,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

k_lo = to_dev_4d(np.random.randn(batch_size, n_kv_split, MAX_SEQ, head_dim).astype(np.float32))
v_lo = to_dev_4d(np.random.randn(batch_size, n_kv_split, MAX_SEQ, head_dim).astype(np.float32))
k_hi = to_dev_4d(np.random.randn(batch_size, n_kv_split, MAX_SEQ, head_dim).astype(np.float32))
v_hi = to_dev_4d(np.random.randn(batch_size, n_kv_split, MAX_SEQ, head_dim).astype(np.float32))
pos_buf = ttnn.from_torch(torch.tensor([100], dtype=torch.int32), device=device)

times_split = []
for _ in range(10):
    t0 = time.perf_counter()
    q_lo = ttnn.slice(q_full, [0,0,0,0], [1,1,n_q_split,head_dim])
    q_hi = ttnn.slice(q_full, [0,0,n_q_split,0], [1,1,32,head_dim])
    attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
        q_lo, k_lo, v_lo, cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
    attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
        q_hi, k_hi, v_hi, cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
    attn = ttnn.concat([attn_lo, attn_hi], dim=2)
    ttnn.synchronize_device(device)
    times_split.append(time.perf_counter() - t0)
avg_split = np.mean(times_split[1:]) * 1000
print(f"           Avg: {avg_split:.2f}ms/call (split = 2 flash_decode + slice + concat)")

ttnn.close_device(device)
print("\nDone!")
