#!/usr/bin/env python3
"""
Latency probe: paged vs non-paged SDPA decode at OUR exact Qwen3.6-27B shape
(N_Q=24, N_KV=4, HEAD_DIM=256). This is the decision-gate for the paged
migration — if paged is much slower at our short-context operating point,
we need a dual-path; if it's close, we can swap unconditionally.

Tests:
  - non-paged @ MAX_POS=256  (current operating point)
  - paged    @ MAX_POS=256, BLOCK_SIZE=64
  - paged    @ MAX_POS=1024, BLOCK_SIZE=64
  - paged    @ MAX_POS=8192, BLOCK_SIZE=64  (the long-context target)

Each op timed with 5 warmup + 50 measured iterations, sync-bounded.

Run on qb2 (server killed):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_vs_nonpaged_sdpa_latency.py
"""
import os, sys, time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

N_Q = 24
N_KV = 4
HEAD_DIM = 256
BLOCK_SIZE = 64
N_WARMUP = 5
N_MEASURE = 50


def alloc(arr, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype, device=device, layout=layout)


def time_op(fn, device):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    times = []
    for _ in range(N_MEASURE):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print(f"Latency probe: paged vs non-paged SDPA at Qwen3.6-27B shape")
    print(f"  N_Q={N_Q} N_KV={N_KV} HEAD_DIM={HEAD_DIM} BLOCK_SIZE={BLOCK_SIZE}")

    device = ttnn.open_device(device_id=device_id)
    try:
        q_np = (np.random.randn(1, 1, N_Q, HEAD_DIM) * 0.05).astype(np.float32)
        q_tt = alloc(q_np, device)

        print(f"\n{'op':<24} {'MAX_POS':>8} {'cur_pos':>8} {'median ms':>10} {'p95 ms':>10}")
        print("-" * 65)

        # NON-PAGED at MAX_POS=256
        for max_pos in [256]:
            cur_pos = max_pos - 8
            k_np = (np.random.randn(1, N_KV, max_pos, HEAD_DIM) * 0.05).astype(np.float32)
            v_np = (np.random.randn(1, N_KV, max_pos, HEAD_DIM) * 0.05).astype(np.float32)
            k_tt = alloc(k_np, device); v_tt = alloc(v_np, device)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                          device=device)
            med, p95 = time_op(
                lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, k_tt, v_tt, cur_pos_tensor=cur_pos_tt),
                device)
            print(f"{'non-paged':<24} {max_pos:>8} {cur_pos:>8} {med:>10.3f} {p95:>10.3f}")

        # PAGED at multiple MAX_POS values
        for max_pos in [256, 1024, 8192]:
            n_blocks = max_pos // BLOCK_SIZE
            cur_pos = max_pos - 8
            k_np = (np.random.randn(n_blocks, N_KV, BLOCK_SIZE, HEAD_DIM) * 0.05).astype(np.float32)
            v_np = (np.random.randn(n_blocks, N_KV, BLOCK_SIZE, HEAD_DIM) * 0.05).astype(np.float32)
            page_table_np = np.arange(n_blocks, dtype=np.int32).reshape(1, n_blocks)
            k_tt = alloc(k_np, device); v_tt = alloc(v_np, device)
            page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                              dtype=ttnn.int32, device=device,
                                              layout=ttnn.ROW_MAJOR_LAYOUT)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                          device=device,
                                          layout=ttnn.ROW_MAJOR_LAYOUT)
            try:
                med, p95 = time_op(
                    lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q_tt, k_tt, v_tt, page_table_tt, cur_pos_tensor=cur_pos_tt),
                    device)
                print(f"{'paged':<24} {max_pos:>8} {cur_pos:>8} {med:>10.3f} {p95:>10.3f}")
            except Exception as e:
                print(f"{'paged':<24} {max_pos:>8} {cur_pos:>8}  FAILED: {str(e)[:60]}")

        # === Compute decode-step impact ===
        print(f"\nPer-decode-step impact (16 full-attention layers per token):")
        # Re-run non-paged at 256 to get a fresh number to multiply
        # (using the already-printed values is fine; just compute the comparison)
        print("  Take the (paged - non-paged) ms × 16 layers to see per-token impact.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
