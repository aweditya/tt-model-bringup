"""speculative_feasibility_probe.py - Verify B=2 paged-SDPA viability for MTP verify lane.

What we are answering:
  - Does ttnn.transformer.paged_scaled_dot_product_attention_decode accept B=2
    queries in a single call at our production Qwen3.6 GQA shape
    (N_Q=24, N_KV=4, HEAD_DIM=256)?
  - What does the latency look like vs B=1?
  - Project: MTP draft step ~2.1 ms (roofline 849 MB at 400 GB/s), verifier
    step ~200 ms - so the only remaining unknown is whether B=2 verify
    works AT ALL on our hardware, and if its cost is ~ 2x B=1 or sub-linear.

DeltaNet recurrence is NOT exercised here (separate question, answered in
research notes: state must be snapshot/restore per spec lane - tractable but
not free). This probe only addresses the attention verify path.

Run on qb1 device 3.

Expected outcomes:
  - PASS: B=2 SDPA works. Greenlight Branch D'3 MTP integration.
  - FAIL: shape error / crash. Spec decoding requires more invasive kernel work.
"""

import os
import sys
import time

import numpy as np
import torch

import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Production Qwen3.6-27B attention shape (per layer)
HIDDEN = 5120
N_Q = 24                # query heads
N_KV = 4                # KV heads
HEAD_DIM = 256
MAX_POS = 256           # cache length
BLOCK_SIZE = 32         # paged page size

assert MAX_POS % BLOCK_SIZE == 0
NUM_BLOCKS = MAX_POS // BLOCK_SIZE


def open_dev():
    return ttnn.open_device(device_id=3)


def make_kv_cache(device, B):
    """Allocate paged-KV-cache tensors of shape [num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM]
    big enough to hold B users x MAX_POS tokens each.
    """
    total_blocks = B * NUM_BLOCKS
    np.random.seed(0)
    k_data = (np.random.randn(total_blocks, N_KV, BLOCK_SIZE, HEAD_DIM).astype(np.float32) * 0.02).astype(np.float32)
    v_data = (np.random.randn(total_blocks, N_KV, BLOCK_SIZE, HEAD_DIM).astype(np.float32) * 0.02).astype(np.float32)
    k_tt = ttnn.from_torch(torch.from_numpy(k_data), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(torch.from_numpy(v_data), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    return k_tt, v_tt


def make_page_table(B):
    """Each user (batch row) maps NUM_BLOCKS blocks.
    Shape: [B, NUM_BLOCKS] int32.
    """
    pt = np.zeros((B, NUM_BLOCKS), dtype=np.int32)
    for u in range(B):
        for b in range(NUM_BLOCKS):
            pt[u, b] = u * NUM_BLOCKS + b
    return pt


def run_B(device, B, label, num_iters=20):
    print(f"\n=== B={B} ({label}) ===")
    k_tt, v_tt = make_kv_cache(device, B)
    pt_np = make_page_table(B)
    page_table_tt = ttnn.from_torch(torch.from_numpy(pt_np), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Query layout for paged SDPA decode: [1, B, N_Q, HEAD_DIM] bf16
    np.random.seed(1)
    q_data = (np.random.randn(1, B, N_Q, HEAD_DIM).astype(np.float32) * 0.02).astype(np.float32)
    q_tt = ttnn.from_torch(torch.from_numpy(q_data), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)

    # Each batch row at position 100 (mid-cache)
    cur_pos = np.full((B,), 100, dtype=np.int32)
    cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    try:
        # Single warmup call to surface any shape errors fast
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, k_tt, v_tt,
            cur_pos_tensor=cur_pos_tt,
            page_table_tensor=page_table_tt,
        )
        ttnn.synchronize_device(device)
        print(f"  paged SDPA B={B} CALL OK, output shape={out.shape}, dtype={out.dtype}")
    except Exception as e:
        print(f"  paged SDPA B={B} FAILED: {e}")
        ttnn.deallocate(q_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)
        ttnn.deallocate(page_table_tt)
        ttnn.deallocate(cur_pos_tt)
        return None

    # Timing
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(num_iters):
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, k_tt, v_tt,
            cur_pos_tensor=cur_pos_tt,
            page_table_tensor=page_table_tt,
        )
    ttnn.synchronize_device(device)
    t1 = time.time()
    ms = (t1 - t0) / num_iters * 1000.0
    print(f"  paged SDPA B={B} latency: {ms:.3f} ms/call (mean of {num_iters})")

    ttnn.deallocate(q_tt)
    ttnn.deallocate(out)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(cur_pos_tt)
    return ms


def main():
    print("speculative_feasibility_probe: paged SDPA B=1 vs B=2 at Qwen3.6 shape")
    print(f"  N_Q={N_Q}, N_KV={N_KV}, HEAD_DIM={HEAD_DIM}, MAX_POS={MAX_POS}, BLOCK_SIZE={BLOCK_SIZE}")

    device = open_dev()
    try:
        ms_b1 = run_B(device, 1, "single-lane baseline")
        ms_b2 = run_B(device, 2, "MTP verify shape (B=2)")

        print("\n=== Summary ===")
        if ms_b1 is None or ms_b2 is None:
            print("  At least one config FAILED. See above.")
            return 1
        print(f"  B=1: {ms_b1:.3f} ms")
        print(f"  B=2: {ms_b2:.3f} ms  ({ms_b2/ms_b1:.2f}x B=1)")
        # SDPA budget at full-decode is ~2% (4 ms / 192 ms total). Even at
        # 2x cost B=2 -> +4 ms for all attention layers; budget is healthy.
        delta = ms_b2 - ms_b1
        layers_full_attn = 16
        added_per_tok = delta * layers_full_attn
        print(f"  Extra SDPA cost if MTP verify uses B=2 for all 16 full-attn layers:")
        print(f"    +{added_per_tok:.2f} ms/tok (vs 192 ms/tok baseline = {added_per_tok/192*100:.1f}% overhead)")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
