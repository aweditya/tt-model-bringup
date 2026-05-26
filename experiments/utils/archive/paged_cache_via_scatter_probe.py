#!/usr/bin/env python3
"""
Probe: paged cache writer via ttnn.scatter (workaround for #16674).

Both paged_fused_update_cache and paged_update_cache hang on Blackhole
(0% CPU at 1+ min, killed by timeout). The READ kernel (paged SDPA decode)
is fine. Solution: bypass the broken writer kernel entirely. Use scatter
— which we already validated in C'1 — to update the paged cache.

Cache shape: [max_blocks, N_KV, P, HD]
To write at logical position t:
  block_idx = t // P
  slot_idx  = t % P

Two-scatter on-device pattern:
  1. block_view = ttnn.slice(cache, [block_idx, ..., 0, 0], [block_idx+1, ...])
     → [1, N_KV, P, HD]
  2. block_with_slot = ttnn.scatter(block_view, dim=2, index=slot_idx, src=k_new)
  3. cache_new = ttnn.scatter(cache, dim=0, index=block_idx, src=block_with_slot)

Plus the SDPA-decode read that we've already validated.

This probe writes at positions 0, 1, 32, 63, 64, 100 across multiple blocks,
then reads via paged SDPA decode at cur_pos=100 and compares to a numpy
GQA reference.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_cache_via_scatter_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_Q = 32
N_KV = 4
N_REP = N_Q // N_KV
P = 64
HD = 256
MAX_BLOCKS = 8
TEST_POSITIONS = [0, 1, 32, 63, 64, 100]   # mix of block-start, mid, end, cross-block
CUR_POS = 100                                # read after writing all positions


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def scatter_write(cache_tt, pos, k_new_tt, device):
    """Write k_new (shape [N_KV, HD]) into paged cache at logical position pos."""
    block_idx = pos // P
    slot_idx = pos % P
    # Slice out the relevant block
    block = ttnn.slice(cache_tt, [block_idx, 0, 0, 0],
                       [block_idx + 1, N_KV, P, HD])
    # Reshape k_new to [1, N_KV, 1, HD] for slot scatter
    k_reshaped = ttnn.reshape(k_new_tt, [1, N_KV, 1, HD])
    # Cast to bf16 if needed (scatter refuses fp32+TILE per C'1 lesson)
    k_reshaped = ttnn.typecast(k_reshaped, ttnn.bfloat16)
    # Slot scatter: write k at position slot_idx along dim=2 of the block
    slot_index_np = np.full((1, N_KV, 1, HD), slot_idx, dtype=np.int32)
    slot_index_tt = ttnn.from_torch(torch.from_numpy(slot_index_np),
                                     dtype=ttnn.int32, device=device,
                                     layout=ttnn.TILE_LAYOUT)
    block_new = ttnn.scatter(block, dim=2, index=slot_index_tt, src=k_reshaped)
    # Block scatter: write block_new at position block_idx along dim=0 of the cache
    block_index_np = np.full((1, N_KV, P, HD), block_idx, dtype=np.int32)
    block_index_tt = ttnn.from_torch(torch.from_numpy(block_index_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.TILE_LAYOUT)
    cache_new = ttnn.scatter(cache_tt, dim=0, index=block_index_tt, src=block_new)
    return cache_new


def main():
    print("=" * 64)
    print("Probe: paged cache writer via ttnn.scatter (workaround for #16674)")
    print(f"  N_KV={N_KV}, P={P}, HD={HD}, max_blocks={MAX_BLOCKS}")
    print(f"  test positions: {TEST_POSITIONS}")
    print("=" * 64)

    rng = np.random.default_rng(7)

    # Generate known K/V values for each test position
    k_per_pos = {t: rng.standard_normal((N_KV, HD)).astype(np.float32) * 0.1 for t in TEST_POSITIONS}
    v_per_pos = {t: rng.standard_normal((N_KV, HD)).astype(np.float32) * 0.1 for t in TEST_POSITIONS}

    # Query
    q_np = rng.standard_normal((N_Q, HD)).astype(np.float32) * 0.1

    # Build numpy reference cache (logical [N_KV, max_pos, HD], with the rest zeros)
    max_pos = MAX_BLOCKS * P
    k_logical_ref = np.zeros((N_KV, max_pos, HD), dtype=np.float32)
    v_logical_ref = np.zeros((N_KV, max_pos, HD), dtype=np.float32)
    for t in TEST_POSITIONS:
        k_logical_ref[:, t, :] = k_per_pos[t]
        v_logical_ref[:, t, :] = v_per_pos[t]

    # Build numpy reference output (GQA-aware)
    rep = N_Q // N_KV
    ref_out = np.zeros((N_Q, HD), dtype=np.float32)
    for h in range(N_Q):
        kv_h = h // rep
        k = k_logical_ref[kv_h, :CUR_POS + 1, :]
        v = v_logical_ref[kv_h, :CUR_POS + 1, :]
        scores = (q_np[h] @ k.T) / np.sqrt(HD)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        ref_out[h] = weights @ v

    device = ttnn.open_device(device_id=0)
    try:
        # Initialize empty paged cache
        cache_shape = (MAX_BLOCKS, N_KV, P, HD)
        keys_tt = ttnn.from_torch(torch.from_numpy(np.zeros(cache_shape, dtype=np.float32)),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        vals_tt = ttnn.from_torch(torch.from_numpy(np.zeros(cache_shape, dtype=np.float32)),
                                   dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        # Write each position via the scatter workaround
        print("\nWriting via scatter workaround:")
        write_times = []
        for t in TEST_POSITIONS:
            k_tt = ttnn.from_torch(torch.from_numpy(k_per_pos[t]),
                                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            v_tt = ttnn.from_torch(torch.from_numpy(v_per_pos[t]),
                                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.synchronize_device(device)
            t0 = time.time()
            keys_tt = scatter_write(keys_tt, t, k_tt, device)
            vals_tt = scatter_write(vals_tt, t, v_tt, device)
            ttnn.synchronize_device(device)
            t1 = time.time()
            write_times.append((t, (t1 - t0) * 1000))
            print(f"  pos {t:>3}: K + V written in {(t1-t0)*1000:.2f} ms")

        # Verify cache contents on host first
        print("\nVerifying cache contents (host readback):")
        cache_k_back = ttnn.to_torch(keys_tt).float().cpu().numpy()
        cache_v_back = ttnn.to_torch(vals_tt).float().cpu().numpy()
        for t in TEST_POSITIONS:
            block, slot = t // P, t % P
            written_k = cache_k_back[block, :, slot, :]
            cos = _cosine(written_k, k_per_pos[t])
            md = float(np.abs(written_k - k_per_pos[t]).max())
            print(f"  pos {t:>3} (block {block}, slot {slot}): K cos={cos:.6f}, max|Δ|={md:.4e}")

        # Now read via paged SDPA decode at cur_pos=100
        print("\nReading via paged SDPA decode:")
        q_tt = ttnn.from_torch(torch.from_numpy(q_np.reshape(1, B, N_Q, HD)),
                                dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        page_table_np = np.arange(MAX_BLOCKS, dtype=np.int32).reshape(B, MAX_BLOCKS)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(np.array([CUR_POS], dtype=np.int32)),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)
        hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False,
        )
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, keys_tt, vals_tt, page_table_tt,
            cur_pos_tensor=cur_pos_tt,
            compute_kernel_config=hifi4,
        )
        ttnn.synchronize_device(device)
        out_np = ttnn.to_torch(attn).float().cpu().numpy().reshape(N_Q, HD)

        # Compare to GQA reference
        print("\nNumerical correctness vs numpy GQA reference:")
        n_pass = 0
        for h in range(N_Q):
            cos = _cosine(out_np[h], ref_out[h])
            if cos > 0.99:
                n_pass += 1
        worst_h = min(range(N_Q), key=lambda h: _cosine(out_np[h], ref_out[h]))
        worst_cos = _cosine(out_np[worst_h], ref_out[worst_h])
        print(f"  passing heads (cos > 0.99): {n_pass}/{N_Q}")
        print(f"  worst head: q_head {worst_h}, cos = {worst_cos:.6f}")

        print()
        if n_pass == N_Q:
            print("✓ FULL PASS: scatter-based writer + paged SDPA reader work end-to-end.")
            print("  Workaround for #16674 is correct and unblocks C'0.5 long-context path.")
        else:
            print(f"⚠ Only {n_pass}/{N_Q} heads pass. Investigate.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
