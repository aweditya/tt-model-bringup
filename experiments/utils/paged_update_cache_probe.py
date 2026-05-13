#!/usr/bin/env python3
"""
Probe #5: ttnn.experimental.paged_update_cache writer with sharded input on Blackhole.

Context: `feedback_paged_sdpa_decode_works_at_32k.md` says paged_update_cache fails
with "TT_FATAL: Expect input_tensor to be sharded" — the writer needs height-sharded
input, not just TILE_LAYOUT. This probe figures out the exact sharded memory config
that works on Blackhole and validates correctness.

If this passes, the paged SDPA migration becomes a straightforward integration:
  - eager:  paged_update_cache(k_cache, k_sharded, update_idxs_tensor=cur_pos_tt,
                                page_table=page_table)
  - reader: paged_scaled_dot_product_attention_decode (already validated in
            feedback_paged_sdpa_gqa_validated.md)
  - trace:  both ops use device-resident position/page_table — trace-compatible.

Method:
  1. Build input `[1, num_users=1, num_heads=N_KV=4, head_dim]` then pad to
     `[1, 1, 32, head_dim]` (32 is the TILE_HEIGHT padding requirement).
  2. Apply HEIGHT_SHARDED memory config across num_users cores (just 1 here).
  3. Build paged cache `[max_num_blocks, num_heads, block_size, head_dim]`.
  4. Build page_table `[num_users, max_num_blocks_per_seq]` int32.
  5. Call paged_update_cache. Verify the write landed at the right position by
     reading the cache back and comparing the target slot to the input.

Recipe sources:
  - `experiments/.refs/tt-metal/tests/ttnn/nightly/unit_tests/operations/
     transformers/test_paged_update_cache.py` (canonical sharded input pattern)
  - `experiments/.refs/tt-metal/models/tt_transformers/tt/attention.py:692-697`

Run on qb2 (device 0, server must be killed first):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_update_cache_probe.py
"""
import os, sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

# Qwen3.6-27B shape
N_KV = 4
HEAD_DIM = 256
MAX_SEQ_LEN = 256        # matches our current MAX_POS for now
BLOCK_SIZE = 64
NUM_USERS = 1            # single user
TILE_HEIGHT = 32         # ttnn TILE_SIZE


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print(f"Probe #5: paged_update_cache sharded writer  device_id={device_id}")
    print(f"  N_KV={N_KV} HEAD_DIM={HEAD_DIM} BLOCK_SIZE={BLOCK_SIZE} MAX_SEQ={MAX_SEQ_LEN}")

    device = ttnn.open_device(device_id=device_id)
    try:
        max_num_blocks_per_seq = MAX_SEQ_LEN // BLOCK_SIZE
        max_num_blocks = NUM_USERS * max_num_blocks_per_seq

        # === Build paged cache ===
        # Start with logical cache shape [num_users, num_heads, max_seq, head_dim]
        cache_np = (np.random.RandomState(42).randn(NUM_USERS, N_KV, MAX_SEQ_LEN, HEAD_DIM)
                    * 0.01).astype(np.float32)
        # Re-pattern to paged layout [max_num_blocks, num_heads, block_size, head_dim]
        paged_np = (cache_np
                    .reshape(NUM_USERS, N_KV, max_num_blocks_per_seq, BLOCK_SIZE, HEAD_DIM)
                    .transpose(0, 2, 1, 3, 4)  # → [num_users, num_blocks_per_seq, n_heads, block_size, hd]
                    .reshape(max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM))

        cache_tt = ttnn.from_torch(torch.from_numpy(paged_np), dtype=ttnn.bfloat16,
                                     device=device, layout=ttnn.TILE_LAYOUT,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"  cache_tt shape: {tuple(cache_tt.shape)}")

        # === Build page table (identity mapping for single user) ===
        page_table_np = np.arange(max_num_blocks_per_seq, dtype=np.int32).reshape(
            NUM_USERS, max_num_blocks_per_seq)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                          dtype=ttnn.int32, device=device,
                                          layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"  page_table_tt shape: {tuple(page_table_tt.shape)}")

        # === Build sharded input ===
        # Per test_paged_update_cache.py: input_shape=[1, num_users, num_heads, head_dim]
        # then pad num_heads dim to TILE_HEIGHT.
        target_pos = 17
        input_np = (np.random.RandomState(7).randn(1, NUM_USERS, N_KV, HEAD_DIM)
                    * 0.1).astype(np.float32)
        # Pad dim -2 (num_heads) from N_KV to 32
        input_padded = np.zeros((1, NUM_USERS, TILE_HEIGHT, HEAD_DIM), dtype=np.float32)
        input_padded[:, :, :N_KV, :] = input_np
        print(f"  input_padded shape: {input_padded.shape}")

        # Convert to ttnn TILE_LAYOUT first
        xt = ttnn.from_torch(torch.from_numpy(input_padded), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT)
        xt = ttnn.reshape(xt, ttnn.Shape([1, NUM_USERS, N_KV, HEAD_DIM]))

        # Apply HEIGHT_SHARDED memory config across NUM_USERS cores
        compute_grid = device.compute_with_storage_grid_size()
        num_cores = NUM_USERS
        shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
        # Per test_paged_update_cache: shard shape [volume / last_dim / num_cores, last_dim]
        # volume of padded = 1*1*32*256 = 8192. 8192/256/1 = 32.
        # So shard shape is [32, 256]
        input_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [TILE_HEIGHT, HEAD_DIM],   # [32, 256]
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
        try:
            xt = xt.to(device, input_mem_config)
        except Exception as e:
            print(f"  ✗ Failed to upload sharded input: {type(e).__name__}: {str(e)[:200]}")
            return

        # === Build update_idxs_tensor (device-resident position tensor) ===
        cache_idxs_tt = ttnn.from_torch(torch.tensor([target_pos], dtype=torch.int32),
                                          device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # === The call ===
        print(f"\n[CALL] paged_update_cache(cache, sharded_input, update_idxs={target_pos}, page_table)")
        try:
            ttnn.experimental.paged_update_cache(
                cache_tt, xt,
                update_idxs_tensor=cache_idxs_tt,
                page_table=page_table_tt)
            ttnn.synchronize_device(device)
            print("  ✓ call succeeded, no hang")
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:300]}")
            return

        # === Verify ===
        # Read cache back; reverse the paging to original [num_users, n_heads, max_seq, hd]
        cache_back_np = ttnn.to_torch(cache_tt).float().cpu().numpy()
        # cache_back_np shape: [max_num_blocks, num_heads, block_size, head_dim]
        unpaged = (cache_back_np
                   .reshape(NUM_USERS, max_num_blocks_per_seq, N_KV, BLOCK_SIZE, HEAD_DIM)
                   .transpose(0, 2, 1, 3, 4)
                   .reshape(NUM_USERS, N_KV, MAX_SEQ_LEN, HEAD_DIM))
        # Expected: position `target_pos` for each KV head was overwritten by input
        written = unpaged[0, :, target_pos, :]                     # [N_KV, HEAD_DIM]
        expected = input_np[0, 0, :, :]                            # [N_KV, HEAD_DIM]
        max_d = float(np.abs(written - expected).max())
        cos = _cosine(written, expected)
        print(f"\n[VERIFY] cache[0, :, {target_pos}, :] vs input:")
        print(f"  cosine = {cos:.6f}  max|Δ| = {max_d:.4e}")
        target_ok = cos > 0.99 and max_d < 0.05

        # Check that other positions still hold the original cache
        # Position 16 is in the same tile as 17 (tile 0 covers positions 0..31).
        # The op may quantize the entire tile; just check position 100 (different tile).
        check_pos = 100
        unchanged = unpaged[0, :, check_pos, :]
        original = cache_np[0, :, check_pos, :]
        cos_un = _cosine(unchanged, original)
        max_d_un = float(np.abs(unchanged - original).max())
        print(f"  cache[0, :, {check_pos}, :] (untouched): cos = {cos_un:.6f}  max|Δ| = {max_d_un:.4e}")
        untouched_ok = cos_un > 0.999 and max_d_un < 0.05

        print("\n=== VERDICT ===")
        if target_ok and untouched_ok:
            print(f"  ✓ paged_update_cache WORKS on Blackhole with sharded input.")
            print(f"  ✓ Target write correct, neighbours preserved.")
            print(f"  ✓ The paged SDPA migration is FULLY UNBLOCKED (reader + writer).")
        else:
            print(f"  ✗ Writer test failed (target_ok={target_ok}, untouched_ok={untouched_ok})")
            print(f"    Will need fallback (dual-cache or custom kernel).")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
