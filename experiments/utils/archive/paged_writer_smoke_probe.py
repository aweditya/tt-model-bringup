#!/usr/bin/env python3
"""
Probe #1: paged_fused_update_cache writer smoke test.

CRITICAL: tt-metal issue #16674 reports that paged_update_cache hangs on
Blackhole under certain configs. If THIS test hangs at our shapes, the
whole paged path is blocked — we need a workaround or wait for upstream fix.

Shapes (Qwen3.6-27B target):
  - B=1 (batch)
  - N_KV=4
  - block_size P=64 (agent's sweet spot)
  - HD=256
  - max_num_blocks=8 (small for smoke test)

We write K/V at cur_pos=0 (first slot of first block) and read back to verify.
Timeout set to 60 sec — if the write doesn't complete in that, it's hung.

Run on qb1 (qb2 busy):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_writer_smoke_probe.py
"""
import sys
import time
import signal
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_KV = 4
P = 64               # block_size
HD = 256
MAX_BLOCKS = 8       # = 8 * 64 = 512 max position


def timeout_handler(signum, frame):
    raise TimeoutError("write took longer than 60 seconds — likely hung (#16674)")


def main():
    print("=" * 64)
    print(f"Probe #1: paged_fused_update_cache writer smoke")
    print(f"  shapes: B={B}, N_KV={N_KV}, block_size={P}, HD={HD}, max_blocks={MAX_BLOCKS}")
    print(f"  bug-watch: tt-metal #16674 (Blackhole hang)")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)
    try:
        # Build zero-initialized paged cache
        cache_shape = (MAX_BLOCKS, N_KV, P, HD)
        cache_np = np.zeros(cache_shape, dtype=np.float32)
        keys_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        values_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                     device=device, layout=ttnn.TILE_LAYOUT)
        print(f"  cache shape: {tuple(keys_tt.shape)}, dtype={keys_tt.dtype}")

        # New K/V values for this step — shape [1, B, N_KV, HD]
        # Use distinctive values per (head, dim) so we can verify placement
        rng = np.random.default_rng(0)
        k_new_np = rng.standard_normal((1, B, N_KV, HD)).astype(np.float32) * 0.1
        v_new_np = rng.standard_normal((1, B, N_KV, HD)).astype(np.float32) * 0.1
        k_new_tt = ttnn.from_torch(torch.from_numpy(k_new_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        v_new_tt = ttnn.from_torch(torch.from_numpy(v_new_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        print(f"  new K/V shape: {tuple(k_new_tt.shape)}")

        # Page table: [B, max_blocks_per_seq] = [1, 8]. Identity mapping.
        page_table_np = np.arange(MAX_BLOCKS, dtype=np.int32).reshape(B, MAX_BLOCKS)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"  page_table shape: {tuple(page_table_tt.shape)}, values={page_table_np[0, :].tolist()}")

        # cur_pos tensor: writing to position 0 (start of block 0)
        cur_pos_np = np.array([0], dtype=np.int32)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"  cur_pos: {cur_pos_np.tolist()}")

        # Attempt the write
        print("\nCalling paged_fused_update_cache...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        try:
            t0 = time.time()
            ttnn.experimental.paged_fused_update_cache(
                keys_tt, k_new_tt, values_tt, v_new_tt,
                update_idxs_tensor=cur_pos_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)
            t1 = time.time()
            signal.alarm(0)
            print(f"  ✓ write completed in {(t1-t0)*1000:.2f} ms — NO HANG")
        except TimeoutError as e:
            signal.alarm(0)
            print(f"  ✗ HANG DETECTED: {e}")
            print(f"  #16674 may be active on our shapes/build. Blocked.")
            return

        # Read back and verify
        print("\nReading back and verifying...")
        keys_back = ttnn.to_torch(keys_tt).float().cpu().numpy()
        values_back = ttnn.to_torch(values_tt).float().cpu().numpy()
        # Block 0, position 0 across all N_KV heads should match k_new_np[0, 0]
        written_k = keys_back[0, :, 0, :]    # [N_KV, HD]
        expected_k = k_new_np[0, 0, :, :]    # [N_KV, HD]
        cos_k = float(np.dot(written_k.flatten().astype(np.float64),
                              expected_k.flatten().astype(np.float64)) /
                       (np.linalg.norm(written_k) * np.linalg.norm(expected_k) + 1e-12))
        max_diff_k = float(np.abs(written_k - expected_k).max())
        print(f"  K @ block 0, pos 0: cos={cos_k:.6f}, max|Δ|={max_diff_k:.4e}")
        written_v = values_back[0, :, 0, :]
        expected_v = v_new_np[0, 0, :, :]
        cos_v = float(np.dot(written_v.flatten().astype(np.float64),
                              expected_v.flatten().astype(np.float64)) /
                       (np.linalg.norm(written_v) * np.linalg.norm(expected_v) + 1e-12))
        max_diff_v = float(np.abs(written_v - expected_v).max())
        print(f"  V @ block 0, pos 0: cos={cos_v:.6f}, max|Δ|={max_diff_v:.4e}")
        # Other positions should still be zero
        keys_other = np.concatenate([keys_back[0, :, 1:, :].flatten(),
                                      keys_back[1:, :, :, :].flatten()])
        zero_max = float(np.abs(keys_other).max())
        print(f"  K @ everywhere else: max|·| = {zero_max:.4e} (should be 0)")
        print()
        if cos_k > 0.99 and cos_v > 0.99 and zero_max < 0.01:
            print("  ✓ Writer is CORRECT and does not hang on our shapes.")
            print("  Paged path is unblocked from a writer standpoint. Proceed to probe #2.")
        else:
            print("  ✗ Writer ran but produced wrong data. Investigate.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
