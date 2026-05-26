#!/usr/bin/env python3
"""
Probe: paged_fused_update_cache with K and V on DISJOINT core ranges.

The previous probe's variant A almost worked — the error was "input_tensor1
and input_tensor2 must not overlap". That means the canonical 4×8 grid
shard pattern IS right, but K and V both landed on cores (0..7, 0..3).
This probe places K and V on adjacent disjoint 4×8 core ranges.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_writer_disjoint_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_KV = 4
P = 64
HD = 256
MAX_BLOCKS = 8


def main():
    print("=" * 64)
    print("Probe: paged_fused_update_cache with disjoint K/V core ranges")
    print("=" * 64)

    rng = np.random.default_rng(7)
    k_new_np = rng.standard_normal((1, B, N_KV, HD)).astype(np.float32) * 0.1
    v_new_np = rng.standard_normal((1, B, N_KV, HD)).astype(np.float32) * 0.1
    keys_init = np.zeros((MAX_BLOCKS, N_KV, P, HD), dtype=np.float32)
    vals_init = np.zeros((MAX_BLOCKS, N_KV, P, HD), dtype=np.float32)
    page_table_np = np.arange(MAX_BLOCKS, dtype=np.int32).reshape(B, MAX_BLOCKS)
    cur_pos_np = np.array([0], dtype=np.int32)

    device = ttnn.open_device(device_id=0)
    try:
        # Build TWO sharded configs on disjoint core ranges
        # K: cores (x=0..7, y=0..3) — 32 cores top half
        # V: cores (x=0..7, y=4..7) — 32 cores bottom half
        k_cores = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))
        })
        v_cores = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7, 7))
        })

        shard_spec_k = ttnn.ShardSpec(k_cores, [32, HD], ttnn.ShardOrientation.ROW_MAJOR)
        shard_spec_v = ttnn.ShardSpec(v_cores, [32, HD], ttnn.ShardOrientation.ROW_MAJOR)
        cfg_k = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1, shard_spec_k)
        cfg_v = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1, shard_spec_v)
        print(f"  K cores: (0,0)..(7,3) = 32 cores top")
        print(f"  V cores: (0,4)..(7,7) = 32 cores bottom (disjoint from K)")
        print(f"  shard per core: (32, {HD}) = 1 tile × HD")

        # Upload K and V to disjoint sharded configs
        k_new_tt = ttnn.from_torch(
            torch.from_numpy(k_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=cfg_k)
        v_new_tt = ttnn.from_torch(
            torch.from_numpy(v_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=cfg_v)
        print(f"  k_new shape: {tuple(k_new_tt.shape)}, layout: {k_new_tt.layout}")

        # Caches live in DRAM_INTERLEAVED (default)
        keys_tt = ttnn.from_torch(torch.from_numpy(keys_init), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        vals_tt = ttnn.from_torch(torch.from_numpy(vals_init), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)

        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

        print("\nCalling paged_fused_update_cache with disjoint K/V...")
        try:
            ttnn.experimental.paged_fused_update_cache(
                keys_tt, k_new_tt, vals_tt, v_new_tt,
                update_idxs_tensor=cur_pos_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)
            print("  ✓ Write succeeded!")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  ✗ FAIL: {msg[:200]}")
            return

        # Read back and verify
        print("\nVerifying write...")
        keys_back = ttnn.to_torch(keys_tt).float().cpu().numpy()
        vals_back = ttnn.to_torch(vals_tt).float().cpu().numpy()
        # Block 0, position 0, all N_KV heads should equal k_new[0, 0]
        written_k = keys_back[0, :, 0, :]
        expected_k = k_new_np[0, 0, :, :]
        cos_k = float(np.dot(written_k.flatten().astype(np.float64),
                              expected_k.flatten().astype(np.float64)) /
                       (np.linalg.norm(written_k) * np.linalg.norm(expected_k) + 1e-12))
        max_diff_k = float(np.abs(written_k - expected_k).max())
        written_v = vals_back[0, :, 0, :]
        expected_v = v_new_np[0, 0, :, :]
        cos_v = float(np.dot(written_v.flatten().astype(np.float64),
                              expected_v.flatten().astype(np.float64)) /
                       (np.linalg.norm(written_v) * np.linalg.norm(expected_v) + 1e-12))
        max_diff_v = float(np.abs(written_v - expected_v).max())

        # Untouched positions still zero?
        keys_other = np.concatenate([keys_back[0, :, 1:, :].flatten(),
                                      keys_back[1:, :, :, :].flatten()])
        zero_max = float(np.abs(keys_other).max())

        print(f"  K @ block 0, pos 0: cos={cos_k:.6f}, max|Δ|={max_diff_k:.4e}")
        print(f"  V @ block 0, pos 0: cos={cos_v:.6f}, max|Δ|={max_diff_v:.4e}")
        print(f"  K everywhere else: max|·|={zero_max:.4e}")
        print()
        if cos_k > 0.99 and cos_v > 0.99 and zero_max < 0.01:
            print("  ✓ Writer works correctly. Paged path is fully unblocked.")
        else:
            print("  ⚠ Wrote without crash but content is wrong. Investigate.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
