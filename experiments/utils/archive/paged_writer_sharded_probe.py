#!/usr/bin/env python3
"""
Probe: paged_fused_update_cache with several sharded memory configs.

Previous writer probe (paged_writer_smoke_probe.py) failed with
"Expect input_tensor to be sharded". The k_new/v_new tensors need to be in
a sharded memory config — the standard tt-metal pattern uses HEIGHT_SHARDED
on Blackhole.

This probe sweeps several candidate sharded configs and reports which
ones the kernel accepts. Each variant uploads K/V then calls
paged_fused_update_cache; success = no exception. We don't verify
numerical correctness here (next probe).

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_writer_sharded_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_KV = 4
P = 64
HD = 256
MAX_BLOCKS = 8


def attempt(device, label, sharded_cfg, k_new_np, v_new_np, page_table_np,
            cur_pos_np, keys_init, vals_init):
    """Try one sharded config; print pass/fail."""
    try:
        # Fresh caches each attempt
        keys_tt = ttnn.from_torch(torch.from_numpy(keys_init), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        vals_tt = ttnn.from_torch(torch.from_numpy(vals_init), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)

        # Upload k_new/v_new with the candidate sharded config
        k_new_tt = ttnn.from_torch(
            torch.from_numpy(k_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
        v_new_tt = ttnn.from_torch(
            torch.from_numpy(v_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)

        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

        ttnn.experimental.paged_fused_update_cache(
            keys_tt, k_new_tt, vals_tt, v_new_tt,
            update_idxs_tensor=cur_pos_tt,
            page_table=page_table_tt,
        )
        ttnn.synchronize_device(device)

        # Sanity: read back, check the written slot looks plausible
        keys_back = ttnn.to_torch(keys_tt).float().cpu().numpy()
        # Block 0, position 0, all kv heads should now hold k_new[0, 0, :, :]
        written = keys_back[0, :, 0, :]
        expected = k_new_np.reshape(N_KV, HD)
        cos = float(np.dot(written.flatten().astype(np.float64),
                           expected.flatten().astype(np.float64)) /
                    (np.linalg.norm(written) * np.linalg.norm(expected) + 1e-12))
        return ("PASS", cos, "")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        return ("FAIL", None, msg[:120])


def main():
    print("=" * 64)
    print("Probe: paged_fused_update_cache — find a working sharded config")
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
        candidates = []

        # Variant A: tt-transformers Blackhole canonical pattern
        try:
            cfg_A = ttnn.create_sharded_memory_config(
                shape=(32, HD),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            candidates.append(("A: (32, HD) shard, 4x8 grid (32 cores)", cfg_A))
        except Exception as e:
            print(f"  cfg A construction FAILED: {e}")

        # Variant B: 1x1 core, shard = full tensor (smallest possible parallelism)
        try:
            cfg_B = ttnn.create_sharded_memory_config(
                shape=(N_KV, HD),
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            candidates.append(("B: (N_KV, HD) shard, 1x1 grid (1 core)", cfg_B))
        except Exception as e:
            print(f"  cfg B construction FAILED: {e}")

        # Variant C: N_KV cores, each gets 1 KV head
        try:
            cfg_C = ttnn.create_sharded_memory_config(
                shape=(1, HD),
                core_grid=ttnn.CoreGrid(y=1, x=N_KV),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            candidates.append((f"C: (1, HD) shard, 1x{N_KV} grid ({N_KV} cores)", cfg_C))
        except Exception as e:
            print(f"  cfg C construction FAILED: {e}")

        # Variant D: stock L1_HEIGHT_SHARDED_MEMORY_CONFIG
        try:
            cfg_D = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            candidates.append(("D: ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG (default)", cfg_D))
        except Exception:
            pass

        print(f"\nTrying {len(candidates)} candidate sharded configs:")
        print("-" * 70)
        for label, cfg in candidates:
            print(f"\n[{label}]")
            verdict, cos, err = attempt(device, label, cfg, k_new_np, v_new_np,
                                         page_table_np, cur_pos_np, keys_init, vals_init)
            if verdict == "PASS":
                print(f"  ✓ PASS — write completed. Readback cos vs expected: {cos:.6f}")
            else:
                print(f"  ✗ FAIL: {err}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
