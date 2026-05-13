#!/usr/bin/env python3
"""
Probe: paged_update_cache (NON-fused, separate K and V calls).

The disjoint-cores probe for paged_FUSED_update_cache likely hit issue
#16674 (Blackhole hang) — process ran 14 min at 101% CPU without ever
returning. Killed.

Hypothesis: the non-fused variant avoids the K/V overlap problem
entirely (one tensor at a time), so it might be the safer path. Trade:
2 dispatches per cache update instead of 1 (small).

This probe:
  1. Calls paged_update_cache for K alone (single sharded tensor)
  2. Calls paged_update_cache for V alone
  3. Reads back, verifies

If THIS hangs or fails, the writer path on Blackhole is broken across
all variants and we need to either wait for the upstream fix or use a
workaround (e.g., on-device scatter into the paged cache by hand —
similar to C'1 scatter, but for paged layout).

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_update_cache_unfused_probe.py
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
P = 64
HD = 256
MAX_BLOCKS = 8


def timeout_handler(signum, frame):
    raise TimeoutError("op took > 60s — almost certainly hung (#16674)")


def main():
    print("=" * 64)
    print("Probe: paged_update_cache (unfused, K and V separately)")
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
        # Sharded config (4x8 grid, shard 32 x HD) — canonical tt-transformers pattern
        cfg = ttnn.create_sharded_memory_config(
            shape=(32, HD),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        print(f"  sharded cfg: (32, {HD}), 4x8 grid (32 cores)")

        k_new_tt = ttnn.from_torch(
            torch.from_numpy(k_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=cfg)
        v_new_tt = ttnn.from_torch(
            torch.from_numpy(v_new_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT, memory_config=cfg)

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

        # K write
        print("\n[1/2] Writing K via paged_update_cache...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        try:
            t0 = time.time()
            ttnn.experimental.paged_update_cache(
                keys_tt, k_new_tt,
                update_idxs_tensor=cur_pos_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)
            signal.alarm(0)
            print(f"  ✓ K write OK in {(time.time()-t0)*1000:.2f} ms")
        except TimeoutError as e:
            signal.alarm(0)
            print(f"  ✗ HANG: {e}")
            return
        except Exception as e:
            signal.alarm(0)
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  ✗ FAIL: {msg[:200]}")
            return

        # V write
        print("\n[2/2] Writing V via paged_update_cache...")
        signal.alarm(60)
        try:
            t0 = time.time()
            ttnn.experimental.paged_update_cache(
                vals_tt, v_new_tt,
                update_idxs_tensor=cur_pos_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)
            signal.alarm(0)
            print(f"  ✓ V write OK in {(time.time()-t0)*1000:.2f} ms")
        except TimeoutError as e:
            signal.alarm(0)
            print(f"  ✗ HANG: {e}")
            return
        except Exception as e:
            signal.alarm(0)
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  ✗ FAIL: {msg[:200]}")
            return

        # Verify
        print("\nVerifying writes...")
        keys_back = ttnn.to_torch(keys_tt).float().cpu().numpy()
        vals_back = ttnn.to_torch(vals_tt).float().cpu().numpy()
        wk = keys_back[0, :, 0, :]
        ek = k_new_np[0, 0, :, :]
        cos_k = float(np.dot(wk.flatten().astype(np.float64), ek.flatten().astype(np.float64)) /
                       (np.linalg.norm(wk) * np.linalg.norm(ek) + 1e-12))
        max_diff_k = float(np.abs(wk - ek).max())
        wv = vals_back[0, :, 0, :]
        ev = v_new_np[0, 0, :, :]
        cos_v = float(np.dot(wv.flatten().astype(np.float64), ev.flatten().astype(np.float64)) /
                       (np.linalg.norm(wv) * np.linalg.norm(ev) + 1e-12))
        max_diff_v = float(np.abs(wv - ev).max())
        print(f"  K: cos={cos_k:.6f}, max|Δ|={max_diff_k:.4e}")
        print(f"  V: cos={cos_v:.6f}, max|Δ|={max_diff_v:.4e}")
        print()
        if cos_k > 0.99 and cos_v > 0.99:
            print("  ✓ Unfused writer works. Two dispatches per cache update is acceptable.")
            print("    Paged path is fully unblocked.")
        else:
            print("  ⚠ Wrote but content is wrong. Check page-table / position semantics.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
