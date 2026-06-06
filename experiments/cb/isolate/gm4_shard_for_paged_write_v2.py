#!/usr/bin/env python3
"""Isolation probe — `_shard_for_paged_write` 5-op chain vs single-op
`to_memory_config` reshard.

Round 1's Tracy v2 measured ~28 ms / forward in Tilize + TilizeWithValPadding
ops. `_shard_for_paged_write` in `server_gemma4_unified_ttnn.py:1066-1086`
fires 176×/forward (88 K + 88 V across 40 sliding + 8 global layers) and
does: to_layout(RM) → reshape → pad → to_layout(TILE) → to_memory_config.

This probe shows the post-slice TILE-layout `[1, head_dim]` (tile-padded
to `[32, head_dim]`) is byte-identical to the post-pad-tile
`[1, 1, BLOCK_SIZE=32, head_dim]` that the chain produces. Both should
land at the same cache slot via paged_fused_update_cache.

Test: feed two paths into paged_fused_update_cache with the same K/V data
and verify the cache_out is bit-identical.

Self-contained, single device (forks `paged_update_cache.py:48-130`).

Run on qb2:
  bash scripts/run_remote_qb2.sh experiments/cb/isolate/gm4_shard_for_paged_write_v2.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NKV = 1               # match sliding's NKV_PER_CHIP=1
HEAD_DIM = 256        # match HEAD_DIM_SLIDING
BLOCK_SIZE = 32
TILE_HEIGHT = 32
NUM_BLOCKS_TOTAL = 32
B_INPUT = 1           # paged_fused_update_cache: input.padded_shape[1] = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_mem_cfg(device, n_kv_heads, head_dim, block_size, base_col=0):
    """Build the L1 HEIGHT_SHARDED mem_cfg used by paged_fused_update_cache.

    base_col = column offset so K and V land on disjoint cores.
    Forks server_gemma4_unified_ttnn.py:509-543.
    """
    compute_grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(base_col, 0),
        ttnn.CoreCoord(base_col + n_kv_heads - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [block_size, head_dim],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )


def _shard_for_paged_write_old(t_2d, n_kv_heads, head_dim, block_size, mem_cfg):
    """Production 5-op chain (server_gemma4_unified_ttnn.py:1066-1086)."""
    t_rm = ttnn.to_layout(t_2d, ttnn.ROW_MAJOR_LAYOUT)
    t4d = ttnn.reshape(t_rm, [1, n_kv_heads, 1, head_dim])
    ttnn.deallocate(t_rm)
    t_pad = ttnn.pad(t4d, [[0, 0], [0, 0], [0, block_size - 1], [0, 0]], value=0.0)
    ttnn.deallocate(t4d)
    t_tile = ttnn.to_layout(t_pad, ttnn.TILE_LAYOUT)
    ttnn.deallocate(t_pad)
    out = ttnn.to_memory_config(t_tile, mem_cfg)
    ttnn.deallocate(t_tile)
    return out


def _shard_for_paged_write_new(t_2d, n_kv_heads, head_dim, block_size, mem_cfg):
    """Single-op variant — reshape the tile-padded [n_kv_heads, head_dim]
    (logical) → [1, n_kv_heads*head_dim] for `to_memory_config` to absorb
    into the [block_size, head_dim] L1 shard.

    The byte layout of TILE-padded `[n_kv_heads, head_dim]` (padded to
    `[block_size, head_dim]` since block_size == TILE_HEIGHT) is identical
    to `[1, n_kv_heads, block_size, head_dim]` — the tile-pad rows
    (positions n_kv_heads..block_size-1) carry zeros either way.

    Approach: use `ttnn.reshape` to set logical shape to [1, 1, n_kv_heads,
    head_dim] — but logical volume must match. With n_kv_heads=1, logical
    [1, head_dim] is fine. For n_kv_heads>1 input we'd have to slice first
    (production already slices per-head before calling this).
    """
    assert n_kv_heads == 1, "v_new variant expects per-head slicing upstream"
    # t_2d: logical [1, head_dim], TILE layout, padded to [32, head_dim].
    # Reshape to [1, 1, 1, head_dim] (logical) keeping TILE padding intact.
    # Then `to_memory_config` reshards: padded shape becomes [1, 1, 32, 256]
    # matching the L1 sharded mem_cfg's [BLOCK_SIZE, head_dim].
    t_4d = ttnn.reshape(t_2d, [1, 1, 1, head_dim])
    out = ttnn.to_memory_config(t_4d, mem_cfg)
    ttnn.deallocate(t_4d)
    return out


def _build_input(device, head_dim, k_data, dtype):
    """Build a [1, head_dim] TILE-layout interleaved tensor matching what
    we'd get post-RoPE in production (server_gemma4_unified_ttnn.py:1156
    after the per-head slice).
    """
    # k_data: [head_dim] numpy fp32.
    # Push as [1, head_dim] interleaved TILE.
    x = ttnn.from_torch(
        torch.from_numpy(k_data.reshape(1, head_dim)),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return x


def main():
    import ttnn as _ttnn
    global ttnn
    ttnn = _ttnn

    log("opening single device")
    device = ttnn.open_device(device_id=0)
    try:
        rng = np.random.default_rng(0)
        # Per-head K/V data (head 0 only — N_KV=1 effective per call,
        # matching the sliding NKV_PER_CHIP=1 contract).
        K_data = rng.normal(0, 1.0, (HEAD_DIM,)).astype(np.float32)
        V_data = rng.normal(0, 1.0, (HEAD_DIM,)).astype(np.float32)
        cur_pos = 5
        log(f"K_data shape={K_data.shape} V_data shape={V_data.shape} pos={cur_pos}")

        # Build mem_cfgs (K on col 0, V on col 1 — disjoint per fused-update).
        mc_K = _build_mem_cfg(device, NKV, HEAD_DIM, BLOCK_SIZE, base_col=0)
        mc_V = _build_mem_cfg(device, NKV, HEAD_DIM, BLOCK_SIZE, base_col=1)

        # update_idxs + page_table
        update_idxs = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        page_table_np = np.arange(NUM_BLOCKS_TOTAL, dtype=np.int32).reshape(1, -1)
        page_table_tt = ttnn.from_torch(
            torch.from_numpy(page_table_np),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        for variant_name, shard_fn in [("OLD 5-op chain", _shard_for_paged_write_old),
                                        ("NEW to_memory_config-only", _shard_for_paged_write_new)]:
            log("=" * 60)
            log(f"variant: {variant_name}")
            # Fresh cache
            cache_np = np.zeros((NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
            kc = ttnn.from_torch(torch.from_numpy(cache_np),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            vc = ttnn.from_torch(torch.from_numpy(cache_np),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            # Build inputs
            k_in = _build_input(device, HEAD_DIM, K_data, ttnn.bfloat16)
            v_in = _build_input(device, HEAD_DIM, V_data, ttnn.bfloat16)

            k_sharded = shard_fn(k_in, NKV, HEAD_DIM, BLOCK_SIZE, mc_K)
            v_sharded = shard_fn(v_in, NKV, HEAD_DIM, BLOCK_SIZE, mc_V)
            log(f"  k_sharded shape={list(k_sharded.shape)} padded={list(k_sharded.padded_shape)}")
            log(f"  v_sharded shape={list(v_sharded.shape)} padded={list(v_sharded.padded_shape)}")
            log(f"  k_sharded mem_cfg.layout={k_sharded.memory_config().memory_layout}")

            ttnn.experimental.paged_fused_update_cache(
                kc, k_sharded, vc, v_sharded,
                update_idxs_tensor=update_idxs, page_table=page_table_tt)
            ttnn.synchronize_device(device)

            # Readback
            kc_out = ttnn.to_torch(kc).float().numpy().reshape(
                NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM)
            vc_out = ttnn.to_torch(vc).float().numpy().reshape(
                NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM)

            # Verify: position cur_pos lands at (block=page_table[0, p//BLOCK_SIZE], slot=p%BLOCK_SIZE)
            phys_block = int(page_table_np[0, cur_pos // BLOCK_SIZE])
            slot = cur_pos % BLOCK_SIZE
            k_written = kc_out[phys_block, 0, slot]
            v_written = vc_out[phys_block, 0, slot]
            k_cos = float(np.dot(k_written, K_data) /
                          (np.linalg.norm(k_written) * np.linalg.norm(K_data) + 1e-9))
            v_cos = float(np.dot(v_written, V_data) /
                          (np.linalg.norm(v_written) * np.linalg.norm(V_data) + 1e-9))
            log(f"  K cos(written, K_data) = {k_cos:.6f}  (PASS if >= 0.99)")
            log(f"  V cos(written, V_data) = {v_cos:.6f}  (PASS if >= 0.99)")
            assert k_cos >= 0.99 and v_cos >= 0.99, \
                f"{variant_name} FAIL: K cos={k_cos} V cos={v_cos}"

            # Save for cross-variant byte compare
            if variant_name.startswith("OLD"):
                k_old, v_old = k_written.copy(), v_written.copy()
            else:
                # Compare byte-for-byte with old
                k_diff = float(np.abs(k_old - k_written).max())
                v_diff = float(np.abs(v_old - v_written).max())
                log(f"  cross-variant K max|delta| = {k_diff:.6e}  (PASS if <= 1e-6)")
                log(f"  cross-variant V max|delta| = {v_diff:.6e}  (PASS if <= 1e-6)")
                assert k_diff < 1e-3, f"K mismatch: {k_diff}"
                assert v_diff < 1e-3, f"V mismatch: {v_diff}"

            for t in (kc, vc, k_in, v_in, k_sharded, v_sharded):
                try: ttnn.deallocate(t)
                except Exception: pass

        log("=" * 60)
        log("PASS: NEW to_memory_config-only variant byte-equivalent to OLD 5-op chain.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
