#!/usr/bin/env python3
"""CB2 isolation — batched paged_update_cache WRITE (the last fiddly primitive).

paged SDPA READ at B>1 is validated (cb_paged_sdpa_batch_isolation.py). The
WRITE is the remaining unknown: paged_update_cache wants input
[1, B, 1[32], head_dim] HEIGHT_SHARDED on B cores, with update_idxs [B] +
page_table [B, blocks_per_seq]. The production B=1 path shards on 1 core
(server_tp.py:444-450); this generalizes to B cores.

Test: write B slots' K vectors at distinct positions into a paged cache,
then read the cache back and verify each slot's K landed at the right
(physical_block, slot_in_block) per its page table + position.

Self-contained, single device.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_paged_update_cache_batch_isolation.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NKV = 1
HEAD_DIM = 256
BLOCK_SIZE = 32
TILE_HEIGHT = 32
NUM_BLOCKS_TOTAL = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import ttnn
    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        rng = np.random.default_rng(args.seed)
        B = args.batch
        cur_pos = np.array([5, 33, 12, 40], dtype=np.int32)[:B]
        log(f"B={B}  write positions={cur_pos.tolist()}")

        blocks_per_seq = NUM_BLOCKS_TOTAL // B
        page_table_np = np.stack([
            np.arange(b * blocks_per_seq, (b + 1) * blocks_per_seq, dtype=np.int32)
            for b in range(B)], axis=0)

        # New K vectors to write: one [head_dim] per slot.
        K_new = rng.normal(0, 1.0, (B, HEAD_DIM)).astype(np.float32)

        # Cache starts zeroed: [total_blocks, NKV, BLOCK_SIZE, head_dim].
        cache_np = np.zeros((NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        cache = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)

        # Input [1, B, TILE_HEIGHT, head_dim], real row 0 per slot = K_new[b],
        # rest padding. Height-sharded on B cores (one slot per core).
        inp_np = np.zeros((1, B, TILE_HEIGHT, HEAD_DIM), dtype=np.float32)
        inp_np[0, :, 0, :] = K_new
        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(B, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                    ttnn.ShardOrientation.ROW_MAJOR)
        write_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inp = ttnn.from_torch(torch.from_numpy(inp_np), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=write_mem_cfg)

        update_idxs = ttnn.from_torch(torch.from_numpy(cur_pos.reshape(B)),
                                      dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=device)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=device)

        log("calling batched paged_update_cache…")
        ttnn.experimental.paged_update_cache(
            cache, inp, update_idxs_tensor=update_idxs, page_table=page_table_tt)
        ttnn.synchronize_device(device)

        cache_out = ttnn.to_torch(cache).float().numpy().reshape(
            NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM)

        log("=== verify each slot's K landed at the right (block, slot) ===")
        any_fail = False
        for b in range(B):
            p = int(cur_pos[b])
            phys_block = int(page_table_np[b, p // BLOCK_SIZE])
            slot_in_block = p % BLOCK_SIZE
            written = cache_out[phys_block, 0, slot_in_block]
            cosv = float(np.dot(written, K_new[b]) /
                         (np.linalg.norm(written) * np.linalg.norm(K_new[b]) + 1e-9))
            ok = cosv >= 0.99
            any_fail = any_fail or not ok
            log(f"  slot {b}: pos={p:3d} block={phys_block} slot_in_block={slot_in_block} "
                f"cos(written,K_new)={cosv:.6f}  {'OK' if ok else 'FAIL'}")

        for t in (cache, inp, update_idxs, page_table_tt):
            try: ttnn.deallocate(t)
            except Exception: pass
        if any_fail:
            log("FAIL: batched paged_update_cache write did not land K correctly.")
            raise SystemExit(1)
        log(f"PASS: batched paged_update_cache write correct at B={B}. The CB "
            f"attn step's KV write is de-risked — use num_cores_to_corerangeset(B).")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
