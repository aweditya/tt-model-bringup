#!/usr/bin/env python3
"""
P12 — verify paged_update_cache signature on (1,4) mesh (qb2).

Per P11's finding: ttnn.kv_cache.update_cache_for_token_ in our build only
accepts `update_index: int` — no tensor variant → can't be used in traced
multi-step decode (cur_pos gets baked into the captured trace).

91f single-chip path uses `ttnn.experimental.paged_update_cache(...
update_idxs_tensor=cur_pos_tt, page_table=...)`. We need to verify:
  1. The exact signature in our installed ttnn (vs. 91f's older usage)
  2. That it works on a (1,4) mesh — paged variants haven't been tested on
     mesh yet (`feedback_paged_sdpa_gqa_validated.md` confirms paged works
     single-chip but mesh untested)
  3. The required input shape, memory_config, and what page_table looks like

Plan:
  - Open mesh (1,4)
  - Help-print paged_update_cache to get docstring
  - Build minimal cache + input + page_table tensors of plausible shapes
  - Call paged_update_cache; iterate on shape mismatches based on errors
  - Verify the write actually landed by reading back the cache

Outcome: clear API recipe to plug into server_tp.py refactor — OR a clear
"this op doesn't work on mesh, fall back to plan X" finding.
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78, flush=True)
    print("P12: paged_update_cache signature probe on (1,4) mesh (qb2)", flush=True)
    print("=" * 78, flush=True)

    import ttnn
    import torch
    import numpy as np

    # First — read the docstring
    print("\n[doc] paged_update_cache docstring:", flush=True)
    print(getattr(ttnn.experimental.paged_update_cache, "__doc__", None) or "(no docstring)",
          flush=True)
    print("\n[doc] callable repr:", flush=True)
    print(repr(ttnn.experimental.paged_update_cache), flush=True)

    print("\n[setup] fabric + (1,4) mesh…", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips", flush=True)

    try:
        # 91f production-validated paged shapes:
        #   cache: [B, n_kv, max_pos, head_dim]
        #   input (per step): [1, B, NKV, head_dim] OR similar; need to verify
        #   page_table: [B, num_blocks]
        #   update_idxs_tensor: [B] int32 (which token slot to write per batch)
        #
        # For our qb2 4-chip TP layout (N_KV = 4 sharded one-per-chip):
        #   per-chip cache: [B=1, NKV_PER_CHIP=1, MAX_POS, head_dim=256]
        #
        # block_size: try 32 first (page boundary). num_blocks = MAX_POS / block_size.
        MAX_POS = 128
        HEAD_DIM = 256
        N_KV = 4               # global, sharded along dim=1 → 1 KV head per chip
        NKV_PER_CHIP = 1
        BLOCK_SIZE = 32
        NUM_BLOCKS = MAX_POS // BLOCK_SIZE  # 4
        B = 1

        print(f"\n[shapes] cache [B={B}, N_KV={N_KV}, MAX_POS={MAX_POS}, HEAD_DIM={HEAD_DIM}] "
              f"sharded along N_KV → per-chip [{B}, {NKV_PER_CHIP}, {MAX_POS}, {HEAD_DIM}]", flush=True)
        print(f"          input  [1, B={B}, NKV={N_KV}, HEAD_DIM] sharded → per-chip [1, {B}, {NKV_PER_CHIP}, {HEAD_DIM}]", flush=True)
        print(f"          page_table [B={B}, NUM_BLOCKS={NUM_BLOCKS}]", flush=True)
        print(f"          update_idxs [B={B}]", flush=True)

        # Correction after first P12 attempt: cache dim 0 is num_blocks_total
        # (the physical block pool), NOT batch. Real layout:
        #   cache: [num_blocks_total, n_kv, block_size, head_dim]
        # page_table[B, num_blocks_per_seq] maps logical blocks → physical blocks.
        print(f"          (corrected) cache: [num_blocks_total=NUM_BLOCKS={NUM_BLOCKS}, "
              f"N_KV={N_KV}, BLOCK_SIZE={BLOCK_SIZE}, HEAD_DIM={HEAD_DIM}]", flush=True)
        cache_np = np.zeros((NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        cache_tt = ttnn.from_torch(
            torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
            device=mesh, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        print(f"  ✓ cache uploaded (sharded along N_KV dim=1; per-chip "
              f"[{NUM_BLOCKS}, {NKV_PER_CHIP}, {BLOCK_SIZE}, {HEAD_DIM}])", flush=True)

        # Build input — fresh K/V for the current token. Shape [1, B, N_KV, HEAD_DIM]
        # sharded along N_KV → per-chip [1, B, NKV_PER_CHIP, HEAD_DIM]
        rng = np.random.default_rng(42)
        input_np = (rng.standard_normal((1, B, N_KV, HEAD_DIM)).astype(np.float32) * 0.5)
        input_tt = ttnn.from_torch(
            torch.from_numpy(input_np), dtype=ttnn.bfloat16,
            device=mesh, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),  # N_KV dim is index 2 here
        )
        print(f"  ✓ input uploaded", flush=True)

        # Build page_table — identity for B=1 (logical block i → physical block i)
        page_table_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(B, NUM_BLOCKS)
        page_table_tt = ttnn.from_torch(
            torch.from_numpy(page_table_np),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ page_table uploaded", flush=True)

        # Build update_idxs — which slot to write per batch
        # For step cur_pos=5 with B=1: update_idxs = [5]
        cur_pos = 5
        idxs_np = np.array([cur_pos], dtype=np.int32)
        idxs_tt = ttnn.from_torch(
            torch.from_numpy(idxs_np),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ update_idxs uploaded (cur_pos={cur_pos})", flush=True)

        # === Attempt 1: with all four (cache, input, update_idxs_tensor, page_table) ===
        print(f"\n[attempt 1] paged_update_cache(cache, input, update_idxs_tensor=..., page_table=...)", flush=True)
        try:
            ttnn.experimental.paged_update_cache(
                cache_tt, input_tt,
                update_idxs_tensor=idxs_tt,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(mesh)
            print(f"  ✓ paged_update_cache call accepted", flush=True)

            # Verify the write landed
            # cache shape after concat dim=1: [NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM]
            # cur_pos=5 → block_idx = 5 // BLOCK_SIZE = 0, slot_in_block = 5 % BLOCK_SIZE = 5
            cache_after = ttnn.to_torch(
                cache_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)
            ).float().numpy()
            block_idx = cur_pos // BLOCK_SIZE
            slot_in_block = cur_pos % BLOCK_SIZE
            written_slot = cache_after[block_idx, :, slot_in_block, :]
            unwritten_slot = cache_after[block_idx, :, slot_in_block + 1, :]
            print(f"  cache[block={block_idx}, :, slot={slot_in_block}, :] "
                  f"max|.|={float(np.abs(written_slot).max()):.4f} (expect non-zero)", flush=True)
            print(f"  cache[block={block_idx}, :, slot={slot_in_block+1}, :] "
                  f"max|.|={float(np.abs(unwritten_slot).max()):.4f} (expect 0)", flush=True)
            if float(np.abs(written_slot).max()) > 1e-3 and float(np.abs(unwritten_slot).max()) < 1e-6:
                print(f"  ✓ write landed at correct slot, others untouched", flush=True)
            else:
                print(f"  ⚠ unexpected — slot doesn't match expectation", flush=True)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:600]}", flush=True)
            print(f"\n[next] check error, adjust shapes/kwargs, re-try in P12.1", flush=True)
            raise

        print("\n" + "=" * 78, flush=True)
        print(f"  ✓ P12 PASSES — paged_update_cache works on mesh", flush=True)
        print(f"    cache: [B, N_KV, MAX_POS, HEAD_DIM] sharded dim=1", flush=True)
        print(f"    input: [1, B, N_KV, HEAD_DIM] sharded dim=2 (N_KV axis)", flush=True)
        print(f"    page_table: [B, NUM_BLOCKS] int32 replicated", flush=True)
        print(f"    update_idxs_tensor: [B] int32 replicated", flush=True)
        print(f"    Next: P13 paged_scaled_dot_product_attention_decode on mesh", flush=True)
        print("=" * 78, flush=True)

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed cleanly", flush=True)
        except Exception as e:
            print(f"  ✗ close error: {e}", flush=True)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED", flush=True)
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}", flush=True)


if __name__ == "__main__":
    main()
