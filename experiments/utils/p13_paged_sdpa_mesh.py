#!/usr/bin/env python3
"""
P13 — paged_scaled_dot_product_attention_decode on (1,4) mesh (qb2).

P12.1 validated paged_update_cache works on mesh with the right setup.
P13 tests the READ side — does paged SDPA decode work on mesh when each
chip has NKV_PER_CHIP=1? Per `feedback_p1_sdpa_decode_breaks_on_mesh.md`,
the non-paged variant FAILS on mesh with tree-reduction error.

If paged SDPA works on mesh: ship it (faster, fused).
If paged SDPA also fails: keep manual SDPA but read from paged cache via
  reshape. Still works for trace because the cache layout doesn't require
  cur_pos baked.

Probe:
  - Build paged cache via the validated P12.1 recipe
  - Populate slot 0..K via paged_update_cache
  - Build Q [1, 1, NQ_PER_CHIP, HEAD_DIM] bf16 (per 91f:494-499 shape)
  - Call paged_scaled_dot_product_attention_decode
  - Verify finite output + plausible magnitude

Wall: ~1 min (mesh open + small probe).
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78, flush=True)
    print("P13: paged_scaled_dot_product_attention_decode on (1,4) mesh", flush=True)
    print("=" * 78, flush=True)

    import ttnn
    import torch
    import numpy as np

    # Doc
    print("\n[doc] paged_scaled_dot_product_attention_decode docstring:", flush=True)
    print(getattr(ttnn.transformer.paged_scaled_dot_product_attention_decode,
                   "__doc__", None) or "(no docstring)", flush=True)

    print("\n[setup] fabric + (1,4) mesh…", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips", flush=True)

    try:
        # Production-equivalent shapes
        MAX_POS = 128
        HEAD_DIM = 256
        N_Q = 24
        N_KV = 4
        NQ_PER_CHIP = N_Q // 4  # 6
        NKV_PER_CHIP = N_KV // 4  # 1
        BLOCK_SIZE = 32
        NUM_BLOCKS = MAX_POS // BLOCK_SIZE  # 4
        B = 1
        TILE_HEIGHT = 32
        NUM_USERS = 1

        # === Build cache & populate slot 0..K via paged_update_cache (validated P12.1) ===
        cache_k_np = np.zeros((NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        cache_v_np = np.zeros((NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        cache_k = ttnn.from_torch(torch.from_numpy(cache_k_np), dtype=ttnn.bfloat16,
                                    device=mesh, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        cache_v = ttnn.from_torch(torch.from_numpy(cache_v_np), dtype=ttnn.bfloat16,
                                    device=mesh, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))

        # page_table (identity)
        page_table_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(B, NUM_BLOCKS)
        page_table = ttnn.from_torch(torch.from_numpy(page_table_np),
                                       device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # Compute shard helpers
        compute_grid = mesh.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                       ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

        def shard_for_paged_write(input_np_134):
            """input_np_134: shape [1, 1, N_KV, HEAD_DIM] → sharded per-chip."""
            t = ttnn.from_torch(torch.from_numpy(input_np_134), dtype=ttnn.bfloat16,
                                  device=mesh, layout=ttnn.TILE_LAYOUT,
                                  mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
            t = ttnn.pad(t, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]], value=0.0)
            return ttnn.to_memory_config(t, sharded_mem_cfg)

        # Populate slots 0, 1, 2, 3, 4 (cur_pos 0..4)
        rng = np.random.default_rng(0)
        print(f"\n[populate] writing slots 0..4 via paged_update_cache…", flush=True)
        for cp in range(5):
            k_np = rng.standard_normal((1, 1, N_KV, HEAD_DIM)).astype(np.float32) * 0.3
            v_np = rng.standard_normal((1, 1, N_KV, HEAD_DIM)).astype(np.float32) * 0.3
            k_sharded = shard_for_paged_write(k_np)
            v_sharded = shard_for_paged_write(v_np)
            idxs = ttnn.from_torch(torch.tensor([cp], dtype=torch.int32),
                                     device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.experimental.paged_update_cache(cache_k, k_sharded,
                                                   update_idxs_tensor=idxs,
                                                   page_table=page_table)
            ttnn.experimental.paged_update_cache(cache_v, v_sharded,
                                                   update_idxs_tensor=idxs,
                                                   page_table=page_table)
        ttnn.synchronize_device(mesh)
        print(f"  ✓ 5 slots populated", flush=True)

        # === Build Q for paged SDPA ===
        # Per 91f:494: q_for_sdpa = ttnn.reshape(q, [1, 1, N_Q, HEAD_DIM]) typecast bf16
        # On mesh: per-chip [1, 1, NQ_PER_CHIP=6, HEAD_DIM] sharded along NQ axis
        q_np = rng.standard_normal((1, 1, N_Q, HEAD_DIM)).astype(np.float32) * 0.1
        q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                                 device=mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        print(f"  ✓ Q built per-chip [1, 1, {NQ_PER_CHIP}, {HEAD_DIM}] sharded N_Q axis", flush=True)

        # cur_pos_tensor for SDPA — same shape as update_idxs
        cur_pos = 5  # decoding step 5; SDPA attends positions 0..5
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                       device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # === The critical call ===
        print(f"\n[call] paged_scaled_dot_product_attention_decode(Q, K_cache, V_cache, page_table, cur_pos_tensor)", flush=True)
        try:
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, cache_k, cache_v, page_table,
                cur_pos_tensor=cur_pos_tt,
            )
            ttnn.synchronize_device(mesh)
            print(f"  ✓ paged SDPA call accepted", flush=True)

            out_np = ttnn.to_torch(
                attn_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)
            ).float().numpy()
            finite = bool(np.isfinite(out_np).all())
            mag = float(np.abs(out_np).max())
            print(f"  output shape: {out_np.shape}  finite={finite}  max|.|={mag:.4f}", flush=True)

            if finite and 1e-3 < mag < 1e3:
                print(f"\n  ✓ P13 PASSES — paged SDPA works on mesh!", flush=True)
                print(f"    Refactor server_tp.py: switch from manual SDPA to paged SDPA", flush=True)
            else:
                print(f"\n  ⚠ output looks suspicious — finite={finite}, max|.|={mag}", flush=True)

        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:600]}", flush=True)
            print(f"\n  → fallback: keep manual SDPA reading from paged cache", flush=True)
            print(f"    (cache reshape to flat [MAX_POS, HEAD_DIM] then transpose+matmul)", flush=True)
            print(f"    Refactor server_tp.py: paged_update_cache + manual SDPA hybrid", flush=True)

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
