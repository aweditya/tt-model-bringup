#!/usr/bin/env python3
"""Probe: write+read paged KV cache across block boundaries.

The bug under investigation: server.handle_generate_paged (paged forward)
produces coherent output for ~80 tokens, then degenerates to garbage.
Single-cache `handle_generate` (non-paged forward) produces coherent
output through 200+ tokens for the same prompt. So the paged write+read
loop is corrupting cache contents past some position.

This probe mirrors the EXACT write+read pattern from
`gated_attn_step_ondevice_paged` in experiments/91f_qwen36_27b_full_ondevice.py
— without weights, RoPE, gating. Just:

  for cur_pos in 0..127:
      k_pos = unique_marker(cur_pos)   # easily-identifiable values
      paged_update_cache(cache_k, k_pos_sharded, idx=cur_pos)
  for cur_pos in 0..127:
      out = paged_scaled_dot_product_attention_decode(
                q=delta(cur_pos), cache_k, cache_v, cur_pos=cur_pos)

then compare to a numpy oracle running the SAME math on a non-paged cache.

If the paged path produces correct attention output through cur_pos=63
but wrong at cur_pos>=64 (block 1), we've localized the bug to a block-
boundary issue. If both blocks are written/read correctly, the bug is
elsewhere in the kernel (possibly silent corruption in the SDPA reader
when len > block_size).

Run on qb1:
    ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.utils.paged_write_read_iter_probe'
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import ttnn

N_Q = 24
N_KV = 4
HEAD_DIM = 256
BLOCK_SIZE = 64
MAX_POS = 256          # 4 blocks
TILE_HEIGHT = 32
NUM_USERS = 1


def numpy_attn(q, k_full, v_full, cur_pos):
    """Reference attention. q:[N_Q,HD], k_full/v_full:[N_KV,MAX_POS,HD]."""
    n_rep = N_Q // N_KV
    k_rep = np.repeat(k_full, n_rep, axis=0)  # [N_Q, MAX_POS, HD]
    v_rep = np.repeat(v_full, n_rep, axis=0)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    # scores [N_Q, MAX_POS]
    scores = (q[:, None, :] * k_rep).sum(axis=-1) * scale
    # Causal mask: only positions 0..cur_pos contribute
    scores[:, cur_pos + 1:] = -1e9
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    out = (weights[:, :, None] * v_rep).sum(axis=1)  # [N_Q, HD]
    return out.astype(np.float32)


def upload_sharded_kv_for_paged_write(k_per_head, v_per_head, device):
    """Mirror of `shard_for_paged_write` in 91f_qwen36_27b_full_ondevice.py."""
    # Input: [N_KV, HEAD_DIM] fp32 -> [1, 1, 32, HEAD_DIM] bf16 sharded
    def shard(arr):
        # Reshape to [1, 1, N_KV, HEAD_DIM] then pad to [1, 1, TILE_HEIGHT, HEAD_DIM]
        t = torch.from_numpy(arr).reshape(1, 1, N_KV, HEAD_DIM).to(torch.bfloat16)
        padded = torch.zeros((1, 1, TILE_HEIGHT, HEAD_DIM), dtype=torch.bfloat16)
        padded[:, :, :N_KV, :] = t
        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                     ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                     ttnn.BufferType.L1, shard_spec)
        return ttnn.from_torch(padded, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=mem_cfg)
    return shard(k_per_head), shard(v_per_head)


def main():
    device = ttnn.open_device(device_id=1)  # device 1, server is on 0
    try:
        max_num_blocks = MAX_POS // BLOCK_SIZE
        rng = np.random.default_rng(42)

        # Pre-build ALL K/V vectors for positions 0..MAX_POS-1 (small magnitude
        # like a real attention with normalized Q/K). Then we write them one at
        # a time via paged_update_cache, then read everything via paged SDPA.
        k_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)
        v_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)

        # Allocate paged cache
        paged_zero = np.zeros((max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        cache_k = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cache_v = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Page table identity
        page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                          dtype=ttnn.int32, device=device,
                                          layout=ttnn.ROW_MAJOR_LAYOUT)

        # WRITE phase: write each position one at a time
        print(f"[write] writing {MAX_POS} positions one at a time via paged_update_cache")
        for cur_pos in range(MAX_POS):
            k_slot = k_full[:, cur_pos, :]      # [N_KV, HEAD_DIM]
            v_slot = v_full[:, cur_pos, :]
            k_sharded, v_sharded = upload_sharded_kv_for_paged_write(k_slot, v_slot, device)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                           device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.paged_update_cache(cache_k, k_sharded,
                                                   update_idxs_tensor=cur_pos_tt,
                                                   page_table=page_table_tt)
            ttnn.experimental.paged_update_cache(cache_v, v_sharded,
                                                   update_idxs_tensor=cur_pos_tt,
                                                   page_table=page_table_tt)
        ttnn.synchronize_device(device)

        # Read back the cache, view as non-paged [N_KV, MAX_POS, HEAD_DIM] for inspection.
        # Paged layout is [max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM];
        # the page_table is identity so virtual block i == physical block i.
        # Logical position p -> (block=p//BLOCK_SIZE, slot=p%BLOCK_SIZE).
        cache_k_np = ttnn.to_torch(cache_k).float().cpu().numpy()  # [B, N_KV, BS, HD]
        cache_v_np = ttnn.to_torch(cache_v).float().cpu().numpy()

        # Compare per-position what we WROTE vs what's in the cache.
        # Build a "reconstructed" non-paged view from the cache
        reconstructed_k = cache_k_np.transpose(1, 0, 2, 3).reshape(N_KV, max_num_blocks * BLOCK_SIZE, HEAD_DIM)
        reconstructed_v = cache_v_np.transpose(1, 0, 2, 3).reshape(N_KV, max_num_blocks * BLOCK_SIZE, HEAD_DIM)
        # k_full is fp32 but stored as bf16. cast to bf16 then back to fp32 for comparison
        k_full_bf16 = torch.from_numpy(k_full).to(torch.bfloat16).float().numpy()
        v_full_bf16 = torch.from_numpy(v_full).to(torch.bfloat16).float().numpy()
        print("\n[check] per-position max|Δ| between written K and what's in cache:")
        max_dev_per_pos = []
        for pos in range(MAX_POS):
            d_k = np.max(np.abs(reconstructed_k[:, pos, :] - k_full_bf16[:, pos, :]))
            d_v = np.max(np.abs(reconstructed_v[:, pos, :] - v_full_bf16[:, pos, :]))
            max_dev_per_pos.append((d_k, d_v))
        worst_positions = []
        for pos, (dk, dv) in enumerate(max_dev_per_pos):
            if dk > 1e-3 or dv > 1e-3:
                worst_positions.append((pos, dk, dv))
        if worst_positions:
            print(f"  positions with >1e-3 K or V deviation: {len(worst_positions)}")
            for pos, dk, dv in worst_positions[:20]:
                print(f"    pos={pos:3d}  max|ΔK|={dk:.2e}  max|ΔV|={dv:.2e}")
        else:
            # Show worst few anyway
            print(f"  all positions within 1e-3 of bf16-rounded source (good)")
            top_worst = sorted(enumerate(max_dev_per_pos), key=lambda x: -max(x[1]))[:8]
            for pos, (dk, dv) in top_worst:
                print(f"    pos={pos:3d}  max|ΔK|={dk:.2e}  max|ΔV|={dv:.2e}")

        # READ phase: at each cur_pos, run paged SDPA and compare to numpy oracle.
        print("\n[read] running paged SDPA at varying cur_pos, comparing to numpy oracle:")
        # Build a fresh Q each step too
        q_full = (rng.standard_normal((MAX_POS, N_Q, HEAD_DIM)) * 0.1).astype(np.float32)
        # check at positions that probe block boundaries
        check_positions = [0, 1, 31, 32, 33, 62, 63, 64, 65, 95, 96, 127, 128, 191, 200, 255]
        for cur_pos in check_positions:
            if cur_pos >= MAX_POS:
                continue
            q_np = q_full[cur_pos].astype(np.float32)
            # bf16 round for fair comparison
            q_bf16 = torch.from_numpy(q_np).to(torch.bfloat16).float().numpy()
            ref = numpy_attn(q_bf16, k_full_bf16, v_full_bf16, cur_pos)

            q_tt = ttnn.from_torch(torch.from_numpy(q_np).reshape(1, 1, N_Q, HEAD_DIM),
                                    dtype=ttnn.bfloat16, device=device,
                                    layout=ttnn.TILE_LAYOUT)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                           device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, cache_k, cache_v, page_table_tt, cur_pos_tensor=cur_pos_tt)
            ttnn.synchronize_device(device)
            attn_np = ttnn.to_torch(attn).float().cpu().numpy().reshape(N_Q, HEAD_DIM)
            cos = float((ref.flatten() @ attn_np.flatten()) /
                          (np.linalg.norm(ref) * np.linalg.norm(attn_np) + 1e-12))
            maxd = float(np.max(np.abs(ref - attn_np)))
            print(f"  cur_pos={cur_pos:3d}  cos={cos:.6f}  max|Δ|={maxd:.2e}")

        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
