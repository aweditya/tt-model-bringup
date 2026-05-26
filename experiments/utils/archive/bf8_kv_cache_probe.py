#!/usr/bin/env python3
"""Probe: does ttnn.bfloat8_b KV cache change drift behavior vs bf16 cache?

Doc constraints (verified 2026-05-14):
  - paged_update_cache device op accepts cache dtype in {FLOAT32, BFLOAT16,
    BFLOAT8_B, BFLOAT4_B} (paged_update_cache_device_operation.cpp:43-45);
    input dtype in {FLOAT32, BFLOAT16} (line 186-187).
  - sdpa_decode device op accepts Q/K/V in {BFLOAT16, BFLOAT8_B, BFLOAT4_B}
    (sdpa_decode_device_operation.cpp:38-44).
  - Production reference: models/demos/llama3_70b_galaxy/tests/unit_tests/
    test_qwen_ops.py:434 uses cache_dtype=ttnn.bfloat8_b.

So end-to-end bf8 KV cache works: bf16 in -> bf8 in cache (paged_update_cache
truncates) -> bf8 K/V to SDPA decode (native support).

This probe compares:
  V1: bf16 cache (current production)
  V2: bf8  cache (proposed change)
Both feed bf16 Q to paged_scaled_dot_product_attention_decode for 10 steps,
report per-step cosine vs numpy fp32 oracle, max|Δ|, and latency.

Three outcomes are all valuable:
  A. V2 cos materially worse than V1 -> bf8 is a downgrade, keep bf16.
  B. V2 cos similar to or better than V1 -> bf8 saves 2x cache memory at
     no quality cost (enables larger MAX_POS), worth shipping.
  C. V2 cos materially better than V1 -> different rounding helps drift,
     definitely ship.

Run on qb1, device 2 (devices 0/1/3 may be held by other agents/server):
    ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.utils.bf8_kv_cache_probe'
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
import traceback

import numpy as np
import torch
import ttnn

# Match production GQA ratio; small MAX_POS for fast turnaround.
N_Q = 24
N_KV = 4
HEAD_DIM = 256
BLOCK_SIZE = 64
MAX_POS = 64
N_STEPS = 10
TILE_HEIGHT = 32
NUM_USERS = 1


def numpy_attn(q, k_full, v_full, cur_pos):
    """fp32 reference attention. q:[N_Q,HD], k_full/v_full:[N_KV,MAX_POS,HD]."""
    n_rep = N_Q // N_KV
    k_rep = np.repeat(k_full, n_rep, axis=0)
    v_rep = np.repeat(v_full, n_rep, axis=0)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = (q[:, None, :] * k_rep).sum(axis=-1) * scale
    scores[:, cur_pos + 1:] = -1e9
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    out = (weights[:, :, None] * v_rep).sum(axis=1)
    return out.astype(np.float32)


def shard_for_paged_write(k_slot, v_slot, device):
    """paged_update_cache input must be HEIGHT_SHARDED bf16 tile-padded.
    Input dtype is bf16 regardless of cache dtype (cache write path truncates)."""
    def shard(arr):
        t = torch.from_numpy(arr).reshape(1, 1, N_KV, HEAD_DIM).to(torch.bfloat16)
        padded = torch.zeros((1, 1, TILE_HEIGHT, HEAD_DIM), dtype=torch.bfloat16)
        padded[:, :, :N_KV, :] = t
        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid,
                                                      row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                     ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                     ttnn.BufferType.L1, shard_spec)
        return ttnn.from_torch(padded, dtype=ttnn.bfloat16, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=mem_cfg)
    return shard(k_slot), shard(v_slot)


def run_test(device, cache_dtype, label, q_full, k_full, v_full):
    """Run N_STEPS write+read iters with the given cache dtype. Returns list of
    (cur_pos, cos_vs_numpy, max_abs_delta, write_ms, sdpa_ms, attn_np)."""
    max_num_blocks = MAX_POS // BLOCK_SIZE
    paged_zero = np.zeros((max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM),
                            dtype=np.float32)
    cache_k = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=cache_dtype,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache_v = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=cache_dtype,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)

    page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

    results = []
    for cur_pos in range(N_STEPS):
        k_slot = k_full[:, cur_pos, :]
        v_slot = v_full[:, cur_pos, :]

        # WRITE
        try:
            k_sharded, v_sharded = shard_for_paged_write(k_slot, v_slot, device)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                            device=device,
                                            layout=ttnn.ROW_MAJOR_LAYOUT)
            t0 = time.time()
            ttnn.experimental.paged_update_cache(cache_k, k_sharded,
                                                   update_idxs_tensor=cur_pos_tt,
                                                   page_table=page_table_tt)
            ttnn.experimental.paged_update_cache(cache_v, v_sharded,
                                                   update_idxs_tensor=cur_pos_tt,
                                                   page_table=page_table_tt)
            ttnn.synchronize_device(device)
            write_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            print(f"[{label}] WRITE FAIL at cur_pos={cur_pos}: {e}")
            traceback.print_exc()
            return None

        # READ via paged SDPA
        try:
            q_np = q_full[cur_pos]
            q_tt = ttnn.from_torch(torch.from_numpy(q_np).reshape(1, 1, N_Q, HEAD_DIM),
                                    dtype=ttnn.bfloat16, device=device,
                                    layout=ttnn.TILE_LAYOUT)
            cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                            device=device,
                                            layout=ttnn.ROW_MAJOR_LAYOUT)
            t0 = time.time()
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, cache_k, cache_v, page_table_tt,
                cur_pos_tensor=cur_pos_tt)
            ttnn.synchronize_device(device)
            sdpa_ms = (time.time() - t0) * 1000.0
            attn_np = ttnn.to_torch(attn).float().cpu().numpy().reshape(N_Q, HEAD_DIM)
        except Exception as e:
            print(f"[{label}] SDPA FAIL at cur_pos={cur_pos}: {e}")
            traceback.print_exc()
            return None

        # numpy oracle (fp32 K/V/Q)
        ref = numpy_attn(q_np, k_full, v_full, cur_pos)
        cos = float((ref.flatten() @ attn_np.flatten()) /
                      (np.linalg.norm(ref) * np.linalg.norm(attn_np) + 1e-12))
        maxd = float(np.max(np.abs(ref - attn_np)))
        results.append((cur_pos, cos, maxd, write_ms, sdpa_ms, attn_np))

    return results


def main():
    device = ttnn.open_device(device_id=2)
    try:
        rng = np.random.default_rng(0xBF8CAC)
        k_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)
        v_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)
        q_full = (rng.standard_normal((MAX_POS, N_Q, HEAD_DIM)) * 0.1).astype(np.float32)

        print("=" * 70)
        print("Probe: bf8 vs bf16 paged KV cache (paged SDPA decode)")
        print(f"  N_Q={N_Q} N_KV={N_KV} HEAD_DIM={HEAD_DIM} BLOCK={BLOCK_SIZE}")
        print(f"  MAX_POS={MAX_POS} N_STEPS={N_STEPS}")
        print("=" * 70)

        # === V1: bf16 cache (baseline)
        print("\n[V1] bf16 cache (baseline)")
        v1 = run_test(device, ttnn.bfloat16, "bf16", q_full, k_full, v_full)
        if v1:
            for r in v1:
                print(f"  cur_pos={r[0]:2d}  cos={r[1]:.6f}  max|Δ|={r[2]:.4e}"
                      f"  write={r[3]:.2f}ms  sdpa={r[4]:.2f}ms")

        # === V2: bf8 cache (proposed)
        print("\n[V2] bf8_b cache (proposed)")
        v2 = run_test(device, ttnn.bfloat8_b, "bf8", q_full, k_full, v_full)
        if v2:
            for r in v2:
                print(f"  cur_pos={r[0]:2d}  cos={r[1]:.6f}  max|Δ|={r[2]:.4e}"
                      f"  write={r[3]:.2f}ms  sdpa={r[4]:.2f}ms")

        # === Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if v1 and v2:
            avg_cos_v1 = sum(r[1] for r in v1) / len(v1)
            avg_maxd_v1 = sum(r[2] for r in v1) / len(v1)
            avg_write_v1 = sum(r[3] for r in v1) / len(v1)
            avg_sdpa_v1 = sum(r[4] for r in v1) / len(v1)
            print(f"V1 (bf16): avg_cos={avg_cos_v1:.6f}  avg_max|Δ|={avg_maxd_v1:.4e}  "
                  f"avg_write={avg_write_v1:.2f}ms  avg_sdpa={avg_sdpa_v1:.2f}ms")
            avg_cos_v2 = sum(r[1] for r in v2) / len(v2)
            avg_maxd_v2 = sum(r[2] for r in v2) / len(v2)
            avg_write_v2 = sum(r[3] for r in v2) / len(v2)
            avg_sdpa_v2 = sum(r[4] for r in v2) / len(v2)
            print(f"V2 (bf8):  avg_cos={avg_cos_v2:.6f}  avg_max|Δ|={avg_maxd_v2:.4e}  "
                  f"avg_write={avg_write_v2:.2f}ms  avg_sdpa={avg_sdpa_v2:.2f}ms")
            delta_cos = avg_cos_v2 - avg_cos_v1
            print(f"\nDELTA cos (V2 - V1): {delta_cos:+.6f}")

            # Also compute V1 vs V2 direct cosine to capture rounding-pattern diff
            v1_concat = np.concatenate([r[5].flatten() for r in v1])
            v2_concat = np.concatenate([r[5].flatten() for r in v2])
            v1v2_cos = float(v1_concat @ v2_concat /
                              (np.linalg.norm(v1_concat) * np.linalg.norm(v2_concat) + 1e-12))
            v1v2_maxd = float(np.max(np.abs(v1_concat - v2_concat)))
            print(f"V1<->V2 cos (rounding profile diff): {v1v2_cos:.6f}  max|Δ|={v1v2_maxd:.4e}")

            if abs(delta_cos) < 1e-4:
                print("\nVerdict: NEUTRAL — |Δcos| < 1e-4 (within bf16 step-to-step variance).")
                print("  -> bf8 saves 2x cache memory at effectively no quality cost.")
                print("  -> 2x cache memory savings enable 2x larger MAX_POS at same RAM budget.")
            elif delta_cos < -1e-4:
                print("\nVerdict: WORSE — bf8 has materially lower cosine vs numpy.")
                print("  -> Keep bf16.")
            else:
                print("\nVerdict: BETTER — bf8 has materially higher cosine vs numpy.")
                print("  -> Definitely ship; different rounding profile helps.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
