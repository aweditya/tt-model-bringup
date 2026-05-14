#!/usr/bin/env python3
"""Probe: can paged KV cache be stored in fp32, and does paged SDPA decode
read it?  If yes, what's the latency + noise improvement vs bf16?

Hypothesis (per feedback_paged_greedy_drift.md): bf16 KV cache noise
compounds across decode steps; storing fp32 should eliminate the noise floor.

Doc constraints (verified 2026-05-13):
  - paged_update_cache device op (ttnn/cpp/.../paged_update_cache_device_operation.cpp)
    accepts cache dtype in {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B}; input dtype
    in {FLOAT32, BFLOAT16}.
  - sdpa_decode device op (ttnn/cpp/.../sdpa_decode_device_operation.cpp) REJECTS
    fp32 input tensors — only {BFLOAT16, BFLOAT8_B, BFLOAT4_B}.

So end-to-end fp32 KV cache requires either:
  (a) typecast cache to bf16 at read time (kills the precision benefit);
  (b) custom SDPA decode kernel supporting fp32 (deferred);
  (c) some hybrid (fp32 cache writes, periodic re-quantize) — not in scope.

This probe tests (a) and the baseline bf16 path side-by-side and reports:
  1. Does paged_update_cache with fp32 cache + fp32 input work?
  2. Does paged SDPA decode work on the fp32 cache (expect: NO)?
  3. If we typecast fp32 cache to bf16 in-line, does cosine vs numpy improve
     vs the bf16-only baseline over 10 simulated decode steps?
  4. Latency penalty?

Run on qb1 device 1:
    ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.utils.fp32_kv_cache_probe'
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
import traceback

import numpy as np
import torch
import ttnn

# Small shape that still matches the production GQA ratio.
N_Q = 24
N_KV = 4
HEAD_DIM = 256
BLOCK_SIZE = 64
MAX_POS = 64           # one block — keeps the test fast
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


def shard_for_paged_write(k_slot, v_slot, device, ttnn_dtype):
    """Mirror of shard_for_paged_write in 91f. ttnn_dtype is what we send to
    paged_update_cache. cache must accept this dtype."""
    def shard(arr):
        torch_dt = torch.float32 if ttnn_dtype == ttnn.float32 else torch.bfloat16
        t = torch.from_numpy(arr).reshape(1, 1, N_KV, HEAD_DIM).to(torch_dt)
        padded = torch.zeros((1, 1, TILE_HEIGHT, HEAD_DIM), dtype=torch_dt)
        padded[:, :, :N_KV, :] = t
        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid,
                                                      row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                     ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                     ttnn.BufferType.L1, shard_spec)
        return ttnn.from_torch(padded, dtype=ttnn_dtype, device=device,
                                layout=ttnn.TILE_LAYOUT, memory_config=mem_cfg)
    return shard(k_slot), shard(v_slot)


def run_test(device, cache_dtype, label, q_full, k_full, v_full,
             typecast_for_sdpa=False):
    """Run N_STEPS write+read iters with the given cache dtype. Returns list of
    (cur_pos, cos_vs_numpy, max_abs_delta, write_ms, sdpa_ms)."""
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

    # Input dtype must match cache for paged_update_cache (or at least pair).
    # cache_dtype FLOAT32 → input FLOAT32; cache_dtype BFLOAT16 → input BFLOAT16.
    input_dtype = ttnn.float32 if cache_dtype == ttnn.float32 else ttnn.bfloat16

    results = []
    for cur_pos in range(N_STEPS):
        k_slot = k_full[:, cur_pos, :]
        v_slot = v_full[:, cur_pos, :]

        # WRITE
        try:
            k_sharded, v_sharded = shard_for_paged_write(k_slot, v_slot, device,
                                                          input_dtype)
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
            if typecast_for_sdpa:
                # Path (a): fp32 cache → bf16 just before SDPA.
                cache_k_bf16 = ttnn.typecast(cache_k, ttnn.bfloat16)
                cache_v_bf16 = ttnn.typecast(cache_v, ttnn.bfloat16)
                sdpa_k, sdpa_v = cache_k_bf16, cache_v_bf16
            else:
                sdpa_k, sdpa_v = cache_k, cache_v

            t0 = time.time()
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, sdpa_k, sdpa_v, page_table_tt,
                cur_pos_tensor=cur_pos_tt)
            ttnn.synchronize_device(device)
            sdpa_ms = (time.time() - t0) * 1000.0
            attn_np = ttnn.to_torch(attn).float().cpu().numpy().reshape(N_Q, HEAD_DIM)
        except Exception as e:
            print(f"[{label}] SDPA FAIL at cur_pos={cur_pos}: {e}")
            traceback.print_exc()
            return None

        # numpy oracle (fp32 K/V, fp32 Q)
        ref = numpy_attn(q_np, k_full, v_full, cur_pos)
        cos = float((ref.flatten() @ attn_np.flatten()) /
                      (np.linalg.norm(ref) * np.linalg.norm(attn_np) + 1e-12))
        maxd = float(np.max(np.abs(ref - attn_np)))
        results.append((cur_pos, cos, maxd, write_ms, sdpa_ms))

    return results


def main():
    device = ttnn.open_device(device_id=1)
    try:
        rng = np.random.default_rng(0xC0FFEE)
        # Generate ALL K/V/Q used over N_STEPS at fp32 precision.
        k_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)
        v_full = (rng.standard_normal((N_KV, MAX_POS, HEAD_DIM)) * 0.1).astype(np.float32)
        q_full = (rng.standard_normal((MAX_POS, N_Q, HEAD_DIM)) * 0.1).astype(np.float32)

        print("=" * 70)
        print("Probe: fp32 vs bf16 paged KV cache, paged SDPA decode")
        print(f"  N_Q={N_Q} N_KV={N_KV} HEAD_DIM={HEAD_DIM} BLOCK={BLOCK_SIZE}")
        print(f"  MAX_POS={MAX_POS} N_STEPS={N_STEPS}")
        print("=" * 70)

        # === Variant 1: bf16 cache (baseline)
        print("\n[V1] bf16 cache (baseline)")
        v1 = run_test(device, ttnn.bfloat16, "bf16", q_full, k_full, v_full,
                      typecast_for_sdpa=False)
        if v1:
            for r in v1:
                print(f"  cur_pos={r[0]:2d}  cos={r[1]:.6f}  max|Δ|={r[2]:.4e}"
                      f"  write={r[3]:.2f}ms  sdpa={r[4]:.2f}ms")

        # === Variant 2: fp32 cache, raw read (expected to fail at SDPA)
        print("\n[V2] fp32 cache, raw read (expect SDPA validation to reject)")
        v2 = run_test(device, ttnn.float32, "fp32_raw", q_full, k_full, v_full,
                      typecast_for_sdpa=False)
        if v2 is None:
            print("  V2 failed — confirms SDPA rejects fp32 cache.")
        elif v2:
            for r in v2:
                print(f"  cur_pos={r[0]:2d}  cos={r[1]:.6f}  max|Δ|={r[2]:.4e}"
                      f"  write={r[3]:.2f}ms  sdpa={r[4]:.2f}ms")

        # === Variant 3: fp32 cache, typecast-to-bf16 at read time
        print("\n[V3] fp32 cache + ttnn.typecast(bf16) at SDPA read")
        v3 = run_test(device, ttnn.float32, "fp32_typecast", q_full, k_full, v_full,
                      typecast_for_sdpa=True)
        if v3:
            for r in v3:
                print(f"  cur_pos={r[0]:2d}  cos={r[1]:.6f}  max|Δ|={r[2]:.4e}"
                      f"  write={r[3]:.2f}ms  sdpa={r[4]:.2f}ms")

        # === Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if v1:
            avg_cos_v1 = sum(r[1] for r in v1) / len(v1)
            avg_write_v1 = sum(r[3] for r in v1) / len(v1)
            avg_sdpa_v1 = sum(r[4] for r in v1) / len(v1)
            print(f"V1 (bf16 baseline):    avg_cos={avg_cos_v1:.6f}  "
                  f"avg_write={avg_write_v1:.2f}ms  avg_sdpa={avg_sdpa_v1:.2f}ms")
        if v2:
            avg_cos_v2 = sum(r[1] for r in v2) / len(v2)
            avg_write_v2 = sum(r[3] for r in v2) / len(v2)
            avg_sdpa_v2 = sum(r[4] for r in v2) / len(v2)
            print(f"V2 (fp32 cache raw):   avg_cos={avg_cos_v2:.6f}  "
                  f"avg_write={avg_write_v2:.2f}ms  avg_sdpa={avg_sdpa_v2:.2f}ms")
        else:
            print("V2 (fp32 cache raw):   FAILED (expected — SDPA rejects fp32)")
        if v3:
            avg_cos_v3 = sum(r[1] for r in v3) / len(v3)
            avg_write_v3 = sum(r[3] for r in v3) / len(v3)
            avg_sdpa_v3 = sum(r[4] for r in v3) / len(v3)
            print(f"V3 (fp32 + typecast):  avg_cos={avg_cos_v3:.6f}  "
                  f"avg_write={avg_write_v3:.2f}ms  avg_sdpa={avg_sdpa_v3:.2f}ms")
            if v1:
                delta_cos = avg_cos_v3 - avg_cos_v1
                print(f"\nDELTA cos (V3 - V1): {delta_cos:+.6f}")
                if delta_cos > 1e-4:
                    print("  fp32 cache + typecast IMPROVES precision (worth integration test)")
                else:
                    print("  No measurable precision benefit — fp32 cache is a dead end")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
