#!/usr/bin/env python3
"""
P18 — paged_scaled_dot_product_attention_decode on (1,4) mesh at REAL Qwen3.6-27B
gated_attn_step_tp production shapes.

Prior P13 attempted this with mock shapes (HEAD_DIM=256, MAX_POS=128, BLOCK_SIZE=32)
WITHOUT passing program_config or compute_kernel_config. P1's tree-reduction error
was caused by the kernel auto-allocating ~110 user cores per head when no
program_config constrained the core grid.

P18 strategy (informed by Galaxy reference llama_attention.py:523-533 +
model_config.py:1181-1207):
  - Pass an explicit ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        q_chunk_size=0, k_chunk_size=0,            # 0 = match Galaxy "paged" path
        max_cores_per_head_batch=16,                # default but explicit
        exp_approx_mode=False)
    that LIMITS cores. Galaxy uses (8, 6) = 48 cores.
  - Pass ttnn.WormholeComputeKernelConfig (HiFi2, no fp32_dest, no packer_l1_acc).
  - Try multiple grid sizes; for each grid that COMPILES, verify cosine vs
    a per-chip manual SDPA on the same K/V/Q.
  - Try MAX_POS ∈ {256, 1024, 8192} once a working config is found.

Production constants (verified from .cache config.json on qb2):
  HEAD_DIM = 256, N_Q = 24, N_KV = 4, partial_rotary_factor = 0.25
  NQ_PER_CHIP = 6, NKV_PER_CHIP = 1
  BLOCK_SIZE = 32, NUM_BLOCKS = MAX_POS / 32

Pass criteria:
  - paged SDPA call ACCEPTED (no exception)
  - per-chip cosine vs manual SDPA ≥ 0.999
  - finite, non-trivial magnitude output

If pass → measure 30-iter median latency vs manual SDPA path → save JSON in
.cache/p18_mesh_paged_sdpa/results.json.

Mesh recovery: if wedged, `ssh qb2 'tt-smi -r 0,1,2,3'`.
"""
import os
import sys
import time
import json
import traceback

sys.stdout.reconfigure(line_buffering=True)


# Production shapes (verified from Qwen3.6-27B config.json)
HEAD_DIM = 256
N_Q = 24
N_KV = 4
NCHIPS = 4
NQ_PER_CHIP = N_Q // NCHIPS  # 6
NKV_PER_CHIP = N_KV // NCHIPS  # 1
BLOCK_SIZE = 32
TILE_HEIGHT = 32
NUM_USERS = 1
B = 1


def _cosine(a, b):
    import numpy as np
    a = a.astype("float64").flatten()
    b = b.astype("float64").flatten()
    n = float((a * a).sum() ** 0.5 * (b * b).sum() ** 0.5)
    if n < 1e-30:
        return 0.0
    return float((a * b).sum() / n)


def _build_sharded_paged_cache(ttnn, torch, np, mesh, max_pos, rng):
    """Allocate paged cache, page table, and populate cur_pos∈[0, K) with random K/V.

    Returns:
      cache_k, cache_v, page_table, k_per_pos_np, v_per_pos_np, num_populated
    """
    num_blocks = max_pos // BLOCK_SIZE
    cache_k_np = np.zeros((num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    cache_v_np = np.zeros((num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    cache_k = ttnn.from_torch(
        torch.from_numpy(cache_k_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    cache_v = ttnn.from_torch(
        torch.from_numpy(cache_v_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )

    # page_table: identity for batch=1
    page_table_np = np.arange(num_blocks, dtype=np.int32).reshape(B, num_blocks)
    page_table = ttnn.from_torch(
        torch.from_numpy(page_table_np),
        device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # Sharded write helper
    compute_grid = mesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def shard_for_paged_write(arr_1_1_NKV_HD):
        t = ttnn.from_torch(
            torch.from_numpy(arr_1_1_NKV_HD), dtype=ttnn.bfloat16,
            device=mesh, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        t = ttnn.pad(t, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]], value=0.0)
        return ttnn.to_memory_config(t, sharded_mem_cfg)

    # Populate cur_pos 0..K with random K/V; remember full per-pos K/V (all 4 heads, before sharding)
    NUM_POPULATED = min(8, max_pos - 1)  # leave at least 1 slot uninitialized; cur_pos=NUM_POPULATED
    k_per_pos_np = np.zeros((NUM_POPULATED, N_KV, HEAD_DIM), dtype=np.float32)
    v_per_pos_np = np.zeros((NUM_POPULATED, N_KV, HEAD_DIM), dtype=np.float32)
    for cp in range(NUM_POPULATED):
        k_np = (rng.standard_normal((1, 1, N_KV, HEAD_DIM)) * 0.3).astype(np.float32)
        v_np = (rng.standard_normal((1, 1, N_KV, HEAD_DIM)) * 0.3).astype(np.float32)
        k_per_pos_np[cp] = k_np[0, 0]
        v_per_pos_np[cp] = v_np[0, 0]
        k_sharded = shard_for_paged_write(k_np)
        v_sharded = shard_for_paged_write(v_np)
        idxs = ttnn.from_torch(
            torch.tensor([cp], dtype=torch.int32),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.experimental.paged_update_cache(cache_k, k_sharded,
                                              update_idxs_tensor=idxs,
                                              page_table=page_table)
        ttnn.experimental.paged_update_cache(cache_v, v_sharded,
                                              update_idxs_tensor=idxs,
                                              page_table=page_table)
    ttnn.synchronize_device(mesh)
    return cache_k, cache_v, page_table, k_per_pos_np, v_per_pos_np, NUM_POPULATED


def _manual_sdpa_per_chip_np(q_np, k_per_pos_np, v_per_pos_np, cur_pos, max_pos):
    """Compute manual SDPA on the host (fp32) PER CHIP, to compare against
    paged SDPA output.

    q_np: [N_Q, HEAD_DIM]  (all 24 Q heads; we split per-chip below)
    k_per_pos_np: [num_populated, N_KV, HEAD_DIM]
    v_per_pos_np: [num_populated, N_KV, HEAD_DIM]

    Returns per-chip [NCHIPS, NQ_PER_CHIP, HEAD_DIM] attention output.
    """
    import numpy as np
    num_populated = k_per_pos_np.shape[0]
    # Replicate K/V via GQA: each KV head serves NQ_PER_CHIP=6 Q heads. With N_KV=4 and N_Q=24,
    # ratio = 6 so each Q head shares with 6 others. The per-chip mapping is: chip i takes
    # Q heads [6i..6i+5] and KV head [i] (NKV_PER_CHIP=1).
    out_per_chip = np.zeros((NCHIPS, NQ_PER_CHIP, HEAD_DIM), dtype=np.float32)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # Pad to max_pos. Positions > cur_pos have score = 0 (cache zeroes); softmax mask isn't
    # applied here either since we want bit-equal comparison to the paged op which uses
    # cur_pos_tensor for masking.
    K_full = np.zeros((max_pos, N_KV, HEAD_DIM), dtype=np.float32)
    V_full = np.zeros((max_pos, N_KV, HEAD_DIM), dtype=np.float32)
    K_full[:num_populated] = k_per_pos_np
    V_full[:num_populated] = v_per_pos_np

    for chip in range(NCHIPS):
        q_chip = q_np[chip * NQ_PER_CHIP:(chip + 1) * NQ_PER_CHIP]  # [6, HD]
        # KV head for this chip = chip index
        K_chip = K_full[:, chip, :]  # [max_pos, HD]
        V_chip = V_full[:, chip, :]  # [max_pos, HD]
        scores = (q_chip @ K_chip.T) * scale  # [6, max_pos]
        # Mask positions > cur_pos to -inf (paged SDPA does this internally)
        scores[:, cur_pos + 1:] = -1e9
        # softmax
        scores -= scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores)
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)  # [6, max_pos]
        out_per_chip[chip] = attn @ V_chip  # [6, HD]
    return out_per_chip


def _try_paged_sdpa(ttnn, torch, np, mesh, q_tt, cache_k, cache_v, page_table,
                      cur_pos_tt, program_config, compute_kernel_config, label):
    """One attempt. Returns (success: bool, err: str|None, out_np: ndarray|None)."""
    try:
        kwargs = dict(
            cur_pos_tensor=cur_pos_tt,
            page_table_tensor=page_table,
        )
        if program_config is not None:
            kwargs["program_config"] = program_config
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, cache_k, cache_v, **kwargs
        )
        ttnn.synchronize_device(mesh)
        # Output shape varies by config; concat-mesh on the head axis
        # Q shape is [1, 1, NQ_PER_CHIP, HEAD_DIM] sharded dim=2; expect output same
        out_np = ttnn.to_torch(
            attn_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)
        ).float().numpy()
        return True, None, out_np
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:800]}", None


def _bench_latency(ttnn, mesh, fn, iters=30, warmup=5):
    """Measure median latency (ms) of fn, syncing before+after."""
    import numpy as np
    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(mesh)
    times = []
    for _ in range(iters):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.min(times))


def main():
    print("=" * 78, flush=True)
    print("P18: paged SDPA decode on (1,4) mesh @ Qwen3.6-27B production shapes", flush=True)
    print("=" * 78, flush=True)
    print(f"  HEAD_DIM={HEAD_DIM}  N_Q={N_Q}  N_KV={N_KV}", flush=True)
    print(f"  NQ_PER_CHIP={NQ_PER_CHIP}  NKV_PER_CHIP={NKV_PER_CHIP}", flush=True)
    print(f"  BLOCK_SIZE={BLOCK_SIZE}  NCHIPS={NCHIPS}", flush=True)

    import ttnn
    import torch
    import numpy as np

    results = {
        "shapes": {
            "HEAD_DIM": HEAD_DIM, "N_Q": N_Q, "N_KV": N_KV,
            "NQ_PER_CHIP": NQ_PER_CHIP, "NKV_PER_CHIP": NKV_PER_CHIP,
            "BLOCK_SIZE": BLOCK_SIZE, "NCHIPS": NCHIPS,
        },
        "attempts": [],
        "max_pos_scaling": [],
        "latency_ms": None,
    }

    print("\n[setup] fabric + (1,4) mesh…", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips opened", flush=True)
    print(f"  compute_with_storage_grid_size={mesh.compute_with_storage_grid_size()}", flush=True)

    try:
        rng = np.random.default_rng(42)
        MAX_POS = 256
        cache_k, cache_v, page_table, k_per_pos_np, v_per_pos_np, num_populated = \
            _build_sharded_paged_cache(ttnn, torch, np, mesh, MAX_POS, rng)
        print(f"\n[populate] {num_populated} slots populated at MAX_POS={MAX_POS}", flush=True)

        # Build Q (all 24 heads, then shard dim=2 across 4 chips → 6 Q heads/chip)
        q_np = (rng.standard_normal((1, 1, N_Q, HEAD_DIM)) * 0.1).astype(np.float32)
        q_tt = ttnn.from_torch(
            torch.from_numpy(q_np), dtype=ttnn.bfloat16,
            device=mesh, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )

        # cur_pos
        cur_pos_val = num_populated - 1
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos_val], dtype=torch.int32),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # Host reference
        gold_per_chip = _manual_sdpa_per_chip_np(
            q_np[0, 0], k_per_pos_np, v_per_pos_np, cur_pos_val, MAX_POS,
        )  # [4, 6, HD]

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # === Attempts ===
        attempts = []
        # A: no program_config (reproduces P13 if it still fails)
        attempts.append(("A_no_progcfg", None, None))
        # B: progcfg with grid (4, 4) = 16 cores, q/k chunk 0 (paged-decode default)
        attempts.append((
            "B_grid_4x4_chunks_0",
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
                q_chunk_size=0, k_chunk_size=0,
                exp_approx_mode=False,
            ),
            compute_kernel_config,
        ))
        # C: Galaxy-style progcfg (8, 6) = 48 cores
        attempts.append((
            "C_grid_8x6_chunks_0",
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 6),
                q_chunk_size=0, k_chunk_size=0,
                exp_approx_mode=False,
            ),
            compute_kernel_config,
        ))
        # D: smaller (2, 2) = 4 cores (force max_cores per head very low)
        attempts.append((
            "D_grid_2x2_chunks_0",
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
                q_chunk_size=0, k_chunk_size=0,
                exp_approx_mode=False,
            ),
            compute_kernel_config,
        ))
        # E: explicit max_cores_per_head_batch=8
        attempts.append((
            "E_grid_8x6_max_cores_8",
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 6),
                q_chunk_size=0, k_chunk_size=0,
                exp_approx_mode=False,
                max_cores_per_head_batch=8,
            ),
            compute_kernel_config,
        ))
        # F: smaller progcfg + chunk_size=32 (match block_size)
        attempts.append((
            "F_grid_4x4_chunks_32",
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
                q_chunk_size=32, k_chunk_size=32,
                exp_approx_mode=False,
            ),
            compute_kernel_config,
        ))

        winning_progcfg = None
        winning_compute = None
        for label, progcfg, ckc in attempts:
            print(f"\n[try {label}]", flush=True)
            ok, err, out_np = _try_paged_sdpa(
                ttnn, torch, np, mesh, q_tt, cache_k, cache_v, page_table,
                cur_pos_tt, progcfg, ckc, label,
            )
            attempt_rec = {"label": label, "ok": ok, "err": err}
            if ok:
                # Reshape paged output → per-chip view, compare to gold
                # paged output expected [1, 1, N_Q, HEAD_DIM] composed via ConcatMeshToTensor dim=2.
                # We composed concat dim=2 so out_np shape = [1, 1, NQ_PER_CHIP*4, HEAD_DIM] = [1, 1, 24, 256]
                finite = bool(np.isfinite(out_np).all())
                mag = float(np.abs(out_np).max())
                attempt_rec.update({
                    "out_shape": list(out_np.shape),
                    "finite": finite,
                    "max_abs": mag,
                })
                print(f"    ✓ accepted  shape={out_np.shape}  finite={finite}  max|.|={mag:.4f}", flush=True)
                # Reshape to [NCHIPS, NQ_PER_CHIP, HEAD_DIM]
                if out_np.shape == (1, 1, N_Q, HEAD_DIM):
                    out_view = out_np[0, 0].reshape(NCHIPS, NQ_PER_CHIP, HEAD_DIM)
                    # Cosine per chip + total
                    cos_per_chip = []
                    for c in range(NCHIPS):
                        cos_per_chip.append(_cosine(out_view[c], gold_per_chip[c]))
                    cos_total = _cosine(out_view, gold_per_chip)
                    attempt_rec["cos_per_chip"] = cos_per_chip
                    attempt_rec["cos_total"] = cos_total
                    print(f"    cos_per_chip={['%.5f' % c for c in cos_per_chip]}", flush=True)
                    print(f"    cos_total   = {cos_total:.6f}", flush=True)
                    if cos_total >= 0.999 and winning_progcfg is None:
                        winning_progcfg = progcfg
                        winning_compute = ckc
                        print(f"    → WINNING CONFIG", flush=True)
                else:
                    print(f"    ⚠ unexpected output shape; skipping cosine", flush=True)
            else:
                print(f"    ✗ {err[:200]}", flush=True)
            attempts_rec_short = {k: v for k, v in attempt_rec.items() if k != "err" or v is None}
            if err:
                attempts_rec_short["err_head"] = err[:200]
            results["attempts"].append(attempt_rec)

        # === If we found a winner, benchmark vs manual SDPA ===
        if winning_progcfg is not None:
            print(f"\n[latency] running 30-iter median benchmark with winning config", flush=True)

            def paged_sdpa_call():
                _ = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_tt, cache_k, cache_v,
                    cur_pos_tensor=cur_pos_tt,
                    page_table_tensor=page_table,
                    program_config=winning_progcfg,
                    compute_kernel_config=winning_compute,
                )

            paged_med, paged_min = _bench_latency(ttnn, mesh, paged_sdpa_call, iters=30, warmup=5)

            # Manual SDPA: replicate the gated_attn_step_tp:587-600 path
            # Per-chip cache → reshape → flat MAX_POS → transpose → matmul → softmax → matmul
            num_blocks_cur = MAX_POS // BLOCK_SIZE

            # Build the same Q tensor for the manual path (already sharded as q_tt)
            # But manual SDPA in production expects Q as [NQ_PER_CHIP, HEAD_DIM] (2D) — see line 537.
            # For the latency benchmark we'll use a [NQ_PER_CHIP, HEAD_DIM] sharded Q.
            q2d_np = q_np[0, 0]  # [N_Q, HEAD_DIM]
            q2d_tt = ttnn.from_torch(
                torch.from_numpy(q2d_np), dtype=ttnn.bfloat16,
                device=mesh, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            # Pre-build reshape ops outside the call would alter the graph; here we time the
            # full path identical to server_tp.py:gated_attn_step_tp lines 587-600.
            scale = 1.0 / (HEAD_DIM ** 0.5)

            def manual_sdpa_call():
                kc_3d = ttnn.reshape(cache_k, [num_blocks_cur, BLOCK_SIZE, HEAD_DIM])
                vc_3d = ttnn.reshape(cache_v, [num_blocks_cur, BLOCK_SIZE, HEAD_DIM])
                kc_flat = ttnn.reshape(kc_3d, [MAX_POS, HEAD_DIM])
                vc_flat = ttnn.reshape(vc_3d, [MAX_POS, HEAD_DIM])
                kT = ttnn.transpose(kc_flat, 0, 1)
                scores = ttnn.mul(ttnn.matmul(q2d_tt, kT), scale)
                attn_w = ttnn.softmax(scores, dim=-1)
                _ = ttnn.matmul(attn_w, vc_flat)

            manual_med, manual_min = _bench_latency(ttnn, mesh, manual_sdpa_call, iters=30, warmup=5)
            results["latency_ms"] = {
                "paged_sdpa_median": paged_med,
                "paged_sdpa_min": paged_min,
                "manual_sdpa_median": manual_med,
                "manual_sdpa_min": manual_min,
                "speedup_median": manual_med / paged_med if paged_med > 0 else None,
            }
            print(f"\n  paged SDPA   median {paged_med:.3f} ms  min {paged_min:.3f}", flush=True)
            print(f"  manual SDPA  median {manual_med:.3f} ms  min {manual_min:.3f}", flush=True)
            print(f"  speedup (manual/paged) = {manual_med/paged_med:.2f}×", flush=True)

            # === MAX_POS scaling sweep ===
            print(f"\n[max_pos scaling] re-running paged SDPA at increasing MAX_POS", flush=True)
            for mp in [1024, 4096, 8192]:
                print(f"\n  [MAX_POS={mp}] allocating cache…", flush=True)
                ttnn.deallocate(cache_k)
                ttnn.deallocate(cache_v)
                ttnn.deallocate(page_table)
                try:
                    cache_k, cache_v, page_table, k_pp, v_pp, npop = \
                        _build_sharded_paged_cache(ttnn, torch, np, mesh, mp, rng)
                    cur_pos_tt_mp = ttnn.from_torch(
                        torch.tensor([npop - 1], dtype=torch.int32),
                        device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    )

                    def paged_sdpa_call_mp():
                        _ = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            q_tt, cache_k, cache_v,
                            cur_pos_tensor=cur_pos_tt_mp,
                            page_table_tensor=page_table,
                            program_config=winning_progcfg,
                            compute_kernel_config=winning_compute,
                        )

                    med, mn = _bench_latency(ttnn, mesh, paged_sdpa_call_mp, iters=10, warmup=2)
                    print(f"    ✓ MAX_POS={mp}  median {med:.3f} ms  min {mn:.3f}", flush=True)
                    results["max_pos_scaling"].append({
                        "max_pos": mp, "median_ms": med, "min_ms": mn, "ok": True,
                    })
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    print(f"    ✗ MAX_POS={mp} failed: {err}", flush=True)
                    results["max_pos_scaling"].append({
                        "max_pos": mp, "ok": False, "err": err,
                    })

            results["winner_label"] = next(
                (a["label"] for a in results["attempts"] if a.get("cos_total", 0) >= 0.999),
                None,
            )
        else:
            print(f"\n[no winner] all attempts failed or gave low cosine", flush=True)
            results["winner_label"] = None

    finally:
        # Save results JSON
        out_dir = os.path.join(os.path.expanduser("~/tt-xla"), ".cache", "p18_mesh_paged_sdpa")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "results.json")
        try:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[results] written to {out_path}", flush=True)
        except Exception as e:
            print(f"[results] save error: {e}", flush=True)

        try:
            ttnn.close_mesh_device(mesh)
            print("  ✓ mesh closed", flush=True)
        except Exception as e:
            print(f"  ✗ close error: {e}", flush=True)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric disabled", flush=True)
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
