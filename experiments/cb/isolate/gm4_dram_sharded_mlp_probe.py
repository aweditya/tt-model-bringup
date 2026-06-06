"""Isolation probe — Round 10 — DRAM-sharded MLP weight matmul.

Round 8 DRAM-traffic profile (research/gemma4_perf_qb2_2026-06-05/reports/
round8_matmul_bw_breakdown.txt) showed Matmul = 99.5% of all per-forward
PM-BANDWIDTH-bound time (43.487 ms / forward of 43.695 ms total). The MLP
per-chip `[32, 3840] x [3840, 3840]` triplet alone is 71% of that (30.9 ms
PM-BW × 144 calls / forward = gate+up+down per layer × 48 layers).

Round 8 cut the BYTES per weight read in half via `bfloat8_b` weights and
got -1.86% traced (-0.87 ms/tok). That confirmed the access PATTERN was
the dominant constant — the matmul still pays a full DRAM round-trip per
weight read, just on half the bytes.

This round attacks the access PATTERN by switching the MLP weight memory
config from default `INTERLEAVED DRAM` to `WIDTH_SHARDED DRAM` across all
8 Blackhole P150 DRAM banks, paired with the dedicated
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` matmul program
config that parallelises weight reads across DRAM banks via NoC.

Production precedent in tt-metal:
  - `tt-metal/models/tt_transformers/tt/model_config.py:3067` —
    `create_dram_sharded_mem_config(k, n)` is the canonical helper; pads N
    to a multiple of TILE × num_banks and lays out as WIDTH_SHARDED DRAM.
  - `tt-metal/models/demos/llama3_70b_galaxy/tt/model_config.py:2312` —
    same helper, used at every MLP/Q/K/V/O/lm_head matmul callsite under
    `USE_PREFETCHER` (the Llama Galaxy ships this for Wormhole TG too).
  - `tt-metal/tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py`
    — full reference test with the [M=32, K=8192, N={1280, 4096, ...}]
    decode shapes and the program-config block-size math (lines 50-184).

Per-chip MLP shape on Gemma 4 12B / NCHIPS=4:
  - HIDDEN=3840, INTERMEDIATE=15360
  - gate_proj, up_proj: per-chip [3840, 3840] (sharded along output)
  - down_proj: per-chip [3840, 3840] (sharded along input)
  - K=3840 = TILE(32) × 120; N=3840 = TILE(32) × 8 banks × 15 cols/bank
    → already aligned to TILE × num_banks; no N padding needed.

Gate: cos(default, dram_sharded) >= 0.99999 on all 3 MLP shapes,
max|delta| <= 0.5 (forgiving for [3840] reduction at HIFI4). If all PASS,
land in `server_gemma4_unified_ttnn.py` under env-gate
`TT_GM4_DRAM_PREFETCH=1`.

Forks `experiments/cb/isolate/gm4_bfp8_weights_probe.py` for the probe
scaffold (same Round 8 family, same harness contract, same 3-shape MLP
target). Forks the matmul program-config + memory-config math from
`tt-metal/tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py:50-184`.
"""
from __future__ import annotations

import math
import sys
import time

import torch
import ttnn

HIDDEN = 3840
TILE = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _cfg():
    """Match production HIFI4 + fp32_dest_acc_en (Round 7 NEGATIVE finding
    docblock: HiFi2 doesn't help at B=1, DRAM-bound)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _dram_weight_mem_cfg(mesh, K, N):
    """WIDTH_SHARDED DRAM memory config for a [K, N] weight matrix,
    distributed across all DRAM banks (P150: 8 banks × 1).

    Forks `create_dram_sharded_mem_config` from tt_transformers/model_config.py
    (lines 3067-3076). N is padded to a multiple of TILE × num_banks; if
    N is already aligned (our case: 3840 = 32 × 8 × 15) no pad is added.
    """
    dram_grid_size = mesh.dram_grid_size()
    num_banks = dram_grid_size.x
    assert dram_grid_size.y == 1, f"Expected dram_grid_size.y=1, got {dram_grid_size.y}"
    padded_N = int(math.ceil(N / (TILE * num_banks)) * (TILE * num_banks))
    # Build the bank core grid: dram_grid is on dedicated DRAM cores, given as
    # a CoreRangeSet that the runtime maps to the actual DRAM channels.
    dram_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
        )
    })
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (K, padded_N // num_banks),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def _activation_l1_width_sharded(mesh, M, K, num_cores):
    """L1 WIDTH_SHARDED activation memory config — matches the DRAM-sharded
    matmul's in0 contract (`test_matmul_dram_sharded.py:135-142`).

    in0_block_w = K / num_cores / TILE; shard_shape = [M, in0_block_w * TILE].
    """
    in0_block_w = K // num_cores // TILE
    in0_shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_cores - 1, 0),  # grid (num_cores, 1)
        )
    })
    in0_shard_spec = ttnn.ShardSpec(
        in0_shard_grid,
        [M, in0_block_w * TILE],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        in0_shard_spec,
    )


def _dram_sharded_program_config(M, K, N, num_cores, num_banks):
    """Matmul program config for DRAM-sharded weights.

    Forks `test_matmul_dram_sharded.py:140-145`:
      in0_block_w = K / num_cores / TILE / 4  (sub-block factor)
      per_core_M  = M / TILE
      per_core_N  = N / num_cores / TILE
    """
    out_block_h = M // TILE
    out_block_w = N // num_cores // TILE
    in0_block_w_unscaled = K // num_cores // TILE
    in0_block_w = max(1, in0_block_w_unscaled // 4)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )


def _test_shape(name, M, K, N, mesh, mapper, composer, num_cores=8, n_iters=20):
    """Run one matmul shape both ways: default interleaved bf16 weight vs
    DRAM-sharded bf16 weight + L1-sharded activation + dedicated program
    config. Compare numerics; measure warm per-call dispatch time as a
    quick smoke (real perf delta comes from the v04 validator on the full
    forward).
    """
    log(f"--- {name}: [{M},{K}] x [{K},{N}] = [{M},{N}], num_cores={num_cores} ---")
    torch.manual_seed(0)
    x_t = torch.randn(M, K, dtype=torch.float32) * 0.5
    w_t = torch.randn(K, N, dtype=torch.float32) * 0.05
    y_ref = (x_t @ w_t)  # fp32 ground truth

    # --- Baseline: interleaved DRAM weight, interleaved activation ---
    x_baseline = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=mapper,
    )
    w_baseline = ttnn.from_torch(
        w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=mapper,
    )
    # Warm
    y0 = ttnn.matmul(x_baseline, w_baseline, compute_kernel_config=_cfg())
    ttnn.deallocate(y0)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n_iters):
        y_baseline_tt = ttnn.matmul(x_baseline, w_baseline, compute_kernel_config=_cfg())
        ttnn.deallocate(y_baseline_tt)
    ttnn.synchronize_device(mesh)
    base_ms = (time.time() - t0) * 1000 / n_iters
    # One more for numerics
    y_baseline_tt = ttnn.matmul(x_baseline, w_baseline, compute_kernel_config=_cfg())
    y_baseline = ttnn.to_torch(y_baseline_tt, mesh_composer=composer)[:M].reshape(M, N)
    ttnn.deallocate(y_baseline_tt)
    ttnn.deallocate(x_baseline)
    ttnn.deallocate(w_baseline)

    # --- DRAM-sharded: WIDTH_SHARDED DRAM weight + L1 width-sharded act ---
    dram_grid_size = mesh.dram_grid_size()
    num_banks = dram_grid_size.x
    w_mem_cfg = _dram_weight_mem_cfg(mesh, K, N)
    x_mem_cfg = _activation_l1_width_sharded(mesh, M, K, num_cores)
    prog_cfg = _dram_sharded_program_config(M, K, N, num_cores, num_banks)

    w_dram = ttnn.from_torch(
        w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=mapper, memory_config=w_mem_cfg,
    )
    x_dram = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=mapper, memory_config=x_mem_cfg,
    )
    # Output is L1 WIDTH_SHARDED (matches the dram-sharded matmul out spec)
    out_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
    )
    # Warm
    try:
        y1 = ttnn.matmul(
            x_dram, w_dram,
            program_config=prog_cfg,
            memory_config=out_mem_cfg,
            compute_kernel_config=_cfg(),
        )
        ttnn.deallocate(y1)
        ttnn.synchronize_device(mesh)
    except Exception as e:
        log(f"  ERROR on warm-up: {e}")
        ttnn.deallocate(w_dram)
        ttnn.deallocate(x_dram)
        return False
    t0 = time.time()
    for _ in range(n_iters):
        y_dram_tt = ttnn.matmul(
            x_dram, w_dram,
            program_config=prog_cfg,
            memory_config=out_mem_cfg,
            compute_kernel_config=_cfg(),
        )
        ttnn.deallocate(y_dram_tt)
    ttnn.synchronize_device(mesh)
    dram_ms = (time.time() - t0) * 1000 / n_iters
    # Numerics
    y_dram_tt = ttnn.matmul(
        x_dram, w_dram,
        program_config=prog_cfg,
        memory_config=out_mem_cfg,
        compute_kernel_config=_cfg(),
    )
    # interleaved-to-host for compare
    y_dram_i = ttnn.sharded_to_interleaved(y_dram_tt, ttnn.DRAM_MEMORY_CONFIG)
    y_dram = ttnn.to_torch(y_dram_i, mesh_composer=composer)[:M].reshape(M, N)
    ttnn.deallocate(y_dram_tt)
    ttnn.deallocate(y_dram_i)
    ttnn.deallocate(x_dram)
    ttnn.deallocate(w_dram)

    # Numerics
    cos_base_ref = _cos(y_baseline, y_ref)
    cos_dram_ref = _cos(y_dram, y_ref)
    cos_pair = _cos(y_baseline, y_dram)
    max_abs = float((y_baseline - y_dram).abs().max())
    mad = float((y_baseline - y_dram).abs().mean())

    log(f"  cos(baseline, fp32_ref)    = {cos_base_ref:.7f}")
    log(f"  cos(dram_shd, fp32_ref)    = {cos_dram_ref:.7f}  (delta: {cos_dram_ref - cos_base_ref:+.7f})")
    log(f"  cos(baseline, dram_shd)    = {cos_pair:.7f}")
    log(f"  max|baseline - dram_shd|   = {max_abs:.6f}")
    log(f"  mean|baseline - dram_shd|  = {mad:.6f}")
    log(f"  per-call ms baseline       = {base_ms:.3f}")
    log(f"  per-call ms dram-sharded   = {dram_ms:.3f}  (delta {(dram_ms-base_ms)/base_ms*100:+.1f}%)")

    ok = cos_pair >= 0.99999 and max_abs < 0.5
    log(f"  PASS: {'yes' if ok else 'NO'}")
    return ok


def main(state=None):
    if state is None:
        log("ERR: probe requires a harness mesh; run via gm4 dev harness.")
        return 1
    mesh = state.mesh
    mapper = ttnn.ReplicateTensorToMesh(mesh)
    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

    log(f"dram_grid_size: {mesh.dram_grid_size()}  (Blackhole P150: expect (8, 1))")
    log(f"compute_grid:  {mesh.compute_with_storage_grid_size()}")

    results = {}
    # MLP triplet — per-chip [3840, 3840] after TP=4 sharding.
    # num_cores=8: matches the activation L1 shard count for K=3840
    # (K/num_cores/TILE = 3840/8/32 = 15 tile-cols per core).
    results["gate_proj"] = _test_shape(
        "gate_proj", M=32, K=3840, N=3840,
        mesh=mesh, mapper=mapper, composer=composer, num_cores=8,
    )
    results["up_proj"] = _test_shape(
        "up_proj", M=32, K=3840, N=3840,
        mesh=mesh, mapper=mapper, composer=composer, num_cores=8,
    )
    results["down_proj"] = _test_shape(
        "down_proj", M=32, K=3840, N=3840,
        mesh=mesh, mapper=mapper, composer=composer, num_cores=8,
    )

    log("")
    log("=" * 70)
    all_pass = all(results.values())
    for k, v in results.items():
        log(f"  {k:24s}: {'PASS' if v else 'FAIL'}")
    log(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    log("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
