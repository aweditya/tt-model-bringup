#!/usr/bin/env python3
"""
P23 / Branch C'6 probe: DRAM-sharded MLP matmuls on qb1 (single P150).

Goal: Lift isolated MLP from baseline ~78% DRAM-peak (interleaved bf8, see
`feedback_qb1_mlp_at_78pct_peak.md`) toward 90%+ via DRAM-WIDTH-sharded
weights + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.

Production constants (Qwen3.6-27B text MLP, server.py + 91f path):
    HIDDEN       = 5120
    INTERMEDIATE = 17408     (per `feedback_bf8_mlp_weights.md`)
    weight dtype = ttnn.bfloat8_b   (gate/up/down stored as bf8 in prod)
    activation   = ttnn.bfloat16
    op           = ttnn.linear (gate has activation="silu" in prod)

Four variants benched on the SAME random fp32 weights:

    V0 INTERLEAVED   weights INTERLEAVED in DRAM, activation INTERLEAVED in L1
                     (this matches `mlp_step_ondevice` in 91f — production)

    V1 DRAM-SHARD W  weights DRAM WIDTH-sharded across dram_weight_grid,
                     activation interleaved L1, NO program_config
                     (just sharding the weights; matmul auto-pick)

    V2 FULL DRAM     weights DRAM WIDTH-sharded, activation L1 WIDTH-sharded,
                     output L1 WIDTH-sharded, explicit DRAM-Sharded program
                     config per tt-metal recipe (qwen_model_config.py:1913).
                     This is the canonical DRAM-sharded matmul.

    V3 V2 + TUNE     same as V2 but with hand-tuned in0_block_w / per_core_N
                     sweep to look for a small extra win.

For each variant, measure:
    - per-MLP-step latency (ms, median of 30 iter, after 5 warmup)
    - weight bytes read per step
    - effective GB/s = weight_bytes / step_seconds
    - % of 512 GB/s P150 peak

Plus a cosine correctness check (V1/V2/V3 vs V0).

Pass criterion (per task brief):
    V2 or V3 hits ≥ 90% of 512 GB/s (≥ 461 GB/s)
    AND latency improvement ≥ 0.5 ms per MLP step vs V0.

References:
    - models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:1772 create_dram_sharded_mem_config
    - models/demos/llama3_70b_galaxy/tt/qwen_model_config.py:1913 dram_matmul_config
    - tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md
    - tech_reports/LLMs/llms.md lines 601-618, 1523-1562

Run (standalone — qb1 server MUST be stopped first):
    ssh qb1 'bash /home/aditya/tt-xla/experiments/serve/scripts/serve.sh stop'
    ssh qb1 'cd /home/aditya/tt-xla && .venv/bin/python experiments/utils/p23_dram_sharded_mlp_probe.py'
    ssh qb1 'bash /home/aditya/tt-xla/experiments/serve/scripts/serve.sh start'
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# ---- production constants ----
HIDDEN = 5120
INTERMEDIATE = 17408
WEIGHT_DTYPE = ttnn.bfloat8_b   # production
ACT_DTYPE = ttnn.bfloat16       # production
TILE = 32
BYTES_PER_ELEM_BF8 = 1.0625      # bf8: 1 byte mantissa + 1 byte shared exp per 16-elem block = 17/16 = 1.0625
# More accurate: bf8 stores 32 elements per 32+1B = 33B, but model_config.py uses 1 byte/elem as
# the BW accounting convention. We'll use both and report the ratio.
BYTES_PER_ELEM_BF8_CONSERVATIVE = 1.0     # 1 byte / elem (mantissa)
P150_DRAM_PEAK_GBS = 512.0
OUT_DIR = Path("/home/aditya/tt-xla/.cache/p23_dram_sharded_mlp")


def mb(n_bytes):
    return n_bytes / (1024 * 1024)


def mlp_weight_bytes(bytes_per_elem):
    """gate (HIDDEN×INTERMEDIATE) + up (HIDDEN×INTERMEDIATE) + down (INTERMEDIATE×HIDDEN)."""
    elems = HIDDEN * INTERMEDIATE + HIDDEN * INTERMEDIATE + INTERMEDIATE * HIDDEN
    return int(elems * bytes_per_elem)


def median_ms(times):
    return float(np.median(times) * 1000.0)


def dram_grid_for_device(device):
    """CoreRangeSet covering the entire DRAM core grid on this device (per qwen_model_config.py:311)."""
    g = device.dram_grid_size()
    return ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))
    }), g.x * g.y


def make_dram_sharded_mem_config(K, N, dram_grid, dram_cores):
    """WIDTH_SHARDED in DRAM, padded so N divides evenly by (TILE × dram_cores)."""
    padded_N = math.ceil(N / (TILE * dram_cores)) * (TILE * dram_cores)
    shard_shape = (K, padded_N // dram_cores)
    spec = ttnn.ShardSpec(dram_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec), padded_N


def _find_largest_divisor_le(n, cap):
    """Largest divisor of n that is ≤ cap. From DeepSeek V3's config_helpers."""
    best = 1
    for d in range(1, cap + 1):
        if d > n:
            break
        if n % d == 0:
            best = d
    return best


def _in0_block_w(K, num_cores, cap=8):
    """DRAM-sharded matmul `in0_block_w`: largest divisor of (K_tiles/num_cores) that is ≤ cap.

    Per deepseek_v3/utils/config_helpers.py: in0_block_w must EVENLY divide k_tiles,
    and is capped (typically at 8) to keep per-core CB allocations within L1.
    """
    K_tiles = K // TILE
    per_core_K_tiles = K_tiles // num_cores
    return _find_largest_divisor_le(per_core_K_tiles, cap)


def make_dram_progcfg(M, K, N, num_cores, in0_block_w_cap=8):
    """qwen_model_config.py:1913 / deepseek_v3 recipe.

    in0_block_w capped at 8 tiles (≤16KB per CB) to stay in L1.
    """
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_in0_block_w(K, num_cores, cap=in0_block_w_cap),
        per_core_M=math.ceil(M / TILE),
        per_core_N=math.ceil(N / (TILE * num_cores)),
        fused_activation=None,
    )


def make_dram_progcfg_silu(M, K, N, num_cores, in0_block_w_cap=8):
    """Same but with fused SiLU activation (for gate_proj)."""
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_in0_block_w(K, num_cores, cap=in0_block_w_cap),
        per_core_M=math.ceil(M / TILE),
        per_core_N=math.ceil(N / (TILE * num_cores)),
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    )


def find_grid_k_n(K, N, max_rows=4, max_cols=8):
    """qwen_model_config.py:1876 — find core grid that evenly divides both K and N tiles."""
    K_t = K // TILE
    N_t = N // TILE
    max_cores = max_rows * max_cols
    possible = [c for c in range(1, max_cores + 1) if K_t % c == 0 and N_t % c == 0]
    possible.sort(reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"no grid for K={K} N={N}")


# ---- variant builders ----

def upload_interleaved(np_arr, device, dtype):
    """Production-style interleaved DRAM upload (matches server.py:upload)."""
    return ttnn.from_torch(
        torch.from_numpy(np_arr).float(),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_dram_sharded(np_arr, device, dtype, dram_grid, dram_cores):
    K, N = np_arr.shape
    mem_cfg, padded_N = make_dram_sharded_mem_config(K, N, dram_grid, dram_cores)
    if padded_N != N:
        # zero-pad columns
        pad = np.zeros((K, padded_N - N), dtype=np_arr.dtype)
        np_arr_p = np.concatenate([np_arr, pad], axis=1)
    else:
        np_arr_p = np_arr
    return ttnn.from_torch(
        torch.from_numpy(np_arr_p).float(),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_cfg,
    ), padded_N


def make_l1_width_sharded_activation(x_np, device, dtype, num_cores_x, num_cores_y=1):
    """L1 WIDTH_SHARDED activation across (num_cores_y, num_cores_x).

    x_np shape [M, K]. Pads M to TILE (typically M=1 → M_padded=32) so tile-layout
    shard height is a tile multiple.
    """
    M, K = x_np.shape
    total_cores = num_cores_x * num_cores_y
    assert K % total_cores == 0, f"K={K} not divisible by total_cores={total_cores}"
    # Tile-pad M dim to a tile multiple
    M_padded = math.ceil(M / TILE) * TILE
    if M_padded != M:
        pad = np.zeros((M_padded - M, K), dtype=x_np.dtype)
        x_pad = np.concatenate([x_np, pad], axis=0)
    else:
        x_pad = x_np
    grid = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=(M_padded, K),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    return ttnn.from_torch(
        torch.from_numpy(x_pad).float(),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_cfg,
    )


# ---- forward passes per variant ----

def mlp_v0_interleaved(x_tt, g_tt, u_tt, d_tt, kcfg):
    g = ttnn.linear(x_tt, g_tt, activation="silu", compute_kernel_config=kcfg)
    u = ttnn.linear(x_tt, u_tt, compute_kernel_config=kcfg)
    h = ttnn.mul(g, u)
    out = ttnn.linear(h, d_tt, compute_kernel_config=kcfg)
    return out


def mlp_v1_dram_shard_weights_only(x_tt, g_tt, u_tt, d_tt, kcfg):
    """Weights DRAM-sharded but no explicit progcfg — matmul auto-picks.

    Note: cannot use activation="silu" kwarg here — sharded matmuls require fused
    activations to live inside the program_config, not as an op kwarg. So we
    apply silu as a separate op.
    """
    g = ttnn.linear(x_tt, g_tt, compute_kernel_config=kcfg)
    g = ttnn.silu(g)
    u = ttnn.linear(x_tt, u_tt, compute_kernel_config=kcfg)
    h = ttnn.mul(g, u)
    out = ttnn.linear(h, d_tt, compute_kernel_config=kcfg)
    return out


def mlp_v2_full_dram_sharded(x_tt_sharded, g_tt, u_tt, d_tt, kcfg,
                              pc_gate, pc_up, pc_down,
                              out_mem_cfg=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG):
    """Canonical DRAM-sharded matmul.

    Activations + outputs L1 WIDTH-sharded; weights DRAM WIDTH-sharded; explicit progcfg.
    Note: bumping the mul into L1 too.
    """
    g = ttnn.linear(x_tt_sharded, g_tt, compute_kernel_config=kcfg,
                    program_config=pc_gate, memory_config=out_mem_cfg, dtype=ACT_DTYPE)
    u = ttnn.linear(x_tt_sharded, u_tt, compute_kernel_config=kcfg,
                    program_config=pc_up, memory_config=out_mem_cfg, dtype=ACT_DTYPE)
    # mul on sharded tensors
    h = ttnn.mul(g, u, memory_config=out_mem_cfg)
    # down_proj: input is L1_WIDTH_SHARDED [1, INTERMEDIATE], output L1_WIDTH_SHARDED [1, HIDDEN]
    out = ttnn.linear(h, d_tt, compute_kernel_config=kcfg,
                      program_config=pc_down, memory_config=out_mem_cfg, dtype=ACT_DTYPE)
    return out


# ---- bench helper ----

def bench(fn, sync_fn, n_warmup=5, n_iter=30):
    for _ in range(n_warmup):
        out = fn()
    sync_fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = fn()
        sync_fn()
        times.append(time.perf_counter() - t0)
    return median_ms(times), times, out


def main():
    print("=" * 78)
    print("P23 / C'6 probe: DRAM-sharded MLP matmuls vs INTERLEAVED baseline (qb1)")
    print("=" * 78)
    print(f"Shapes: HIDDEN={HIDDEN}, INTERMEDIATE={INTERMEDIATE}")
    print(f"Dtypes: weights={WEIGHT_DTYPE}, activation={ACT_DTYPE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("\n[1] Open device 0 ...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    results = {
        "constants": {
            "HIDDEN": HIDDEN,
            "INTERMEDIATE": INTERMEDIATE,
            "weight_dtype": "bfloat8_b",
            "act_dtype": "bfloat16",
            "P150_DRAM_PEAK_GBS": P150_DRAM_PEAK_GBS,
        },
        "variants": {},
        "verdict": {},
    }

    try:
        dram_grid, dram_cores = dram_grid_for_device(device)
        print(f"  DRAM grid: {dram_cores} cores  (device.dram_grid_size())")
        results["constants"]["dram_cores"] = dram_cores

        # Compute kernel config (HiFi2 matches 91f hifi4? Let's check both)
        # Production uses hifi4 in 91f mlp_step_ondevice (line 599). Use the same to be apples-apples.
        kcfg_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        kcfg_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        kcfg = kcfg_hifi4  # match production

        # --- weights ---
        print(f"\n[2] Build random weights (seed=42)...")
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1)
        # bf16 activation magnitudes typical of post-rmsnorm
        w_gate = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                  / np.sqrt(HIDDEN))
        w_up = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                / np.sqrt(HIDDEN))
        w_down = (rng.standard_normal((INTERMEDIATE, HIDDEN)).astype(np.float32)
                  / np.sqrt(INTERMEDIATE))
        weight_bytes_bf8 = mlp_weight_bytes(BYTES_PER_ELEM_BF8)
        weight_bytes_bf8_conservative = mlp_weight_bytes(BYTES_PER_ELEM_BF8_CONSERVATIVE)
        print(f"  Total MLP weight bytes (bf8, 1.0625 B/elem): {mb(weight_bytes_bf8):.1f} MB")
        print(f"  Total MLP weight bytes (bf8, 1.0 B/elem):    {mb(weight_bytes_bf8_conservative):.1f} MB")

        results["constants"]["weight_bytes_bf8"] = weight_bytes_bf8
        results["constants"]["weight_bytes_bf8_conservative"] = weight_bytes_bf8_conservative

        sync_fn = lambda: ttnn.synchronize_device(device)

        # ============================================================
        # V0: INTERLEAVED (production)
        # ============================================================
        print("\n[V0] INTERLEAVED weights, INTERLEAVED activation")
        g0 = upload_interleaved(w_gate, device, WEIGHT_DTYPE)
        u0 = upload_interleaved(w_up,   device, WEIGHT_DTYPE)
        d0 = upload_interleaved(w_down, device, WEIGHT_DTYPE)
        x0 = ttnn.from_torch(torch.from_numpy(x), dtype=ACT_DTYPE, device=device,
                             layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        v0_fn = lambda: mlp_v0_interleaved(x0, g0, u0, d0, kcfg)
        v0_ms, _, v0_out = bench(v0_fn, sync_fn)
        v0_out_np = ttnn.to_torch(v0_out).float().cpu().numpy().reshape(-1)[:HIDDEN]
        v0_bw = weight_bytes_bf8 / (v0_ms / 1000.0) / 1e9
        v0_pct = v0_bw / P150_DRAM_PEAK_GBS * 100
        print(f"  V0: {v0_ms:.3f} ms/step  →  {v0_bw:.1f} GB/s = {v0_pct:.1f}% of {P150_DRAM_PEAK_GBS:.0f} GB/s peak")
        results["variants"]["V0_interleaved"] = {
            "ms": v0_ms, "gbs_bf8": v0_bw, "pct_peak": v0_pct,
            "cos_vs_v0": 1.0,
        }
        # Free V0 weights
        del g0, u0, d0
        ttnn.synchronize_device(device)

        # ============================================================
        # V1: SKIPPED — DRAM WIDTH-sharded weights cannot be paired with
        # interleaved L1 activation + default progcfg. ttnn enforces:
        #   "Input B memory layout must be INTERLEAVED, got: WIDTH_SHARDED"
        # at matmul_device_operation.cpp:1089 unless explicit DRAMSharded
        # program_config is passed. So V1 is structurally invalid; V2 is the
        # minimum valid DRAM-sharded weight configuration.
        # ============================================================
        print("\n[V1] SKIPPED: DRAM WIDTH-sharded weights require explicit progcfg")
        print("     (ttnn rejects WIDTH-SHARDED weight tensor without DRAMShardedProgramConfig)")
        results["variants"]["V1_dram_sharded_weights_only"] = {
            "skipped": True,
            "reason": ("DRAM WIDTH-sharded weight tensor requires explicit "
                       "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; "
                       "auto-pick matmul rejects with 'Input B memory layout must be "
                       "INTERLEAVED' (matmul_device_operation.cpp:1089)."),
        }

        # ============================================================
        # V1.5: SINGLE DRAM-sharded matmul (just gate) — isolate the recipe.
        # If this fails, V2 chain has no hope. If it works, the failure in V2
        # is about L1 budget for chained ops, not the recipe.
        # ============================================================
        print("\n[V1.5] SINGLE DRAM-sharded matmul (gate only)")
        try:
            act_cores = dram_cores
            g15, _ = upload_dram_sharded(w_gate, device, WEIGHT_DTYPE, dram_grid, dram_cores)
            x_sharded15 = make_l1_width_sharded_activation(
                x, device, ACT_DTYPE, num_cores_x=act_cores, num_cores_y=1
            )
            M_padded = TILE
            for cap in [8, 4, 2]:
                try:
                    pc = make_dram_progcfg_silu(M_padded, K=HIDDEN, N=INTERMEDIATE,
                                                num_cores=dram_cores, in0_block_w_cap=cap)
                    def vfn15(pcv=pc):
                        out = ttnn.linear(x_sharded15, g15, compute_kernel_config=kcfg,
                                          program_config=pcv,
                                          memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                          dtype=ACT_DTYPE)
                        return out
                    v_ms, _, v_out = bench(vfn15, sync_fn, n_warmup=3, n_iter=10)
                    # bytes for ONE matmul (gate): HIDDEN*INTERMEDIATE*bf8 elem * 1.0625 byte
                    bytes_g = HIDDEN * INTERMEDIATE * BYTES_PER_ELEM_BF8
                    bw = bytes_g / (v_ms / 1000.0) / 1e9
                    pct = bw / P150_DRAM_PEAK_GBS * 100
                    print(f"  V1.5 cap={cap}: {v_ms:.3f} ms (just gate)  {bw:.1f} GB/s = {pct:.1f}% peak")
                    del v_out
                except Exception as e:
                    print(f"  V1.5 cap={cap} FAILED: {type(e).__name__}: {str(e)[:200]}")
            del g15, x_sharded15
            ttnn.synchronize_device(device)
        except Exception as e:
            import traceback
            print(f"  ✗ V1.5 outer FAILED: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()

        # ============================================================
        # V2: full DRAM-sharded matmul (canonical recipe)
        # ============================================================
        print("\n[V2] DRAM WIDTH-sharded weights + L1 WIDTH-sharded activation + progcfg")
        v2_ok = False
        try:
            # Use dram_cores for weight sharding
            # For activation L1 sharding, pick num_cores that divides HIDDEN evenly.
            # HIDDEN=5120, TILE=32 → 160 tiles. 160/8=20 → 8 cores works.
            # We need the activation grid to match the WEIGHT progcfg core count.
            # Per llms.md 1546: "The core_grid is the same core grid the activation is
            # width-sharded on". With dram_cores=8 we use 8-core 1x8 grid for activation too.
            act_cores = dram_cores

            assert HIDDEN % (TILE * act_cores) == 0, \
                f"HIDDEN={HIDDEN} not divisible by TILE*act_cores={TILE*act_cores}"
            assert INTERMEDIATE % (TILE * act_cores) == 0, \
                f"INTERMEDIATE={INTERMEDIATE} not divisible by TILE*act_cores={TILE*act_cores}"

            # Activations: x is [1, HIDDEN] going into gate/up. After mul output is [1, INTERMEDIATE]
            # which goes into down. For tile-aligned width sharding the per-core width must be a tile multiple.
            print(f"  act_cores={act_cores}, "
                  f"HIDDEN/(TILE*ac)={HIDDEN//(TILE*act_cores)}, "
                  f"INTERMEDIATE/(TILE*ac)={INTERMEDIATE//(TILE*act_cores)}")

            # Re-upload DRAM-sharded weights
            g2, gate_padded_N = upload_dram_sharded(w_gate, device, WEIGHT_DTYPE, dram_grid, dram_cores)
            u2, up_padded_N   = upload_dram_sharded(w_up,   device, WEIGHT_DTYPE, dram_grid, dram_cores)
            d2, down_padded_N = upload_dram_sharded(w_down, device, WEIGHT_DTYPE, dram_grid, dram_cores)
            print(f"  weight padded_N: gate={gate_padded_N}, up={up_padded_N}, down={down_padded_N}")

            # M is tile-padded batch rows. batch=1 → tile-pad to TILE=32.
            M_padded = TILE  # ttnn pads height to a tile

            # cap=4 is the largest that fit in L1 in V1.5 single-op test; use it as V2 default.
            V2_CAP = 4
            pc_gate = make_dram_progcfg_silu(M_padded, K=HIDDEN, N=INTERMEDIATE, num_cores=dram_cores, in0_block_w_cap=V2_CAP)
            pc_up   = make_dram_progcfg(M_padded, K=HIDDEN, N=INTERMEDIATE, num_cores=dram_cores, in0_block_w_cap=V2_CAP)
            pc_down = make_dram_progcfg(M_padded, K=INTERMEDIATE, N=HIDDEN, num_cores=dram_cores, in0_block_w_cap=V2_CAP)

            # Activation x: width-shard across act_cores
            x_sharded = make_l1_width_sharded_activation(
                x, device, ACT_DTYPE, num_cores_x=act_cores, num_cores_y=1
            )

            # Forward: gate has fused silu inside progcfg, so we replace ttnn.linear(activation="silu")
            # with a single matmul whose progcfg has fused SILU.
            # Aggressive ttnn.deallocate() between ops to keep L1 within 1.5MB per core
            # (Galaxy pattern from llama_mlp.py).
            def v2_fn():
                g = ttnn.linear(x_sharded, g2, compute_kernel_config=kcfg,
                                program_config=pc_gate,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                dtype=ACT_DTYPE)
                u = ttnn.linear(x_sharded, u2, compute_kernel_config=kcfg,
                                program_config=pc_up,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                dtype=ACT_DTYPE)
                h = ttnn.mul(g, u, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
                ttnn.deallocate(g)
                ttnn.deallocate(u)
                out = ttnn.linear(h, d2, compute_kernel_config=kcfg,
                                  program_config=pc_down,
                                  memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                  dtype=ACT_DTYPE)
                ttnn.deallocate(h)
                return out

            v2_ms, _, v2_out = bench(v2_fn, sync_fn)
            # V2 output is L1 width-sharded [M_padded, HIDDEN]; take row 0 (real row).
            v2_out_t = ttnn.to_torch(v2_out).float().cpu().numpy()
            v2_out_np = v2_out_t.reshape(-1, HIDDEN)[0]
            v2_bw = weight_bytes_bf8 / (v2_ms / 1000.0) / 1e9
            v2_pct = v2_bw / P150_DRAM_PEAK_GBS * 100
            v2_cos = float(np.dot(v0_out_np, v2_out_np) /
                           (np.linalg.norm(v0_out_np) * np.linalg.norm(v2_out_np) + 1e-12))
            print(f"  V2: {v2_ms:.3f} ms/step  →  {v2_bw:.1f} GB/s = {v2_pct:.1f}% of peak  cos vs V0={v2_cos:.6f}")
            results["variants"]["V2_full_dram_sharded"] = {
                "ms": v2_ms, "gbs_bf8": v2_bw, "pct_peak": v2_pct, "cos_vs_v0": v2_cos,
                "act_cores": act_cores, "weight_dram_cores": dram_cores,
            }
            v2_ok = True
            del g2, u2, d2, x_sharded, v2_out
            ttnn.synchronize_device(device)
        except Exception as e:
            import traceback
            print(f"  ✗ V2 FAILED: {type(e).__name__}: {str(e)[:400]}")
            traceback.print_exc()
            results["variants"]["V2_full_dram_sharded"] = {"error": str(e)[:500]}

        # ============================================================
        # V3: V2 + in0_block_w cap sweep (1, 2, 4, 8)
        # ============================================================
        # Only run V3 if V2 succeeded; otherwise it's pointless.
        if v2_ok:
            print("\n[V3] V2 + in0_block_w cap sweep (1, 2, 4, 8) on down_proj")
            try:
                act_cores = dram_cores
                # Reuse DRAM-sharded weights
                g3, _ = upload_dram_sharded(w_gate, device, WEIGHT_DTYPE, dram_grid, dram_cores)
                u3, _ = upload_dram_sharded(w_up,   device, WEIGHT_DTYPE, dram_grid, dram_cores)
                d3, _ = upload_dram_sharded(w_down, device, WEIGHT_DTYPE, dram_grid, dram_cores)
                x_sharded3 = make_l1_width_sharded_activation(
                    x, device, ACT_DTYPE, num_cores_x=act_cores, num_cores_y=1
                )
                M_padded = TILE

                sweep_results = []
                for cap in [1, 2, 4, 8]:
                    try:
                        # Gate/up: K=HIDDEN=5120 → K_tiles=160, per-core=20. cap_8 → 8, cap_4 → 4, cap_2 → 2, cap_1 → 1
                        # Down: K=INTERMEDIATE=17408 → K_tiles=544, per-core=68. cap_8 → 4 (best ≤8 dividing 68: 4),
                        #       cap_4 → 4, cap_2 → 2, cap_1 → 1.
                        pc_gate_v = make_dram_progcfg_silu(M_padded, K=HIDDEN, N=INTERMEDIATE,
                                                          num_cores=dram_cores, in0_block_w_cap=cap)
                        pc_up_v = make_dram_progcfg(M_padded, K=HIDDEN, N=INTERMEDIATE,
                                                    num_cores=dram_cores, in0_block_w_cap=cap)
                        pc_down_v = make_dram_progcfg(M_padded, K=INTERMEDIATE, N=HIDDEN,
                                                       num_cores=dram_cores, in0_block_w_cap=cap)
                        gate_bw = _in0_block_w(HIDDEN, dram_cores, cap)
                        down_bw = _in0_block_w(INTERMEDIATE, dram_cores, cap)

                        def vfn(pcg=pc_gate_v, pcu=pc_up_v, pcd=pc_down_v):
                            g = ttnn.linear(x_sharded3, g3, compute_kernel_config=kcfg,
                                            program_config=pcg,
                                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                            dtype=ACT_DTYPE)
                            u = ttnn.linear(x_sharded3, u3, compute_kernel_config=kcfg,
                                            program_config=pcu,
                                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                            dtype=ACT_DTYPE)
                            h = ttnn.mul(g, u, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
                            ttnn.deallocate(g)
                            ttnn.deallocate(u)
                            out = ttnn.linear(h, d3, compute_kernel_config=kcfg,
                                              program_config=pcd,
                                              memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                              dtype=ACT_DTYPE)
                            ttnn.deallocate(h)
                            return out

                        v_ms, _, v_out = bench(vfn, sync_fn)
                        v_bw = weight_bytes_bf8 / (v_ms / 1000.0) / 1e9
                        v_pct = v_bw / P150_DRAM_PEAK_GBS * 100
                        v_out_t = ttnn.to_torch(v_out).float().cpu().numpy()
                        v_out_np = v_out_t.reshape(-1, HIDDEN)[0]
                        v_cos = float(np.dot(v0_out_np, v_out_np) /
                                      (np.linalg.norm(v0_out_np) * np.linalg.norm(v_out_np) + 1e-12))
                        print(f"  V3 cap={cap}  gate_bw={gate_bw}  down_bw={down_bw}  "
                              f"{v_ms:.3f} ms  {v_bw:.1f} GB/s = {v_pct:.1f}%  cos={v_cos:.6f}")
                        sweep_results.append({
                            "cap": cap, "gate_bw": gate_bw, "down_bw": down_bw,
                            "ms": v_ms, "gbs_bf8": v_bw, "pct_peak": v_pct, "cos_vs_v0": v_cos,
                        })
                    except Exception as e:
                        print(f"  V3 cap={cap} FAILED: {type(e).__name__}: {str(e)[:200]}")
                        sweep_results.append({"cap": cap, "error": str(e)[:300]})

                results["variants"]["V3_tuned"] = {"sweep": sweep_results}
                # Pick best
                successful = [r for r in sweep_results if "error" not in r]
                if successful:
                    best_v3 = min(successful, key=lambda r: r["ms"])
                    results["variants"]["V3_tuned"]["best"] = best_v3
                    print(f"  V3 best: cap={best_v3['cap']}  {best_v3['ms']:.3f} ms  "
                          f"{best_v3['pct_peak']:.1f}%")
                del g3, u3, d3, x_sharded3
                ttnn.synchronize_device(device)
            except Exception as e:
                import traceback
                print(f"  ✗ V3 FAILED: {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
                results["variants"]["V3_tuned"] = {"error": str(e)[:500]}

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  {'variant':<32}  {'ms/step':>8}  {'GB/s':>7}  {'%peak':>6}  {'cos vs V0':>10}")
        for name in ["V0_interleaved", "V1_dram_sharded_weights_only", "V2_full_dram_sharded"]:
            v = results["variants"].get(name)
            if v is None:
                print(f"  {name:<32}  missing")
                continue
            if "error" in v:
                print(f"  {name:<32}  ERROR: {v['error'][:60]}")
                continue
            if "skipped" in v:
                print(f"  {name:<32}  SKIPPED: {v.get('reason','')[:60]}")
                continue
            print(f"  {name:<32}  {v['ms']:>8.3f}  {v['gbs_bf8']:>7.1f}  "
                  f"{v['pct_peak']:>5.1f}%  {v['cos_vs_v0']:>10.6f}")
        # V3 sweep summary
        v3 = results["variants"].get("V3_tuned", {})
        if isinstance(v3, dict) and "sweep" in v3:
            print("  V3_tuned sweep:")
            for s in v3["sweep"]:
                if "error" in s:
                    print(f"    cap={s['cap']:>2}  ERROR: {s['error'][:60]}")
                else:
                    print(f"    cap={s['cap']:>2}  {s['ms']:>8.3f}  {s['gbs_bf8']:>7.1f}  "
                          f"{s['pct_peak']:>5.1f}%  {s['cos_vs_v0']:>10.6f}")

        # Verdict
        v0 = results["variants"]["V0_interleaved"]
        best_name, best = None, None
        for name in ["V1_dram_sharded_weights_only", "V2_full_dram_sharded"]:
            v = results["variants"].get(name)
            if v and "error" not in v and "ms" in v:
                if best is None or v["ms"] < best["ms"]:
                    best = v
                    best_name = name
        # V3 has its own "best" entry
        v3 = results["variants"].get("V3_tuned", {})
        if isinstance(v3, dict) and "best" in v3:
            v3b = v3["best"]
            if best is None or v3b["ms"] < best["ms"]:
                best = v3b
                best_name = f"V3_tuned[cap={v3b['cap']}]"
        if best:
            delta_ms = v0["ms"] - best["ms"]
            speedup = v0["ms"] / best["ms"] if best["ms"] > 0 else 0.0
            pass_bw = best["pct_peak"] >= 90.0
            pass_lat = delta_ms >= 0.5
            verdict = "PASS" if (pass_bw and pass_lat) else "PARTIAL" if (pass_bw or pass_lat) else "FAIL"
            print(f"\n  Best non-V0: {best_name}")
            print(f"    {best['ms']:.3f} ms  vs V0 {v0['ms']:.3f} ms  (Δ={delta_ms:+.3f} ms, {speedup:.2f}×)")
            print(f"    {best['pct_peak']:.1f}% of peak  (target ≥ 90%)")
            print(f"    cos vs V0 = {best['cos_vs_v0']:.6f}  (target ≥ 0.999)")
            print(f"    Verdict: {verdict}")
            print(f"    At 64 MLPs/tok: V0={v0['ms']*64:.1f} ms, best={best['ms']*64:.1f} ms, Δ={delta_ms*64:+.1f} ms")
            results["verdict"] = {
                "best_variant": best_name,
                "delta_ms_per_step": delta_ms,
                "delta_ms_per_token_at_64_mlps": delta_ms * 64,
                "speedup": speedup,
                "best_pct_peak": best["pct_peak"],
                "pass_bw_90pct": pass_bw,
                "pass_lat_half_ms": pass_lat,
                "cos_ok": best["cos_vs_v0"] >= 0.999,
                "verdict": verdict,
            }

        out_json = OUT_DIR / "results.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ wrote {out_json}")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
