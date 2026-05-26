#!/usr/bin/env python3
"""
C'6 probe: DRAM-sharded MLP weights on qb1 (single P150).

Hypothesis: moving MLP gate/up/down weights from INTERLEAVED → DRAM-sharded
lifts effective DRAM bandwidth from ~333 GB/s baseline toward the achievable
ceiling of ~470 GB/s (Saturating_DRAM_bandwidth tech report). For BW-bound
single-MLP step this could be a 30-40% latency win.

Probe scope: ISOLATED single-MLP forward, not full model. Validates the
hypothesis before we touch 91f / production code.

Shapes (Qwen3.6-27B text MLP):
  HIDDEN = 5120, INTERMEDIATE = 25600
  gate_proj, up_proj: [5120, 25600]  → 250 MB per matrix in bf8 (we test bf16
    for parity with current production: 500 MB per matrix)
  down_proj: [25600, 5120]            → same.

We compare:
  A) baseline:  gate/up/down all INTERLEAVED (production default)
  B) variant:   gate/up/down all DRAM-sharded (split across 8 DRAM channels)

For each, measure:
  - per-step latency (warm)
  - throughput in GB/s (weight-bytes / step-time)

Single-P150 DRAM peak per Tenstorrent specs:
  https://docs.tenstorrent.com/aibs/blackhole/specifications.html → **512 GB/s peak**
Achievable single-stream is lower; Saturating_DRAM_bandwidth tech report shows
DRAM-sharded multi-bank streaming gets within ~70% of peak (~350 GB/s typical).

Revised ceiling (per external assessment 2026-05-13):
  - 198 ms/tok current  → ~140 GB/s effective (only 27% of 512 peak)
  - 250 GB/s effective  → ~109 ms/tok = 9.2 tok/s (plausible near-term)
  - 350 GB/s effective  → ~78 ms/tok  = 12.8 tok/s (good kernels/layouts)
  - 512 GB/s ideal      → ~53.5 ms/tok = 18.7 tok/s (theoretical single-card)

Run:
    ssh qb1 'cd ~/tt-xla && pkill -f serve.server || true; .venv/bin/python experiments/utils/mlp_dram_sharded_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


HIDDEN = 5120
INTERMEDIATE = 25600
DTYPE = ttnn.bfloat16
BYTES_PER_ELEM = 2  # bf16


def mb(n):
    return n / (1024 * 1024)


def mlp_forward(x_tt, g_tt, u_tt, d_tt):
    g = ttnn.linear(x_tt, g_tt, activation="silu")
    u = ttnn.linear(x_tt, u_tt)
    h = ttnn.mul(g, u)
    return ttnn.linear(h, d_tt)


def bench(device, x_tt, g_tt, u_tt, d_tt, label, n_warmup=5, n_iter=50):
    # Warmup
    for _ in range(n_warmup):
        _ = mlp_forward(x_tt, g_tt, u_tt, d_tt)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = mlp_forward(x_tt, g_tt, u_tt, d_tt)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    # Weight bytes read per step: gate + up + down
    weight_bytes = (HIDDEN * INTERMEDIATE * 2 + INTERMEDIATE * HIDDEN) * BYTES_PER_ELEM
    gbs = weight_bytes / (elapsed_ms / 1000.0) / 1e9

    print(f"  {label}: {elapsed_ms:.3f} ms/step  →  {gbs:.1f} GB/s effective "
          f"(weight bytes / step = {weight_bytes / 1e9:.2f} GB)")
    return elapsed_ms, gbs, out


def main():
    print("=" * 72)
    print("C'6 probe: DRAM-sharded MLP weights vs INTERLEAVED baseline (qb1)")
    print("=" * 72)
    print(f"Shapes: HIDDEN={HIDDEN}, INTERMEDIATE={INTERMEDIATE}, dtype={DTYPE}")
    print(f"Weight footprint per MLP: gate={mb(HIDDEN * INTERMEDIATE * BYTES_PER_ELEM):.1f} MB, "
          f"up={mb(HIDDEN * INTERMEDIATE * BYTES_PER_ELEM):.1f} MB, "
          f"down={mb(INTERMEDIATE * HIDDEN * BYTES_PER_ELEM):.1f} MB")

    print("\n[1] Open device 0...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print(f"\n[2] Build random fp32 weights (deterministic seed)...")
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1)
        w_gate = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                  / np.sqrt(HIDDEN))
        w_up   = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                  / np.sqrt(HIDDEN))
        w_down = (rng.standard_normal((INTERMEDIATE, HIDDEN)).astype(np.float32)
                  / np.sqrt(INTERMEDIATE))

        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=DTYPE,
                               device=device, layout=ttnn.TILE_LAYOUT)

        # --- A) INTERLEAVED baseline ---
        print("\n[3a] Baseline: INTERLEAVED weights (production default)")
        g_tt = ttnn.from_torch(torch.from_numpy(w_gate), dtype=DTYPE,
                               device=device, layout=ttnn.TILE_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u_tt = ttnn.from_torch(torch.from_numpy(w_up), dtype=DTYPE,
                               device=device, layout=ttnn.TILE_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        d_tt = ttnn.from_torch(torch.from_numpy(w_down), dtype=DTYPE,
                               device=device, layout=ttnn.TILE_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ms_baseline, gbs_baseline, out_baseline = bench(
            device, x_tt, g_tt, u_tt, d_tt, label="INTERLEAVED"
        )
        out_b_np = ttnn.to_torch(out_baseline).float().cpu().numpy().reshape(-1)[:HIDDEN]

        # Free baseline weights before allocating sharded
        del g_tt, u_tt, d_tt

        # --- B) DRAM-sharded variant ---
        print("\n[3b] Variant: DRAM-sharded weights (split across 8 channels)")
        # DRAM sharded memory config: shard the matrix across all 8 DRAM channels.
        # The shape is split along dim=1 (output dim) for gate/up, dim=0 for down.
        # Each shard lives on one DRAM channel. ttnn uses a `dram_sharded` builder
        # — we go via the explicit memory_config builder.
        try:
            from ttnn import CoreGrid
            # ttnn DRAM sharded weight pattern from tt_transformers:
            # weight stored across DRAM banks, accessed by all cores via a fast
            # reader. The simplest call is ttnn.create_sharded_memory_config_ with
            # explicit grid + shard shape.
            # Helper: use the canonical DRAM sharded preset if available.
            ncores = 8  # 8 DRAM channels on Blackhole
            shard_w_int = INTERMEDIATE // ncores  # 25600/8 = 3200
            shard_h = HIDDEN
            # DRAM sharded for gate/up: shape [HIDDEN, INTERMEDIATE], shard
            # along output dim (col), shard_shape = (HIDDEN, INTERMEDIATE/8).
            from ttnn import ShardSpec, ShardOrientation, MemoryConfig, BufferType, ShardStrategy
            dram_shard_g = ttnn.create_sharded_memory_config(
                shape=(HIDDEN, INTERMEDIATE),
                core_grid=ttnn.CoreGrid(y=1, x=ncores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
            # For down: shape [INTERMEDIATE, HIDDEN], shard along input dim (row)
            dram_shard_d = ttnn.create_sharded_memory_config(
                shape=(INTERMEDIATE, HIDDEN),
                core_grid=ttnn.CoreGrid(y=1, x=ncores),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )

            g_tt = ttnn.from_torch(torch.from_numpy(w_gate), dtype=DTYPE,
                                   device=device, layout=ttnn.TILE_LAYOUT,
                                   memory_config=dram_shard_g)
            u_tt = ttnn.from_torch(torch.from_numpy(w_up), dtype=DTYPE,
                                   device=device, layout=ttnn.TILE_LAYOUT,
                                   memory_config=dram_shard_g)
            d_tt = ttnn.from_torch(torch.from_numpy(w_down), dtype=DTYPE,
                                   device=device, layout=ttnn.TILE_LAYOUT,
                                   memory_config=dram_shard_d)
            print("  ✓ weights uploaded with DRAM sharded memory config")

            ms_shard, gbs_shard, out_shard = bench(
                device, x_tt, g_tt, u_tt, d_tt, label="DRAM-SHARDED"
            )
            out_s_np = ttnn.to_torch(out_shard).float().cpu().numpy().reshape(-1)[:HIDDEN]

            # Correctness: numerics should match within bf16 rounding (cos ≥ 0.999)
            cos_eq = float(np.dot(out_b_np, out_s_np) /
                            (np.linalg.norm(out_b_np) * np.linalg.norm(out_s_np) + 1e-12))
            print(f"\n  cos(INTERLEAVED out, DRAM-SHARDED out) = {cos_eq:.6f}  (should be ≥ 0.9999)")

            print("\n" + "=" * 72)
            print("RESULTS")
            print("=" * 72)
            print(f"  INTERLEAVED   {ms_baseline:.3f} ms  {gbs_baseline:.1f} GB/s")
            print(f"  DRAM-SHARDED  {ms_shard:.3f} ms  {gbs_shard:.1f} GB/s")
            speedup = ms_baseline / ms_shard
            bw_gain = gbs_shard / gbs_baseline
            print(f"  speedup: {speedup:.2f}×  (BW gain: {bw_gain:.2f}×)")
            print(f"  per-token at 64 MLPs/token:")
            print(f"    INTERLEAVED   = {ms_baseline * 64:.1f} ms")
            print(f"    DRAM-SHARDED  = {ms_shard * 64:.1f} ms  (Δ = {(ms_baseline - ms_shard) * 64:.1f} ms)")

        except Exception as e:
            import traceback
            print(f"  ✗ DRAM-sharded path failed: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
            print(f"\n  Falling back: report INTERLEAVED baseline only")
            print(f"  Baseline: {ms_baseline:.3f} ms/step at {gbs_baseline:.1f} GB/s "
                  f"({gbs_baseline / 470 * 100:.0f}% of 470 GB/s achievable ceiling)")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
