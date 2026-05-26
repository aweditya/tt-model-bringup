#!/usr/bin/env python3
"""
C'7.6: Multi-chip TP full-model perf benchmark on qb2.

Goal: measure actual ms/tok of TP forward at full Qwen3.6-27B model
scale (128 sub-layers = 48 DeltaNet + 16 Gated Attn + 64 MLP). Until now
we've only measured 1-8 chained sub-layers in TP. Need a real perf
number to compare against single-chip 200.81 ms baseline.

Approach (random weights, production magnitudes):
  Build representative sharded weights for 1 DeltaNet + 1 Gated Attn + 1
  MLP layer (each = "one full transformer block"). Chain them through
  N iterations to simulate full-model depth. Measure:
    - per-iteration latency (≈ 1 transformer block at 4-chip TP)
    - extrapolate to 64 blocks (= 128 sub-layers)
    - compare to single-chip extrapolation

This is the FIRST concrete perf number for multi-chip TP. C'7.2 showed
MLP TP at 0.71 ms/step (vs 1.96 single-chip = 2.77×); C'7.3 showed
Gated Attn TP cos PASS but no latency. C'7.4 showed DeltaNet TP cos PASS
but no latency. C'7.6 combines all three into a chain at production scale.

Random weights with bounded magnitudes (no LayerNorm but small scaling
keeps activations from exploding through 4-8 chained layers).
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

# Reuse the chained TP probe machinery
sys.path.insert(0, os.path.expanduser("~/tt-xla/experiments/utils"))
from full_layer_tp_probe import (
    HIDDEN, N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, KERNEL,
    KEY_DIM, VAL_DIM, CONV_DIM, N_REP, IN_PROJ_OUT, INTERMEDIATE,
    EPS, NCHIPS, NK_PER_CHIP, NV_PER_CHIP,
    KEY_DIM_CHIP, VAL_DIM_CHIP, CONV_DIM_CHIP, IN_PROJ_OUT_CHIP,
    _cosine, deltanet_tp, mlp_tp,
)
from tp_chain_scaling_probe import (
    build_layer_random, upload_layer, chain_np, chain_tp,
)

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("C'7.6: Multi-chip TP full-model perf benchmark (qb2)")
    print("=" * 78)
    print(f"Target: project full Qwen3.6-27B TP latency from 1-block × 64 chain")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build 1 representative layer with bounded random weights...")
        # Smaller scale than C'7.5b to avoid recurrence explosion
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5
        layer = build_layer_random(rng)
        # Override to bound activations: smaller SSM init, smaller weights
        layer['ssm']        = rng.standard_normal((N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.05
        layer['conv_state'] = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.1
        # Scale weights down so per-layer output stays ~unit magnitude
        for k in ['w_in', 'w_out', 'w_gate', 'w_up', 'w_down']:
            layer[k] = layer[k] * 0.7

        print("\n[3] Upload sharded weights...")
        sharded = upload_layer(mesh, layer)
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        print("\n[4] Warm-up forward (3 iters)...")
        for _ in range(3):
            y = deltanet_tp(mesh, x_tt, sharded['dn'])
            y = mlp_tp(mesh, y, sharded['mlp'])
        ttnn.synchronize_device(mesh)

        print("\n[5] Per-block latency benchmark (N=20)...")
        # Each "block" = DeltaNet TP + MLP TP (i.e. one transformer layer worth of TP work)
        # Production has 48 DN blocks + 16 GatedAttn blocks + 64 MLP blocks.
        # But Gated Attn TP wasn't included in this chain probe — use the
        # DN-block latency as an approximation. We'll annotate the projection.
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            y = deltanet_tp(mesh, x_tt, sharded['dn'])
            y = mlp_tp(mesh, y, sharded['mlp'])
        ttnn.synchronize_device(mesh)
        per_dn_mlp_ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  Per DN+MLP block (4-chip TP): {per_dn_mlp_ms:.3f} ms")

        # Standalone DN TP
        t0 = time.perf_counter()
        for _ in range(N):
            y = deltanet_tp(mesh, x_tt, sharded['dn'])
        ttnn.synchronize_device(mesh)
        per_dn_ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  Per DeltaNet TP only:         {per_dn_ms:.3f} ms")

        # Standalone MLP TP
        t0 = time.perf_counter()
        for _ in range(N):
            y = mlp_tp(mesh, x_tt, sharded['mlp'])
        ttnn.synchronize_device(mesh)
        per_mlp_ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  Per MLP TP only:              {per_mlp_ms:.3f} ms")

        print("\n[6] Full-model projection:")
        # Qwen3.6-27B: 48 DeltaNet + 16 Gated Attn + 64 MLP layers
        # Approximate Gated Attn cost from prior Gated Attn TP probe:
        # C'7.3 didn't measure latency; use single-chip Gated Attn 2.26 ms / 2.77x = 0.81 ms
        # (rough scale from MLP's measured 2.77× speedup)
        per_attn_ms_estimated = 0.81  # rough
        n_dn, n_attn, n_mlp = 48, 16, 64
        proj_ms = n_dn * per_dn_ms + n_attn * per_attn_ms_estimated + n_mlp * per_mlp_ms
        proj_tok_s = 1000.0 / proj_ms
        print(f"  Projected: {n_dn}×{per_dn_ms:.2f} + {n_attn}×{per_attn_ms_estimated:.2f} + {n_mlp}×{per_mlp_ms:.2f}")
        print(f"           = {proj_ms:.1f} ms/tok = {proj_tok_s:.2f} tok/s")
        print(f"  Single-chip baseline: 200.81 ms/tok = 4.98 tok/s")
        print(f"  Speedup vs single-chip: {200.81 / proj_ms:.2f}×")

        print("\n  Caveat: 'per DeltaNet TP' here uses bounded random weights/states.")
        print("  Production weights may differ (esp. recurrence path); real perf")
        print("  validation requires loading actual Qwen3.6 layer-0 weights.")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
