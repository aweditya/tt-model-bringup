#!/usr/bin/env python3
"""
C'7.5b: TP chain scaling probe — chain N TP layers, validate cosine + measure scaling.

Goal: does the residual stream chain hold across MANY TP layer boundaries, or
does precision drift accumulate? Chain N×(DeltaNet TP + MLP TP) blocks and
check cosine vs numpy gold at N ∈ {1, 2, 4, 8}.

If cosine stays high through 8 layers, the full 128-layer scale (C'7.6) is
purely engineering — no per-iteration math validation needed.

Reuses the validated TP forward functions from full_layer_tp_probe.py.
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

# Reuse logic from full_layer_tp_probe.py
sys.path.insert(0, os.path.expanduser("~/tt-xla/experiments/utils"))
from full_layer_tp_probe import (
    HIDDEN, N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, KERNEL,
    KEY_DIM, VAL_DIM, CONV_DIM, N_REP, IN_PROJ_OUT, INTERMEDIATE,
    EPS, NCHIPS, NK_PER_CHIP, NV_PER_CHIP,
    KEY_DIM_CHIP, VAL_DIM_CHIP, CONV_DIM_CHIP, IN_PROJ_OUT_CHIP,
    _cosine, silu_np, deltanet_np, mlp_np, full_layer_np,
    relayout_in_proj, relayout_conv,
    deltanet_tp, mlp_tp,
)

sys.stdout.reconfigure(line_buffering=True)


def chain_np(x, layers):
    """N-layer numpy chain: each layer = (DeltaNet weights + MLP weights)."""
    cur = x
    for L in layers:
        cur = full_layer_np(cur, L['w_in'], L['w_conv'], L['dt_bias'], L['A_log'],
                            L['w_out'], L['ssm'], L['conv_state'],
                            L['w_gate'], L['w_up'], L['w_down'])
    return cur


def chain_tp(mesh, x_tt, sharded_layers):
    """Run N-layer TP chain. Each layer goes DeltaNet → MLP."""
    cur = x_tt
    for sl in sharded_layers:
        cur = deltanet_tp(mesh, cur, sl['dn'])
        cur = mlp_tp(mesh, cur, sl['mlp'])
    return cur


def build_layer_random(rng):
    return {
        'w_in':       rng.standard_normal((HIDDEN, IN_PROJ_OUT)).astype(np.float32) / np.sqrt(HIDDEN),
        'w_conv':     rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3,
        'dt_bias':    rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.1,
        'A_log':      rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.5,
        'w_out':      rng.standard_normal((VAL_DIM, HIDDEN)).astype(np.float32) / np.sqrt(VAL_DIM),
        'ssm':        rng.standard_normal((N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.3,
        'conv_state': rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.5,
        'w_gate':     rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32) / np.sqrt(HIDDEN),
        'w_up':       rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32) / np.sqrt(HIDDEN),
        'w_down':     rng.standard_normal((INTERMEDIATE, HIDDEN)).astype(np.float32) / np.sqrt(INTERMEDIATE),
    }


def upload_layer(mesh, L):
    """Upload one layer's weights to the mesh with proper sharding."""
    def sh(arr, dim):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim))

    w_in_sh = relayout_in_proj(L['w_in'])
    w_conv_sh = relayout_conv(L['w_conv'])
    conv_state_sh = relayout_conv(L['conv_state'])

    return {
        'dn': {
            'w_in':    sh(w_in_sh, dim=1),
            'w_conv':  sh(w_conv_sh, dim=0),
            'conv_st': sh(conv_state_sh, dim=0),
            'dt_bias': sh(L['dt_bias'], dim=0),
            'A_log':   sh(L['A_log'], dim=0),
            'w_out':   sh(L['w_out'], dim=0),
            'ssm':     sh(L['ssm'], dim=0),
        },
        'mlp': {
            'w_gate':  sh(L['w_gate'], dim=1),
            'w_up':    sh(L['w_up'], dim=1),
            'w_down':  sh(L['w_down'], dim=0),
        },
    }


def main():
    print("=" * 78)
    print("C'7.5b: TP chain scaling (N ∈ {1, 2, 4, 8} layers)")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build 8 random layers + replicated input (seed=42)...")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        layers = [build_layer_random(rng) for _ in range(8)]
        print(f"  built {len(layers)} layers")

        print("\n[3] Upload all 8 sharded layers to mesh...")
        sharded_layers = [upload_layer(mesh, L) for L in layers]
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        print("\n[4] Chain N layers and validate cosine at N ∈ {1, 2, 4, 8}...")
        results = []
        for n in [1, 2, 4, 8]:
            # Numpy gold for N layers
            y_gold = chain_np(x, layers[:n])
            # TP forward for N layers
            t0 = time.perf_counter()
            y_tp_out = chain_tp(mesh, x_tt, sharded_layers[:n])
            stacked = ttnn.to_torch(
                y_tp_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000.0
            y_chip0 = stacked[0].flatten()
            y_chip3 = stacked[-1].flatten()
            cos_inter = _cosine(y_chip0, y_chip3)
            cos_gold = _cosine(y_chip0, y_gold.flatten())
            print(f"  N={n}: cos_vs_gold={cos_gold:.6f}  cos_inter={cos_inter:.6f}  "
                  f"wall={ms:.1f}ms  std_gold={y_gold.std():.4f}")
            results.append((n, cos_gold, cos_inter, ms))

        print("\n[5] Latency at N=8 (warmed, N=5 iters)...")
        # Re-warm
        for _ in range(3):
            _ = chain_tp(mesh, x_tt, sharded_layers)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = chain_tp(mesh, x_tt, sharded_layers)
        ttnn.synchronize_device(mesh)
        ms_per_8layer = (time.perf_counter() - t0) * 1000.0 / 5
        print(f"  8-layer TP chain: {ms_per_8layer:.2f} ms")
        print(f"  Per-layer (8-layer avg): {ms_per_8layer / 8:.3f} ms")
        # Projection: Qwen3.6-27B is 48 DeltaNet + 16 Gated Attn + 64 MLP layers total.
        # 8 layers of (DeltaNet+MLP) chains = 16 sub-layers. Full model = 128 sub-layers.
        # Scale: 128 / 16 = 8× this.
        print(f"  Projected for full 128 sub-layer model: {ms_per_8layer * 8:.1f} ms/tok")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        all_ok = all(cg >= 0.99 and ci >= 0.99 for _, cg, ci, _ in results)
        print(f"  Cosines hold through 8 layers: {'✓ PASS' if all_ok else '✗ FAIL'}")
        for n, cg, ci, _ in results:
            mark = "✓" if cg >= 0.99 else "✗"
            print(f"    N={n}: gold={cg:.6f} inter={ci:.6f} {mark}")
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
