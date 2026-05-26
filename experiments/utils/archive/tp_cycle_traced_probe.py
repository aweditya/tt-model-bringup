#!/usr/bin/env python3
"""
Multi-chip TP transformer-cycle traced probe — closes the projection gap.

Status check:
  - DN+MLP traced @ K=1..8: 1.20 ms/block flat (linear ✓)
  - Attn TP traced @ K=1..4: 0.234 ms/block (linear ✓)
  - Layer 0 real-weight forward cos vs gold: 0.999997 ✓

Open question: does the FULL Qwen3.6-27B forward fit in trace memory + scale
linearly? The model interleaves DN and attn (3 DN + 1 attn per "cycle"; 16
cycles = 64 transformer blocks = full model).

This probe builds K transformer cycles and traces them as one graph.
A "cycle" = 3 DeltaNet sub-layers + 1 Gated Attn sub-layer, each with MLP.
K = {1, 2, 4} cycles → {4, 8, 16} transformer blocks.

If linear at K=4, the 48 DN + 16 attn = 64-block full-model projection
(real extrapolation, not speculation) has the strongest possible backing
without actually running the full thing.

Test at random weights, bounded magnitudes (no LayerNorm bounding here).
Real-shape constants from full_layer_tp_probe.py (corrected 2026-05-13).
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.path.insert(0, os.path.expanduser("~/tt-xla/experiments/utils"))
from full_layer_tp_probe import (
    HIDDEN, N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, KERNEL,
    KEY_DIM, VAL_DIM, CONV_DIM, N_REP, IN_PROJ_OUT, INTERMEDIATE,
    EPS, NCHIPS, NK_PER_CHIP, NV_PER_CHIP,
    KEY_DIM_CHIP, VAL_DIM_CHIP, CONV_DIM_CHIP, IN_PROJ_OUT_CHIP,
    deltanet_tp, mlp_tp,
    relayout_in_proj, relayout_conv,
)
from tp_chain_scaling_probe import build_layer_random, upload_layer
from tp_attn_traced_probe import (
    HEAD_DIM as ATTN_HEAD_DIM, N_Q, N_KV,
    ATTN_QKV_DIM, O_DIM as ATTN_O_DIM,
    NQ_PER_CHIP, NKV_PER_CHIP, MAX_POS as ATTN_MAX_POS,
    relayout_attn_qkv, relayout_o,
    attn_tp_forward, build_attn_block_random, upload_attn_layer,
)

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("Transformer cycle TP traced probe (qb2)")
    print("=" * 78)
    print(f"  cycle = 3 DN + 1 attn (+MLP each)")
    print(f"  K=1 cycle = 4 transformer blocks; K=4 = 16 blocks = 25% of full model")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    try:
        K_MAX = 4  # 4 cycles = 16 transformer blocks
        rng = np.random.default_rng(42)

        print(f"\n[1] Build K_MAX={K_MAX} cycles of (3 DN + 1 attn) blocks with MLP...")
        # For each cycle: 4 layers (3 DN, 1 attn). MLP weights per layer.
        # We need: 3*K_MAX DN+MLP blocks, K_MAX attn+MLP blocks.
        dn_blocks = []
        for _ in range(3 * K_MAX):
            L = build_layer_random(rng)
            L['ssm']        = rng.standard_normal(L['ssm'].shape).astype(np.float32) * 0.05
            L['conv_state'] = rng.standard_normal(L['conv_state'].shape).astype(np.float32) * 0.1
            for k in ['w_in', 'w_out', 'w_gate', 'w_up', 'w_down']:
                L[k] = L[k] * 0.7
            dn_blocks.append(L)
        attn_blocks = [build_attn_block_random(rng) for _ in range(K_MAX)]
        # MLP for attn layers — reuse build_layer_random but only keep MLP weights
        attn_mlp_blocks = []
        for _ in range(K_MAX):
            L = build_layer_random(rng)
            for k in ['w_gate', 'w_up', 'w_down']:
                L[k] = L[k] * 0.7
            attn_mlp_blocks.append(L)

        print(f"\n[2] Upload all sharded weights...")
        sharded_dn = [upload_layer(mesh, L) for L in dn_blocks]
        sharded_attn = []
        for b in attn_blocks:
            w_qkv_sh = relayout_attn_qkv(b['w_qkv'], NQ_PER_CHIP, NKV_PER_CHIP)
            w_o_sh = relayout_o(b['w_o'])
            k_cache_chip = b['k_cache'].transpose(1, 0, 2)
            v_cache_chip = b['v_cache'].transpose(1, 0, 2)
            sharded_attn.append(upload_attn_layer(mesh, w_qkv_sh, w_o_sh, k_cache_chip, v_cache_chip))
        sharded_attn_mlp = [upload_layer(mesh, L) for L in attn_mlp_blocks]
        print("  ✓ uploaded")

        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def cycle_forward(cur, cycle_idx):
            """3 DN+MLP + 1 attn+MLP."""
            base = cycle_idx * 3
            for j in range(3):
                cur = deltanet_tp(mesh, cur, sharded_dn[base + j]['dn'])
                cur = mlp_tp(mesh, cur, sharded_dn[base + j]['mlp'])
            cur = attn_tp_forward(mesh, cur, sharded_attn[cycle_idx])
            cur = mlp_tp(mesh, cur, sharded_attn_mlp[cycle_idx]['mlp'])
            return cur

        def forward_K(x_in, K):
            cur = x_in
            for k in range(K):
                cur = cycle_forward(cur, k)
            return cur

        # Warmup (JIT)
        print("\n[3] Warmup (K=1, 3 iters)...")
        for _ in range(3):
            _ = forward_K(x_tt, 1)
        ttnn.synchronize_device(mesh)
        for _ in range(2):
            _ = forward_K(x_tt, K_MAX)
        ttnn.synchronize_device(mesh)
        print("  ✓ warmup")

        results = []
        for K in [1, 2, 4]:
            print(f"\n[K={K} cycles = {4*K} transformer blocks]")
            N = max(5, 20 // K)
            t0 = time.perf_counter()
            for _ in range(N):
                _ = forward_K(x_tt, K)
            ttnn.synchronize_device(mesh)
            eager_ms = (time.perf_counter() - t0) * 1000.0 / N

            try:
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                _ = forward_K(x_tt, K)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            except Exception as e:
                print(f"  ✗ trace K={K}: {type(e).__name__}: {str(e)[:300]}")
                results.append((K, eager_ms, None))
                continue

            for _ in range(3):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            traced_ms = (time.perf_counter() - t0) * 1000.0 / 20

            try:
                ttnn.release_trace(mesh, trace_id)
            except Exception:
                pass

            speedup = eager_ms / traced_ms if traced_ms > 0 else float('nan')
            per_cycle_traced = traced_ms / K
            per_block_traced = per_cycle_traced / 4  # 4 transformer blocks per cycle
            print(f"  K={K}: eager={eager_ms:.3f} ms, traced={traced_ms:.3f} ms, "
                  f"speedup={speedup:.2f}×, per-cycle={per_cycle_traced:.3f} ms, "
                  f"per-block={per_block_traced:.3f} ms")
            results.append((K, eager_ms, traced_ms))

        print("\n" + "=" * 78)
        print("CYCLE TRACED RESULTS — real model interleave pattern (3 DN + 1 attn)")
        print("=" * 78)
        print(f"{'K cycles':>9s} {'blocks':>7s} {'eager (ms)':>12s} {'traced (ms)':>12s} "
              f"{'speedup':>10s} {'per-cycle':>10s}")
        for K, eager_ms, traced_ms in results:
            if traced_ms is None:
                print(f"{K:>9d} {4*K:>7d} {eager_ms:>12.3f} {'FAILED':>12s} {'-':>10s} {'-':>10s}")
            else:
                sp = eager_ms / traced_ms if traced_ms > 0 else float('nan')
                per_cycle = traced_ms / K
                print(f"{K:>9d} {4*K:>7d} {eager_ms:>12.3f} {traced_ms:>12.3f} "
                      f"{sp:>10.2f} {per_cycle:>10.3f}")

        valid = [(K, t) for K, _, t in results if t is not None]
        if len(valid) >= 2:
            per_cycle_values = [t / K for K, t in valid]
            min_p, max_p = min(per_cycle_values), max(per_cycle_values)
            print(f"\nPer-cycle traced range: {min_p:.3f}–{max_p:.3f} ms (ratio {max_p/min_p:.2f})")

        if valid:
            K_last, traced_last = valid[-1]
            per_cycle_real = traced_last / K_last
            # 16 cycles = full Qwen3.6-27B (48 DN + 16 attn = 64 transformer blocks)
            full_projection_ms = per_cycle_real * 16
            print(f"\n[Honest projection]")
            print(f"  Per-cycle traced at K={K_last}: {per_cycle_real:.3f} ms (3 DN + 1 attn, each +MLP)")
            print(f"  Full Qwen3.6 (16 cycles): {full_projection_ms:.1f} ms model forward")
            print(f"  STILL missing for real tok/s claim: embedding lookup, lm_head matmul,")
            print(f"  argmax sampling, KV cache writes during decode. Need C'7.8 to measure these.")
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:
            print(f"close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as e:
            print(f"fabric reset error: {e}")


if __name__ == "__main__":
    main()
