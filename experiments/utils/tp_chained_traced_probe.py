#!/usr/bin/env python3
"""
TP chained traced forward — real multi-chip latency scaling probe on qb2.

The honest question: our 1.21 ms/block measurement (C'7.6.1 at real shapes)
was for ONE DN+MLP block. Extrapolating to 64 transformer blocks gave us
the "16 tok/s projection" that the user correctly called out as not real.

This probe gives a REAL number for chained traced forward:
  - Build K different random-weight blocks (at real Qwen3.6 shapes)
  - Chain them inside a single trace (state threads through residual stream)
  - Measure execute_trace latency
  - Compare to K × 1.21 ms (the naive extrapolation)
  - Find if trace has fixed overhead, memory limits, dispatch jitter

K ∈ {1, 2, 4, 8} gives us a scaling curve. If linear, the per-block
measurement does extrapolate. If superlinear (e.g. trace memory pressure)
or with a high fixed cost, we'll know.

Output: actual ms per chain depth K → projection arithmetic for full model
is now grounded in a chain measurement, not single-block.

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/tp_chained_traced_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.path.insert(0, os.path.expanduser("~/tt-xla/experiments/utils"))
from full_layer_tp_probe import (
    HIDDEN, deltanet_tp, mlp_tp,
)
from tp_chain_scaling_probe import build_layer_random, upload_layer

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("Chained traced TP probe — real scaling measurement on qb2")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build K=8 distinct random layers (real Qwen3.6 shapes)...")
        rng = np.random.default_rng(42)
        # Bounded magnitudes so state doesn't blow up over 8 blocks
        K_MAX = 8
        layers = []
        for i in range(K_MAX):
            L = build_layer_random(rng)
            L['ssm']        = rng.standard_normal(L['ssm'].shape).astype(np.float32) * 0.05
            L['conv_state'] = rng.standard_normal(L['conv_state'].shape).astype(np.float32) * 0.1
            for k in ['w_in', 'w_out', 'w_gate', 'w_up', 'w_down']:
                L[k] = L[k] * 0.7
            layers.append(L)

        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5

        print(f"\n[3] Upload all {K_MAX} layers sharded...")
        sharded_layers = [upload_layer(mesh, L) for L in layers]
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def forward_K(x_in, K):
            cur = x_in
            for j in range(K):
                cur = deltanet_tp(mesh, cur, sharded_layers[j]['dn'])
                cur = mlp_tp(mesh, cur, sharded_layers[j]['mlp'])
            return cur

        print("\n[4] Warmup (K=1 eager, 3 iters; per feedback_c4v4_validated JIT must run before trace)...")
        for _ in range(3):
            _ = forward_K(x_tt, 1)
        ttnn.synchronize_device(mesh)
        # Also warm up for larger K (different op fingerprint = different JIT)
        for _ in range(2):
            _ = forward_K(x_tt, 4)
            _ = forward_K(x_tt, 8)
        ttnn.synchronize_device(mesh)
        print("  ✓ warmup done")

        # Measure eager + traced for each K
        results = []
        for K in [1, 2, 4, 8]:
            print(f"\n[K={K}] Measure eager + traced...")

            # Eager
            N_eager = max(5, 20 // K)  # fewer iters for higher K (each iter is bigger)
            t0 = time.perf_counter()
            for _ in range(N_eager):
                _ = forward_K(x_tt, K)
            ttnn.synchronize_device(mesh)
            eager_ms = (time.perf_counter() - t0) * 1000.0 / N_eager

            # Traced
            try:
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                _ = forward_K(x_tt, K)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            except Exception as e:
                print(f"  ✗ trace capture for K={K} FAILED: {type(e).__name__}: {str(e)[:300]}")
                results.append((K, eager_ms, None))
                continue

            # Warmup trace
            for _ in range(3):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            N_trace = 20
            t0 = time.perf_counter()
            for _ in range(N_trace):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            traced_ms = (time.perf_counter() - t0) * 1000.0 / N_trace

            try:
                ttnn.release_trace(mesh, trace_id)
            except Exception as e:
                print(f"  (release_trace warning: {e})")

            speedup = eager_ms / traced_ms if traced_ms > 0 else float('nan')
            per_block_traced = traced_ms / K
            print(f"  K={K}: eager={eager_ms:.3f} ms, traced={traced_ms:.3f} ms, "
                  f"speedup={speedup:.2f}×, per-block traced={per_block_traced:.3f} ms")
            results.append((K, eager_ms, traced_ms))

        print("\n" + "=" * 78)
        print("REAL SCALING RESULTS")
        print("=" * 78)
        print(f"{'K':>4s} {'eager (ms)':>12s} {'traced (ms)':>12s} {'speedup':>10s} {'traced/K (ms)':>14s}")
        for K, eager_ms, traced_ms in results:
            if traced_ms is None:
                print(f"{K:>4d} {eager_ms:>12.3f} {'FAILED':>12s} {'-':>10s} {'-':>14s}")
            else:
                speedup = eager_ms / traced_ms if traced_ms > 0 else float('nan')
                per_block = traced_ms / K
                print(f"{K:>4d} {eager_ms:>12.3f} {traced_ms:>12.3f} {speedup:>10.2f} "
                      f"{per_block:>14.3f}")

        # Linear-fit check: if traced/K is constant, scaling is linear.
        valid = [(K, t) for K, _, t in results if t is not None]
        if len(valid) >= 2:
            per_block_traced_values = [t / K for K, t in valid]
            min_per = min(per_block_traced_values)
            max_per = max(per_block_traced_values)
            ratio = max_per / min_per
            print(f"\nPer-block traced ms range: {min_per:.3f} – {max_per:.3f} (ratio {ratio:.2f})")
            print(f"If ratio ~1.0, scaling is linear → can project to full model.")
            print(f"If ratio >> 1.0, there's overhead/dispatch behavior we're missing.")

        # Real projection (if linear): use K=8 per-block traced cost × 64 transformer blocks
        if valid and valid[-1][1] is not None:
            K8, traced_K8 = valid[-1]
            per_block_real = traced_K8 / K8
            # Each Qwen3.6 'transformer block' = DN-OR-attn + MLP. 64 transformer blocks total.
            # This probe only chains DN+MLP, not attn+MLP. So projection is rough.
            print(f"\n[Honest projection caveat]")
            print(f"  This probe only chained DN+MLP. Real model = 48 DN+MLP + 16 attn+MLP.")
            print(f"  Per-block traced at K={K8}: {per_block_real:.3f} ms (DN+MLP block).")
            print(f"  Naive 64-block extrapolation: {per_block_real * 64:.1f} ms/tok = "
                  f"{1000 / (per_block_real * 64):.2f} tok/s.")
            print(f"  Still NOT a real tok/s — missing: Gated Attn TP, embedding, lm_head, sampling.")
            print(f"  The HONEST number requires C'7.8 (multi-chip persistent server).")

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
