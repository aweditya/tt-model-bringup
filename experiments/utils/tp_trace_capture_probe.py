#!/usr/bin/env python3
"""
C'7.6.1: Can we trace-capture a multi-chip TP forward on qb2?

If `ttnn.begin_trace_capture` + `ttnn.execute_trace` work on a mesh device
the way they work on a single device, we could eliminate per-step Python
dispatch overhead from TP. C'7.6 showed TP alone was slower than single-
chip — partly due to ~50 dispatched ops/layer × 48 layers × Python loop
overhead. Trace capture amortizes that.

Reference: C'4 v4 single-chip trace capture validated (cosine=1.0 across
3 traced steps, execute_trace alone 198 ms/tok = 5.5% faster than eager).
Does the same machinery work on a (1, 4) mesh device?

This probe:
  1. Open mesh, allocate persistent buffers (x, sharded weights)
  2. Run warmup eager forward (per memory note: JIT must happen before trace)
  3. ttnn.begin_trace_capture(mesh, cq_id=0)
  4. Run the DN+MLP TP chain
  5. ttnn.end_trace_capture(mesh, trace_id)
  6. ttnn.execute_trace(mesh, trace_id) — measure latency
  7. Compare to eager TP latency

Open questions tested:
  - Does begin_trace_capture accept a mesh device?
  - Does the all_reduce collective work inside the trace?
  - Is replicated/sharded input update inside the trace?
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
    print("C'7.6.1: Multi-chip TP trace capture probe (qb2)")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build 1 representative layer...")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5
        layer = build_layer_random(rng)
        # Bound magnitudes
        layer['ssm']        = rng.standard_normal(layer['ssm'].shape).astype(np.float32) * 0.05
        layer['conv_state'] = rng.standard_normal(layer['conv_state'].shape).astype(np.float32) * 0.1
        for k in ['w_in', 'w_out', 'w_gate', 'w_up', 'w_down']:
            layer[k] = layer[k] * 0.7

        sharded = upload_layer(mesh, layer)
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        print("\n[3] Warmup eager forward (3 iters; per feedback_c4v4_validated, JIT must run before trace)...")
        for _ in range(3):
            y = deltanet_tp(mesh, x_tt, sharded['dn'])
            y = mlp_tp(mesh, y, sharded['mlp'])
        ttnn.synchronize_device(mesh)
        print("  ✓ warmup done")

        # Eager latency for baseline
        N = 10
        t0 = time.perf_counter()
        for _ in range(N):
            y = deltanet_tp(mesh, x_tt, sharded['dn'])
            y = mlp_tp(mesh, y, sharded['mlp'])
        ttnn.synchronize_device(mesh)
        eager_ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"\n[4] Eager TP latency (baseline): {eager_ms:.3f} ms/block")

        # Try trace capture
        print("\n[5] Attempting ttnn.begin_trace_capture on mesh...")
        try:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            print(f"  ✓ begin_trace_capture returned trace_id={trace_id}")
        except Exception as e:
            print(f"  ✗ begin_trace_capture failed: {type(e).__name__}: {str(e)[:200]}")
            return

        try:
            print("\n[6] Running TP forward inside trace...")
            y_traced = deltanet_tp(mesh, x_tt, sharded['dn'])
            y_traced = mlp_tp(mesh, y_traced, sharded['mlp'])
            print("  ✓ TP forward inside trace ran without error")
        except Exception as e:
            print(f"  ✗ TP forward inside trace FAILED: {type(e).__name__}: {str(e)[:300]}")
            try:
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            except Exception:
                pass
            return

        try:
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            print("  ✓ end_trace_capture succeeded")
        except Exception as e:
            print(f"  ✗ end_trace_capture failed: {type(e).__name__}: {str(e)[:200]}")
            return

        # Execute the trace
        print("\n[7] execute_trace latency benchmark (N=20, warmup=5)...")
        for _ in range(5):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        traced_ms = (time.perf_counter() - t0) * 1000.0 / N

        print(f"\n  Eager TP:    {eager_ms:.3f} ms/block")
        print(f"  Traced TP:   {traced_ms:.3f} ms/block")
        speedup = eager_ms / traced_ms
        print(f"  Trace speedup: {speedup:.2f}×")

        # Project to full model: 48 DN + 64 MLP traced blocks (skip Gated Attn for now)
        # (Each "block" in this probe = 1 DN + 1 MLP)
        # If trace gives N× speedup, the full model projection improves proportionally
        new_proj = 245.0 / speedup if speedup > 1 else 245.0
        new_tok_s = 1000.0 / new_proj
        print(f"\n  C'7.6 projection was 245 ms/tok = 4.08 tok/s (eager)")
        print(f"  With trace: {new_proj:.1f} ms/tok = {new_tok_s:.2f} tok/s")
        print(f"  Single-chip baseline: 200.81 ms/tok = 4.98 tok/s")
        if new_proj < 200.81:
            print(f"  ✓ Traced TP would BEAT single-chip ({200.81 - new_proj:.1f} ms/tok faster)")
        else:
            print(f"  ✗ Traced TP still trails single-chip by {new_proj - 200.81:.1f} ms/tok")

        # Release trace
        try:
            ttnn.release_trace(mesh, trace_id)
            print("\n  ✓ trace released")
        except Exception as e:
            print(f"\n  ✗ release_trace failed: {type(e).__name__}: {str(e)[:200]}")

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
