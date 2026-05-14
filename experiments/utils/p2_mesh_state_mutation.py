#!/usr/bin/env python3
"""
P2 — in-trace state mutation on mesh-sharded SSM-shape buffer (qb2).

Validates that ttnn.copy(new_value, existing_buffer) propagates to a
mesh-sharded tensor. server_tp.py:deltanet_step_tp relies on this pattern
to commit H_new → dn['ssm'] each decode step. If broken, DeltaNet
recurrence state never advances and the model produces garbage.

Single-chip pattern validated in feedback_trace_state_threading_works.md.
Mesh version is unproven — P2 is the probe.

Buffer shape (per chip):
  [N_V_PER_CHIP=12, K_DIM=128, V_DIM=128]  ≈ 393 KB bf16 per chip
Sharded along dim=0 (head axis) from global [N_V=48, K_DIM, V_DIM].

Step function (closed-form, easy to verify):
  H_next = H * 0.5 + 1.0
With H_0 = 0:
  H_1 = 1.0
  H_2 = 1.5
  H_3 = 1.75
  H_4 = 1.875

Three tests:
  T1 EAGER one-step: build buffer, run step, ttnn.copy(new, state), read
     back from each chip → values should be 1.0
  T2 EAGER multi-step (3 calls): verify state actually advances 1.0 → 1.5 → 1.75
  T3 TRACED (wrap step in begin/end_trace, execute_trace 3×):
     verify trace also propagates state across executions

Pass: all three tests produce expected per-chip values.

Wall: ~3-5 min (mesh open + JIT + 3 small step invocations).
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


N_V_HEADS = 48
N_V_PER_CHIP = 12
K_DIM = 128
V_DIM = 128
NCHIPS = 4


def step_eager(state_tt, mesh):
    """H_next = H * 0.5 + 1.0; commit back to state_tt in place."""
    h_decayed = ttnn.mul(state_tt, 0.5)
    h_new = ttnn.add(h_decayed, 1.0)
    ttnn.copy(h_new, state_tt)


def read_back(state_tt, mesh):
    """Read state from all chips → numpy [NCHIPS, N_V_PER_CHIP, K_DIM, V_DIM]."""
    t = ttnn.to_torch(state_tt,
                       mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return t.float().cpu().numpy()


def check(name, got_np, expected_val, tol=1e-2):
    actual_mean = got_np.mean()
    actual_min = got_np.min()
    actual_max = got_np.max()
    ok = abs(actual_mean - expected_val) < tol and abs(actual_min - expected_val) < tol
    sign = "✓" if ok else "✗"
    print(f"  {sign} {name}: expected ≈ {expected_val:.4f}  "
          f"got mean={actual_mean:.4f} min={actual_min:.4f} max={actual_max:.4f}")
    return ok


def main():
    print("=" * 78)
    print("P2: mesh-sharded state mutation (ttnn.copy) probe (qb2)")
    print("=" * 78)
    print(f"Per-chip buffer: [{N_V_PER_CHIP}, {K_DIM}, {V_DIM}] sharded along dim=0 "
          f"from global [{N_V_HEADS}, {K_DIM}, {V_DIM}]")

    print("\n[1] Init fabric + open mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    overall_pass = True
    try:
        # --- T1: EAGER one-step ---
        print("\n[2] T1 EAGER one-step (H_0 = 0 → H_1 should be 1.0)")
        state_np = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
        state_tt = ttnn.from_torch(torch.from_numpy(state_np),
                                     dtype=ttnn.bfloat16, device=mesh,
                                     layout=ttnn.TILE_LAYOUT,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        # Sanity: confirm initial state is zeros on all chips
        readback = read_back(state_tt, mesh)
        print(f"  initial state shape (concat): {readback.shape}, mean={readback.mean():.4f}")
        # Run one step
        step_eager(state_tt, mesh)
        ttnn.synchronize_device(mesh)
        readback = read_back(state_tt, mesh)
        if not check("T1", readback, expected_val=1.0):
            overall_pass = False
            print("  → ttnn.copy did NOT propagate to mesh-sharded buffer in eager mode")

        # --- T2: EAGER multi-step (3 calls) ---
        print("\n[3] T2 EAGER multi-step (3 calls; expect 1.0 → 1.5 → 1.75)")
        # Reset state
        state_np = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
        state_tt = ttnn.from_torch(torch.from_numpy(state_np),
                                     dtype=ttnn.bfloat16, device=mesh,
                                     layout=ttnn.TILE_LAYOUT,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        expected = [1.0, 1.5, 1.75]
        for i, exp in enumerate(expected, 1):
            step_eager(state_tt, mesh)
            ttnn.synchronize_device(mesh)
            readback = read_back(state_tt, mesh)
            ok_i = check(f"T2 step {i}", readback, expected_val=exp)
            if not ok_i:
                overall_pass = False
                print(f"  → state did not advance correctly at step {i}")
                break

        # --- T3: TRACED multi-step ---
        print("\n[4] T3 TRACED (capture step, execute_trace 3×; expect 1.0 → 1.5 → 1.75)")
        # Reset state
        state_np = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
        state_tt = ttnn.from_torch(torch.from_numpy(state_np),
                                     dtype=ttnn.bfloat16, device=mesh,
                                     layout=ttnn.TILE_LAYOUT,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        # Warmup (JIT must run before trace capture, per feedback_c4v4_validated)
        step_eager(state_tt, mesh)
        ttnn.synchronize_device(mesh)
        # Reset state again for the actual trace test
        state_np = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
        state_tt2 = ttnn.from_torch(torch.from_numpy(state_np),
                                      dtype=ttnn.bfloat16, device=mesh,
                                      layout=ttnn.TILE_LAYOUT,
                                      mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        # Need to warm up with the new buffer too (different tensor addresses)
        step_eager(state_tt2, mesh)
        ttnn.synchronize_device(mesh)
        # Reset state values (the warmup advanced it)
        state_zero = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
        ttnn.copy(
            ttnn.from_torch(torch.from_numpy(state_zero),
                             dtype=ttnn.bfloat16, device=mesh,
                             layout=ttnn.TILE_LAYOUT,
                             mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)),
            state_tt2,
        )
        ttnn.synchronize_device(mesh)
        readback = read_back(state_tt2, mesh)
        print(f"  pre-trace state mean={readback.mean():.4f} (should be 0.0)")

        try:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            step_eager(state_tt2, mesh)
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            print(f"  ✓ trace captured (id={trace_id})")
        except Exception as e:
            print(f"  ✗ trace capture FAILED: {type(e).__name__}: {str(e)[:300]}")
            overall_pass = False
            trace_id = None

        if trace_id is not None:
            try:
                for i, exp in enumerate(expected, 1):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    readback = read_back(state_tt2, mesh)
                    ok_i = check(f"T3 trace step {i}", readback, expected_val=exp)
                    if not ok_i:
                        overall_pass = False
                        print(f"  → trace did not propagate state at step {i}")
                        break
                ttnn.release_trace(mesh, trace_id)
            except Exception as e:
                print(f"  ✗ execute_trace FAILED: {type(e).__name__}: {str(e)[:300]}")
                overall_pass = False

        # --- Verdict ---
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        if overall_pass:
            print("  ✓ P2 PASSES — ttnn.copy threads state correctly on mesh")
            print("    (eager + traced both verified). server_tp.py:deltanet_step_tp")
            print("    state-thread pattern is viable.")
        else:
            print("  ✗ P2 PARTIAL/FAIL — state mutation pattern needs alternative")
            print("    on mesh. See above for the failing test.")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed cleanly")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
