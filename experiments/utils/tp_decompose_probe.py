#!/usr/bin/env python3
"""TP per-step decomposition probe on qb2.

Goal: explain why 4-chip TP traced gives 7.02 tok/s (142 ms/tok) instead of
the naive 1.20 ms/block x 64 = 76.8 ms compute-only projection (~13 tok/s).

Decomposition: each decode step does:
  1. update_input_buffers - host writes 4x from_torch + 4x copy_host_to_device_tensor
                            (token embedding, cur_pos, cos, sin) onto a replicated
                            (1, 4) mesh.
  2. execute_trace        - device-side N-layer forward.
  3. synchronize_device   - wait for trace to finish.
  4. (next iter only)      to_torch(logits) - read 152064-elem fp32 logits from chip 0.

This probe builds a K=8-block traced TP forward (DN+MLP per block - same pattern as
tp_chained_traced_probe with K_MAX=8) and measures each component independently with
sync barriers. Extrapolates compute linearly (K-scaling proven flat in
feedback_tp_trace_scales_linearly.md); measures non-compute per-step overhead
directly (it does NOT scale with K).

Output: a numeric breakdown matching the 142 ms/tok envelope.

Usage:
  ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/tp_decompose_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.path.insert(0, os.path.expanduser("~/tt-xla/experiments/utils"))
from full_layer_tp_probe import HIDDEN, deltanet_tp, mlp_tp
from tp_chain_scaling_probe import build_layer_random, upload_layer

sys.stdout.reconfigure(line_buffering=True)


K_PROBE = 8                  # chained DN+MLP layers in the trace
K_FULL  = 64                 # production model total transformer blocks
N_LOGITS = 152064            # vocab; lm_head readback shape
N_TIMING_ITERS = 30          # iters to average each component over
WARMUP_TRACE = 5


def time_op(fn, n_iter, mesh):
    """Call fn() n_iter times with sync barriers before+after; return (median, mean, std) ms."""
    samples = []
    for _ in range(n_iter):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples)), float(np.mean(samples)), float(np.std(samples))


def main():
    print("=" * 78)
    print("TP per-step decomposition probe - qb2 (1,4) mesh")
    print(f"K_PROBE={K_PROBE} chained DN+MLP blocks; K_FULL={K_FULL} for extrapolation")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("\n[1] Open (1,4) mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ok {mesh.get_num_devices()} chips")

    try:
        print(f"\n[2] Build + upload K={K_PROBE} random DN+MLP blocks (real Qwen3.6 shapes)...")
        rng = np.random.default_rng(42)
        layers = []
        for i in range(K_PROBE):
            L = build_layer_random(rng)
            L['ssm']        = rng.standard_normal(L['ssm'].shape).astype(np.float32) * 0.05
            L['conv_state'] = rng.standard_normal(L['conv_state'].shape).astype(np.float32) * 0.1
            for k in ['w_in', 'w_out', 'w_gate', 'w_up', 'w_down']:
                L[k] = L[k] * 0.7
            layers.append(L)
        sharded = [upload_layer(mesh, L) for L in layers]

        # x_buf - pre-allocated replicated input (mirrors server_tp.state.x_buf)
        x_np = (rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5)
        x_buf = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                 device=mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def forward_K():
            cur = x_buf
            for j in range(K_PROBE):
                cur = deltanet_tp(mesh, cur, sharded[j]['dn'])
                cur = mlp_tp(mesh, cur, sharded[j]['mlp'])
            return cur

        print("\n[3] Warmup eager forwards (JIT)...")
        for _ in range(3):
            _ = forward_K()
        ttnn.synchronize_device(mesh)

        print(f"\n[4] Capture trace of K={K_PROBE} block forward...")
        t0 = time.perf_counter()
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        out_tt = forward_K()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        print(f"  ok captured in {(time.perf_counter()-t0)*1000:.1f} ms")

        print(f"\n[5] Trace warmup (x{WARMUP_TRACE})...")
        for _ in range(WARMUP_TRACE):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)

        # === Component A: execute_trace + sync (compute) ===
        print(f"\n[6] Measure component A: execute_trace + sync (K={K_PROBE} compute)...")
        med_A, mean_A, std_A = time_op(
            lambda: ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False),
            N_TIMING_ITERS, mesh=mesh)
        per_block_A = med_A / K_PROBE
        proj_A_full = per_block_A * K_FULL
        print(f"  median {med_A:.3f} ms (mean {mean_A:.3f}, sigma {std_A:.3f})")
        print(f"  per-block: {per_block_A:.4f} ms -> K={K_FULL} extrap: {proj_A_full:.2f} ms")

        # === Component B: synchronize_device alone (sync overhead) ===
        print(f"\n[7] Measure component B: synchronize_device alone (idle barrier)...")
        med_B, mean_B, std_B = time_op(lambda: None, N_TIMING_ITERS, mesh=mesh)
        print(f"  median {med_B:.3f} ms (mean {mean_B:.3f}, sigma {std_B:.3f})")
        print(f"  (this is the cost of the trailing sync barrier alone)")

        # === Component C: update_input_buffers cost ===
        # Mirror server_tp.update_input_buffers: 4x from_torch+copy_host_to_device.
        print(f"\n[8] Measure component C: update_input_buffers (4x from_torch + copy_host_to_device_tensor)...")
        embed_buf = ttnn.from_torch(torch.zeros(1, HIDDEN, dtype=torch.float32),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=mesh,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        curpos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32),
                                      dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=mesh,
                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        HEAD_DIM = 256  # Qwen3.6 head dim
        cos_buf = ttnn.from_torch(torch.zeros(1, HEAD_DIM, dtype=torch.float32),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=mesh,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sin_buf = ttnn.from_torch(torch.zeros(1, HEAD_DIM, dtype=torch.float32),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=mesh,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        embed_src = np.random.randn(1, HIDDEN).astype(np.float32)
        cos_src   = np.random.randn(1, HEAD_DIM).astype(np.float32)
        sin_src   = np.random.randn(1, HEAD_DIM).astype(np.float32)

        def do_update():
            embed_host = ttnn.from_torch(torch.from_numpy(embed_src),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(embed_host, embed_buf)
            cp_host = ttnn.from_torch(torch.tensor([0], dtype=torch.int32),
                                       dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(cp_host, curpos_buf)
            cos_host = ttnn.from_torch(torch.from_numpy(cos_src),
                                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(cos_host, cos_buf)
            sin_host = ttnn.from_torch(torch.from_numpy(sin_src),
                                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(sin_host, sin_buf)

        med_C, mean_C, std_C = time_op(do_update, N_TIMING_ITERS, mesh=mesh)
        print(f"  median {med_C:.3f} ms (mean {mean_C:.3f}, sigma {std_C:.3f})")
        print(f"  (4 host->device copies per step, replicated 4-way)")

        # === Component D: to_torch(logits) - read 152064 fp32 from chip 0 ===
        print(f"\n[9] Measure component D: to_torch(logits) readback (152064 fp32)...")
        med_D_small, mean_D_small, std_D_small = time_op(
            lambda: ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)),
            N_TIMING_ITERS, mesh=mesh)
        # Build a logits-shaped tensor (replicated) to get realistic readback.
        logits_buf = ttnn.from_torch(torch.zeros(1, N_LOGITS, dtype=torch.float32),
                                      dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=mesh,
                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        med_D, mean_D, std_D = time_op(
            lambda: ttnn.to_torch(logits_buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)),
            N_TIMING_ITERS, mesh=mesh)
        print(f"  HIDDEN-wide ({HIDDEN}) median: {med_D_small:.3f} ms")
        print(f"  LOGITS-wide ({N_LOGITS}) median: {med_D:.3f} ms (mean {mean_D:.3f}, sigma {std_D:.3f})")

        # === Component E: just argmax() over the cpu logits (for completeness)
        print(f"\n[10] Measure component E: numpy argmax over composed logits (host)...")
        composed = ttnn.to_torch(logits_buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        composed_np = composed.float().cpu().numpy().reshape(-1)[:N_LOGITS]
        def do_argmax():
            _ = int(np.argmax(composed_np))
        med_E, mean_E, std_E = time_op(do_argmax, N_TIMING_ITERS, mesh=mesh)
        print(f"  median {med_E:.3f} ms (CPU work; usually trivial)")

        # === Full step: update + execute + sync (what server times) ===
        print(f"\n[11] Measure FULL per-step (update + execute_trace + sync)...")
        def full_step():
            do_update()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        med_F, mean_F, std_F = time_op(full_step, N_TIMING_ITERS, mesh=mesh)
        print(f"  median {med_F:.3f} ms")

        # === Summary ===
        print("\n" + "=" * 78)
        print("DECOMPOSITION SUMMARY (per decode step)")
        print("=" * 78)
        print(f"  K={K_PROBE} chained DN+MLP TP trace:")
        print(f"    A  execute_trace + sync          : {med_A:.3f} ms")
        print(f"    B  pure sync barrier             : {med_B:.3f} ms")
        print(f"    A-B  execute_trace dispatch      : {med_A - med_B:.3f} ms")
        print(f"    C  update_input_buffers          : {med_C:.3f} ms")
        print(f"    D  logits to_torch (152064)      : {med_D:.3f} ms")
        print(f"    E  numpy argmax (host)           : {med_E:.3f} ms")
        print(f"    F  full step (C + A)             : {med_F:.3f} ms")
        print()
        print(f"  Per-block traced compute (A-B)/K : {(med_A - med_B)/K_PROBE:.4f} ms/block")
        print()
        print(f"  Projection to {K_FULL}-block model:")
        per_block_compute = (med_A - med_B) / K_PROBE
        proj_compute_64 = per_block_compute * K_FULL
        proj_step = med_C + proj_compute_64 + med_B + med_D
        print(f"    compute (per-block x {K_FULL})  : {proj_compute_64:.2f} ms")
        print(f"  + update_input_buffers           : {med_C:.2f} ms")
        print(f"  + sync barrier                   : {med_B:.2f} ms")
        print(f"  + logits readback                : {med_D:.2f} ms")
        print(f"  --------")
        print(f"  predicted ms/tok                 : {proj_step:.2f} ms")
        print(f"  predicted tok/s                  : {1000/proj_step:.2f}")
        print()
        print(f"  measured baseline (commit 9369e1b): 142.36 ms/tok = 7.02 tok/s")
        print(f"  predicted - measured gap         : {proj_step - 142.36:+.2f} ms")
        print()
        print(f"  HEADLINE ATTRIBUTION (assuming linear K scaling):")
        share_compute = proj_compute_64 / proj_step * 100
        share_update  = med_C / proj_step * 100
        share_sync    = med_B / proj_step * 100
        share_logits  = med_D / proj_step * 100
        print(f"    compute     {proj_compute_64:6.2f} ms ({share_compute:5.1f}%)")
        print(f"    update      {med_C:6.2f} ms ({share_update:5.1f}%)")
        print(f"    sync        {med_B:6.2f} ms ({share_sync:5.1f}%)")
        print(f"    logits      {med_D:6.2f} ms ({share_logits:5.1f}%)")

        ttnn.release_trace(mesh, trace_id)
    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ok mesh closed")
        except Exception as e:
            print(f"  err close: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ok fabric reset")
        except Exception as e:
            print(f"  err fabric reset: {e}")


if __name__ == "__main__":
    main()
