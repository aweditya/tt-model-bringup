#!/usr/bin/env python3
"""Isolated test for async all_reduce overlap with shared-expert compute.

The 35B MoE batched forward currently serializes:
    routed_local = matmul(rw, expert_out_2d)
    routed_full  = all_reduce(routed_local)              # BLOCKS
    gated_shared = _moe_shared_expert(h_tt, w, mesh)     # 4 matmuls + 1 all_reduce
    final        = add(routed_full, gated_shared)

If `routed_full = ttnn.experimental.all_reduce_async(...)` is fired first and
we wait at `add`, the shared-expert compute window can hide the reduce. Memory
note `feedback_async_ccl_negative` says async lost on 27B's serial residual
stream; here the shared-expert path is independent compute.

Semaphore-pool pattern + signature copied verbatim from
experiments/serve/server_tp.py:handle_probe_async_ccl_components_tp.

Gate: cos(serial, async) > 0.9999 AND async ms/iter < serial.

Run (qb1): see HANDOFF.md for env-var bootstrap; then
  .venv/bin/python -u experiments/utils/test_async_all_reduce_overlap_isolated.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

HIDDEN = 2048
NCHIPS = 4
N_ITERS = 50
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_np(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_ttnn_replicated(arr_np, mesh):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build_sem_pool(mesh, n_sets=4):
    """Build n_sets of (barrier×2, rs×3, ag×2) global semaphores. Same shape
    as server_tp.handle_probe_async_ccl_components_tp uses for double-buffered
    back-to-back launches."""
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
    })
    make_sem = lambda: ttnn.create_global_semaphore(mesh, cores, 0)
    return [
        {
            "barrier": [make_sem() for _ in range(2)],
            "rs":      [make_sem() for _ in range(3)],
            "ag":      [make_sem() for _ in range(2)],
        }
        for _ in range(n_sets)
    ]


def ar_sync(x):
    return ttnn.all_reduce(
        x, cluster_axis=1, memory_config=x.memory_config(),
        num_links=2, topology=ttnn.Topology.Ring,
    )


def ar_async(x, mesh, sem_set):
    return ttnn.experimental.all_reduce_async(
        x,
        cluster_axis=1, mesh_device=mesh,
        barrier_semaphores=sem_set["barrier"],
        rs_global_semaphores=sem_set["rs"],
        ag_global_semaphores=sem_set["ag"],
        math_op=ttnn.ReduceType.Sum,
        num_links=2,
        memory_config=x.memory_config(),
        topology=ttnn.Topology.Linear,
    )


def main():
    rng = np.random.default_rng(0)
    h_np = rng.normal(0, 1.0, size=(1, HIDDEN)).astype(np.float32)
    W_routed_np = rng.normal(0, 0.05, size=(HIDDEN, HIDDEN)).astype(np.float32)
    W_shared_np = [rng.normal(0, 0.05, size=(HIDDEN, HIDDEN)).astype(np.float32) for _ in range(4)]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        log(f"mesh open: {mesh}")
        sems = build_sem_pool(mesh)
        ttnn.synchronize_device(mesh)
        log(f"semaphore pool ready ({len(sems)} sets)")

        h_tt = to_ttnn_replicated(h_np, mesh)
        W_routed_tt = to_ttnn_replicated(W_routed_np, mesh)
        W_shared_tt = [to_ttnn_replicated(W, mesh) for W in W_shared_np]

        def serial_seq():
            R = ttnn.matmul(h_tt, W_routed_tt, compute_kernel_config=HIFI4)
            R_full = ar_sync(R)
            ttnn.deallocate(R)
            x = h_tt
            for W in W_shared_tt[:-1]:
                m = ttnn.matmul(x, W, compute_kernel_config=HIFI4)
                ttnn.deallocate(m)
            S_partial = ttnn.matmul(x, W_shared_tt[-1], compute_kernel_config=HIFI4)
            S_full = ar_sync(S_partial)
            ttnn.deallocate(S_partial)
            out = ttnn.add(R_full, S_full)
            ttnn.deallocate(R_full); ttnn.deallocate(S_full)
            return out

        def async_seq(sem_idx):
            R = ttnn.matmul(h_tt, W_routed_tt, compute_kernel_config=HIFI4)
            R_full_fut = ar_async(R, mesh, sems[sem_idx])
            ttnn.deallocate(R)
            x = h_tt
            for W in W_shared_tt[:-1]:
                m = ttnn.matmul(x, W, compute_kernel_config=HIFI4)
                ttnn.deallocate(m)
            S_partial = ttnn.matmul(x, W_shared_tt[-1], compute_kernel_config=HIFI4)
            S_full = ar_sync(S_partial)
            ttnn.deallocate(S_partial)
            out = ttnn.add(R_full_fut, S_full)
            ttnn.deallocate(R_full_fut); ttnn.deallocate(S_full)
            return out

        log("warmup 3 of each…")
        for _ in range(3):
            ttnn.deallocate(serial_seq())
        for i in range(3):
            ttnn.deallocate(async_seq(i % len(sems)))
        ttnn.synchronize_device(mesh)

        log("correctness check…")
        out_serial = serial_seq()
        out_async = async_seq(0)
        ttnn.synchronize_device(mesh)
        s_np = ttnn.to_torch(
            out_serial, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()[0]
        a_np = ttnn.to_torch(
            out_async, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()[0]
        ttnn.deallocate(out_serial); ttnn.deallocate(out_async)
        c = cos_np(s_np, a_np)
        log(f"cos(serial, async) = {c:.8f}")
        if c < 0.9999:
            log(f"FAIL: cos {c:.6f} < 0.9999")
            return

        def time_fn(label, fn):
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for i in range(N_ITERS):
                out = fn(i) if fn.__code__.co_argcount > 0 else fn()
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            ms = (time.time() - t0) * 1000.0 / N_ITERS
            log(f"  {label}: {ms:.3f} ms/iter")
            return ms

        t_serial = time_fn("serial", lambda: serial_seq())
        # alternate semaphore sets to avoid sem-counter contention back-to-back
        t_async = time_fn("async ", lambda i: async_seq(i % len(sems)))
        delta_pct = 100.0 * (t_serial - t_async) / t_serial
        log(f"delta: {delta_pct:+.1f}%  ({t_serial:.3f} -> {t_async:.3f} ms)")
        if t_async < t_serial:
            log("PASS  async wins.")
        else:
            log("REJECT  async did not win — comm/compute overlap insufficient.")

        ttnn.deallocate(h_tt); ttnn.deallocate(W_routed_tt)
        for W in W_shared_tt:
            ttnn.deallocate(W)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
