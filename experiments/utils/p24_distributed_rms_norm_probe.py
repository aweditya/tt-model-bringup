#!/usr/bin/env python3
"""
P24: Distributed RMSNorm probe on (1,4) mesh.

GOAL — verify two-version equivalence + measure latency:
  V1 (current prod): ttnn.rms_norm(replicated_x, weight=replicated_w, eps)
       — each chip redundantly computes full reduction.
  V2 (target):
       stats = ttnn.rms_norm_pre_all_gather(fractured_x)   # per-chip partial stats
       gath  = ttnn.all_gather(stats, dim=-1)              # ~1 tile payload
       out   = ttnn.rms_norm_post_all_gather(fractured_x, gath, weight=fractured_w, eps)
       — each chip reduces HIDDEN/4 only.

PASS criteria:
  - V1 vs V2 cosine ≥ 0.999 (after gathering V2's width-fractured output).
  - Both V1 and V2 cos ≥ 0.999 vs numpy fp64 oracle.
  - V2 latency ≤ V1 latency + 1 ms (architectural change without measurable win
    is not worth shipping at this scope).

References:
  - Galaxy recipe: experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390
  - Integration plan: research/integration_distributed_rms_norm.md
  - Correction note: feedback_distributed_rms_norm_corrections.md (DO NOT use
    residual_input_tensor= for residual fusion; semantics are wrong).

Workload: Qwen3.6-27B HIDDEN=5120 (per-chip 1280); also test final_norm-shape
(same HIDDEN). The narrow per-head norms (HEAD_DIM=128) are EXCLUDED — they
won't amortize the AG fixed cost per W's analysis.

Output: .cache/p24_distributed_rms_norm/results.json + log.

Run on qb2 (mesh — server must be stopped):
    cd ~/tt-xla && .venv/bin/python experiments/utils/p24_distributed_rms_norm_probe.py
"""
import os
import sys
import json
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


HIDDEN = 5120
NCHIPS = 4
EPS = 1e-6
N_TRIALS = 5     # number of random hidden states + weights to compare
N_BENCH = 30     # benchmark iterations


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def numpy_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """fp64 oracle: y = x / sqrt(mean(x^2) + eps) * w."""
    x64 = x.astype(np.float64)
    w64 = w.astype(np.float64)
    var = (x64 ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x64 * inv * w64).astype(np.float32)


def v2_distributed(x_fractured, w_fractured, eps):
    """V2: pre_all_gather → all_gather(stats) → post_all_gather.

    x_fractured + w_fractured are per-chip [..., HIDDEN/4] tensors on a (1,4) mesh.
    Returns a per-chip [..., HIDDEN/4] tensor (fractured output).
    """
    stats = ttnn.rms_norm_pre_all_gather(x_fractured, dtype=ttnn.bfloat16)
    gathered = ttnn.all_gather(stats, dim=3)
    out = ttnn.rms_norm_post_all_gather(
        x_fractured, gathered, epsilon=eps, weight=w_fractured)
    return out


def main():
    out_dir = os.path.expanduser("~/tt-xla/.cache/p24_distributed_rms_norm")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    results_path = os.path.join(out_dir, "results.json")

    print(f"=== P24: distributed RMSNorm probe (HIDDEN={HIDDEN}, NCHIPS={NCHIPS}) ===")
    print(f"Output dir: {out_dir}")
    print()

    # === Init fabric + open mesh ===
    print("[setup] FABRIC_1D + open_mesh_device(1,4)…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"  ✓ mesh opened, n_devices={mesh.get_num_devices()}")

    results = {
        "trials": [],
        "bench": {},
    }

    try:
        rng = np.random.default_rng(42)

        for trial in range(N_TRIALS):
            print(f"\n--- trial {trial} ---")
            # Random hidden state + weight (same scale as Qwen3.6 RMSNorm acts on)
            # NOTE: rms_norm_pre_all_gather requires 4D input (it accesses shape[3])
            # so shape data as [1, 1, 1, HIDDEN]. The norm dim is the last (HIDDEN).
            x_np = rng.standard_normal((1, 1, 1, HIDDEN)).astype(np.float32) * 0.5
            w_np = (rng.standard_normal((HIDDEN,)).astype(np.float32) * 0.02
                    + 1.0).astype(np.float32)  # weights cluster near 1.0

            # Numpy oracle (operate on the last axis)
            oracle = numpy_rms_norm(x_np, w_np, EPS)

            # V1: replicated x + replicated w + ttnn.rms_norm (current prod path)
            x_rep = ttnn.from_torch(
                torch.from_numpy(x_np),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            w_rep = ttnn.from_torch(
                torch.from_numpy(w_np.reshape(1, 1, 1, HIDDEN)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

            v1_tt = ttnn.rms_norm(x_rep, weight=w_rep, epsilon=EPS)
            # All chips agree post-V1 (replicated). Read one chip's view.
            v1_concat = ttnn.to_torch(
                v1_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().cpu().numpy()
            # Shape per-chip [1,1,1,HIDDEN], concatenated on dim=0 → [NCHIPS,1,1,HIDDEN]
            v1_chip0 = v1_concat[0:1].reshape(1, 1, 1, HIDDEN)
            chip_max_drift = float(
                np.abs(v1_concat - v1_concat[0:1]).max())

            cos_v1_oracle = cosine(v1_chip0, oracle)

            # V2: fractured x + fractured w + distributed pre/post pattern
            x_frac = ttnn.from_torch(
                torch.from_numpy(x_np),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
            w_frac = ttnn.from_torch(
                torch.from_numpy(w_np.reshape(1, 1, 1, HIDDEN)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))

            v2_tt = v2_distributed(x_frac, w_frac, EPS)
            # v2_tt is fractured → concat across mesh on dim=-1
            v2_full_torch = ttnn.to_torch(
                v2_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)
            ).float().cpu().numpy()
            # Shape is [1, 1, 1, HIDDEN] (chips concatenated along dim=-1)
            v2_full = v2_full_torch.reshape(1, 1, 1, HIDDEN)

            cos_v2_oracle = cosine(v2_full, oracle)
            cos_v1_v2 = cosine(v1_chip0, v2_full)
            max_abs_v1_v2 = float(np.abs(v1_chip0 - v2_full).max())

            trial_res = {
                "trial": trial,
                "cos_v1_vs_oracle": cos_v1_oracle,
                "cos_v2_vs_oracle": cos_v2_oracle,
                "cos_v1_vs_v2": cos_v1_v2,
                "max_abs_v1_v2": max_abs_v1_v2,
                "v1_chip_max_drift": chip_max_drift,
            }
            results["trials"].append(trial_res)
            print(f"  cos V1 vs oracle: {cos_v1_oracle:.6f}")
            print(f"  cos V2 vs oracle: {cos_v2_oracle:.6f}")
            print(f"  cos V1 vs V2:     {cos_v1_v2:.6f}  (max|Δ|={max_abs_v1_v2:.4e})")
            print(f"  V1 chip-to-chip drift: {chip_max_drift:.4e}")

            # Cleanup
            ttnn.deallocate(v1_tt)
            ttnn.deallocate(v2_tt)
            ttnn.deallocate(x_rep)
            ttnn.deallocate(w_rep)
            ttnn.deallocate(x_frac)
            ttnn.deallocate(w_frac)

        # === Bench V1 vs V2 ===
        print("\n--- bench (median over 30 iters, fixed input) ---")
        x_np = rng.standard_normal((1, 1, 1, HIDDEN)).astype(np.float32) * 0.5
        w_np = (rng.standard_normal((HIDDEN,)).astype(np.float32) * 0.02
                + 1.0).astype(np.float32)

        x_rep = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        w_rep = ttnn.from_torch(
            torch.from_numpy(w_np.reshape(1, 1, 1, HIDDEN)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_frac = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        w_frac = ttnn.from_torch(
            torch.from_numpy(w_np.reshape(1, 1, 1, HIDDEN)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))

        # Warmup
        for _ in range(5):
            _ = ttnn.rms_norm(x_rep, weight=w_rep, epsilon=EPS)
            _ = v2_distributed(x_frac, w_frac, EPS)
        ttnn.synchronize_device(mesh)

        # V1 bench
        v1_times = []
        for _ in range(N_BENCH):
            t0 = time.perf_counter()
            o1 = ttnn.rms_norm(x_rep, weight=w_rep, epsilon=EPS)
            ttnn.synchronize_device(mesh)
            v1_times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(o1)

        # V2 bench
        v2_times = []
        for _ in range(N_BENCH):
            t0 = time.perf_counter()
            o2 = v2_distributed(x_frac, w_frac, EPS)
            ttnn.synchronize_device(mesh)
            v2_times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(o2)

        v1_med = float(np.median(v1_times))
        v1_p10 = float(np.percentile(v1_times, 10))
        v2_med = float(np.median(v2_times))
        v2_p10 = float(np.percentile(v2_times, 10))

        results["bench"] = {
            "v1_median_ms": v1_med,
            "v1_p10_ms": v1_p10,
            "v2_median_ms": v2_med,
            "v2_p10_ms": v2_p10,
            "delta_median_ms": v2_med - v1_med,
        }
        print(f"  V1 median: {v1_med:.4f} ms  (p10 {v1_p10:.4f})")
        print(f"  V2 median: {v2_med:.4f} ms  (p10 {v2_p10:.4f})")
        print(f"  delta (V2 - V1): {v2_med - v1_med:+.4f} ms")

        # === Verdict ===
        print("\n=== verdict ===")
        all_cos = [t["cos_v1_vs_v2"] for t in results["trials"]]
        worst_cos = min(all_cos)
        v1_oracle_all = [t["cos_v1_vs_oracle"] for t in results["trials"]]
        v2_oracle_all = [t["cos_v2_vs_oracle"] for t in results["trials"]]

        pass_correctness = worst_cos >= 0.999
        pass_latency = (v2_med - v1_med) <= 1.0

        results["pass_correctness"] = pass_correctness
        results["pass_latency"] = pass_latency
        results["worst_cos_v1_v2"] = worst_cos
        results["worst_cos_v1_oracle"] = min(v1_oracle_all)
        results["worst_cos_v2_oracle"] = min(v2_oracle_all)

        print(f"  worst V1 vs V2 cos:    {worst_cos:.6f}  (pass ≥ 0.999: {pass_correctness})")
        print(f"  worst V1 vs oracle:    {min(v1_oracle_all):.6f}")
        print(f"  worst V2 vs oracle:    {min(v2_oracle_all):.6f}")
        print(f"  latency Δ (V2 - V1):   {v2_med - v1_med:+.4f} ms  (pass ≤ 1.0: {pass_latency})")

        if pass_correctness and pass_latency:
            print("  ✓ PROBE PASSES — proceed to Phase 2 / Phase 3 ship.")
        elif pass_correctness:
            print("  ⚠ correctness OK but no latency win — DO NOT SHIP this scope.")
        else:
            print("  ✗ CORRECTNESS FAIL — investigate before any ship.")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"  ⚠ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as e:
            print(f"  ⚠ fabric reset error: {e}")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  results → {results_path}")


if __name__ == "__main__":
    main()
