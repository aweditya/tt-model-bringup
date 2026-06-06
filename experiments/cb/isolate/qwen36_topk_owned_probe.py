#!/usr/bin/env python3
"""qwen36_topk_owned validation probe.

Forks `nemotron3_v040hc_topk_tiebreak_probe.py`.

Compares `ttnn.experimental.qwen36_topk_owned` (stable-sort fork of
`ttnn.topk`) against:
  - `ttnn.topk` baseline (unstable; same problem the v040hc probe ran)
  - `numpy.argpartition` ground truth (full fp32)
  - `numpy.argpartition` over bf16-rounded scores (what the device sees)

The probe also runs the owned op 10 times in a row on the same input and
asserts every result is bit-identical (the determinism guarantee that the
stable flag is supposed to give).

Run from the dev-harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/qwen36_topk_owned_probe'
  ssh qb1 'cat ~/tt-xla/.cache/nm3_runtime/trig/last.log'

Or standalone (slow — opens its own mesh):
  PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
    ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/cb/isolate/qwen36_topk_owned_probe.py

Pass criteria:
  - owned vs numpy fp32-ref idx_set_match  = 8/8
  - owned vs numpy bf16-ref idx_set_match  = 8/8
  - owned determinism across 10 calls       = 10/10 bit-identical
  - owned weights cos vs numpy fp32-ref     >= 0.9999
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

# Same parameters as the production Nemotron-3 router.
B = 1
S_PADDED = 8
NUM_EXPERTS = 128
TOP_K = 6
ROUTED_SCALING = 2.5

# Determinism gate: how many times to call the op back-to-back on the
# same input and check for bit-identical output.
N_DETERMINISM_CALLS = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def numpy_reference(scores_np, bias_np):
    """Full-fp32 numpy reference. argpartition breaks ties by source idx."""
    scores_biased = scores_np + bias_np[None, None, :]
    topk_idx = np.argpartition(
        -scores_biased.reshape(-1, NUM_EXPERTS), TOP_K, axis=-1
    )[:, :TOP_K].reshape(B, S_PADDED, TOP_K)
    rows = np.arange(B * S_PADDED).reshape(B, S_PADDED, 1)
    topk_weights = np.take_along_axis(
        scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
        topk_idx.reshape(B * S_PADDED, TOP_K),
        axis=-1,
    ).reshape(B, S_PADDED, TOP_K)
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom * ROUTED_SCALING
    return topk_idx, topk_weights


def bf16_round(x: np.ndarray) -> np.ndarray:
    return (
        torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
        .to(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def numpy_bf16_reference(scores_np, bias_np):
    """bf16-rounded numpy reference — the achievable ground truth given
    the device's dtype contract.
    """
    scores_bf16 = bf16_round(scores_np)
    bias_bf16 = bf16_round(bias_np)
    scores_biased_bf16 = bf16_round(scores_bf16 + bias_bf16[None, None, :])
    topk_idx = np.argpartition(
        -scores_biased_bf16.reshape(-1, NUM_EXPERTS), TOP_K, axis=-1
    )[:, :TOP_K].reshape(B, S_PADDED, TOP_K)
    return topk_idx, scores_biased_bf16


def _to_tt_replicated(arr, ttnn, mesh, dtype, layout=None):
    layout = layout or ttnn.TILE_LAYOUT
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_np(t, ttnn, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    return arr[:1].float().numpy()


def evaluate_idx_match(name, ref_idx, dev_idx):
    """Returns (idx_set_match_count, total)."""
    idx_match = 0
    idx_total = B * S_PADDED
    for b in range(B):
        for s in range(S_PADDED):
            ref_set = set(ref_idx[b, s].tolist())
            dev_set = set(dev_idx[b, s].tolist())
            if ref_set == dev_set:
                idx_match += 1
    log(f"  {name:36s} idx_set_match={idx_match}/{idx_total}")
    return idx_match, idx_total


def evaluate_weights(name, ref_idx, dev_idx, scores_np, ref_weights):
    """Gather scores at device-picked idxs, normalise, compare cosine."""
    dev_weights = np.take_along_axis(
        scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
        dev_idx.reshape(B * S_PADDED, TOP_K),
        axis=-1,
    ).reshape(B, S_PADDED, TOP_K)
    denom = dev_weights.sum(axis=-1, keepdims=True) + 1e-20
    dev_weights = dev_weights / denom * ROUTED_SCALING

    # Sort-by-idx for shape-agnostic cosine.
    ref_sorted = np.zeros_like(ref_weights)
    dev_sorted = np.zeros_like(dev_weights)
    for b in range(B):
        for s in range(S_PADDED):
            ref_order = np.argsort(ref_idx[b, s])
            dev_order = np.argsort(dev_idx[b, s])
            ref_sorted[b, s] = ref_weights[b, s, ref_order]
            dev_sorted[b, s] = dev_weights[b, s, dev_order]
    cos = float(
        np.dot(ref_sorted.reshape(-1), dev_sorted.reshape(-1))
        / (np.linalg.norm(ref_sorted) * np.linalg.norm(dev_sorted) + 1e-12)
    )
    mad = float(np.mean(np.abs(ref_sorted - dev_sorted)))
    log(f"  {name:36s} weights cos={cos:.6f}  mad={mad:.4e}")
    return cos, mad


def main(state=None) -> int:
    import ttnn

    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=50_000_000,
        )

    try:
        # Check the op is loaded.
        if not hasattr(ttnn.experimental, "qwen36_topk_owned"):
            log("FAIL: ttnn.experimental.qwen36_topk_owned not found")
            log("Have you rebuilt + copied _ttnn.so + _ttnncpp.so?")
            log("See experiments/owned_ops/qwen36_topk_owned/INTEGRATION.md")
            return 2

        rng = np.random.default_rng(seed=99)
        scores_np = rng.uniform(0.0, 1.0, size=(B, S_PADDED, NUM_EXPERTS)).astype(np.float32)
        bias_np = rng.standard_normal((NUM_EXPERTS,), dtype=np.float32) * 0.1

        ref_idx, ref_weights = numpy_reference(scores_np, bias_np)
        ref_bf16_idx, scores_biased_bf16 = numpy_bf16_reference(scores_np, bias_np)

        # Diagnostic: how many rows have a true bf16-level tie at the K-th
        # rank? Those are the rows where stable_sort matters.
        tied_rows = 0
        for b_i in range(B):
            for s_i in range(S_PADDED):
                row = scores_biased_bf16[b_i, s_i]
                sorted_vals = np.sort(row)[::-1]
                kth = sorted_vals[TOP_K - 1]
                if int(np.sum(row == kth)) > 1:
                    tied_rows += 1
        log(f"rows with EXACT bf16 tie at K-th rank: {tied_rows}/{B * S_PADDED}")

        # Upload.
        scores_tt = _to_tt_replicated(scores_np, ttnn, mesh, ttnn.bfloat16)
        bias_tt = _to_tt_replicated(
            bias_np.reshape(1, 1, NUM_EXPERTS), ttnn, mesh, ttnn.bfloat16,
        )
        scores_biased_tt = ttnn.add(scores_tt, bias_tt)

        log("")
        log("=" * 70)
        log("VARIANT EVALUATION (B*S = 8 rows, K=6, 128 experts, seed=99)")
        log("=" * 70)

        # A. baseline (unstable ttnn.topk).
        vals_a, idxs_a = ttnn.topk(scores_biased_tt, k=TOP_K, dim=-1)
        idxs_a_np = _to_np(idxs_a, ttnn, mesh).astype(np.int64)
        a_fp32 = evaluate_idx_match("A baseline   vs fp32-ref ", ref_idx, idxs_a_np)
        a_bf16 = evaluate_idx_match("A baseline   vs bf16-ref ", ref_bf16_idx, idxs_a_np)
        a_cos, _ = evaluate_weights("A baseline   weights", ref_idx, idxs_a_np, scores_np, ref_weights)
        ttnn.deallocate(vals_a); ttnn.deallocate(idxs_a)

        # E. owned (stable_sort=true).
        vals_e, idxs_e = ttnn.experimental.qwen36_topk_owned(scores_biased_tt, k=TOP_K, dim=-1)
        idxs_e_np = _to_np(idxs_e, ttnn, mesh).astype(np.int64)
        e_fp32 = evaluate_idx_match("E owned      vs fp32-ref ", ref_idx, idxs_e_np)
        e_bf16 = evaluate_idx_match("E owned      vs bf16-ref ", ref_bf16_idx, idxs_e_np)
        e_cos, _ = evaluate_weights("E owned      weights", ref_idx, idxs_e_np, scores_np, ref_weights)
        ttnn.deallocate(vals_e); ttnn.deallocate(idxs_e)

        # Determinism: call owned 10x, check all returned idxs are bit-identical
        # to the first.
        log("")
        log("=" * 70)
        log(f"DETERMINISM ({N_DETERMINISM_CALLS} consecutive calls to owned)")
        log("=" * 70)
        det_match = 0
        baseline_idxs = None
        for i in range(N_DETERMINISM_CALLS):
            v, x = ttnn.experimental.qwen36_topk_owned(scores_biased_tt, k=TOP_K, dim=-1)
            x_np = _to_np(x, ttnn, mesh).astype(np.int64)
            if baseline_idxs is None:
                baseline_idxs = x_np
                det_match += 1
            elif np.array_equal(x_np, baseline_idxs):
                det_match += 1
            ttnn.deallocate(v); ttnn.deallocate(x)
        log(f"  bit-identical: {det_match}/{N_DETERMINISM_CALLS}")

        ttnn.deallocate(scores_biased_tt)
        ttnn.deallocate(scores_tt)
        ttnn.deallocate(bias_tt)

        log("")
        log("=" * 70)
        log("REPORT")
        log("=" * 70)
        log(f"  A baseline (ttnn.topk):       {a_fp32[0]}/{a_fp32[1]} fp32-ref, {a_bf16[0]}/{a_bf16[1]} bf16-ref, cos={a_cos:.6f}")
        log(f"  E owned (qwen36_topk_owned):  {e_fp32[0]}/{e_fp32[1]} fp32-ref, {e_bf16[0]}/{e_bf16[1]} bf16-ref, cos={e_cos:.6f}")
        log(f"  determinism (owned):          {det_match}/{N_DETERMINISM_CALLS}")
        log("")

        passed = (
            e_bf16[0] == e_bf16[1]
            and e_cos >= 0.9999
            and det_match == N_DETERMINISM_CALLS
        )
        log(f"qwen36_topk_owned probe: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
