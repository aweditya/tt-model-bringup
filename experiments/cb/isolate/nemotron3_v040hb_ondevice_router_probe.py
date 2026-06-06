#!/usr/bin/env python3
"""MM7 v0.4.0h.b — on-device DeepSeek-V3 MoE router probe.

Validates the on-device router path matches the numpy reference:

  Numpy reference:
    scores_biased = scores + bias
    topk_idxs = argpartition(-scores_biased, K)[:K]
    topk_weights = scores[topk_idxs]  # UNBIASED scores at top indices
    topk_weights /= sum(topk_weights) + 1e-20
    topk_weights *= ROUTED_SCALING

  On-device candidate:
    scores_biased_tt = ttnn.add(scores_tt, bias_tt)
    top_vals, top_idxs = ttnn.topk(scores_biased_tt, k=K, dim=-1)
    bias_at_idx = ttnn.embedding(top_idxs, bias_table)  # gather bias at idx
    weights_tt = ttnn.sub(top_vals, bias_at_idx)  # = scores at top_idx
    denom = ttnn.sum(weights_tt, dim=-1, keepdim=True) + 1e-20
    weights_tt = ttnn.div(weights_tt, denom) * ROUTED_SCALING

Gate (per Nemotron-3 shape):
  - top_idxs same SET as numpy (order may differ; we compare sets)
  - weights cos ≥ 0.999 after sorting by index

If PASS: integrate into moe_block_eager_ep_tt, eliminate scores readback +
host argpartition + topk_indices re-upload.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
S_PADDED = 8  # decode-step prompt padded
NUM_EXPERTS = 128
TOP_K = 6
ROUTED_SCALING = 2.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def numpy_reference(scores_np, bias_np):
    """[B, S, 128], [128] → (indices [B, S, K], weights [B, S, K])."""
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


def device_router(scores_tt, bias_tt, bias_table_tt, ttnn, mesh):
    """ttnn-pure equivalent. scores_tt [B, S, 128] TILE bf16 + pre-uploaded
    bias_tt [1, 1, 128] + bias_table_tt [128, 32] (padded to TILE) →
    (topk_idxs, topk_weights) both [B, S, K] TILE.

    Returns ttnn.Tensors. Caller deallocates.
    """
    scores_biased_tt = ttnn.add(scores_tt, bias_tt)
    top_vals_tt, top_idxs_tt = ttnn.topk(scores_biased_tt, k=TOP_K, dim=-1)
    ttnn.deallocate(scores_biased_tt)
    # ttnn.embedding requires UINT32 indices; ttnn.topk returns UINT16
    # by default. Cast.
    top_idxs_u32 = ttnn.typecast(top_idxs_tt, ttnn.uint32)
    # Gather bias at top_idxs: bias_table_tt is [128, embed_dim] where
    # embed_dim=32 (to make tile-aligned). The bias value lives at col 0
    # of each row; the other cols are zero. After embedding we slice col 0.
    # ttnn.embedding with [B, S, K] indices apparently collapses to
    # [B, K, embed_dim] (ignoring S). Reshape indices to 2D [B*S, K]
    # so embedding produces [B*S, K, embed_dim] cleanly.
    top_idxs_2d = ttnn.reshape(top_idxs_u32, [B * S_PADDED, TOP_K])
    ttnn.deallocate(top_idxs_u32)
    bias_at_idx_full = ttnn.embedding(top_idxs_2d, bias_table_tt)
    ttnn.deallocate(top_idxs_2d)
    log(f"  embedding output shape: {list(bias_at_idx_full.shape)}")
    # Expected shape: [B*S, K, embed_dim=32]. Slice col 0 of embed_dim.
    bias_at_idx = ttnn.slice(
        bias_at_idx_full, [0, 0, 0], [B * S_PADDED, TOP_K, 1],
    )
    ttnn.deallocate(bias_at_idx_full)
    bias_at_idx_3d = ttnn.reshape(bias_at_idx, [B, S_PADDED, TOP_K])
    weights_tt = ttnn.subtract(top_vals_tt, bias_at_idx_3d)
    ttnn.deallocate(top_vals_tt)
    ttnn.deallocate(bias_at_idx_3d)
    denom_tt = ttnn.sum(weights_tt, dim=-1, keepdim=True)
    denom_eps = ttnn.add(denom_tt, 1e-20)
    ttnn.deallocate(denom_tt)
    weights_norm = ttnn.div(weights_tt, denom_eps)
    ttnn.deallocate(weights_tt)
    ttnn.deallocate(denom_eps)
    weights_scaled = ttnn.multiply(weights_norm, ROUTED_SCALING)
    ttnn.deallocate(weights_norm)
    return top_idxs_tt, weights_scaled


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


def main(state=None) -> int:
    import ttnn

    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh ✓")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=50_000_000,
        )

    try:
        rng = np.random.default_rng(seed=99)
        scores_np = rng.uniform(0.0, 1.0, size=(B, S_PADDED, NUM_EXPERTS)).astype(np.float32)
        bias_np = rng.standard_normal((NUM_EXPERTS,), dtype=np.float32) * 0.1

        # Numpy reference
        ref_idx, ref_weights = numpy_reference(scores_np, bias_np)
        log(f"numpy ref idx shape: {ref_idx.shape}  weights shape: {ref_weights.shape}")

        # Upload scores [B, S, 128] TILE
        scores_tt = _to_tt_replicated(scores_np, ttnn, mesh, ttnn.bfloat16)

        # Upload bias as broadcast [1, 1, 128] TILE
        bias_tt = _to_tt_replicated(
            bias_np.reshape(1, 1, NUM_EXPERTS), ttnn, mesh, ttnn.bfloat16,
        )

        # Upload bias table for embedding gather: [128, embed_dim=32].
        # Place bias[i] at row i col 0; other cols zero.
        bias_table_np = np.zeros((NUM_EXPERTS, 32), dtype=np.float32)
        bias_table_np[:, 0] = bias_np
        bias_table_tt = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(bias_table_np)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # On-device router
        top_idxs_tt, weights_tt = device_router(
            scores_tt, bias_tt, bias_table_tt, ttnn, mesh,
        )
        top_idxs_np = _to_np(top_idxs_tt, ttnn, mesh).astype(np.int64)
        weights_np = _to_np(weights_tt, ttnn, mesh)
        ttnn.deallocate(top_idxs_tt)
        ttnn.deallocate(weights_tt)
        ttnn.deallocate(scores_tt)
        ttnn.deallocate(bias_tt)
        ttnn.deallocate(bias_table_tt)

        log(f"device top_idxs shape: {top_idxs_np.shape}")
        log(f"device weights shape: {weights_np.shape}")

        # Compare: indices should be same SET per (b, s) row (order may differ)
        # because argpartition and ttnn.topk may break ties differently.
        idx_match = 0
        idx_total = B * S_PADDED
        for b in range(B):
            for s in range(S_PADDED):
                ref_set = set(ref_idx[b, s].tolist())
                dev_set = set(top_idxs_np[b, s].tolist())
                if ref_set == dev_set:
                    idx_match += 1
        log(f"index sets matched: {idx_match}/{idx_total}")

        # Compare weights — sort both by index ascending, then compare.
        ref_sorted = np.zeros_like(ref_weights)
        dev_sorted = np.zeros_like(weights_np)
        for b in range(B):
            for s in range(S_PADDED):
                ref_order = np.argsort(ref_idx[b, s])
                dev_order = np.argsort(top_idxs_np[b, s])
                ref_sorted[b, s] = ref_weights[b, s, ref_order]
                dev_sorted[b, s] = weights_np[b, s, dev_order]
        cos = float(
            np.dot(ref_sorted.reshape(-1), dev_sorted.reshape(-1))
            / (np.linalg.norm(ref_sorted) * np.linalg.norm(dev_sorted) + 1e-12)
        )
        mad = float(np.mean(np.abs(ref_sorted - dev_sorted)))
        log(f"weights cos={cos:.6f}  mad={mad:.4e}")

        log("")
        log("=" * 60)
        log("REPORT")
        log("=" * 60)
        log(f"  index sets matched:  {idx_match}/{idx_total}")
        log(f"  weights cos:         {cos:.6f}  (gate 0.999)")
        log(f"  weights mad:         {mad:.4e}")
        all_pass = idx_match == idx_total and cos >= 0.999
        log("")
        log(f"v0.4.0h.b probe {'PASS ✓' if all_pass else 'FAIL ✗'}")
        return 0 if all_pass else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
