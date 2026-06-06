#!/usr/bin/env python3
"""MM7 v0.4.1.d — does all_to_all_dispatch tolerate replicated input?

The reshard probe (v0.4.1.c) showed no ttnn primitive converts
replicated→sharded on-device. Option B from that probe's commit:
just pass the REPLICATED h directly to all_to_all_dispatch and see
what happens. If it works, we eliminate the trace blocker entirely
(at the cost of 4x bandwidth on the dispatch input — which is small
since h is only [1, S=8, 2688] bf16 = ~43 KB).

This probe runs the dispatch op two ways:

  A. REFERENCE: sharded input (current production path)
  B. CANDIDATE: replicated input (no resharder needed)

Compares outputs cos vs each other. PASS if cos ≥ 0.999.

If PASS: blocker #3 is GONE — just delete the h readback + re-upload
in moe_block_eager_ep_tt and pass h_norm_tt (replicated) directly to
all_to_all_dispatch.

If FAIL: dispatch hard-rejects replicated input; need a real reshard.
The research agent's findings will guide the next path.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
S = 8
HIDDEN = 2688
NCHIPS = 4
S_PER_CHIP = S // NCHIPS
NUM_EXPERTS = 128
TOP_K = 6
E_LOCAL = NUM_EXPERTS // NCHIPS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        rng = np.random.default_rng(seed=23)
        # Inputs needed for all_to_all_dispatch:
        # - hidden states (the thing we want to vary: shard vs replicate)
        # - topk_indices (uint16, dim 2 sharded)
        # - expert_mapping (replicated; [num_devices, num_experts] uint16)
        h_np = rng.standard_normal((B, 1, S, HIDDEN), dtype=np.float32) * 0.1
        topk_np = rng.integers(0, NUM_EXPERTS,
                               size=(B, 1, S, TOP_K), dtype=np.int32)
        # Each chip owns experts [chip_idx*E_LOCAL : (chip_idx+1)*E_LOCAL].
        # expert_mapping[chip, expert] = 1 if expert is local to chip.
        expert_mapping_np = np.zeros((NCHIPS, NUM_EXPERTS), dtype=np.int32)
        for c in range(NCHIPS):
            expert_mapping_np[c, c * E_LOCAL:(c + 1) * E_LOCAL] = 1

        # Reshape h to the dispatch input shape: [B, 1, S, HIDDEN]
        h_rm = torch.from_numpy(np.ascontiguousarray(h_np))

        log("Building tensors…")

        h_sharded_tt = ttnn.from_torch(
            h_rm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        h_replicated_tt = ttnn.from_torch(
            h_rm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # all_to_all_dispatch existing call uses ROW_MAJOR for h.
        h_sharded_rm_tt = ttnn.to_layout(h_sharded_tt, ttnn.ROW_MAJOR_LAYOUT)
        h_replicated_rm_tt = ttnn.to_layout(h_replicated_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(h_sharded_tt)
        ttnn.deallocate(h_replicated_tt)

        topk_tt = ttnn.from_torch(
            torch.from_numpy(topk_np.astype(np.int32)),
            device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        expert_mapping_tt = ttnn.from_torch(
            torch.from_numpy(expert_mapping_np.astype(np.int32)),
            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # ── REFERENCE: sharded input ─────────────────────────────────
        log("REFERENCE: all_to_all_dispatch on SHARDED h…")
        try:
            ref_out_tt, ref_meta_tt = ttnn.all_to_all_dispatch(
                h_sharded_rm_tt, topk_tt, expert_mapping_tt,
                cluster_axis=1, output_concat_dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ref_out_np = ttnn.to_torch(
                ref_out_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
            ).float().numpy()
            log(f"  ref output shape: {list(ref_out_np.shape)}")
            ttnn.deallocate(ref_out_tt)
            ttnn.deallocate(ref_meta_tt)
            ref_ok = True
        except Exception as e:
            log(f"  ✗ REFERENCE FAILED (this is a setup bug, not the test): "
                f"{type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            ref_ok = False

        # ── CANDIDATE: replicated input ──────────────────────────────
        log("CANDIDATE: all_to_all_dispatch on REPLICATED h…")
        try:
            cand_out_tt, cand_meta_tt = ttnn.all_to_all_dispatch(
                h_replicated_rm_tt, topk_tt, expert_mapping_tt,
                cluster_axis=1, output_concat_dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cand_out_np = ttnn.to_torch(
                cand_out_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
            ).float().numpy()
            log(f"  candidate output shape: {list(cand_out_np.shape)}")
            ttnn.deallocate(cand_out_tt)
            ttnn.deallocate(cand_meta_tt)
            cand_ok = True
        except Exception as e:
            log(f"  ✗ CANDIDATE FAILED: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            cand_ok = False

        ttnn.deallocate(h_sharded_rm_tt)
        ttnn.deallocate(h_replicated_rm_tt)
        ttnn.deallocate(topk_tt)
        ttnn.deallocate(expert_mapping_tt)

        log("")
        log("=" * 60)
        log("REPORT")
        log("=" * 60)
        if ref_ok and cand_ok:
            cos = float(
                np.dot(ref_out_np.reshape(-1), cand_out_np.reshape(-1))
                / (np.linalg.norm(ref_out_np)
                   * np.linalg.norm(cand_out_np) + 1e-12)
            )
            mad = float(np.mean(np.abs(ref_out_np - cand_out_np)))
            log(f"  sharded ref output:    shape {list(ref_out_np.shape)}")
            log(f"  replicated cand output: shape {list(cand_out_np.shape)}")
            log(f"  cos: {cos:.6f}  mad: {mad:.4e}")
            ok = cos >= 0.999
            log("")
            if ok:
                log("  → all_to_all_dispatch ACCEPTS replicated input.")
                log("    Blocker #3 GONE. Just pass h_norm_tt directly to")
                log("    all_to_all_dispatch in moe_block_eager_ep_tt;")
                log("    delete the h readback + sharded re-upload.")
            else:
                log("  → all_to_all_dispatch runs on replicated input but")
                log("    produces DIFFERENT output. The op interprets")
                log("    REPLICATED data differently from SHARDED. Need a")
                log("    real reshard.")
            return 0 if ok else 1
        else:
            log("  ref or candidate didn't run; see traceback above.")
            return 2
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
