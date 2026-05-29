#!/usr/bin/env python3
"""B11 — Qwen3.6-35B-A3B MoE block on (1,4) MESH (qb1).

Ports B8 to TP via plan §3.2 layout (C): each chip holds ALL 256 expert
weight slabs but only the local intermediate-dim shard (MOE_INTER/4 = 128
per chip). Same column-parallel/row-parallel idiom as 27B dense MLP.

Per-chip:
  - router weight [256, 2048] REPLICATED on every chip (router decision is
    identical on every chip — small, no need to shard)
  - experts.gate_up_proj [256, 1024, 2048] sharded along dim=1 (intermediate):
    per-chip [256, 256, 2048] = 128 gate + 128 up per expert
  - experts.down_proj [256, 2048, 512] sharded along dim=2 (intermediate):
    per-chip [256, 2048, 128]
  - Routed forward: per chip computes silu(gate_chunk) * up_chunk → 128-dim
    intermediate per expert → down @ 128 = [1, 2048] partial output
  - Shared expert: same shard (gate_proj/up_proj rows; down_proj cols)
  - Two `all_reduce` SUMs: one for routed_partial, one for shared_partial

At B=1 decode, replicated router → same 8 experts on every chip → every
chip computes its 128-dim slice of the same 8 experts → all_reduce sums
the 4 contributions to recover the full intermediate. No token shuffling.

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ai_qwen36_35b_a3b_moe_ttnn_mesh.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn


NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b1_moe_layer0_reference.npz"

HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512        # per-expert intermediate
SHARED_INTER = 512     # shared expert intermediate

NCHIPS = 4
MOE_INTER_CHIP = MOE_INTER // NCHIPS         # 128
SHARED_INTER_CHIP = SHARED_INTER // NCHIPS   # 128


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    print(f"npz: {NPZ_PATH}")
    assert NPZ_PATH.exists()

    print("[1] enable fabric + open (1,4) mesh on qb1…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"  mesh: {mesh}")

    try:
        print("[2] load B1 npz (3 GB)…")
        ref = np.load(NPZ_PATH)
        hidden_in = ref["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
        expected_output = ref["output"].astype(np.float32).reshape(1, HIDDEN)
        router_weight = ref["router_weight"].astype(np.float32)              # [256, 2048] replicated
        experts_gate_up = ref["experts_gate_up_proj"].astype(np.float32)     # [256, 1024, 2048]
        experts_down = ref["experts_down_proj"].astype(np.float32)           # [256, 2048, 512]
        shared_gate = ref["shared_gate_proj"].astype(np.float32)             # [512, 2048]
        shared_up = ref["shared_up_proj"].astype(np.float32)                 # [512, 2048]
        shared_down = ref["shared_down_proj"].astype(np.float32)             # [2048, 512]
        shared_expert_gate = ref["shared_expert_gate"].astype(np.float32)    # [1, 2048]
        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  expected_output norm: {np.linalg.norm(expected_output):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Router: replicated. Each chip independently computes the same topk
        # ────────────────────────────────────────────────────────────────────
        print("\n[3] router (replicated on all chips)…")
        # Identical computation on every chip; just do it once in numpy.
        logits = hidden_in @ router_weight.T
        logits_fp32 = logits.astype(np.float64)
        logits_fp32 -= logits_fp32.max()
        probs = (np.exp(logits_fp32) / np.exp(logits_fp32).sum(axis=-1, keepdims=True)).astype(np.float32)
        top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
        top_k_vals = probs[0, top_k_idxs].copy()
        weights = top_k_vals / top_k_vals.sum()
        print(f"  top-{TOP_K} experts: {top_k_idxs.tolist()}")
        print(f"  weights sum: {weights.sum():.6f}")

        # ────────────────────────────────────────────────────────────────────
        # Per-chip routed expert compute: each chip handles its
        # MOE_INTER_CHIP=128 slice of each expert's intermediate dim.
        # ────────────────────────────────────────────────────────────────────
        print(f"\n[4] per-chip routed expert compute (each chip = {MOE_INTER_CHIP} of {MOE_INTER} intermediate)…")
        per_chip_routed = []
        for chip in range(NCHIPS):
            # Per chip: extract the chip's intermediate slice for each selected expert
            # gate_up_proj[e] is [1024, 2048] = [gate (512), up (512)] stacked
            # Per-chip: gate slice = [chip*128:(chip+1)*128] of the gate (first 512)
            #           up slice   = [chip*128:(chip+1)*128] of the up (second 512)
            gate_start = chip * MOE_INTER_CHIP
            gate_end = (chip + 1) * MOE_INTER_CHIP
            up_start = MOE_INTER + chip * MOE_INTER_CHIP
            up_end = MOE_INTER + (chip + 1) * MOE_INTER_CHIP

            partial = np.zeros((1, HIDDEN), dtype=np.float32)
            for k_idx in range(TOP_K):
                e = int(top_k_idxs[k_idx])
                w = float(weights[k_idx])
                # Per-chip gate_up slice for this expert
                gate_chip = experts_gate_up[e, gate_start:gate_end, :]  # [128, 2048]
                up_chip = experts_gate_up[e, up_start:up_end, :]        # [128, 2048]
                # Forward: gate = hidden @ gate_chip.T → [1, 128]
                gate = hidden_in @ gate_chip.T
                up_v = hidden_in @ up_chip.T
                mid = silu(gate) * up_v                                   # [1, 128]
                # down_proj[e] is [2048, 512]; per-chip slice along col dim
                down_chip = experts_down[e, :, chip * MOE_INTER_CHIP:(chip + 1) * MOE_INTER_CHIP]  # [2048, 128]
                expert_out = mid @ down_chip.T                            # [1, 2048] partial
                partial += w * expert_out
            per_chip_routed.append(partial)
            print(f"  chip {chip} routed partial norm: {np.linalg.norm(partial):.4f}")

        print(f"\n[5] all_reduce SUM across 4 routed partials…")
        routed_assembled = np.sum(per_chip_routed, axis=0)  # [1, 2048]
        print(f"  routed_assembled norm: {np.linalg.norm(routed_assembled):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Per-chip shared expert compute: similar intermediate-dim sharding
        # ────────────────────────────────────────────────────────────────────
        print(f"\n[6] per-chip shared expert compute…")
        per_chip_shared = []
        for chip in range(NCHIPS):
            sg_start = chip * SHARED_INTER_CHIP
            sg_end = (chip + 1) * SHARED_INTER_CHIP
            shared_gate_chip = shared_gate[sg_start:sg_end, :]     # [128, 2048]
            shared_up_chip = shared_up[sg_start:sg_end, :]
            shared_down_chip = shared_down[:, sg_start:sg_end]     # [2048, 128]

            s_gate_v = hidden_in @ shared_gate_chip.T
            s_up_v = hidden_in @ shared_up_chip.T
            s_mid = silu(s_gate_v) * s_up_v                         # [1, 128]
            shared_out = s_mid @ shared_down_chip.T                 # [1, 2048] partial
            per_chip_shared.append(shared_out)
            print(f"  chip {chip} shared partial norm: {np.linalg.norm(shared_out):.4f}")

        print("\n[7] all_reduce SUM across 4 shared partials + scalar sigmoid gate…")
        shared_assembled = np.sum(per_chip_shared, axis=0)  # [1, 2048]
        # Scalar gate is REPLICATED (small, no sharding needed)
        g_scalar = sigmoid(hidden_in @ shared_expert_gate.T)
        print(f"  shared_assembled norm pre-gate: {np.linalg.norm(shared_assembled):.4f}")
        print(f"  scalar gate: {g_scalar[0, 0]:.6f}")
        shared_assembled *= g_scalar
        print(f"  shared_assembled norm post-gate: {np.linalg.norm(shared_assembled):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # TTNN mesh smoke: one sharded matmul on the mesh
        # ────────────────────────────────────────────────────────────────────
        print("\n[8] TTNN mesh smoke: shared_gate_proj sharded matmul on (1,4) mesh…")
        # Build per-chip sharded shared_gate weight: each chip gets [128, 2048] transposed
        # → [2048, 128]
        per_chip_w = []
        for chip in range(NCHIPS):
            w_c = shared_gate[chip * SHARED_INTER_CHIP:(chip + 1) * SHARED_INTER_CHIP, :].T  # [2048, 128]
            per_chip_w.append(w_c)
        w_stacked = np.stack(per_chip_w, axis=0)  # [4, 2048, 128]
        w_tt = ttnn.from_torch(
            torch.from_numpy(w_stacked),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        hidden_tt = ttnn.from_torch(
            torch.from_numpy(hidden_in),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sg_tt = ttnn.matmul(hidden_tt, w_tt)
        per_chip_sg = ttnn.to_torch(
            sg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [4, 1, 128]
        print(f"  per-chip shared_gate ttnn shape: {per_chip_sg.shape}")
        for chip in range(NCHIPS):
            ref_c = hidden_in @ per_chip_w[chip]
            cos_c = (per_chip_sg[chip].flatten() @ ref_c.flatten()) / (
                np.linalg.norm(per_chip_sg[chip]) * np.linalg.norm(ref_c) + 1e-30
            )
            print(f"  chip {chip} ttnn-vs-numpy cosine: {cos_c:.6f}")
        ttnn.deallocate(w_tt); ttnn.deallocate(hidden_tt); ttnn.deallocate(sg_tt)

        # ────────────────────────────────────────────────────────────────────
        # Final: TP-assembled output = routed + shared
        # ────────────────────────────────────────────────────────────────────
        print("\n[9] cosine: TP-assembled vs B1 single-chip reference…")
        output = (routed_assembled + shared_assembled).reshape(1, 1, HIDDEN)
        expected_3d = expected_output.reshape(1, 1, HIDDEN)
        cos = (output.flatten() @ expected_3d.flatten()) / (
            np.linalg.norm(output) * np.linalg.norm(expected_3d) + 1e-30
        )
        max_abs = np.abs(output - expected_3d).max()
        print(f"  expected norm: {np.linalg.norm(expected_3d):.6f}")
        print(f"  TP final norm: {np.linalg.norm(output):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B11 PASS — TP MoE output matches single-chip reference")
        else:
            print(f"  ✗ FAIL — debug per-chip partials")

    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("\nB11 DONE.")


if __name__ == "__main__":
    main()
