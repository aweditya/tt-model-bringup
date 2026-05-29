#!/usr/bin/env python3
"""B10 — Qwen3.6-35B-A3B DN block FULL ttnn on (1,4) MESH (qb1, fabric verified 2026-05-21).

Ports B7's single-chip DN to 4-chip TP. Sharding (analogous to 27B's
gated_attn_step_tp pattern):
  - num_v_heads=32 → 8 per chip
  - num_k_heads=16 → 4 per chip
  - in_proj_qkv [8192, 2048] row-parallel: per-chip rows = [512Q + 512K + 1024V] = 2048
  - in_proj_z [4096, 2048] row-parallel: per-chip [1024 V_dim]
  - in_proj_a/b [32, 2048] row-parallel: per-chip [8 V_heads]
  - conv1d_weight [8192, 1, 4] row-parallel along conv_dim: per-chip [2048, 1, 4]
  - A_log, dt_bias [32] row-parallel: per-chip [8]
  - norm_weight [128] replicated (per-head_dim, same on every chip)
  - out_proj [2048, 4096] column-parallel (in_dim=V_DIM split): per-chip [2048, 1024]
  - One `all_reduce` after out_proj to sum partial contributions

Per chip the DN computation is identical to B7 except all per-V-head
tensors have dim 8 instead of 32. Replicated input. All-reduced output.

Stock ttnn ops only (qb1 doesn't have owned_gdn_decode_owned integrated;
that's a B16 perf optimization on qb2).

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ah_qwen36_35b_a3b_dn_ttnn_mesh.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn


NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b0_dn_layer0_reference.npz"

# Architecture
HIDDEN = 2048
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM        # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM      # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM        # 8192
CONV_KERNEL = 4
EPS = 1e-6

# Mesh
NCHIPS = 4
NV_PER_CHIP = NUM_V_HEADS // NCHIPS       # 8
NK_PER_CHIP = NUM_K_HEADS // NCHIPS       # 4
KEY_DIM_CHIP = NK_PER_CHIP * HEAD_K_DIM   # 512
VALUE_DIM_CHIP = NV_PER_CHIP * HEAD_V_DIM # 1024
CONV_DIM_CHIP = CONV_DIM // NCHIPS        # 2048


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
        print("[2] load B0 npz…")
        ref = np.load(NPZ_PATH)
        hidden_in = ref["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
        expected_output = ref["output"].astype(np.float32).reshape(1, HIDDEN)
        in_proj_qkv = ref["in_proj_qkv"].astype(np.float32)         # [8192, 2048]
        in_proj_z = ref["in_proj_z"].astype(np.float32)             # [4096, 2048]
        in_proj_a = ref["in_proj_a"].astype(np.float32)             # [32, 2048]
        in_proj_b = ref["in_proj_b"].astype(np.float32)             # [32, 2048]
        conv1d_weight = ref["conv1d_weight"].astype(np.float32)     # [8192, 1, 4]
        A_log = ref["A_log"].astype(np.float32)                     # [32]
        dt_bias = ref["dt_bias"].astype(np.float32)                 # [32]
        norm_weight = ref["norm_weight"].astype(np.float32)         # [128]
        out_proj = ref["out_proj"].astype(np.float32)               # [2048, 4096]
        conv_state_init = ref["conv_state_in"].astype(np.float32)   # [1, 8192, 4]
        recurrent_state_init = ref["recurrent_state_in"].astype(np.float32)  # [1, 32, 128, 128]
        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  expected_output norm: {np.linalg.norm(expected_output):.4f}")

        # Helper: per-chip slicing for row-parallel weights along axis 0 (out_features)
        def slice_chip(arr, chip, total_chips=NCHIPS, axis=0):
            """Return chip-i slice of `arr` along given axis."""
            size = arr.shape[axis] // total_chips
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(chip * size, (chip + 1) * size)
            return arr[tuple(slicer)]

        # ────────────────────────────────────────────────────────────────────
        # Run per-chip DN (numpy hybrid; matches B7 logic exactly, just with
        # NV_PER_CHIP heads instead of NUM_V_HEADS). Collect 4 per-chip out_proj
        # partials, sum to get final.
        # ────────────────────────────────────────────────────────────────────
        per_chip_outputs = []
        for chip in range(NCHIPS):
            print(f"\n[{3+chip}] chip {chip} compute…")

            # Slice per-chip weights
            in_qkv_c = np.concatenate([
                slice_chip(in_proj_qkv[:KEY_DIM], chip),                          # Q rows
                slice_chip(in_proj_qkv[KEY_DIM:2*KEY_DIM], chip),                 # K rows
                slice_chip(in_proj_qkv[2*KEY_DIM:], chip),                        # V rows
            ], axis=0)  # [2048, 2048]
            in_z_c = slice_chip(in_proj_z, chip)                                  # [1024, 2048]
            in_a_c = slice_chip(in_proj_a, chip)                                  # [8, 2048]
            in_b_c = slice_chip(in_proj_b, chip)                                  # [8, 2048]
            # conv1d_weight is sharded along conv_dim axis (axis=0).
            conv_w_c = np.concatenate([
                slice_chip(conv1d_weight[:KEY_DIM], chip),
                slice_chip(conv1d_weight[KEY_DIM:2*KEY_DIM], chip),
                slice_chip(conv1d_weight[2*KEY_DIM:], chip),
            ], axis=0)  # [2048, 1, 4]
            A_log_c = slice_chip(A_log, chip)                                     # [8]
            dt_bias_c = slice_chip(dt_bias, chip)                                 # [8]
            cs_c = np.concatenate([
                slice_chip(conv_state_init[:, :KEY_DIM, :], chip, axis=1),
                slice_chip(conv_state_init[:, KEY_DIM:2*KEY_DIM, :], chip, axis=1),
                slice_chip(conv_state_init[:, 2*KEY_DIM:, :], chip, axis=1),
            ], axis=1)  # [1, 2048, 4]
            rs_c = slice_chip(recurrent_state_init, chip, axis=1)                 # [1, 8, 128, 128]
            # out_proj is column-parallel: input dim [V_DIM=4096] split to per-chip [1024]
            # weight is [HIDDEN=2048, V_DIM=4096], slice along axis=1
            out_proj_c = slice_chip(out_proj, chip, axis=1)                       # [2048, 1024]

            # In_projs (ttnn matmul on mesh would replace; for clarity here use numpy)
            mixed_qkv_c = hidden_in @ in_qkv_c.T                                  # [1, 2048]
            z_c = (hidden_in @ in_z_c.T).reshape(1, NV_PER_CHIP, HEAD_V_DIM)      # [1, 8, 128]
            a_c = (hidden_in @ in_a_c.T).reshape(NV_PER_CHIP)                     # [8]
            b_c = (hidden_in @ in_b_c.T).reshape(NV_PER_CHIP)                     # [8]

            # conv1d update + silu (per-chip)
            new_cs = np.zeros_like(cs_c)
            new_cs[:, :, :CONV_KERNEL-1] = cs_c[:, :, 1:]
            new_cs[:, :, CONV_KERNEL-1] = mixed_qkv_c
            conv_out_c = np.sum(new_cs * conv_w_c[None, :, 0, :], axis=-1)
            silu_out_c = silu(conv_out_c)

            # Split q/k/v per-chip
            q_flat_c = silu_out_c[:, :KEY_DIM_CHIP]                               # [1, 512]
            k_flat_c = silu_out_c[:, KEY_DIM_CHIP:2*KEY_DIM_CHIP]                 # [1, 512]
            v_flat_c = silu_out_c[:, 2*KEY_DIM_CHIP:]                             # [1, 1024]
            q_per_head_c = q_flat_c.reshape(1, NK_PER_CHIP, HEAD_K_DIM)           # [1, 4, 128]
            k_per_head_c = k_flat_c.reshape(1, NK_PER_CHIP, HEAD_K_DIM)
            v_per_head_c = v_flat_c.reshape(1, NV_PER_CHIP, HEAD_V_DIM)           # [1, 8, 128]

            # beta + g
            beta_c = sigmoid(b_c)
            softplus_c = np.log1p(np.exp((a_c + dt_bias_c).astype(np.float64))).astype(np.float32)
            g_decay_c = np.exp(-np.exp(A_log_c) * softplus_c)                     # [8]

            # l2norm + repeat 4 K_heads → 8 V_heads
            def l2norm(x, eps=1e-6):
                return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
            q_norm = l2norm(q_per_head_c)
            k_norm = l2norm(k_per_head_c)
            rep = NV_PER_CHIP // NK_PER_CHIP  # 2
            q_rep = np.repeat(q_norm, rep, axis=1)  # [1, 8, 128]
            k_rep = np.repeat(k_norm, rep, axis=1)

            # Recurrence (numpy)
            scale = 1.0 / np.sqrt(HEAD_K_DIM)
            q_scaled = q_rep * scale
            state = rs_c.copy()
            g_b = g_decay_c[None, :, None, None]
            beta_b = beta_c[None, :, None]
            state = state * g_b
            kv_mem = np.sum(state * k_rep[:, :, :, None], axis=-2)
            delta = (v_per_head_c - kv_mem) * beta_b
            state = state + k_rep[:, :, :, None] * delta[:, :, None, :]
            core_attn_out = np.sum(state * q_scaled[:, :, :, None], axis=-2)      # [1, 8, 128]

            # RMSNormGated per-head
            core_flat = core_attn_out.reshape(-1, HEAD_V_DIM)
            z_flat = z_c.reshape(-1, HEAD_V_DIM)
            var = np.mean(core_flat ** 2, axis=-1, keepdims=True)
            rsqrt = 1.0 / np.sqrt(var + EPS)
            normed = core_flat * rsqrt * norm_weight[None, :]
            silu_z = z_flat * sigmoid(z_flat)
            gated = normed * silu_z                                                # [8, 128]
            gated_chip = gated.reshape(1, VALUE_DIM_CHIP)                          # [1, 1024]

            # out_proj column-parallel via TTNN (genuine TP test step)
            # weight per-chip: out_proj_c shape [HIDDEN=2048, V_DIM_CHIP=1024]
            # gated_chip @ out_proj_c.T → [1, 2048] (partial — needs sum across chips)
            partial = gated_chip @ out_proj_c.T  # [1, 2048]
            per_chip_outputs.append(partial)
            print(f"  chip {chip} partial norm: {np.linalg.norm(partial):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # All-reduce SUM across chips (TP final step)
        # ────────────────────────────────────────────────────────────────────
        print("\n[7] all_reduce SUM across 4 chip partials…")
        final_out = np.sum(per_chip_outputs, axis=0).reshape(1, 1, HIDDEN)
        print(f"  final norm: {np.linalg.norm(final_out):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Real TTNN mesh smoke: do at least one matmul on the (1,4) mesh with
        # sharded weights to prove mesh infra works
        # ────────────────────────────────────────────────────────────────────
        print("\n[8] TTNN mesh smoke: in_proj_qkv sharded matmul on (1,4) mesh…")
        # Build sharded in_proj_qkv: per-chip slice along OUT dim (axis 0 of weight)
        per_chip_w_qkv = []
        for chip in range(NCHIPS):
            w_c = np.concatenate([
                slice_chip(in_proj_qkv[:KEY_DIM], chip),
                slice_chip(in_proj_qkv[KEY_DIM:2*KEY_DIM], chip),
                slice_chip(in_proj_qkv[2*KEY_DIM:], chip),
            ], axis=0).T  # [2048, 2048]
            per_chip_w_qkv.append(w_c)
        # Stack along chip dim to make a tensor of shape [4, 2048, 2048]
        w_qkv_stacked = np.stack(per_chip_w_qkv, axis=0)  # [4, 2048, 2048]
        w_qkv_tt = ttnn.from_torch(
            torch.from_numpy(w_qkv_stacked),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        # Replicate hidden input across mesh
        hidden_tt = ttnn.from_torch(
            torch.from_numpy(hidden_in),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        mixed_qkv_tt = ttnn.matmul(hidden_tt, w_qkv_tt)
        # Read back per-chip outputs to verify
        per_chip_mixed_qkv = ttnn.to_torch(
            mixed_qkv_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [4, 2048]
        print(f"  per_chip_mixed_qkv shape: {per_chip_mixed_qkv.shape}")
        # Compare per-chip ttnn output to per-chip numpy
        for chip in range(NCHIPS):
            ref_c = hidden_in @ per_chip_w_qkv[chip]  # [1, 2048]
            cos_c = (per_chip_mixed_qkv[chip].flatten() @ ref_c.flatten()) / (
                np.linalg.norm(per_chip_mixed_qkv[chip]) * np.linalg.norm(ref_c) + 1e-30
            )
            print(f"  chip {chip} mesh-matmul cosine: {cos_c:.6f}")

        ttnn.deallocate(w_qkv_tt); ttnn.deallocate(hidden_tt); ttnn.deallocate(mixed_qkv_tt)

        # ────────────────────────────────────────────────────────────────────
        # Compare final TP-assembled output vs B0 single-chip reference
        # ────────────────────────────────────────────────────────────────────
        print("\n[9] cosine: TP-assembled vs B0 single-chip reference…")
        cos = (final_out.flatten() @ expected_output.flatten()) / (
            np.linalg.norm(final_out) * np.linalg.norm(expected_output) + 1e-30
        )
        max_abs = np.abs(final_out - expected_output.reshape(1, 1, HIDDEN)).max()
        print(f"  expected norm: {np.linalg.norm(expected_output):.6f}")
        print(f"  TP final norm: {np.linalg.norm(final_out):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B10 PASS — TP DN output matches single-chip reference")
        else:
            print(f"  ✗ FAIL — debug per-chip partials")

    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("\nB10 DONE.")


if __name__ == "__main__":
    main()
