#!/usr/bin/env python3
"""B12.8 — Gated attention block on (1,4) MESH (qb1).

Ports B12.5/B12.7 attention to TP. Strategy (plan §7 risk 5 resolution):
**Q-head sharded, KV replicated**. The 2 KV heads don't divide 4 chips
cleanly, so the simplest correct approach is to replicate KV across all
4 chips (TT-Metal Galaxy llama3_70b does the same for similar GQA shapes).

Sharding:
  - q_proj [8192, 2048] row-sharded along Q axis:
      per-chip [2048, 2048] (4 Q heads × head_dim × 2 → split Q + gate)
  - k_proj [512, 2048] REPLICATED on every chip (both 2 KV heads everywhere)
  - v_proj [512, 2048] REPLICATED
  - q_norm, k_norm REPLICATED (per-head_dim, tiny)
  - o_proj [2048, 4096] column-sharded along input (V_dim split):
      per-chip [2048, 1024] (corresponds to 4 Q heads × head_dim)
  - One all_reduce SUM after o_proj

Per-chip GQA:
  - Q heads per chip = 4 (16 / 4)
  - KV heads per chip = 2 (replicated, both heads present everywhere)
  - GQA repeat per chip: K/V each repeated 2× (4 Q heads = 2 KV head groups × 2)
  - Single-token attn: out_per_chip = v_per_chip_repeated (size [1, 4, 256])
  - attn_output_gate per chip: gate has 4 heads × head_dim = 1024 dims
  - o_proj_c: chip's slice of [V_dim → hidden]; partial summed via all_reduce

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91an_qwen36_35b_a3b_attn_ttnn_mesh.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn

NPZ_INTER = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b3p_layer3_intermediates.npz"
NPZ_B3 = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b3_layer3_full_reference.npz"

HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 8
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64
EPS = 1e-6

NCHIPS = 4
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS         # 4
# KV stays replicated (NUM_KV_HEADS=2 doesn't divide NCHIPS=4)


def silu(x): return x * (1.0 / (1.0 + np.exp(-x)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def qwen35_rms_norm(x, w, eps=EPS):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * (1.0 + w)


def rms_norm_head(x, w, eps=EPS):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def main():
    assert NPZ_INTER.exists() and NPZ_B3.exists()

    print("[1] enable fabric + open (1,4) mesh on qb1…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"  mesh: {mesh}")

    try:
        print("[2] load npzs…")
        inter = np.load(NPZ_INTER)
        b3 = np.load(NPZ_B3)
        hidden_in = inter["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
        attn_intermediate = inter["attn_intermediate"].astype(np.float32).reshape(1, HIDDEN)
        input_ln_w = b3["input_layernorm_weight"].astype(np.float32)
        q_proj = b3["q_proj"].astype(np.float32)        # [8192, 2048]
        k_proj = b3["k_proj"].astype(np.float32)        # [512, 2048] (replicated)
        v_proj = b3["v_proj"].astype(np.float32)        # [512, 2048] (replicated)
        o_proj = b3["o_proj"].astype(np.float32)        # [2048, 4096]
        q_norm_w = b3["q_norm"].astype(np.float32)      # [256]
        k_norm_w = b3["k_norm"].astype(np.float32)      # [256]
        cos_hf = b3["cos"].astype(np.float32).reshape(1, 1, ROTARY_DIM)
        sin_hf = b3["sin"].astype(np.float32).reshape(1, 1, ROTARY_DIM)

        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  attn_intermediate norm: {np.linalg.norm(attn_intermediate):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Replicated input_layernorm (output same on every chip)
        # ────────────────────────────────────────────────────────────────────
        h = qwen35_rms_norm(hidden_in, input_ln_w)
        print(f"  post-input_layernorm norm: {np.linalg.norm(h):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Per-chip compute: shard Q (and gate) along Q-head axis;
        # replicate KV. Each chip independently runs single-token GQA on its
        # 4-of-16 Q heads against the full 2 KV heads.
        # ────────────────────────────────────────────────────────────────────
        per_chip_o_partials = []
        for chip in range(NCHIPS):
            # q_proj per chip: rows for [chip*4:(chip+1)*4] Q heads × head_dim × 2
            # Output layout: [num_q_heads, head_dim, 2] flat → chip slice along
            # heads. The full q_proj outputs in_features=2048 → out_features=8192
            # where 8192 = num_q_heads × head_dim × 2. To get chip's Q rows,
            # slice axis=0 of q_proj into per-head blocks.
            # q_proj reshape: [num_q_heads, head_dim*2, hidden] for slicing.
            q_proj_r = q_proj.reshape(NUM_Q_HEADS, HEAD_DIM * 2, HIDDEN)
            q_proj_c = q_proj_r[chip*NQ_PER_CHIP:(chip+1)*NQ_PER_CHIP].reshape(NQ_PER_CHIP * HEAD_DIM * 2, HIDDEN)
            # Per-chip Q+gate compute
            q_full_c = h @ q_proj_c.T  # [1, NQ_PER_CHIP*HEAD_DIM*2 = 2048]
            q_full_c_h = q_full_c.reshape(1, NQ_PER_CHIP, HEAD_DIM * 2)
            q_c = q_full_c_h[..., :HEAD_DIM]                                 # [1, 4, 256]
            gate_c = q_full_c_h[..., HEAD_DIM:]                              # [1, 4, 256]
            gate_flat_c = gate_c.reshape(1, NQ_PER_CHIP * HEAD_DIM)          # [1, 1024]

            # KV: replicated weights, every chip computes the same
            k = (h @ k_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM)            # [1, 2, 256]
            v = (h @ v_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM)

            # q_norm / k_norm (replicated weights)
            q_c = rms_norm_head(q_c, q_norm_w)
            k = rms_norm_head(k, k_norm_w)

            # RoPE
            q_rot = q_c[..., :ROTARY_DIM]; q_pass = q_c[..., ROTARY_DIM:]
            k_rot = k[..., :ROTARY_DIM]; k_pass = k[..., ROTARY_DIM:]
            q_rot = q_rot * cos_hf + rotate_half(q_rot) * sin_hf
            k_rot = k_rot * cos_hf + rotate_half(k_rot) * sin_hf
            q_final = np.concatenate([q_rot, q_pass], axis=-1)
            k_final = np.concatenate([k_rot, k_pass], axis=-1)

            # GQA: each Q head g_global maps to KV head g_global // GQA_GROUP
            # = g_global // 8. With 16 Q heads and 2 KV heads:
            #   Q heads 0-7  → KV head 0
            #   Q heads 8-15 → KV head 1
            # With chip c holding Q heads [c*4 : c*4+4]:
            #   chip 0 (Q 0-3) → KV head 0 (all 4 Q heads)
            #   chip 1 (Q 4-7) → KV head 0
            #   chip 2 (Q 8-11) → KV head 1
            #   chip 3 (Q 12-15) → KV head 1
            # So per chip: pick ONE KV head (chip // (NCHIPS // NUM_KV_HEADS))
            # and broadcast it across all NQ_PER_CHIP Q heads.
            chip_kv_idx = chip // (NCHIPS // NUM_KV_HEADS)                 # 0 or 1
            v_chip = v[:, chip_kv_idx:chip_kv_idx + 1, :]                  # [1, 1, 256]
            v_per_q_c = np.broadcast_to(v_chip, (1, NQ_PER_CHIP, HEAD_DIM)).copy()
            # (single-token attention → output = v_per_q_c)
            attn_out_c = v_per_q_c                                          # [1, 4, 256]
            attn_flat_c = attn_out_c.reshape(1, NQ_PER_CHIP * HEAD_DIM)    # [1, 1024]

            # attn_output_gate
            gated_c = attn_flat_c * sigmoid(gate_flat_c)

            # o_proj column-sharded along input dim: per-chip slice
            # o_proj is [hidden, num_q_heads*head_dim] = [2048, 4096]
            # Per-chip columns: chip's Q heads contribute their head_dim of input dim
            o_proj_c = o_proj[:, chip*NQ_PER_CHIP*HEAD_DIM:(chip+1)*NQ_PER_CHIP*HEAD_DIM]  # [2048, 1024]
            partial = gated_c @ o_proj_c.T  # [1, 2048]
            per_chip_o_partials.append(partial)
            print(f"  chip {chip} partial norm: {np.linalg.norm(partial):.4f}")

        print("\n[3] all_reduce SUM…")
        attn_assembled = np.sum(per_chip_o_partials, axis=0)
        h_after_attn = hidden_in + attn_assembled  # residual
        print(f"  attn_assembled norm: {np.linalg.norm(attn_assembled):.4f}")
        print(f"  h_after_attn norm: {np.linalg.norm(h_after_attn):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # TTNN mesh smoke: K matmul replicated weight (no shard) on (1,4)
        # ────────────────────────────────────────────────────────────────────
        print("\n[4] TTNN mesh smoke: k_proj REPLICATED matmul on (1,4)…")
        h_tt = ttnn.from_torch(
            torch.from_numpy(h.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        k_proj_tt = ttnn.from_torch(
            torch.from_numpy(k_proj.T.astype(np.float32)),  # [2048, 512]
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        k_tt = ttnn.matmul(h_tt, k_proj_tt)
        k_per_chip = ttnn.to_torch(
            k_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [4, 1, 512] (concat of 4 replicated outputs)
        # Each chip's K should be identical (replicated weight + replicated input)
        for chip in range(NCHIPS):
            ref_k = h @ k_proj.T
            cos_c = (k_per_chip[chip].flatten() @ ref_k.flatten()) / (
                np.linalg.norm(k_per_chip[chip]) * np.linalg.norm(ref_k) + 1e-30
            )
            print(f"  chip {chip} K mesh cos: {cos_c:.6f}")
        ttnn.deallocate(h_tt); ttnn.deallocate(k_proj_tt); ttnn.deallocate(k_tt)

        # ────────────────────────────────────────────────────────────────────
        # Final cosine vs HF attn_intermediate
        # ────────────────────────────────────────────────────────────────────
        print("\n[5] cosine: TP attn-only vs HF attn_intermediate…")
        cos = (h_after_attn.flatten() @ attn_intermediate.flatten()) / (
            np.linalg.norm(h_after_attn) * np.linalg.norm(attn_intermediate) + 1e-30
        )
        max_abs = np.abs(h_after_attn - attn_intermediate).max()
        print(f"  expected norm: {np.linalg.norm(attn_intermediate):.6f}")
        print(f"  TP final norm: {np.linalg.norm(h_after_attn):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B12.8 PASS — TP attn output matches single-chip reference")
        else:
            print(f"  ✗ FAIL")

    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("\nB12.8 DONE.")


if __name__ == "__main__":
    main()
