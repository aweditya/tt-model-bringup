#!/usr/bin/env python3
"""
C'7.4: DeltaNet 4-chip tensor-parallel correctness probe on qb2.

Tests head-sharded TP for the DeltaNet recurrence. Pattern (canonical):
  N_K_HEADS=16, N_V_HEADS=32 → 4 K heads + 8 V heads per chip
  Recurrent SSM state stays per-chip (no comm inside recurrence body)
  All-reduce only at out_proj exit

Sharding layout (chip i, i ∈ {0..3}):
  in_proj_all input: x replicated [1, HIDDEN]
  in_proj_all weight: per-chip slab.
    Production layout: [Q (KEY_DIM) | K (KEY_DIM) | V (VAL_DIM) | Z (VAL_DIM)
                      | A (N_V_HEADS) | B (N_V_HEADS)]
    Per-chip: Q heads i*4..(i+1)*4 (K_DIM each) +
              K heads same (K_DIM each) +
              V heads i*8..(i+1)*8 (V_DIM each) +
              Z heads same +
              A entries i*8..(i+1)*8 +
              B entries same.
    Slab width per chip:
       K * 4 + K * 4 + V * 8 + V * 8 + 8 + 8
     = 128*4 + 128*4 + 128*8 + 128*8 + 16
     = 512 + 512 + 1024 + 1024 + 16
     = 3088 cols per chip; total 12352 cols
     (vs production 2*KEY_DIM + 2*VAL_DIM + 2*N_V_HEADS = 4096+8192+64 = 12352 ✓)

  conv1d_weight: [CONV_DIM, KERNEL]. CONV_DIM = 2*KEY_DIM + VAL_DIM = 4096+4096 = 8192.
    Per-chip slab: 2*(4*K_DIM) + 8*V_DIM = 1024 + 1024 = 2048 cols.
    Conv state per chip: [2048, KERNEL].

  out_proj weight: [VAL_DIM, HIDDEN]. Row-parallel: each chip has [V_DIM*8, HIDDEN] = [1024, 5120].
  SSM state: per-chip [N_V_PER_CHIP, K_DIM, V_DIM] = [8, 128, 128].

This is the BIG one — DeltaNet has more state than attention. If this passes,
the full 4-chip TP forward (C'7.5) is just plumbing.
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B DeltaNet config
HIDDEN = 5120
N_K_HEADS = 16
N_V_HEADS = 32
K_DIM = 128
V_DIM = 128
KERNEL = 3
KEY_DIM = N_K_HEADS * K_DIM      # 2048
VAL_DIM = N_V_HEADS * V_DIM      # 4096
CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 8192
N_REP = N_V_HEADS // N_K_HEADS    # 2
IN_PROJ_OUT = CONV_DIM + VAL_DIM + 2 * N_V_HEADS  # 12352
EPS = 1e-6

NCHIPS = 4
NK_PER_CHIP = N_K_HEADS // NCHIPS  # 4
NV_PER_CHIP = N_V_HEADS // NCHIPS  # 8
KEY_DIM_CHIP = NK_PER_CHIP * K_DIM  # 512
VAL_DIM_CHIP = NV_PER_CHIP * V_DIM  # 1024
CONV_DIM_CHIP = 2 * KEY_DIM_CHIP + VAL_DIM_CHIP  # 2048
IN_PROJ_OUT_CHIP = CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP  # 3088

assert N_K_HEADS % NCHIPS == 0
assert N_V_HEADS % NCHIPS == 0
assert IN_PROJ_OUT_CHIP * NCHIPS == IN_PROJ_OUT, (
    f"in_proj layout mismatch: {IN_PROJ_OUT_CHIP * NCHIPS} != {IN_PROJ_OUT}")


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def deltanet_np_forward(x, w_in, w_conv, dt_bias, A_log, w_out, ssm, conv_state):
    """
    Pure numpy fp32 DeltaNet step. Mirrors gated_attn_step shape but for the
    recurrence math, sans z (gate) and per-head rms_norm.

    Simplified vs production:
      - skip input rms_norm (random in_ln weight = 1)
      - skip per-head rms_norm of recurrence output (norm_w = 1, identity)
      - z gate replaced by direct silu(z)
    """
    # in_proj
    all_p = x @ w_in  # [1, IN_PROJ_OUT]
    mixed_qkv = all_p[:, :CONV_DIM]             # [1, 8192]
    z_p     = all_p[:, CONV_DIM:CONV_DIM + VAL_DIM]   # [1, 4096]
    a_p     = all_p[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + N_V_HEADS]    # [1, 32]
    b_p     = all_p[:, CONV_DIM + VAL_DIM + N_V_HEADS:]                       # [1, 32]

    # conv1d
    mixed_col = mixed_qkv.reshape(CONV_DIM, 1)
    conv_input = np.concatenate([conv_state, mixed_col], axis=-1)  # [CONV_DIM, KERNEL]
    conv_prod = conv_input * w_conv  # [CONV_DIM, KERNEL]
    conv_out = conv_prod.sum(axis=-1)  # [CONV_DIM]
    conv_out = conv_out * (1.0 / (1.0 + np.exp(-conv_out)))  # silu

    # Split into Q, K, V (flat)
    q_flat = conv_out[:KEY_DIM]
    k_flat = conv_out[KEY_DIM:2 * KEY_DIM]
    v_flat = conv_out[2 * KEY_DIM:]

    # GQA broadcast: repeat-interleave Q, K from N_K to N_V
    def interleave(t, n_kh, d, n_rep):
        t2 = t.reshape(n_kh, 1, d)
        t3 = np.broadcast_to(t2, (n_kh, n_rep, d)).copy()
        return t3.reshape(n_kh * n_rep, d)

    q = interleave(q_flat, N_K_HEADS, K_DIM, N_REP)  # [N_V_HEADS, K_DIM]
    k = interleave(k_flat, N_K_HEADS, K_DIM, N_REP)
    v = v_flat.reshape(N_V_HEADS, V_DIM)

    # Q/K L2 normalize + Q-scaling
    q_n = q / np.sqrt((q * q).sum(axis=-1, keepdims=True) + EPS)
    k_n = k / np.sqrt((k * k).sum(axis=-1, keepdims=True) + EPS)
    q_n = q_n * (1.0 / np.sqrt(K_DIM))

    # gate/decay/beta
    softplus_a = np.log(np.exp(a_p[0] + dt_bias) + 1.0)  # [N_V_HEADS]
    g = -np.exp(A_log) * softplus_a
    beta = 1.0 / (1.0 + np.exp(-b_p[0]))
    decay = np.exp(g).reshape(N_V_HEADS, 1, 1)

    # Recurrence
    H = ssm.copy()  # [N_V_HEADS, K_DIM, V_DIM]
    H_decayed = H * decay
    k_col = k_n.reshape(N_V_HEADS, K_DIM, 1)
    kv_mem = (H_decayed * k_col).sum(axis=-2)  # [N_V_HEADS, V_DIM]
    delta = (v - kv_mem) * beta.reshape(N_V_HEADS, 1)
    H_new = H_decayed + k_col * delta.reshape(N_V_HEADS, 1, V_DIM)
    q_col = q_n.reshape(N_V_HEADS, K_DIM, 1)
    out = (H_new * q_col).sum(axis=-2)  # [N_V_HEADS, V_DIM]

    # Gate by silu(z)
    z = z_p.reshape(N_V_HEADS, V_DIM)
    silu_z = z * (1.0 / (1.0 + np.exp(-z)))
    out_gated = out * silu_z

    out_flat = out_gated.reshape(1, VAL_DIM)
    final = out_flat @ w_out  # [1, HIDDEN]
    return final, H_new


def relayout_in_proj_weight(w_in):
    """
    Re-arrange production in_proj layout [Q|K|V|Z|A|B] into per-chip-contiguous
    layout [chip0_slab | chip1_slab | ...] so ShardTensorToMesh(dim=1) gives
    each chip its own [Q_chip | K_chip | V_chip | Z_chip | A_chip | B_chip].
    """
    # Slice production layout by group
    Q = w_in[:, :KEY_DIM].reshape(HIDDEN, N_K_HEADS, K_DIM)
    K = w_in[:, KEY_DIM:2 * KEY_DIM].reshape(HIDDEN, N_K_HEADS, K_DIM)
    V = w_in[:, 2 * KEY_DIM:CONV_DIM].reshape(HIDDEN, N_V_HEADS, V_DIM)
    Z = w_in[:, CONV_DIM:CONV_DIM + VAL_DIM].reshape(HIDDEN, N_V_HEADS, V_DIM)
    A = w_in[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + N_V_HEADS]  # [HIDDEN, N_V_HEADS]
    B = w_in[:, CONV_DIM + VAL_DIM + N_V_HEADS:]                     # [HIDDEN, N_V_HEADS]

    slabs = []
    for chip_i in range(NCHIPS):
        q_chip = Q[:, chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP, :]
        k_chip = K[:, chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP, :]
        v_chip = V[:, chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP, :]
        z_chip = Z[:, chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP, :]
        a_chip = A[:, chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP]
        b_chip = B[:, chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP]
        slab = np.concatenate([
            q_chip.reshape(HIDDEN, KEY_DIM_CHIP),
            k_chip.reshape(HIDDEN, KEY_DIM_CHIP),
            v_chip.reshape(HIDDEN, VAL_DIM_CHIP),
            z_chip.reshape(HIDDEN, VAL_DIM_CHIP),
            a_chip,
            b_chip,
        ], axis=1)
        assert slab.shape[1] == IN_PROJ_OUT_CHIP, f"{slab.shape[1]} != {IN_PROJ_OUT_CHIP}"
        slabs.append(slab)
    return np.concatenate(slabs, axis=1)  # [HIDDEN, IN_PROJ_OUT]


def relayout_conv_weight(w_conv):
    """conv1d weight [CONV_DIM, KERNEL] split per-chip by head group.
    Per-chip rows: 2*KEY_DIM_CHIP + VAL_DIM_CHIP = 2*512 + 1024 = 2048.
    """
    Q = w_conv[:KEY_DIM].reshape(N_K_HEADS, K_DIM, KERNEL)
    K = w_conv[KEY_DIM:2 * KEY_DIM].reshape(N_K_HEADS, K_DIM, KERNEL)
    V = w_conv[2 * KEY_DIM:].reshape(N_V_HEADS, V_DIM, KERNEL)
    slabs = []
    for chip_i in range(NCHIPS):
        slab = np.concatenate([
            Q[chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP].reshape(KEY_DIM_CHIP, KERNEL),
            K[chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP].reshape(KEY_DIM_CHIP, KERNEL),
            V[chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP].reshape(VAL_DIM_CHIP, KERNEL),
        ], axis=0)
        assert slab.shape == (CONV_DIM_CHIP, KERNEL), f"got {slab.shape}"
        slabs.append(slab)
    return np.concatenate(slabs, axis=0)  # [CONV_DIM, KERNEL]


def relayout_conv_state(conv_state):
    """conv_state [CONV_DIM, KERNEL-1] must follow same head-group order as conv_weight."""
    Q = conv_state[:KEY_DIM].reshape(N_K_HEADS, K_DIM, KERNEL - 1)
    K = conv_state[KEY_DIM:2 * KEY_DIM].reshape(N_K_HEADS, K_DIM, KERNEL - 1)
    V = conv_state[2 * KEY_DIM:].reshape(N_V_HEADS, V_DIM, KERNEL - 1)
    slabs = []
    for chip_i in range(NCHIPS):
        slab = np.concatenate([
            Q[chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP].reshape(KEY_DIM_CHIP, KERNEL - 1),
            K[chip_i * NK_PER_CHIP:(chip_i + 1) * NK_PER_CHIP].reshape(KEY_DIM_CHIP, KERNEL - 1),
            V[chip_i * NV_PER_CHIP:(chip_i + 1) * NV_PER_CHIP].reshape(VAL_DIM_CHIP, KERNEL - 1),
        ], axis=0)
        slabs.append(slab)
    return np.concatenate(slabs, axis=0)


def tp_forward(mesh, x, w_in_sharded_np, w_conv_sharded_np, dt_bias_np,
               A_log_np, w_out_np, ssm_np, conv_state_np):
    """
    4-chip TP DeltaNet forward (no z-gate / per-head rms_norm for probe).
    Returns stacked [4, 1, HIDDEN] (each chip should hold identical result
    after all_reduce).
    """
    # Upload sharded weights
    x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                           device=mesh, layout=ttnn.TILE_LAYOUT,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    w_in_tt = ttnn.from_torch(torch.from_numpy(w_in_sharded_np),
                              dtype=ttnn.bfloat16, device=mesh,
                              layout=ttnn.TILE_LAYOUT,
                              mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
    w_conv_tt = ttnn.from_torch(torch.from_numpy(w_conv_sharded_np),
                                dtype=ttnn.bfloat16, device=mesh,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    # dt_bias [N_V_HEADS] → per-chip [NV_PER_CHIP], split along dim=0
    dt_bias_tt = ttnn.from_torch(torch.from_numpy(dt_bias_np),
                                  dtype=ttnn.bfloat16, device=mesh,
                                  layout=ttnn.TILE_LAYOUT,
                                  mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    A_log_tt = ttnn.from_torch(torch.from_numpy(A_log_np),
                                dtype=ttnn.bfloat16, device=mesh,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    # out_proj [VAL_DIM, HIDDEN] → per-chip [VAL_DIM_CHIP, HIDDEN], split dim=0
    w_out_tt = ttnn.from_torch(torch.from_numpy(w_out_np),
                                dtype=ttnn.bfloat16, device=mesh,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    # SSM state [N_V_HEADS, K_DIM, V_DIM] → per-chip [NV_PER_CHIP, K_DIM, V_DIM], dim=0
    ssm_tt = ttnn.from_torch(torch.from_numpy(ssm_np),
                              dtype=ttnn.bfloat16, device=mesh,
                              layout=ttnn.TILE_LAYOUT,
                              mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    # Conv state [CONV_DIM, KERNEL-1] → per-chip [CONV_DIM_CHIP, KERNEL-1], dim=0
    conv_state_tt = ttnn.from_torch(torch.from_numpy(conv_state_np),
                                     dtype=ttnn.bfloat16, device=mesh,
                                     layout=ttnn.TILE_LAYOUT,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

    # === per-chip forward ===
    # in_proj
    all_tt = ttnn.linear(x_tt, w_in_tt)  # [1, IN_PROJ_OUT_CHIP]

    # slice into qkv | z | a | b (per-chip widths)
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])

    # conv1d (per-chip)
    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
    conv_input = ttnn.concat([conv_state_tt, mixed_col], dim=-1)
    conv_prod = ttnn.mul(conv_input, w_conv_tt)
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))

    # Split q | k | v
    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])

    # GQA repeat-interleave (Q, K)
    def gqa(t_flat, n_kh, d):
        t = ttnn.reshape(t_flat, [n_kh, 1, d])
        t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t, [n_kh * N_REP, d])

    q = gqa(q_flat, NK_PER_CHIP, K_DIM)  # [NV_PER_CHIP, K_DIM]
    k = gqa(k_flat, NK_PER_CHIP, K_DIM)
    v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])

    # Q/K L2 normalize + Q-scaling
    qq = ttnn.mul(q, q)
    q_n = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
    kk = ttnn.mul(k, k)
    k_n = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
    q_n = ttnn.mul(q_n, 1.0 / (K_DIM ** 0.5))

    # gate/decay/beta (per-chip on NV_PER_CHIP heads)
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, dt_bias_tt)), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(A_log_tt)), softplus_a)
    beta = ttnn.sigmoid(b_tt)
    decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])

    # Recurrence
    H_4d = ttnn.reshape(ssm_tt, [1, NV_PER_CHIP, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay)
    k_col = ttnn.reshape(k_n, [1, NV_PER_CHIP, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [1, NV_PER_CHIP, V_DIM])
    v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                     ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
    H_new = ttnn.add(H_decayed,
                     ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])))
    q_col = ttnn.reshape(q_n, [1, NV_PER_CHIP, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2),
                       [1, VAL_DIM_CHIP])

    # silu(z) gate (skip per-head rms_norm for probe simplicity)
    silu_z = ttnn.silu(z_tt)
    out_gated = ttnn.mul(out, silu_z)

    # out_proj (row-parallel): per-chip [1, VAL_DIM_CHIP] @ [VAL_DIM_CHIP, HIDDEN]
    partial = ttnn.linear(out_gated, w_out_tt)

    # All-reduce
    try:
        result = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        result = ttnn.all_gather(scattered, dim=1)

    stacked = ttnn.to_torch(
        result, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().cpu().numpy()
    return stacked


def main():
    print("=" * 78)
    print("C'7.4: DeltaNet 4-chip TP correctness probe")
    print("=" * 78)
    print(f"N_K_HEADS={N_K_HEADS}  N_V_HEADS={N_V_HEADS}")
    print(f"Per chip: {NK_PER_CHIP} K heads, {NV_PER_CHIP} V heads")
    print(f"         IN_PROJ_OUT_CHIP={IN_PROJ_OUT_CHIP}  CONV_DIM_CHIP={CONV_DIM_CHIP}")

    print("\n[0] FABRIC_1D init...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build random weights + state (seed=42)...")
        rng = np.random.default_rng(42)
        # Use magnitudes representative of production: residual stream has std~1
        # (post RMSNorm), SSM state grows toward unit-magnitude over many steps,
        # conv state matches mixed_qkv post-silu (also unit-magnitude). Random
        # 0.01 scaling makes y_gold tiny (~1e-5) so bf16 quantization dominates.
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        w_in = rng.standard_normal((HIDDEN, IN_PROJ_OUT)).astype(np.float32) / np.sqrt(HIDDEN)
        w_conv = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
        dt_bias = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.1
        A_log = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.5
        w_out = rng.standard_normal((VAL_DIM, HIDDEN)).astype(np.float32) / np.sqrt(VAL_DIM)
        ssm = rng.standard_normal((N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.3
        conv_state = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.5

        print(f"  shapes built")

        print("\n[3] Numpy fp32 gold forward...")
        y_gold, _ = deltanet_np_forward(x, w_in, w_conv, dt_bias, A_log,
                                        w_out, ssm, conv_state)
        print(f"  y_gold: {y_gold.shape}, std={y_gold.std():.6f}, max|y|={np.abs(y_gold).max():.6f}")

        print("\n[4] Re-layout weights for per-chip sharding...")
        w_in_sharded = relayout_in_proj_weight(w_in)
        w_conv_sharded = relayout_conv_weight(w_conv)
        conv_state_sharded = relayout_conv_state(conv_state)
        print(f"  in_proj re-laid: {w_in_sharded.shape}")
        print(f"  conv re-laid:    {w_conv_sharded.shape}")
        print(f"  conv_state re-laid: {conv_state_sharded.shape}")

        print("\n[5] TP forward (4-chip)...")
        y_tp = tp_forward(mesh, x, w_in_sharded, w_conv_sharded, dt_bias,
                          A_log, w_out, ssm, conv_state_sharded)
        print(f"  stacked: {y_tp.shape}")

        y_chip0 = y_tp[0].flatten()
        y_chip3 = y_tp[-1].flatten()
        cos_inter = _cosine(y_chip0, y_chip3)
        print(f"  cos(chip0, chip3) = {cos_inter:.6f}  (should be 1.0)")

        cos_tp = _cosine(y_chip0, y_gold.flatten())
        max_diff = float(np.abs(y_chip0 - y_gold.flatten()).max())
        print(f"  cos(TP, gold) = {cos_tp:.6f}  max|Δ| = {max_diff:.4e}")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        ok = cos_tp >= 0.998 and cos_inter >= 0.999
        print(f"  TP cos: {cos_tp:.6f}  inter-chip cos: {cos_inter:.6f}")
        print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'}")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
