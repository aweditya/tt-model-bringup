#!/usr/bin/env python3
"""
C'7.5a: Chain DeltaNet TP + MLP TP on synthetic "layer" — multi-stage TP plumbing.

Goal: confirm the residual stream chain works across layers. The TP pattern
is: after every sub-layer (DeltaNet, MLP), all_reduce produces a replicated
residual on every chip. Next sub-layer takes that replicated input as its
own replicated x. If that chain breaks (e.g. layouts incompatible, or
collective output isn't really replicated), we'd catch it here BEFORE
loading 27B real weights.

This is C'7.5a — synthetic chain, random weights, validates the plumbing.
C'7.5b will load layer-0 real weights and validate cosine vs Branch III
single-chip ref.

Pipeline:
  x_replicated [1, HIDDEN]
    → DeltaNet TP (head-sharded) → all_reduce → residual_after_dn
    → MLP TP (column-parallel gate/up + row-parallel down + all_reduce) → residual_after_mlp

Numpy gold: same math in fp32 (skips per-head rms_norm + z-gate variant
that differs in production; this is plumbing, not full math validation).
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
KEY_DIM = N_K_HEADS * K_DIM        # 2048
VAL_DIM = N_V_HEADS * V_DIM        # 4096
CONV_DIM = 2 * KEY_DIM + VAL_DIM    # 8192
N_REP = N_V_HEADS // N_K_HEADS      # 2
IN_PROJ_OUT = CONV_DIM + VAL_DIM + 2 * N_V_HEADS  # 12352
INTERMEDIATE = 25600
EPS = 1e-6

NCHIPS = 4
NK_PER_CHIP = N_K_HEADS // NCHIPS
NV_PER_CHIP = N_V_HEADS // NCHIPS
KEY_DIM_CHIP = NK_PER_CHIP * K_DIM
VAL_DIM_CHIP = NV_PER_CHIP * V_DIM
CONV_DIM_CHIP = 2 * KEY_DIM_CHIP + VAL_DIM_CHIP
IN_PROJ_OUT_CHIP = CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP
INTERMEDIATE_CHIP = INTERMEDIATE // NCHIPS


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


# === Numpy gold ===

def deltanet_np(x, w_in, w_conv, dt_bias, A_log, w_out, ssm, conv_state):
    all_p = x @ w_in
    mixed_qkv = all_p[:, :CONV_DIM]
    z_p = all_p[:, CONV_DIM:CONV_DIM + VAL_DIM]
    a_p = all_p[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + N_V_HEADS]
    b_p = all_p[:, CONV_DIM + VAL_DIM + N_V_HEADS:]

    mixed_col = mixed_qkv.reshape(CONV_DIM, 1)
    conv_input = np.concatenate([conv_state, mixed_col], axis=-1)
    conv_prod = conv_input * w_conv
    conv_out = silu_np(conv_prod.sum(axis=-1))

    q_flat = conv_out[:KEY_DIM]
    k_flat = conv_out[KEY_DIM:2 * KEY_DIM]
    v_flat = conv_out[2 * KEY_DIM:]

    def interleave(t, n_kh, d):
        t2 = t.reshape(n_kh, 1, d)
        t3 = np.broadcast_to(t2, (n_kh, N_REP, d)).copy()
        return t3.reshape(n_kh * N_REP, d)

    q = interleave(q_flat, N_K_HEADS, K_DIM)
    k = interleave(k_flat, N_K_HEADS, K_DIM)
    v = v_flat.reshape(N_V_HEADS, V_DIM)

    q_n = q / np.sqrt((q * q).sum(axis=-1, keepdims=True) + EPS)
    k_n = k / np.sqrt((k * k).sum(axis=-1, keepdims=True) + EPS)
    q_n = q_n * (1.0 / np.sqrt(K_DIM))

    softplus_a = np.log(np.exp(a_p[0] + dt_bias) + 1.0)
    g = -np.exp(A_log) * softplus_a
    beta = 1.0 / (1.0 + np.exp(-b_p[0]))
    decay = np.exp(g).reshape(N_V_HEADS, 1, 1)

    H = ssm.copy()
    H_decayed = H * decay
    k_col = k_n.reshape(N_V_HEADS, K_DIM, 1)
    kv_mem = (H_decayed * k_col).sum(axis=-2)
    delta = (v - kv_mem) * beta.reshape(N_V_HEADS, 1)
    H_new = H_decayed + k_col * delta.reshape(N_V_HEADS, 1, V_DIM)
    q_col = q_n.reshape(N_V_HEADS, K_DIM, 1)
    out = (H_new * q_col).sum(axis=-2)

    z = z_p.reshape(N_V_HEADS, V_DIM)
    silu_z = silu_np(z)
    out_gated = (out * silu_z).reshape(1, VAL_DIM)

    return x + out_gated @ w_out


def mlp_np(x, w_gate, w_up, w_down):
    h = silu_np(x @ w_gate) * (x @ w_up)
    return x + h @ w_down


def full_layer_np(x, w_in, w_conv, dt_bias, A_log, w_out, ssm, conv_state,
                  w_gate, w_up, w_down):
    after_dn = deltanet_np(x, w_in, w_conv, dt_bias, A_log, w_out, ssm, conv_state)
    return mlp_np(after_dn, w_gate, w_up, w_down)


# === TP re-layout helpers (from existing probes) ===

def relayout_in_proj(w_in):
    Q = w_in[:, :KEY_DIM].reshape(HIDDEN, N_K_HEADS, K_DIM)
    K = w_in[:, KEY_DIM:2 * KEY_DIM].reshape(HIDDEN, N_K_HEADS, K_DIM)
    V = w_in[:, 2 * KEY_DIM:CONV_DIM].reshape(HIDDEN, N_V_HEADS, V_DIM)
    Z = w_in[:, CONV_DIM:CONV_DIM + VAL_DIM].reshape(HIDDEN, N_V_HEADS, V_DIM)
    A = w_in[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + N_V_HEADS]
    B = w_in[:, CONV_DIM + VAL_DIM + N_V_HEADS:]
    slabs = []
    for i in range(NCHIPS):
        slabs.append(np.concatenate([
            Q[:, i*NK_PER_CHIP:(i+1)*NK_PER_CHIP].reshape(HIDDEN, KEY_DIM_CHIP),
            K[:, i*NK_PER_CHIP:(i+1)*NK_PER_CHIP].reshape(HIDDEN, KEY_DIM_CHIP),
            V[:, i*NV_PER_CHIP:(i+1)*NV_PER_CHIP].reshape(HIDDEN, VAL_DIM_CHIP),
            Z[:, i*NV_PER_CHIP:(i+1)*NV_PER_CHIP].reshape(HIDDEN, VAL_DIM_CHIP),
            A[:, i*NV_PER_CHIP:(i+1)*NV_PER_CHIP],
            B[:, i*NV_PER_CHIP:(i+1)*NV_PER_CHIP],
        ], axis=1))
    return np.concatenate(slabs, axis=1)


def relayout_conv(w_conv):
    Q = w_conv[:KEY_DIM].reshape(N_K_HEADS, K_DIM, -1)
    K = w_conv[KEY_DIM:2*KEY_DIM].reshape(N_K_HEADS, K_DIM, -1)
    V = w_conv[2*KEY_DIM:].reshape(N_V_HEADS, V_DIM, -1)
    last_dim = w_conv.shape[1]
    slabs = []
    for i in range(NCHIPS):
        slabs.append(np.concatenate([
            Q[i*NK_PER_CHIP:(i+1)*NK_PER_CHIP].reshape(KEY_DIM_CHIP, last_dim),
            K[i*NK_PER_CHIP:(i+1)*NK_PER_CHIP].reshape(KEY_DIM_CHIP, last_dim),
            V[i*NV_PER_CHIP:(i+1)*NV_PER_CHIP].reshape(VAL_DIM_CHIP, last_dim),
        ], axis=0))
    return np.concatenate(slabs, axis=0)


# === TP forward functions ===

def deltanet_tp(mesh, x_tt, sharded_state):
    """One DeltaNet step on the mesh. Takes x_tt replicated, returns residual replicated."""
    all_tt = ttnn.linear(x_tt, sharded_state['w_in'])
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])

    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
    conv_input = ttnn.concat([sharded_state['conv_st'], mixed_col], dim=-1)
    conv_prod = ttnn.mul(conv_input, sharded_state['w_conv'])
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))

    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])

    def gqa(t_flat, n_kh, d):
        t = ttnn.reshape(t_flat, [n_kh, 1, d])
        t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t, [n_kh * N_REP, d])

    q = gqa(q_flat, NK_PER_CHIP, K_DIM)
    k = gqa(k_flat, NK_PER_CHIP, K_DIM)
    v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])

    qq = ttnn.mul(q, q)
    q_n = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
    kk = ttnn.mul(k, k)
    k_n = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
    q_n = ttnn.mul(q_n, 1.0 / (K_DIM ** 0.5))

    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, sharded_state['dt_bias'])), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(sharded_state['A_log'])), softplus_a)
    beta = ttnn.sigmoid(b_tt)
    decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])

    H_4d = ttnn.reshape(sharded_state['ssm'], [1, NV_PER_CHIP, K_DIM, V_DIM])
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
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM_CHIP])

    silu_z = ttnn.silu(z_tt)
    out_gated = ttnn.mul(out, silu_z)
    partial = ttnn.linear(out_gated, sharded_state['w_out'])

    # All-reduce → replicated residual
    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    return ttnn.add(x_tt, reduced)


def mlp_tp(mesh, x_tt, sharded_state):
    """MLP TP step. Takes replicated x, returns replicated residual."""
    g = ttnn.linear(x_tt, sharded_state['w_gate'], activation="silu")
    u = ttnn.linear(x_tt, sharded_state['w_up'])
    h = ttnn.mul(g, u)
    partial = ttnn.linear(h, sharded_state['w_down'])
    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    return ttnn.add(x_tt, reduced)


def main():
    print("=" * 78)
    print("C'7.5a: Full layer TP chain (DeltaNet → MLP)")
    print("=" * 78)

    print("\n[0] FABRIC_1D init...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    print("[1] Open mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print("\n[2] Build random weights + state (seed=42)...")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32)
        # DeltaNet weights
        w_in = rng.standard_normal((HIDDEN, IN_PROJ_OUT)).astype(np.float32) / np.sqrt(HIDDEN)
        w_conv = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
        dt_bias = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.1
        A_log = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.5
        w_dn_out = rng.standard_normal((VAL_DIM, HIDDEN)).astype(np.float32) / np.sqrt(VAL_DIM)
        ssm = rng.standard_normal((N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.3
        conv_state = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.5
        # MLP weights
        w_gate = rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32) / np.sqrt(HIDDEN)
        w_up = rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32) / np.sqrt(HIDDEN)
        w_down = rng.standard_normal((INTERMEDIATE, HIDDEN)).astype(np.float32) / np.sqrt(INTERMEDIATE)

        print("\n[3] Numpy fp32 gold (DeltaNet → MLP)...")
        y_gold = full_layer_np(x, w_in, w_conv, dt_bias, A_log, w_dn_out, ssm, conv_state,
                                w_gate, w_up, w_down)
        print(f"  y_gold: {y_gold.shape}, std={y_gold.std():.4f}, max|y|={np.abs(y_gold).max():.4f}")

        print("\n[4] Re-layout weights for per-chip sharding...")
        w_in_sh = relayout_in_proj(w_in)
        w_conv_sh = relayout_conv(w_conv)
        conv_state_sh = relayout_conv(conv_state)

        print("\n[5] Upload sharded state...")
        def upload_shard(arr, dim, dtype=ttnn.bfloat16):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                    device=mesh, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim))
        def upload_repl(arr, dtype=ttnn.bfloat16):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                    device=mesh, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        dn_state = {
            'w_in':    upload_shard(w_in_sh, dim=1),
            'w_conv':  upload_shard(w_conv_sh, dim=0),
            'conv_st': upload_shard(conv_state_sh, dim=0),
            'dt_bias': upload_shard(dt_bias, dim=0),
            'A_log':   upload_shard(A_log, dim=0),
            'w_out':   upload_shard(w_dn_out, dim=0),
            'ssm':     upload_shard(ssm, dim=0),
        }
        mlp_state = {
            'w_gate':  upload_shard(w_gate, dim=1),
            'w_up':    upload_shard(w_up, dim=1),
            'w_down':  upload_shard(w_down, dim=0),
        }
        x_tt = upload_repl(x)

        print("\n[6] Run TP chain (DeltaNet → MLP)...")
        after_dn = deltanet_tp(mesh, x_tt, dn_state)
        after_mlp = mlp_tp(mesh, after_dn, mlp_state)
        stacked = ttnn.to_torch(
            after_mlp, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().cpu().numpy()
        print(f"  stacked shape: {stacked.shape}")

        chip0 = stacked[0].flatten()
        chip3 = stacked[-1].flatten()
        cos_inter = _cosine(chip0, chip3)
        cos_vs_gold = _cosine(chip0, y_gold.flatten())
        max_diff = float(np.abs(chip0 - y_gold.flatten()).max())

        print(f"\n  cos(chip0, chip3) = {cos_inter:.6f}  (should be 1.0)")
        print(f"  cos(TP chain, numpy gold) = {cos_vs_gold:.6f}  max|Δ| = {max_diff:.4e}")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        ok = cos_vs_gold >= 0.998 and cos_inter >= 0.999
        print(f"  TP chain: cos={cos_vs_gold:.6f}  inter-chip={cos_inter:.6f}")
        print(f"  Result: {'✓ PASS — full TP layer chain works' if ok else '✗ FAIL'}")
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
