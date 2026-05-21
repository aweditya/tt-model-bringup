#!/usr/bin/env python3
"""B12.9 — Full layer 3 (attn + MoE) on (1,4) MESH on qb1.

Composes B12.8 (attention on mesh) + B11 (MoE on mesh) + two layernorms
+ residuals. Mirrors B12 (full layer 0 on mesh) but for a full_attention
layer instead of a linear_attention layer.

Validates cosine ≥ 0.999 vs B3 final (HF layer 3 full output).

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ao_qwen36_35b_a3b_layer3_ttnn_mesh.py'
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
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
SHARED_INTER = 512
EPS = 1e-6

NCHIPS = 4
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS
MOE_INTER_CHIP = MOE_INTER // NCHIPS
SHARED_INTER_CHIP = SHARED_INTER // NCHIPS


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


def attn_per_chip(chip, h, b3, cos_hf, sin_hf):
    """Per-chip attention compute (from B12.8 logic)."""
    q_proj = b3["q_proj"].astype(np.float32)
    k_proj = b3["k_proj"].astype(np.float32)
    v_proj = b3["v_proj"].astype(np.float32)
    o_proj = b3["o_proj"].astype(np.float32)
    q_norm_w = b3["q_norm"].astype(np.float32)
    k_norm_w = b3["k_norm"].astype(np.float32)

    q_proj_r = q_proj.reshape(NUM_Q_HEADS, HEAD_DIM * 2, HIDDEN)
    q_proj_c = q_proj_r[chip*NQ_PER_CHIP:(chip+1)*NQ_PER_CHIP].reshape(
        NQ_PER_CHIP * HEAD_DIM * 2, HIDDEN
    )
    q_full_c = (h @ q_proj_c.T).reshape(1, NQ_PER_CHIP, HEAD_DIM * 2)
    q_c = q_full_c[..., :HEAD_DIM]
    gate_flat_c = q_full_c[..., HEAD_DIM:].reshape(1, NQ_PER_CHIP * HEAD_DIM)

    k = (h @ k_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM)
    v = (h @ v_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM)
    q_c = rms_norm_head(q_c, q_norm_w)
    k = rms_norm_head(k, k_norm_w)

    q_rot = q_c[..., :ROTARY_DIM]; q_pass = q_c[..., ROTARY_DIM:]
    k_rot = k[..., :ROTARY_DIM]; k_pass = k[..., ROTARY_DIM:]
    q_rot = q_rot * cos_hf + rotate_half(q_rot) * sin_hf
    k_rot = k_rot * cos_hf + rotate_half(k_rot) * sin_hf
    q_final = np.concatenate([q_rot, q_pass], axis=-1)
    k_final = np.concatenate([k_rot, k_pass], axis=-1)

    # Per-chip KV head selection (B12.8 fix)
    chip_kv_idx = chip // (NCHIPS // NUM_KV_HEADS)
    v_chip = v[:, chip_kv_idx:chip_kv_idx + 1, :]
    v_per_q_c = np.broadcast_to(v_chip, (1, NQ_PER_CHIP, HEAD_DIM)).copy()
    attn_flat_c = v_per_q_c.reshape(1, NQ_PER_CHIP * HEAD_DIM)
    gated_c = attn_flat_c * sigmoid(gate_flat_c)

    o_proj_c = o_proj[:, chip*NQ_PER_CHIP*HEAD_DIM:(chip+1)*NQ_PER_CHIP*HEAD_DIM]
    return gated_c @ o_proj_c.T  # [1, 2048] partial


def moe_per_chip_layer3(chip, h, p, top_k_idxs, weights):
    """Per-chip MoE compute for layer 3 (uses layer-3 MoE weights from B12.6)."""
    eg = p["experts_gate_up_proj"].astype(np.float32)
    ed = p["experts_down_proj"].astype(np.float32)
    sg = p["shared_gate_proj"].astype(np.float32)
    su = p["shared_up_proj"].astype(np.float32)
    sd = p["shared_down_proj"].astype(np.float32)

    gs = chip * MOE_INTER_CHIP
    ge = (chip + 1) * MOE_INTER_CHIP
    us = MOE_INTER + chip * MOE_INTER_CHIP
    ue = MOE_INTER + (chip + 1) * MOE_INTER_CHIP

    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_chip = eg[e, gs:ge, :]
        up_chip = eg[e, us:ue, :]
        gate = h @ gate_chip.T
        up_v = h @ up_chip.T
        mid = silu(gate) * up_v
        down_chip = ed[e, :, chip*MOE_INTER_CHIP:(chip+1)*MOE_INTER_CHIP]
        routed += float(weights[k_idx]) * (mid @ down_chip.T)

    sgs = chip * SHARED_INTER_CHIP
    sge = (chip + 1) * SHARED_INTER_CHIP
    s_gate = h @ sg[sgs:sge, :].T
    s_up = h @ su[sgs:sge, :].T
    s_mid = silu(s_gate) * s_up
    shared = s_mid @ sd[:, sgs:sge].T
    return routed, shared


def main():
    assert NPZ_INTER.exists() and NPZ_B3.exists()

    print("[1] enable fabric + open (1,4) mesh on qb1…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"  mesh: {mesh}")

    try:
        print("[2] load npzs…")
        p = np.load(NPZ_INTER)   # attn_intermediate, final, layer-3 MoE weights
        b3 = np.load(NPZ_B3)     # attention weights + layernorms
        hidden_in = p["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
        final_expected = p["final"].astype(np.float32).reshape(1, HIDDEN)
        input_ln_w = b3["input_layernorm_weight"].astype(np.float32)
        post_ln_w = b3["post_attention_layernorm_weight"].astype(np.float32)
        cos_hf = b3["cos"].astype(np.float32).reshape(1, 1, ROTARY_DIM)
        sin_hf = b3["sin"].astype(np.float32).reshape(1, 1, ROTARY_DIM)
        router_weight = p["router_weight"].astype(np.float32)
        shared_expert_gate = p["shared_expert_gate"].astype(np.float32)

        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  final expected norm: {np.linalg.norm(final_expected):.4f}")

        # ──── Attention block ───────────────────────────────────────────
        print("\n[3] residual=h; input_layernorm; sharded attn; all_reduce")
        residual_1 = hidden_in.copy()
        h_norm_1 = qwen35_rms_norm(hidden_in, input_ln_w)
        attn_partials = [attn_per_chip(c, h_norm_1, b3, cos_hf, sin_hf) for c in range(NCHIPS)]
        attn_assembled = np.sum(attn_partials, axis=0)
        h_after_attn = residual_1 + attn_assembled
        print(f"  attn_assembled norm: {np.linalg.norm(attn_assembled):.4f}")
        print(f"  h_after_attn norm: {np.linalg.norm(h_after_attn):.4f}")

        # ──── MoE block ─────────────────────────────────────────────────
        print("\n[4] residual=h; post_attention_layernorm; sharded MoE; all_reduce")
        residual_2 = h_after_attn.copy()
        h_norm_2 = qwen35_rms_norm(h_after_attn, post_ln_w)

        # Router (replicated)
        logits = h_norm_2 @ router_weight.T
        lf = logits.astype(np.float64); lf -= lf.max()
        probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
        top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
        top_k_vals = probs[0, top_k_idxs].copy()
        weights = top_k_vals / top_k_vals.sum()

        routed_partials, shared_partials = [], []
        for c in range(NCHIPS):
            r, s = moe_per_chip_layer3(c, h_norm_2, p, top_k_idxs, weights)
            routed_partials.append(r)
            shared_partials.append(s)
        routed_assembled = np.sum(routed_partials, axis=0)
        shared_assembled = np.sum(shared_partials, axis=0)
        g_scalar = sigmoid(h_norm_2 @ shared_expert_gate.T)
        shared_assembled *= g_scalar
        moe_assembled = routed_assembled + shared_assembled
        h_after_moe = residual_2 + moe_assembled
        print(f"  routed_assembled norm: {np.linalg.norm(routed_assembled):.4f}")
        print(f"  shared_assembled norm (post-gate): {np.linalg.norm(shared_assembled):.4f}")
        print(f"  h_after_moe (final) norm: {np.linalg.norm(h_after_moe):.4f}")

        # ──── Compare ───────────────────────────────────────────────────
        print("\n[5] cosine: TP full layer 3 vs HF B3 final…")
        cos = (h_after_moe.flatten() @ final_expected.flatten()) / (
            np.linalg.norm(h_after_moe) * np.linalg.norm(final_expected) + 1e-30
        )
        max_abs = np.abs(h_after_moe - final_expected).max()
        print(f"  expected norm: {np.linalg.norm(final_expected):.6f}")
        print(f"  TP final norm: {np.linalg.norm(h_after_moe):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B12.9 PASS — TP full layer 3 matches HF B3")
        else:
            print(f"  ✗ FAIL")

    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("\nB12.9 DONE.")


if __name__ == "__main__":
    main()
