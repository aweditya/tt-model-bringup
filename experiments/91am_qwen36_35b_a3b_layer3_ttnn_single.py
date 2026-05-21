#!/usr/bin/env python3
"""B12.7 — Full layer 3 (attn + MoE) on qb1 single-chip ttnn.

Uses B12.6's npz (`b3p_layer3_intermediates.npz`) which has both:
  - `attn_intermediate`: post-attn pre-MoE state (tight cosine target for
    the attention block alone)
  - `final`: full layer-3 output (matches B3's final)
  - Layer 3 MoE weights (router, experts, shared)

Validates TWO gates:
  1. Attention-only cosine: our ttnn attn output vs HF attn_intermediate
  2. Full layer-3 cosine: our ttnn (attn + MoE) vs HF final

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91am_qwen36_35b_a3b_layer3_ttnn_single.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn

NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b3p_layer3_intermediates.npz"
B3_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b3_layer3_full_reference.npz"

HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
EPS = 1e-6


def to_ttnn(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(arr.astype(np.float32))
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def from_ttnn(t):
    return ttnn.to_torch(t).float().numpy()


def matmul_ttnn(h_np, w_in_to_out_np, device):
    h_tt = to_ttnn(h_np.reshape(1, h_np.shape[-1]), device)
    w_tt = to_ttnn(w_in_to_out_np, device)
    o_tt = ttnn.matmul(h_tt, w_tt)
    o_np = from_ttnn(o_tt).reshape(1, -1)
    ttnn.deallocate(h_tt); ttnn.deallocate(w_tt); ttnn.deallocate(o_tt)
    return o_np


def silu(x): return x * (1.0 / (1.0 + np.exp(-x)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def qwen35_rms_norm(x, w, eps=EPS):
    """Layer-level Qwen3_5MoeRMSNorm: output * (1 + weight)."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * (1.0 + w)


def rms_norm_head(x, w, eps=EPS):
    """Attention-internal q_norm/k_norm: standard output * weight."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def attn_forward(h_in, b3, device):
    """Single-chip ttnn attention forward (B12.5 logic)."""
    q_proj = b3["q_proj"].astype(np.float32)
    k_proj = b3["k_proj"].astype(np.float32)
    v_proj = b3["v_proj"].astype(np.float32)
    o_proj = b3["o_proj"].astype(np.float32)
    q_norm_w = b3["q_norm"].astype(np.float32)
    k_norm_w = b3["k_norm"].astype(np.float32)
    cos_hf = b3["cos"].astype(np.float32).reshape(1, 1, ROTARY_DIM)
    sin_hf = b3["sin"].astype(np.float32).reshape(1, 1, ROTARY_DIM)

    q_full = matmul_ttnn(h_in, q_proj.T, device).reshape(1, NUM_Q_HEADS, HEAD_DIM * 2)
    q = q_full[..., :HEAD_DIM]
    gate_flat = q_full[..., HEAD_DIM:].reshape(1, NUM_Q_HEADS * HEAD_DIM)

    k = matmul_ttnn(h_in, k_proj.T, device).reshape(1, NUM_KV_HEADS, HEAD_DIM)
    v = matmul_ttnn(h_in, v_proj.T, device).reshape(1, NUM_KV_HEADS, HEAD_DIM)

    q = rms_norm_head(q, q_norm_w)
    k = rms_norm_head(k, k_norm_w)

    # Partial RoPE
    q_rot = q[..., :ROTARY_DIM]; q_pass = q[..., ROTARY_DIM:]
    k_rot = k[..., :ROTARY_DIM]; k_pass = k[..., ROTARY_DIM:]
    q_rotated = q_rot * cos_hf + rotate_half(q_rot) * sin_hf
    k_rotated = k_rot * cos_hf + rotate_half(k_rot) * sin_hf
    q_final = np.concatenate([q_rotated, q_pass], axis=-1)
    k_final = np.concatenate([k_rotated, k_pass], axis=-1)

    # GQA + single-token attention (output = v_per_q)
    v_per_q = np.repeat(v, GQA_GROUP, axis=1)
    attn_flat = v_per_q.reshape(1, NUM_Q_HEADS * HEAD_DIM)
    gated = attn_flat * sigmoid(gate_flat)
    attn_out = matmul_ttnn(gated, o_proj.T, device)
    return attn_out


def moe_forward(h_in, p, device):
    """Layer-3 MoE forward (from p = B12.6 npz)."""
    router_w = p["router_weight"].astype(np.float32)
    eg = p["experts_gate_up_proj"].astype(np.float32)
    ed = p["experts_down_proj"].astype(np.float32)
    sg = p["shared_gate_proj"].astype(np.float32)
    su = p["shared_up_proj"].astype(np.float32)
    sd = p["shared_down_proj"].astype(np.float32)
    seg = p["shared_expert_gate"].astype(np.float32)

    # Router
    logits = matmul_ttnn(h_in, router_w.T, device)
    lf = logits.astype(np.float64); lf -= lf.max()
    probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
    top_k_vals = probs[0, top_k_idxs].copy()
    weights = top_k_vals / top_k_vals.sum()

    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_up = matmul_ttnn(h_in, eg[e].T, device)
        g = gate_up[:, :MOE_INTER]; u = gate_up[:, MOE_INTER:]
        mid = silu(g) * u
        exp_out = matmul_ttnn(mid, ed[e].T, device)
        routed += float(weights[k_idx]) * exp_out

    sg_v = matmul_ttnn(h_in, sg.T, device)
    su_v = matmul_ttnn(h_in, su.T, device)
    s_mid = silu(sg_v) * su_v
    shared = matmul_ttnn(s_mid, sd.T, device)
    g_scalar = sigmoid(matmul_ttnn(h_in, seg.T, device))
    shared *= g_scalar
    return routed + shared


def main():
    assert NPZ_PATH.exists() and B3_PATH.exists()
    print("[1] open device on qb1…")
    device = ttnn.open_device(device_id=0)
    try:
        print("[2] load npzs…")
        p = np.load(NPZ_PATH)   # B12.6: attn_intermediate, final, MoE weights
        b3 = np.load(B3_PATH)   # B3: attn weights + layernorm weights
        hidden_in = p["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
        attn_intermediate = p["attn_intermediate"].astype(np.float32).reshape(1, HIDDEN)
        final_expected = p["final"].astype(np.float32).reshape(1, HIDDEN)
        input_ln_w = b3["input_layernorm_weight"].astype(np.float32)
        post_ln_w = b3["post_attention_layernorm_weight"].astype(np.float32)
        print(f"  hidden_in norm:        {np.linalg.norm(hidden_in):.4f}")
        print(f"  attn_intermediate norm: {np.linalg.norm(attn_intermediate):.4f}")
        print(f"  final norm:            {np.linalg.norm(final_expected):.4f}")

        print("\n[3] Stage 1 — attn-only: residual + attn(input_layernorm(h))")
        residual_1 = hidden_in.copy()
        h_norm_1 = qwen35_rms_norm(hidden_in, input_ln_w)
        attn_proj = attn_forward(h_norm_1, b3, device)
        h_after_attn = residual_1 + attn_proj
        cos_attn = (h_after_attn.flatten() @ attn_intermediate.flatten()) / (
            np.linalg.norm(h_after_attn) * np.linalg.norm(attn_intermediate) + 1e-30
        )
        max_attn = np.abs(h_after_attn - attn_intermediate).max()
        print(f"  ttnn attn norm:   {np.linalg.norm(h_after_attn):.4f}")
        print(f"  HF attn_intermediate: {np.linalg.norm(attn_intermediate):.4f}")
        print(f"  cosine (attn): {cos_attn:.6f}  max|Δ|: {max_attn:.4f}")
        attn_pass = cos_attn > 0.999

        print("\n[4] Stage 2 — full layer 3: + MoE")
        residual_2 = h_after_attn.copy()
        h_norm_2 = qwen35_rms_norm(h_after_attn, post_ln_w)
        moe_out = moe_forward(h_norm_2, p, device)
        h_final = residual_2 + moe_out
        cos_final = (h_final.flatten() @ final_expected.flatten()) / (
            np.linalg.norm(h_final) * np.linalg.norm(final_expected) + 1e-30
        )
        max_final = np.abs(h_final - final_expected).max()
        print(f"  ttnn final norm: {np.linalg.norm(h_final):.4f}")
        print(f"  HF final norm:   {np.linalg.norm(final_expected):.4f}")
        print(f"  cosine (full): {cos_final:.6f}  max|Δ|: {max_final:.4f}")
        full_pass = cos_final > 0.999

        print()
        print(f"  attn: {'✓ PASS' if attn_pass else '✗ FAIL'}  "
              f"full: {'✓ PASS' if full_pass else '✗ FAIL'}")
        if attn_pass and full_pass:
            print("  ✓ B12.7 PASS — full layer 3 ttnn output matches HF reference")

    finally:
        ttnn.close_device(device)
    print("\nB12.7 DONE.")


if __name__ == "__main__":
    main()
