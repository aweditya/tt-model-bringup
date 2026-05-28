#!/usr/bin/env python3
"""B9 — Qwen3.6-35B-A3B FULL layer 0 ttnn forward on qb1 single chip.

Composes B7 (DN block) + B8 (MoE block) with the two RMSNorms + residual
adds that make a complete `Qwen3_5MoeDecoderLayer.forward()`. Validates
cosine ≥ 0.999 vs B2's HF reference output.

This is the first end-to-end ttnn equivalent of one transformer block in
our backbone. Once this passes, B12 (mesh) and B14 (server integration)
are mostly composition.

Layer 0 forward (HF source, modeling_qwen3_5_moe.py:848-881):

    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)          # Qwen3_5MoeRMSNorm:
                                                            #   norm * (1.0 + weight)
    hidden_states = linear_attn(hidden_states, cache)       # DN (B7)
    hidden_states = residual + hidden_states                # residual

    residual = hidden_states
    hidden_states = post_attention_layernorm(hidden_states) # same Qwen3_5MoeRMSNorm
    hidden_states = mlp(hidden_states)                      # MoE (B8)
    hidden_states = residual + hidden_states                # residual

    return hidden_states

Key gotcha: Qwen3_5MoeRMSNorm applies `(1.0 + weight)`, NOT `weight`.
(Different from typical Llama-style RMSNorm.) Weights stored bf16 near
zero; the +1.0 supplies the multiplicative offset.

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ag_qwen36_35b_a3b_layer0_ttnn_full.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn


CACHE = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
NPZ_B0 = CACHE / "b0_dn_layer0_reference.npz"   # DN weights + input + DN-only output
NPZ_B1 = CACHE / "b1_moe_layer0_reference.npz"  # MoE weights + MoE-only output
NPZ_B2 = CACHE / "b2_layer0_full_reference.npz" # full-layer reference + layernorm weights

HIDDEN = 2048
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_KERNEL = 4
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
EPS = 1e-6


def to_ttnn(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(arr.astype(np.float32))
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def from_ttnn(t):
    return ttnn.to_torch(t).float().numpy()


def matmul_ttnn(hidden_np, weight_np_in_to_out, device):
    """[1,D] @ [D,F] → [1,F] via ttnn.matmul. weight already in [in, out]."""
    h_tt = to_ttnn(hidden_np.reshape(1, hidden_np.shape[-1]), device)
    w_tt = to_ttnn(weight_np_in_to_out, device)
    out_tt = ttnn.matmul(h_tt, w_tt)
    out_np = from_ttnn(out_tt).reshape(1, -1)
    ttnn.deallocate(h_tt); ttnn.deallocate(w_tt); ttnn.deallocate(out_tt)
    return out_np


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def qwen35_rms_norm(x_np, weight_np, eps=EPS):
    """Qwen3_5MoeRMSNorm — `output * (1.0 + weight)` convention.

    HF source:
      output = x * rsqrt(mean(x^2) + eps)
      output = output * (1.0 + weight.float())
    """
    var = np.mean(x_np ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(var + eps)
    return x_np * rsqrt * (1.0 + weight_np)


# ── DN forward (from B7) ────────────────────────────────────────────────
def dn_forward(hidden_in_np, ref, device,
               conv_state_in_np, recurrent_state_in_np):
    """Full DN block: returns output [1, 1, 2048]."""
    # Project
    mixed_qkv = matmul_ttnn(hidden_in_np, ref["in_proj_qkv"].astype(np.float32).T, device)  # [1, 8192]
    z = matmul_ttnn(hidden_in_np, ref["in_proj_z"].astype(np.float32).T, device).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
    a = matmul_ttnn(hidden_in_np, ref["in_proj_a"].astype(np.float32).T, device).reshape(NUM_V_HEADS)
    b = matmul_ttnn(hidden_in_np, ref["in_proj_b"].astype(np.float32).T, device).reshape(NUM_V_HEADS)
    # Conv1d update + silu (numpy)
    new_cs = np.zeros_like(conv_state_in_np)
    new_cs[:, :, :CONV_KERNEL - 1] = conv_state_in_np[:, :, 1:]
    new_cs[:, :, CONV_KERNEL - 1] = mixed_qkv
    conv_w = ref["conv1d_weight"].astype(np.float32)[:, 0, :]  # [8192, 4]
    conv_out = np.sum(new_cs * conv_w[None, :, :], axis=-1)
    silu_out = silu(conv_out)
    # Split q/k/v
    q_flat = silu_out[:, :KEY_DIM]
    k_flat = silu_out[:, KEY_DIM:2 * KEY_DIM]
    v_flat = silu_out[:, 2 * KEY_DIM:]
    q_per_head = q_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)
    k_per_head = k_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)
    v_per_head = v_flat.reshape(1, NUM_V_HEADS, HEAD_V_DIM)
    # beta + g (decay)
    beta = sigmoid(b)
    softplus_ab = np.log1p(np.exp((a + ref["dt_bias"].astype(np.float32)).astype(np.float64))).astype(np.float32)
    g_decay = np.exp(-np.exp(ref["A_log"].astype(np.float32)) * softplus_ab)
    # l2_norm + repeat 16→32
    def l2norm(x, eps=1e-6):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    q_norm = l2norm(q_per_head)
    k_norm = l2norm(k_per_head)
    rep = NUM_V_HEADS // NUM_K_HEADS
    q_rep = np.repeat(q_norm, rep, axis=1)
    k_rep = np.repeat(k_norm, rep, axis=1)
    # Recurrence (numpy, single token)
    scale = 1.0 / np.sqrt(HEAD_K_DIM)
    q_scaled = q_rep * scale
    state = recurrent_state_in_np.copy()
    g_b = g_decay[None, :, None, None]
    beta_b = beta[None, :, None]
    state = state * g_b
    kv_mem = np.sum(state * k_rep[:, :, :, None], axis=-2)
    delta = (v_per_head - kv_mem) * beta_b
    state = state + k_rep[:, :, :, None] * delta[:, :, None, :]
    core_attn_out = np.sum(state * q_scaled[:, :, :, None], axis=-2)
    # RMSNormGated (out, z): per-head rms norm * silu(z), weight applied as `weight` (NOT 1+weight)
    norm_weight = ref["norm_weight"].astype(np.float32)
    core_flat = core_attn_out.reshape(-1, HEAD_V_DIM)
    z_flat = z.reshape(-1, HEAD_V_DIM)
    var = np.mean(core_flat ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(var + EPS)
    normed = core_flat * rsqrt * norm_weight[None, :]
    silu_z = z_flat * sigmoid(z_flat)
    gated = normed * silu_z
    gated_reshaped = gated.reshape(1, VALUE_DIM)
    # out_proj
    out = matmul_ttnn(gated_reshaped, ref["out_proj"].astype(np.float32).T, device)
    return out.reshape(1, 1, HIDDEN), state


# ── MoE forward (from B8) ───────────────────────────────────────────────
def moe_forward(hidden_in_np, ref, device):
    """Full MoE block: returns output [1, 1, 2048]."""
    hidden_flat = hidden_in_np.reshape(1, HIDDEN)
    router_weight = ref["router_weight"].astype(np.float32)
    experts_gate_up = ref["experts_gate_up_proj"].astype(np.float32)
    experts_down = ref["experts_down_proj"].astype(np.float32)
    shared_gate = ref["shared_gate_proj"].astype(np.float32)
    shared_up = ref["shared_up_proj"].astype(np.float32)
    shared_down = ref["shared_down_proj"].astype(np.float32)
    shared_expert_gate = ref["shared_expert_gate"].astype(np.float32)
    # Router
    logits = matmul_ttnn(hidden_flat, router_weight.T, device)
    logits_fp32 = logits.astype(np.float64)
    logits_fp32 -= logits_fp32.max()
    exps = np.exp(logits_fp32)
    probs = (exps / exps.sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
    top_k_vals = probs[0, top_k_idxs].copy()
    weights = top_k_vals / top_k_vals.sum()
    # Routed
    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_up = matmul_ttnn(hidden_flat, experts_gate_up[e].T, device)
        gate = gate_up[:, :MOE_INTER]; up = gate_up[:, MOE_INTER:]
        mid = silu(gate) * up
        expert_out = matmul_ttnn(mid, experts_down[e].T, device)
        routed += float(weights[k_idx]) * expert_out
    # Shared
    s_gate = matmul_ttnn(hidden_flat, shared_gate.T, device)
    s_up = matmul_ttnn(hidden_flat, shared_up.T, device)
    s_mid = silu(s_gate) * s_up
    shared = matmul_ttnn(s_mid, shared_down.T, device)
    g_scalar = sigmoid(matmul_ttnn(hidden_flat, shared_expert_gate.T, device))
    shared *= g_scalar
    return (routed + shared).reshape(1, 1, HIDDEN)


def main():
    assert NPZ_B0.exists() and NPZ_B1.exists() and NPZ_B2.exists(), \
        "need all three npzs at " + str(CACHE)

    print("[1] open device on qb1…")
    device = ttnn.open_device(device_id=0)

    try:
        print("[2] load all 3 npzs…")
        b0 = np.load(NPZ_B0)
        b1 = np.load(NPZ_B1)
        b2 = np.load(NPZ_B2)

        hidden_in = b2["hidden_in"].astype(np.float32)              # [1, 1, 2048]
        expected_output = b2["output"].astype(np.float32)           # [1, 1, 2048]
        input_ln_w = b2["input_layernorm_weight"].astype(np.float32)
        post_ln_w = b2["post_attention_layernorm_weight"].astype(np.float32)
        conv_state_init = b0["conv_state_in"].astype(np.float32)
        recurrent_state_init = b0["recurrent_state_in"].astype(np.float32)

        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  expected_output norm: {np.linalg.norm(expected_output):.4f}")
        print(f"  input_layernorm_weight stats: "
              f"min={input_ln_w.min():.4f} max={input_ln_w.max():.4f} (applied as 1+w)")
        print(f"  post_attention_layernorm_weight stats: "
              f"min={post_ln_w.min():.4f} max={post_ln_w.max():.4f}")

        # ── DN block ───────────────────────────────────────────────────
        print("\n[3] residual = hidden; input_layernorm(hidden)…")
        residual_1 = hidden_in.copy()
        h_norm_1 = qwen35_rms_norm(hidden_in.reshape(1, HIDDEN), input_ln_w).reshape(1, 1, HIDDEN)
        print(f"  post-input_layernorm norm: {np.linalg.norm(h_norm_1):.4f}")

        print("[4] DN forward (B7 logic)…")
        dn_out, _ = dn_forward(h_norm_1.reshape(1, HIDDEN), b0, device,
                                conv_state_init, recurrent_state_init)
        h_after_dn = residual_1 + dn_out
        print(f"  dn_out norm: {np.linalg.norm(dn_out):.4f}")
        print(f"  h after residual 1: {np.linalg.norm(h_after_dn):.4f}")

        # ── MoE block ──────────────────────────────────────────────────
        print("\n[5] residual = h_after_dn; post_attention_layernorm(h)…")
        residual_2 = h_after_dn.copy()
        h_norm_2 = qwen35_rms_norm(h_after_dn.reshape(1, HIDDEN), post_ln_w).reshape(1, 1, HIDDEN)
        print(f"  post-post-attn-layernorm norm: {np.linalg.norm(h_norm_2):.4f}")

        print("[6] MoE forward (B8 logic)…")
        moe_out = moe_forward(h_norm_2.reshape(1, HIDDEN), b1, device)
        h_after_moe = residual_2 + moe_out
        print(f"  moe_out norm: {np.linalg.norm(moe_out):.4f}")
        print(f"  h after residual 2 (final): {np.linalg.norm(h_after_moe):.4f}")

        # ── Compare ────────────────────────────────────────────────────
        print("\n[7] cosine vs B2 expected output…")
        cos = (h_after_moe.flatten() @ expected_output.flatten()) / (
            np.linalg.norm(h_after_moe) * np.linalg.norm(expected_output) + 1e-30
        )
        max_abs = np.abs(h_after_moe - expected_output).max()
        print(f"  expected norm: {np.linalg.norm(expected_output):.6f}")
        print(f"  ttnn final norm: {np.linalg.norm(h_after_moe):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B9 PASS — full layer 0 ttnn output matches HF reference")
        elif cos > 0.99:
            print(f"  ⚠ cos {cos:.4f} above 0.99 but below 0.999 — debug per-step")
        else:
            print("  ✗ FAIL")

    finally:
        ttnn.close_device(device)

    print("\nB9 DONE.")


if __name__ == "__main__":
    main()
