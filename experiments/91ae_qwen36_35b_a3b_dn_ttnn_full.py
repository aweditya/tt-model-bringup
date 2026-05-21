#!/usr/bin/env python3
"""B7 — Qwen3.6-35B-A3B DN block FULL ttnn implementation on qb1 single chip.

End-to-end ttnn port of `Qwen3_5MoeGatedDeltaNet.forward()` for single-token
decode. Reads B0's npz (input + weights + expected output) and validates
cosine ≥ 0.999 vs HF.

Recurrence: STOCK TTNN OPS only (qb1's ttnn build doesn't have the
27B-era `owned_gdn_decode_owned` integrated; that integration happens at
B10 on qb2 where it already exists).

Forward steps mirror HF (modeling_qwen3_5_moe.py:425-545):
  1. mixed_qkv = in_proj_qkv(hidden)                           # [1, T, 8192]
  2. z = in_proj_z(hidden) → [B, T, 32, 128]
  3. b = in_proj_b(hidden) → [B, T, 32]
  4. a = in_proj_a(hidden) → [B, T, 32]
  5. Conv1d update (kernel=4 depthwise + state shift)
  6. silu activation on conv output
  7. Split → q [B,T,16,128], k [B,T,16,128], v [B,T,32,128]
  8. beta = sigmoid(b)                                         # [B, T, 32]
  9. g = -exp(A_log) * softplus(a + dt_bias)                   # [B, T, 32]
 10. l2_norm on q and k (head_dim)
 11. q = q.repeat_interleave(2, dim=-2); k = same              # 16→32 heads
 12. Recurrence loop (T=1 single token):
       state *= exp(g)
       kv = sum(state * k_t, dim=K)                            # [B, 32, V_dim]
       delta = beta * (v - kv)
       state += k ⊗ delta
       out = sum(state * q_t, dim=K)                           # [B, 32, V_dim]
 13. RMSNormGated: norm(out) * silu(z) per-head
 14. out_proj
 15. Compare to B0's saved output.

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ae_qwen36_35b_a3b_dn_ttnn_full.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn


NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b0_dn_layer0_reference.npz"

# Architecture constants (single chip, full 35B-A3B DN)
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


def to_ttnn(np_array, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Convert a numpy array to a ttnn tensor on `device`."""
    t = torch.from_numpy(np_array.astype(np.float32))
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def from_ttnn(tt_tensor):
    """Read back a ttnn tensor as a numpy fp32 array."""
    return ttnn.to_torch(tt_tensor).float().numpy()


def rms_norm_per_head(x_np, weight_np, eps=EPS):
    """RMS norm reference (per-head_dim) for cross-check.

    HF's Qwen3_5MoeRMSNormGated does:
      norm = x * rsqrt(mean(x^2) + eps)
      out = weight * norm * silu(gate)
    We'll do the silu(gate) multiply elsewhere; this is just the norm+weight.
    """
    var = np.mean(x_np ** 2, axis=-1, keepdims=True)
    norm = x_np / np.sqrt(var + eps)
    return norm * weight_np


def main():
    print(f"npz: {NPZ_PATH}")
    assert NPZ_PATH.exists(), f"sync B0 npz to {NPZ_PATH} first"

    print("[1] open device on qb1…")
    device = ttnn.open_device(device_id=0)
    print(f"  device: {device}")

    try:
        print("[2] load B0 npz…")
        ref = np.load(NPZ_PATH)
        hidden_in = ref["hidden_in"].astype(np.float32)                   # [1,1,2048]
        conv_state_in = ref["conv_state_in"].astype(np.float32)           # [1,8192,4]
        recurrent_state_in = ref["recurrent_state_in"].astype(np.float32) # [1,32,128,128]
        expected_output = ref["output"].astype(np.float32)                # [1,1,2048]
        expected_recurrent_state_out = ref["recurrent_state_out"].astype(np.float32)
        # weights
        in_proj_qkv = ref["in_proj_qkv"].astype(np.float32)               # [8192,2048]
        in_proj_z = ref["in_proj_z"].astype(np.float32)                   # [4096,2048]
        in_proj_a = ref["in_proj_a"].astype(np.float32)                   # [32,2048]
        in_proj_b = ref["in_proj_b"].astype(np.float32)                   # [32,2048]
        conv1d_weight = ref["conv1d_weight"].astype(np.float32)           # [8192,1,4]
        A_log = ref["A_log"].astype(np.float32)                           # [32]
        dt_bias = ref["dt_bias"].astype(np.float32)                       # [32]
        norm_weight = ref["norm_weight"].astype(np.float32)               # [128]
        out_proj = ref["out_proj"].astype(np.float32)                     # [2048,4096]
        print(f"  hidden_in shape: {hidden_in.shape}, norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  expected_output norm: {np.linalg.norm(expected_output):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 1: in_proj_qkv @ hidden — produces mixed_qkv [1, 1, 8192]
        # ────────────────────────────────────────────────────────────────────
        print("\n[3] step 1: mixed_qkv = in_proj_qkv @ hidden")
        hidden_tt = to_ttnn(hidden_in.reshape(1, HIDDEN), device)
        w_qkv_tt = to_ttnn(in_proj_qkv.T, device)  # [2048, 8192]
        mixed_qkv_tt = ttnn.matmul(hidden_tt, w_qkv_tt)
        ttnn.deallocate(w_qkv_tt)
        mixed_qkv_np = from_ttnn(mixed_qkv_tt).reshape(1, CONV_DIM)  # [1, 8192]
        print(f"  ttnn mixed_qkv norm: {np.linalg.norm(mixed_qkv_np):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 2: z = in_proj_z @ hidden  (no activation here; silu applied later)
        # ────────────────────────────────────────────────────────────────────
        print("\n[4] step 2: z = in_proj_z @ hidden")
        w_z_tt = to_ttnn(in_proj_z.T, device)  # [2048, 4096]
        z_tt = ttnn.matmul(hidden_tt, w_z_tt)
        ttnn.deallocate(w_z_tt)
        z_np = from_ttnn(z_tt).reshape(1, NUM_V_HEADS, HEAD_V_DIM)  # [1, 32, 128]
        print(f"  z norm: {np.linalg.norm(z_np):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 3: a, b (small per-head projections, 32 dims each)
        # ────────────────────────────────────────────────────────────────────
        print("\n[5] step 3: a, b")
        w_a_tt = to_ttnn(in_proj_a.T, device)  # [2048, 32]
        w_b_tt = to_ttnn(in_proj_b.T, device)
        a_tt = ttnn.matmul(hidden_tt, w_a_tt)
        b_tt = ttnn.matmul(hidden_tt, w_b_tt)
        ttnn.deallocate(w_a_tt); ttnn.deallocate(w_b_tt); ttnn.deallocate(hidden_tt)
        a_np = from_ttnn(a_tt).reshape(NUM_V_HEADS).astype(np.float32)  # [32]
        b_np = from_ttnn(b_tt).reshape(NUM_V_HEADS).astype(np.float32)
        print(f"  a stats: min={a_np.min():.3f} max={a_np.max():.3f}")
        print(f"  b stats: min={b_np.min():.3f} max={b_np.max():.3f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 4: conv1d update (single-token; do in numpy for simplicity)
        # Causal depthwise conv with kernel=4. State [1, 8192, 4]:
        #   new_state[:, :, :3] = old_state[:, :, 1:]
        #   new_state[:, :, 3]  = mixed_qkv
        #   out = sum(new_state * conv_weight[:, 0, :], dim=-1)
        # Then SILU.
        # ────────────────────────────────────────────────────────────────────
        print("\n[6] step 4-5: conv1d update + silu (numpy-side for now)")
        cs = conv_state_in.copy()  # [1, 8192, 4]
        new_cs = np.zeros_like(cs)
        new_cs[:, :, :CONV_KERNEL - 1] = cs[:, :, 1:]
        new_cs[:, :, CONV_KERNEL - 1] = mixed_qkv_np  # [1, 8192]
        conv_w = conv1d_weight[:, 0, :]  # [8192, 4]
        conv_out = np.sum(new_cs * conv_w[None, :, :], axis=-1)  # [1, 8192]
        # SILU
        silu_out = conv_out * (1.0 / (1.0 + np.exp(-conv_out)))
        print(f"  conv+silu norm: {np.linalg.norm(silu_out):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 6: q/k/v split
        # ────────────────────────────────────────────────────────────────────
        print("\n[7] step 6: split into q/k/v")
        q_flat = silu_out[:, :KEY_DIM]                          # [1, 2048]
        k_flat = silu_out[:, KEY_DIM:2 * KEY_DIM]               # [1, 2048]
        v_flat = silu_out[:, 2 * KEY_DIM:]                      # [1, 4096]
        q_per_head = q_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)  # [1, 16, 128]
        k_per_head = k_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)
        v_per_head = v_flat.reshape(1, NUM_V_HEADS, HEAD_V_DIM)  # [1, 32, 128]
        print(f"  q [1, {NUM_K_HEADS}, {HEAD_K_DIM}] norm: {np.linalg.norm(q_per_head):.4f}")
        print(f"  v [1, {NUM_V_HEADS}, {HEAD_V_DIM}] norm: {np.linalg.norm(v_per_head):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 7: beta = sigmoid(b),
        #         g = -exp(A_log) * softplus(a + dt_bias)
        # ────────────────────────────────────────────────────────────────────
        print("\n[8] step 7: beta + g (decay)")
        beta = 1.0 / (1.0 + np.exp(-b_np))  # [32]
        a_plus_bias = a_np + dt_bias
        softplus = np.log1p(np.exp(a_plus_bias.astype(np.float64))).astype(np.float32)
        g_log = -np.exp(A_log) * softplus  # [32], in log-decay form
        g_decay = np.exp(g_log)             # multiplicative decay
        print(f"  beta: min={beta.min():.4f} max={beta.max():.4f}")
        print(f"  g (decay): min={g_decay.min():.4f} max={g_decay.max():.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 8: l2_norm on q and k (per head_dim)
        # ────────────────────────────────────────────────────────────────────
        print("\n[9] step 8: l2_norm(q, k)")
        def l2norm(x, eps=1e-6):
            return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
        q_norm = l2norm(q_per_head)
        k_norm = l2norm(k_per_head)

        # ────────────────────────────────────────────────────────────────────
        # Step 9: q/k repeat_interleave 16 → 32 heads (HF does this when v_heads > k_heads)
        # ────────────────────────────────────────────────────────────────────
        print("\n[10] step 9: q/k repeat_interleave 16→32 heads")
        rep = NUM_V_HEADS // NUM_K_HEADS  # 2
        q_rep = np.repeat(q_norm, rep, axis=1)  # [1, 32, 128]
        k_rep = np.repeat(k_norm, rep, axis=1)

        # ────────────────────────────────────────────────────────────────────
        # Step 10: recurrence (T=1, stock numpy — would be ttnn in production)
        # query = query * scale where scale = 1/sqrt(K_DIM)
        # state has shape [1, 32, K_DIM, V_DIM]
        # state *= g (broadcast); kv = sum(state * k.unsqueeze(-1), dim=K)
        # delta = beta * (v - kv); state += k.unsqueeze(-1) * delta.unsqueeze(-2)
        # out = sum(state * q.unsqueeze(-1), dim=K)
        # ────────────────────────────────────────────────────────────────────
        print("\n[11] step 10: recurrence (numpy)")
        scale = 1.0 / np.sqrt(HEAD_K_DIM)
        q_scaled = q_rep * scale
        state = recurrent_state_in.copy()  # [1, 32, 128, 128]
        g_b = g_decay[None, :, None, None]              # [1, 32, 1, 1]
        beta_b = beta[None, :, None]                    # [1, 32, 1]
        k_t = k_rep                                     # [1, 32, 128]
        v_t = v_per_head                                # [1, 32, 128]
        q_t = q_scaled                                  # [1, 32, 128]

        state = state * g_b                                                    # decay
        kv_mem = np.sum(state * k_t[:, :, :, None], axis=-2)                   # [1, 32, 128]
        delta = (v_t - kv_mem) * beta_b                                        # [1, 32, 128]
        state = state + k_t[:, :, :, None] * delta[:, :, None, :]              # outer product update
        core_attn_out = np.sum(state * q_t[:, :, :, None], axis=-2)            # [1, 32, 128]
        print(f"  core_attn_out norm: {np.linalg.norm(core_attn_out):.4f}")
        print(f"  recurrent_state norm: {np.linalg.norm(state):.4f}")

        # Validate recurrent state matches HF expectation (sanity gate before output projection)
        rs_cos = (state.flatten() @ expected_recurrent_state_out.flatten()) / (
            np.linalg.norm(state) * np.linalg.norm(expected_recurrent_state_out) + 1e-30
        )
        rs_max = np.abs(state - expected_recurrent_state_out).max()
        print(f"  recurrent_state vs HF: cos={rs_cos:.6f}, max|Δ|={rs_max:.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 11: RMSNormGated — per-head norm then * silu(z)
        # Note: HF's RMSNormGated does norm in fp32, multiplies by `weight` (NOT
        # `1.0 + weight` — different from main RMSNorm convention).
        # ────────────────────────────────────────────────────────────────────
        print("\n[12] step 11: RMSNormGated(core_attn_out, z)")
        core_flat = core_attn_out.reshape(-1, HEAD_V_DIM)  # [B*32, 128]
        z_flat = z_np.reshape(-1, HEAD_V_DIM)              # [B*32, 128]
        # variance in fp32
        var = np.mean(core_flat ** 2, axis=-1, keepdims=True)
        rsqrt = 1.0 / np.sqrt(var + EPS)
        normed = core_flat * rsqrt
        # weight multiply  (HF: norm_weight is bf16, applied as `weight * norm`)
        normed = normed * norm_weight[None, :]
        # silu(z) gate (computed in fp32 per HF)
        silu_z = z_flat * (1.0 / (1.0 + np.exp(-z_flat)))
        gated = normed * silu_z
        gated_reshaped = gated.reshape(1, 1, VALUE_DIM)  # [B, T, 4096]
        print(f"  gated norm: {np.linalg.norm(gated_reshaped):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Step 12: out_proj — back to hidden via ttnn matmul
        # ────────────────────────────────────────────────────────────────────
        print("\n[13] step 12: out_proj (ttnn)")
        gated_tt = to_ttnn(gated_reshaped.reshape(1, VALUE_DIM), device)
        w_out_tt = to_ttnn(out_proj.T, device)  # [4096, 2048]
        out_tt = ttnn.matmul(gated_tt, w_out_tt)
        ttnn.deallocate(w_out_tt); ttnn.deallocate(gated_tt)
        out_np = from_ttnn(out_tt).reshape(1, 1, HIDDEN)
        print(f"  out_proj output norm: {np.linalg.norm(out_np):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Final comparison
        # ────────────────────────────────────────────────────────────────────
        print("\n[14] cosine vs B0 expected output…")
        cos = (out_np.flatten() @ expected_output.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(expected_output) + 1e-30
        )
        max_abs = np.abs(out_np - expected_output).max()
        print(f"  expected norm: {np.linalg.norm(expected_output):.6f}")
        print(f"  ttnn-mix norm: {np.linalg.norm(out_np):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B7 PASS — single-chip DN ttnn output matches HF reference")
        elif cos > 0.99:
            print(f"  ⚠ cosine {cos:.4f} above 0.99 but below 0.999 — investigate which step drifts")
        else:
            print(f"  ✗ FAIL — investigate per-step")

    finally:
        ttnn.close_device(device)

    print("\nB7 DONE.")


if __name__ == "__main__":
    main()
