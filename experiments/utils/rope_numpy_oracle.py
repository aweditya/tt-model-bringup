#!/usr/bin/env python3
"""Numpy reimplementation of Qwen3.6 partial RoPE for cross-checking my
server_35b_ttnn._apply_partial_rope.

Background: my naive _apply_partial_rope enable regressed cosine probe
L39 pos≥1 from 0.95 → 0.81. Need to determine whether:
  (a) my math/algorithm is wrong (numpy reimpl ≠ HF), or
  (b) my ttnn implementation has an op-level bug (numpy reimpl = HF
      but ttnn output ≠ numpy).

This script tests (a). It runs HF's actual apply_rotary_pos_emb on
fixed Q, K, and runs my-formula in numpy, then computes cosine of
both outputs. If cosine = 1.0, math is right; debug shifts to ttnn ops.

Run (qb1):
  cd ~/tt-xla && .venv/bin/python -u experiments/utils/rope_numpy_oracle.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb,
    rotate_half,
)

# Same constants as server_35b_ttnn.py
HEAD_DIM_ATTN = 256
ROTARY_DIM = 64  # = HEAD_DIM_ATTN * partial_rotary_factor (0.25)
ROPE_THETA = 10_000_000.0
NQ_PER_CHIP = 4


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def my_apply_partial_rope_numpy(x, cos, sin):
    """Reproduce server_35b_ttnn._apply_partial_rope exactly in numpy.

    x: [N, HEAD_DIM_ATTN=256]   (per-chip, post rms_norm)
    cos, sin: [1, ROTARY_DIM=64]
    Returns: x_embed [N, HEAD_DIM_ATTN]
    """
    x_rot = x[:, :ROTARY_DIM]               # [N, 64]
    x_pass = x[:, ROTARY_DIM:]              # [N, 192]
    half = ROTARY_DIM // 2                  # 32
    x1 = x_rot[:, :half]                    # [N, 32]
    x2 = x_rot[:, half:]                    # [N, 32]
    neg_x2 = -x2                            # [N, 32]
    rotated = np.concatenate([neg_x2, x1], axis=-1)  # [N, 64]
    x_rot_cos = x_rot * cos                 # [N, 64] * [1, 64] → [N, 64]
    rotated_sin = rotated * sin
    x_rot_embed = x_rot_cos + rotated_sin   # [N, 64]
    return np.concatenate([x_rot_embed, x_pass], axis=-1)  # [N, 256]


def compute_cos_sin_numpy(pos):
    """Match my server impl: inv_freq = theta^(-2i/R) for i in [0, R/2)."""
    inv_freq = 1.0 / (ROPE_THETA ** (
        np.arange(0, ROTARY_DIM, 2).astype(np.float64) / ROTARY_DIM))
    angle = float(pos) * inv_freq
    cos_half = np.cos(angle).astype(np.float32)
    sin_half = np.sin(angle).astype(np.float32)
    cos = np.concatenate([cos_half, cos_half]).reshape(1, ROTARY_DIM)
    sin = np.concatenate([sin_half, sin_half]).reshape(1, ROTARY_DIM)
    return cos, sin


def main():
    np.random.seed(42)
    # Fake q (NQ_PER_CHIP heads, post rms_norm — values typically ~N(0,1) ish)
    q_np = np.random.randn(NQ_PER_CHIP, HEAD_DIM_ATTN).astype(np.float32)
    k_np = np.random.randn(1, HEAD_DIM_ATTN).astype(np.float32)

    for pos in [0, 1, 2, 3, 5, 10, 50]:
        cos_np, sin_np = compute_cos_sin_numpy(pos)

        # --- variant A: my impl in numpy ---
        q_out_my = my_apply_partial_rope_numpy(q_np, cos_np, sin_np)
        k_out_my = my_apply_partial_rope_numpy(k_np, cos_np, sin_np)

        # --- variant B: HF apply_rotary_pos_emb ---
        # HF expects q [B, H, S, head_dim]; reshape to [1, NQ_PER_CHIP, 1, 256]
        q_t = torch.from_numpy(q_np).reshape(1, NQ_PER_CHIP, 1, HEAD_DIM_ATTN)
        k_t = torch.from_numpy(k_np).reshape(1, 1, 1, HEAD_DIM_ATTN)
        cos_t = torch.from_numpy(cos_np).reshape(1, 1, ROTARY_DIM)  # [B=1, S=1, R]
        sin_t = torch.from_numpy(sin_np).reshape(1, 1, ROTARY_DIM)
        q_hf, k_hf = apply_rotary_pos_emb(q_t, k_t, cos_t, sin_t, unsqueeze_dim=1)
        q_out_hf = q_hf.numpy().reshape(NQ_PER_CHIP, HEAD_DIM_ATTN)
        k_out_hf = k_hf.numpy().reshape(1, HEAD_DIM_ATTN)

        cos_q = cosine(q_out_my, q_out_hf)
        cos_k = cosine(k_out_my, k_out_hf)
        max_diff_q = np.abs(q_out_my - q_out_hf).max()
        max_diff_k = np.abs(k_out_my - k_out_hf).max()
        flag = "✓" if (cos_q > 0.9999 and cos_k > 0.9999) else "✗"
        print(f"  pos {pos:3d}: q_cos={cos_q:.6f} (max|Δ|={max_diff_q:.6f})  "
              f"k_cos={cos_k:.6f} (max|Δ|={max_diff_k:.6f})  {flag}")


if __name__ == "__main__":
    main()
