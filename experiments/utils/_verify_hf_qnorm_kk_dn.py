#!/usr/bin/env python3
"""Verify whether Qwen3.6 q_norm, k_norm, and linear_attn.norm use
y = x/rms(x) * gamma (standard) or y = x/rms(x) * (1+gamma) (zero-centered).

Compares the captured outputs to manually-computed RMSNorm with both
variants on a single test pair of (input, captured_output).

For q_norm/k_norm: input = q_proj/k_proj output (post-reshape to heads).
For linear_attn.norm (= "norm" inside DN): input is captured as
'dn_core_attn_out_L<N>' (pre-hook).
"""
import sys
import json
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

hf_dir = Path(sys.argv[1])
EPS = 1e-6


def rms_norm(x, w, eps):
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return (x.astype(np.float64) / rms * w.astype(np.float64)).astype(np.float32)


def fetch_weight(key):
    idx_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "model.safetensors.index.json")
    idx = json.loads(Path(idx_path).read_text())
    shard = idx["weight_map"][key]
    shard_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", shard)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def check(label, x_pre, x_post, gamma):
    print(f"--- {label} ---")
    print(f"  shapes: pre={list(x_pre.shape)}, post={list(x_post.shape)}, gamma={list(gamma.shape)}")
    a = rms_norm(x_pre, gamma, EPS)
    b = rms_norm(x_pre, 1.0 + gamma, EPS)
    for name, recon in [("variant A (y=x/rms*γ)", a), ("variant B (y=x/rms*(1+γ))", b)]:
        c = recon.reshape(-1)
        d = x_post.reshape(-1)
        cos = float(c @ d / (np.linalg.norm(c) * np.linalg.norm(d)))
        print(f"  {name:>28}  |recon|={np.linalg.norm(c):8.4f}  |captured|={np.linalg.norm(d):8.4f}  cos={cos:.8f}")


# Try L31 attn (has q_norm/k_norm hooks via --hook-attn-layer 31)
hf_dir31 = Path(str(hf_dir).replace("L32", "L31"))
if (hf_dir31 / "L0_attn_L31_q_proj.npy").exists():
    print("==== L31 q_norm / k_norm ====")
    N = 31
    q_proj = np.load(hf_dir31 / "L0_attn_L31_q_proj.npy")
    q_norm_out = np.load(hf_dir31 / "L0_attn_L31_q_norm.npy")
    k_proj = np.load(hf_dir31 / "L0_attn_L31_k_proj.npy")
    k_norm_out = np.load(hf_dir31 / "L0_attn_L31_k_norm.npy")
    print(f"q_proj shape={q_proj.shape}, q_norm shape={q_norm_out.shape}")
    # q_norm gamma shape: [head_dim] (per-head RMSNorm)
    q_norm_w = fetch_weight(f"model.language_model.layers.{N}.self_attn.q_norm.weight")
    k_norm_w = fetch_weight(f"model.language_model.layers.{N}.self_attn.k_norm.weight")
    print(f"q_norm gamma shape={q_norm_w.shape}, mean|γ|={np.mean(np.abs(q_norm_w)):.4f}")
    # q_norm_out shape: [1, seq, num_heads, head_dim]. Pick pos 1.
    p = 1
    # Reshape q_proj if needed; q_proj is the raw projection [batch,seq,nq*head_dim]
    # but q_norm input is post-reshape to [batch, seq, n_heads, head_dim].
    head_dim = q_norm_w.shape[0]
    # Strip leading batch dims (file may have been saved with or without batch)
    q_pre = q_proj if q_proj.ndim == 2 else q_proj[0]
    q_post = q_norm_out if q_norm_out.ndim == 3 else q_norm_out[0]
    k_pre = k_proj if k_proj.ndim == 2 else k_proj[0]
    k_post = k_norm_out if k_norm_out.ndim == 3 else k_norm_out[0]
    q_pre_p = q_pre[p].reshape(-1, head_dim)  # [n_heads, head_dim]
    q_post_p = q_post[p]                       # [n_heads, head_dim]
    check("q_norm L31 pos 1", q_pre_p, q_post_p, q_norm_w)
    k_pre_p = k_pre[p].reshape(-1, head_dim)
    k_post_p = k_post[p]
    check("k_norm L31 pos 1", k_pre_p, k_post_p, k_norm_w)

# DN norm at L32
if (hf_dir / "L0_dn_core_attn_out_L32.npy").exists():
    print()
    print("==== L32 linear_attn.norm ====")
    N = 32
    dn_norm_input = np.load(hf_dir / "L0_dn_core_attn_out_L32.npy")  # [seq*NV, head_dim] usually
    dn_norm_output = np.load(hf_dir / "L0_dn_norm_L32.npy")
    print(f"dn_norm input shape={dn_norm_input.shape}, output shape={dn_norm_output.shape}")
    dn_norm_w = fetch_weight(f"model.language_model.layers.{N}.linear_attn.norm.weight")
    print(f"dn_norm gamma shape={dn_norm_w.shape}, mean|γ|={np.mean(np.abs(dn_norm_w)):.4f}")
    # NOTE: DN norm has a gate too — y = RMSNorm(core_attn_out) * SiLU(gate_z). The
    # captured `dn_norm_L32` is the FINAL output (with gate). So pure-RMSNorm reference
    # won't match exactly. But we can check the RATIO of magnitudes / cos sign.
    p = 1
    # Just check first NV rows for position 1
    NV = dn_norm_input.shape[0] // 97
    pre_p = dn_norm_input[p * NV:(p + 1) * NV]
    post_p = dn_norm_output[p * NV:(p + 1) * NV]
    check("dn_norm L32 pos 1 (rms-only, ignores gate)", pre_p, post_p, dn_norm_w)
