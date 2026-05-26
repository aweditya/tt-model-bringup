#!/usr/bin/env python3
"""Numpy-reimplementation of Qwen3_5MoeRMSNormGated for isolating the on-device
divergence (dn_norm cos 0.9494 vs HF at pos 0 even though inputs match at cos 1.0).

Method:
  - Load HF's captured `dn_core_attn_out` (= input x), `dn_norm_gate_z` (= gate z),
    and `dn_norm` (= expected output) at L0
  - Load HF norm.weight from the saved model
  - Compute three numpy variants:
      (a) "fp32_path"  : exact HF impl in fp32 (variance/rsqrt/silu(gate) in fp32)
      (b) "all_bf16"   : everything in bf16 (the naive bf16 baseline)
      (c) "weight_plus_one": (a) but with weight += 1.0 (sanity check for the
                             (1+w) convention we apply elsewhere)
  - Report cosine of each variant vs HF's `dn_norm`

If (a) matches HF cos ≈ 1.0 we have the right formula. If (b) is 0.94-ish, the
gap is bf16 precision (need fp32 path on-device). If (c) matches HF, the +1
convention applies and our upload is wrong.

Run (qb1):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/rmsnormgated_numpy_oracle.py
"""
import numpy as np
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
ORACLE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "hf_oracle_35b"


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else 0.0


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype)


def rmsnormgated_fp32(x, gate, weight, eps):
    """HF Qwen3_5MoeRMSNormGated forward, exact."""
    input_dtype = x.dtype
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    x32 = x32 * (1.0 / np.sqrt(variance + eps))
    x_bf16 = x32.astype(input_dtype)
    x_weighted = weight.astype(input_dtype) * x_bf16
    gate32 = gate.astype(np.float32)
    silu_g = gate32 * (1.0 / (1.0 + np.exp(-gate32)))
    out = x_weighted.astype(np.float32) * silu_g
    return out.astype(input_dtype)


def rmsnormgated_all_bf16(x, gate, weight, eps):
    """Naive all-bf16 (cast intermediates back to bf16 aggressively)."""
    input_dtype = x.dtype
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True).astype(input_dtype)
    inv_rms = (1.0 / np.sqrt(variance.astype(np.float32) + eps)).astype(input_dtype)
    x_normed = x.astype(np.float32) * inv_rms.astype(np.float32)
    x_normed = x_normed.astype(input_dtype)
    x_weighted = weight.astype(input_dtype) * x_normed
    silu_g = silu(gate)
    out = x_weighted.astype(np.float32) * silu_g.astype(np.float32)
    return out.astype(input_dtype)


def main():
    # Load HF model just to extract Layer 0 linear_attn.norm.weight + epsilon.
    print(f"loading {MODEL_ID} (bf16, just for norm.weight)…")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
    )
    dn = m.model.layers[0].linear_attn
    weight = dn.norm.weight.detach().float().numpy()  # [128]
    eps = float(dn.norm.variance_epsilon)
    print(f"norm.weight shape={weight.shape} dtype={weight.dtype} "
          f"min/max/mean: {weight.min():.4f}/{weight.max():.4f}/{weight.mean():.4f}")
    print(f"variance_epsilon: {eps}")

    # Load oracle artifacts
    x = np.load(ORACLE_DIR / "L0_dn_core_attn_out.npy").astype(np.float32)  # [seq, NV_HEADS*head_v_dim]
    gate = np.load(ORACLE_DIR / "L0_dn_norm_gate_z.npy").astype(np.float32)
    expected = np.load(ORACLE_DIR / "L0_dn_norm.npy").astype(np.float32)
    print(f"loaded oracle: x={x.shape} gate={gate.shape} expected={expected.shape}")

    # Reshape from [seq, NV_HEADS*head_v_dim=4096] to [seq*NV_HEADS, head_v_dim=128]
    NV_HEADS = 32
    HEAD_V_DIM = 128
    seq = x.shape[0]
    x = x.reshape(seq * NV_HEADS, HEAD_V_DIM)
    gate = gate.reshape(seq * NV_HEADS, HEAD_V_DIM)
    expected = expected.reshape(seq * NV_HEADS, HEAD_V_DIM)

    # Convert to bf16 first (mirroring on-device storage)
    x_bf = x.astype(np.float32).astype(np.dtype('float32'))  # numpy can't represent bf16 directly
    x_bf = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    gate_bf = torch.from_numpy(gate).to(torch.bfloat16).float().numpy()
    weight_bf = torch.from_numpy(weight).to(torch.bfloat16).float().numpy()
    expected_bf = torch.from_numpy(expected).to(torch.bfloat16).float().numpy()

    # Three variants
    out_fp32 = rmsnormgated_fp32(x_bf, gate_bf, weight_bf, eps)
    out_bf16 = rmsnormgated_all_bf16(x_bf, gate_bf, weight_bf, eps)
    weight_plus_one = (weight_bf + 1.0).astype(np.float32)
    out_w1 = rmsnormgated_fp32(x_bf, gate_bf, weight_plus_one, eps)

    # Save intermediate values for on-device sub-comparison:
    #   rms_only = weight * (x / rms(x))   (i.e. RMSNorm WITHOUT gate)
    #   silu_z_only = silu(gate)
    x32 = x_bf.astype(np.float32)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    x_n = x32 * (1.0 / np.sqrt(var + eps))
    rms_only = weight_bf.astype(np.float32) * x_n.astype(weight_bf.dtype).astype(np.float32)
    silu_z_only = gate_bf.astype(np.float32) * (
        1.0 / (1.0 + np.exp(-gate_bf.astype(np.float32))))
    np.save(ORACLE_DIR / "L0_dn_norm_rms_only.npy",
            rms_only.reshape(seq, -1).astype(np.float32))
    np.save(ORACLE_DIR / "L0_dn_norm_silu_z.npy",
            silu_z_only.reshape(seq, -1).astype(np.float32))
    print(f"\nsaved intermediates to {ORACLE_DIR}: L0_dn_norm_rms_only.npy, L0_dn_norm_silu_z.npy")

    print(f"\ncosines vs HF dn_norm (full {seq * NV_HEADS} rows × {HEAD_V_DIM}):")
    print(f"  fp32_path        : {cosine(out_fp32, expected_bf):.6f}")
    print(f"  all_bf16         : {cosine(out_bf16, expected_bf):.6f}")
    print(f"  weight_plus_one  : {cosine(out_w1, expected_bf):.6f}")

    # Also per-position
    print(f"\ncosines per position (fp32 variant):")
    for p in range(seq):
        row_start = p * NV_HEADS
        row_end = (p + 1) * NV_HEADS
        c = cosine(out_fp32[row_start:row_end], expected_bf[row_start:row_end])
        c_bf = cosine(out_bf16[row_start:row_end], expected_bf[row_start:row_end])
        c_w1 = cosine(out_w1[row_start:row_end], expected_bf[row_start:row_end])
        print(f"  pos {p}: fp32={c:.6f}  bf16={c_bf:.6f}  weight+1={c_w1:.6f}")


if __name__ == "__main__":
    main()
