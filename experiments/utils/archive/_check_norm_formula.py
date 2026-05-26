#!/usr/bin/env python3
"""For each RMSNorm weight, compare captured output magnitude vs predicted
magnitudes for variant A (y=x/rms*γ) and variant B (y=x/rms*(1+γ)).

For a tensor with unit-rms inputs, |y|² ≈ D * E[γ²] (variant A) or D * E[(1+γ)²] (variant B).
"""
import sys, json
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

hf_dir = Path(sys.argv[1])


def fetch_weight(key):
    idx = json.loads(Path(hf_hub_download("Qwen/Qwen3.6-35B-A3B", "model.safetensors.index.json")).read_text())
    shard = hf_hub_download("Qwen/Qwen3.6-35B-A3B", idx["weight_map"][key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def predict(D, gamma):
    a = np.sqrt(D * np.mean(gamma ** 2))
    b = np.sqrt(D * np.mean((1.0 + gamma) ** 2))
    return a, b


def report(label, captured_norms, gamma):
    D = gamma.shape[0]
    a_pred, b_pred = predict(D, gamma)
    obs = float(np.mean(captured_norms))
    err_a = abs(obs - a_pred) / a_pred
    err_b = abs(obs - b_pred) / b_pred
    pick = "B (1+γ)" if err_b < err_a else "A (γ)"
    print(f"{label:>40}  D={D:4d}  γ_mean={np.mean(gamma):+.4f}  γ_std={np.std(gamma):.4f}  "
          f"obs|y|={obs:7.3f}  A_pred={a_pred:7.3f}  B_pred={b_pred:7.3f}  → {pick}")


# q_norm at L31
q_norm_w = fetch_weight("model.language_model.layers.31.self_attn.q_norm.weight")
q_norm_out = np.load(hf_dir.parent / "hf_oracle_35b_needle100_L31" / "L0_attn_L31_q_norm.npy")
# shape [seq, n_heads, head_dim] — magnitude per (seq, head)
q_norms = np.linalg.norm(q_norm_out, axis=-1).flatten()
report("L31 q_norm output magnitude", q_norms, q_norm_w)

# k_norm
k_norm_w = fetch_weight("model.language_model.layers.31.self_attn.k_norm.weight")
k_norm_out = np.load(hf_dir.parent / "hf_oracle_35b_needle100_L31" / "L0_attn_L31_k_norm.npy")
k_norms = np.linalg.norm(k_norm_out, axis=-1).flatten()
report("L31 k_norm output magnitude", k_norms, k_norm_w)

# dn_norm at L32 (= linear_attn.norm); captured is post-gate (× SiLU(z)) so prediction differs
dn_norm_w = fetch_weight("model.language_model.layers.32.linear_attn.norm.weight")
dn_norm_out = np.load(hf_dir / "L0_dn_norm_L32.npy")
dn_norms = np.linalg.norm(dn_norm_out, axis=-1).flatten()
report("L32 dn_norm (post-gate) magnitude", dn_norms, dn_norm_w)
print("  NB: dn_norm is post-gated (× SiLU(z) ∈ ~(0, x_max)), so magnitudes will be smaller than pure-rmsnorm pred.")

# Sanity: input_layernorm at L32 (which we know is variant B)
in_norm_w = fetch_weight("model.language_model.layers.32.input_layernorm.weight")
in_norm_out = np.load(hf_dir / "L0_in_norm_L32.npy")
in_norms = np.linalg.norm(in_norm_out, axis=-1).flatten()
report("[sanity] L32 input_layernorm magnitude", in_norms, in_norm_w)
