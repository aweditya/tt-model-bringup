#!/usr/bin/env python3
"""
Permanent utility — probe softplus numerical stability for DeltaNet's `g` decay.

Our DeltaNet implementation computes:
    softplus(x) = log(exp(x) + 1)

HF's reference uses:
    softplus(x) = F.softplus(x)
                = log(1 + exp(-|x|)) + max(0, x)   (the numerically stable form)

For large positive x:
    log(exp(x)+1) → exp(x) overflows (inf at x>88 in fp32, x>16 in bf16)
    F.softplus(x) → just returns x (no overflow)

This script:
  1. Loads layer 2's `in_proj_a` weight + `dt_bias` from safetensors
  2. Loads HF's hidden_2 (input to layer 2) from the saved oracle npz
  3. Computes `a + dt_bias` for each of the 5 positions
  4. Measures the actual range of x = a + dt_bias values
  5. Computes softplus via both formulas
  6. Reports max|Δ| per position

If max|Δ| is large at any position, our unstable softplus is the bug.
If it's small everywhere, we need to look elsewhere (conv1d state,
recurrence math, etc.).

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/softplus_stability_probe.py
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_HIDDEN_PATH = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")


def our_softplus(x):
    """The unstable form we use: log(exp(x) + 1). NumPy fp32."""
    return np.log(np.exp(x.astype(np.float32)) + 1.0).astype(np.float32)


def stable_softplus(x):
    """Torch's stable softplus."""
    return F.softplus(torch.from_numpy(x.astype(np.float32))).numpy()


def main():
    print("=" * 64)
    print("Softplus stability probe — layer 2's `a + dt_bias` for HF hidden_2")
    print("=" * 64)

    # Load HF hidden_2 (input to layer 2)
    if not os.path.exists(HF_HIDDEN_PATH):
        print(f"missing {HF_HIDDEN_PATH} — run hf_full_model_oracle.py --dump-hidden-states first")
        return
    hf_data = np.load(HF_HIDDEN_PATH)
    hidden_2 = hf_data["hidden_2"][0]   # [seq=5, hidden=5120]
    print(f"\nHF hidden_2 shape: {hidden_2.shape}, ‖·‖ per pos: "
          f"{[f'{np.linalg.norm(hidden_2[p]):.3f}' for p in range(5)]}")

    # Load layer 2 weights we need (in_proj_a, dt_bias) + input_layernorm
    # (because the actual `a` is computed from RMS-normalized hidden)
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    needed = {
        "input_layernorm": "model.language_model.layers.2.input_layernorm.weight",
        "in_proj_a":       "model.language_model.layers.2.linear_attn.in_proj_a.weight",
        "dt_bias":         "model.language_model.layers.2.linear_attn.dt_bias",
    }
    weights = {}
    by_shard = {}
    for k, full in needed.items():
        by_shard.setdefault(weight_map[full], []).append((k, full))
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k, full in items:
                arr = f.get_tensor(full).float().numpy()
                weights[k] = arr

    # Apply (1.0 + w) to input_layernorm (Qwen3.5 parameterization)
    norm_w = 1.0 + weights["input_layernorm"]
    # in_proj_a is stored as [out=48, in=5120]; we use x @ W.T conv to apply
    Wa = weights["in_proj_a"]  # [48, 5120]
    dt_bias = weights["dt_bias"]  # [48]
    print(f"in_proj_a shape: {Wa.shape}  dt_bias shape: {dt_bias.shape}")
    print(f"dt_bias stats:   mean={dt_bias.mean():+.4f}  std={dt_bias.std():.4f}  "
          f"min={dt_bias.min():+.4f}  max={dt_bias.max():+.4f}")

    # For each position, compute a = in_proj_a(rms_norm(hidden_2[pos]))
    print("\nPer-position softplus comparison:")
    print(f"{'pos':>4s} {'‖x‖':>10s} {'x range':>16s} {'max|Δ|':>12s} {'Δ/value%':>10s}")
    for pos in range(5):
        h = hidden_2[pos]                          # [5120]
        # RMSNorm: x / sqrt(mean(x²)+eps) * (1+w)
        rms = np.sqrt(np.mean(h.astype(np.float64) ** 2) + 1e-6)
        h_normed = (h / rms) * norm_w               # [5120]
        a = h_normed @ Wa.T                          # [48]  (in_proj_a)
        x = a + dt_bias                              # [48]  (input to softplus)
        # Compute softplus both ways
        our = our_softplus(x)
        stable = stable_softplus(x)
        max_diff = np.max(np.abs(our - stable))
        max_val = np.max(np.abs(stable))
        rel = max_diff / (max_val + 1e-12) * 100
        x_str = f"[{x.min():.2f}, {x.max():.2f}]"
        print(f"{pos:4d} {np.linalg.norm(x):10.3f} {x_str:>16s} {max_diff:12.6f} {rel:9.3f}%")
        # Flag NaN/Inf
        if not np.isfinite(our).all():
            print(f"      ⚠️ OUR softplus has NaN/Inf at pos {pos}!  inf count: {(~np.isfinite(our)).sum()}")
            print(f"          x values causing overflow: {x[~np.isfinite(our)]}")

    print("\n→ If max|Δ| is large or our softplus has Inf at any position,")
    print("  the unstable formula IS the bug. Fix: switch to stable form.")
    print("→ If max|Δ| is tiny everywhere, softplus is not the bug; look elsewhere.")


if __name__ == "__main__":
    main()
