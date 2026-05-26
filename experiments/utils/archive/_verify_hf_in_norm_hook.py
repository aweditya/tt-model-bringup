#!/usr/bin/env python3
"""Verify HF in_norm_L<N> hook captures RMSNorm(hidden_states[N])."""
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

hf_dir = Path(sys.argv[1])
N = int(sys.argv[2])
positions = [int(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 5]

hf_hidden = np.load(hf_dir / "hidden_states.npy")
captured = np.load(hf_dir / f"L0_in_norm_L{N}.npy")
if captured.ndim == 3:
    captured = captured[0]

print(f"hidden_states shape={list(hf_hidden.shape)}, captured in_norm_L{N} shape={list(captured.shape)}")

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
EPS = cfg.text_config.rms_norm_eps
print(f"rms_norm_eps={EPS}")

# Load just the input_layernorm weight for layer N. Loading the full model is overkill;
# read from safetensors via from_pretrained but only fetch one tensor via safe_open.
from safetensors import safe_open
from huggingface_hub import hf_hub_download

# Find which shard contains the input_layernorm
import json
idx_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "model.safetensors.index.json")
idx = json.loads(Path(idx_path).read_text())
key = f"model.language_model.layers.{N}.input_layernorm.weight"
shard_name = idx["weight_map"][key]
shard_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", shard_name)
with safe_open(shard_path, framework="pt", device="cpu") as f:
    gamma = f.get_tensor(key).float().numpy()
print(f"gamma shape={gamma.shape}, |gamma|_mean={np.mean(np.abs(gamma)):.4f}")

# Compute RMSNorm manually in fp32
def rms_norm(x, w, eps):
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return (x.astype(np.float64) / rms * w.astype(np.float64)).astype(np.float32)

input_hs = hf_hidden[N]  # [seq, hidden]

print("=== variant A: y = x/rms(x) * gamma ===")
recon_a = rms_norm(input_hs, gamma, EPS)
for p in positions:
    a, b = recon_a[p], captured[p]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"pos {p:>3}: |recon|={np.linalg.norm(a):8.4f}  |captured|={np.linalg.norm(b):8.4f}  cos={cos:.8f}")

print("=== variant B: y = x/rms(x) * (1 + gamma) ===")
recon_b = rms_norm(input_hs, 1.0 + gamma, EPS)
for p in positions:
    a, b = recon_b[p], captured[p]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"pos {p:>3}: |recon|={np.linalg.norm(a):8.4f}  |captured|={np.linalg.norm(b):8.4f}  cos={cos:.8f}")
