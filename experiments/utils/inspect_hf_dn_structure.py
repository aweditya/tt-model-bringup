#!/usr/bin/env python3
"""Enumerate Qwen3_5MoeGatedDeltaNet sub-modules + forward signature.

Used to plan DN sub-capture hooks for the cosine probe. Discovers
what's a `nn.Module` (hookable) vs what's inline math (needs
monkey-patch or numpy-reimpl to compare against HF).

Run (qb1):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/inspect_hf_dn_structure.py
"""
import inspect

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def main():
    print(f"loading {MODEL_ID} (bf16)…")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
    )
    # Find first linear_attention layer
    text_cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    layer_types = text_cfg.layer_types
    L_idx = next(i for i, t in enumerate(layer_types) if t == "linear_attention")
    L = m.model.layers[L_idx]
    print(f"layer {L_idx} type: {layer_types[L_idx]}")
    print(f"L type: {type(L).__name__}")
    print(f"L children: {[n for n, _ in L.named_children()]}")

    dn = L.linear_attn
    print(f"\nDN type: {type(dn).__name__}")
    print(f"DN children: {[n for n, _ in dn.named_children()]}")
    print(f"DN named_modules (depth ≤ 2):")
    for n, mod in dn.named_modules():
        if 0 < n.count(".") <= 1 or (n and "." not in n):
            print(f"  {n}: {type(mod).__name__}  params=" +
                  f"{sum(p.numel() for p in mod.parameters()) if list(mod.parameters()) else 0}")

    # DN parameters at top level
    print(f"\nDN top-level parameters:")
    for n, p in dn.named_parameters(recurse=False):
        print(f"  {n}: shape={list(p.shape)} dtype={p.dtype}")

    # DN forward source for understanding inline math
    print(f"\nDN forward signature:")
    print(f"  {inspect.signature(dn.forward)}")
    src = inspect.getsource(dn.forward)
    lines = src.splitlines()
    print(f"DN forward source ({len(lines)} lines, full):")
    for i, line in enumerate(lines):
        print(f"  {i+1:3d}| {line}")

    # Also conv1d weight/bias shapes
    print(f"\nconv1d.weight: shape={list(dn.conv1d.weight.shape)} "
          f"bias={'YES' if dn.conv1d.bias is not None else 'NONE'} "
          f"groups={dn.conv1d.groups} padding={dn.conv1d.padding}")
    print(f"norm weight shape: {list(dn.norm.weight.shape)} "
          f"norm type: {type(dn.norm).__name__}")
    # Inline norm forward source for RMSNormGated
    print(f"\nnorm forward source:")
    nsrc = inspect.getsource(dn.norm.forward)
    for line in nsrc.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
