#!/usr/bin/env python3
"""Inspect Qwen3_5MoeForCausalLM module structure (paths, sub-modules).

Captures the model topology so future probes don't re-discover paths
inline. Discovered (transformers ≥5.0 build on qb1):

  top:                              Qwen3_5MoeForCausalLM
  m.model:                          Qwen3_5MoeTextModel
                                    children = ['embed_tokens','layers','norm','rotary_emb']
  m.model.layers[L]:                Qwen3_5MoeDecoderLayer
    .input_layernorm                Qwen3_5MoeRMSNorm
    .linear_attn or .self_attn      (depends on layer_types[L])
    .post_attention_layernorm       Qwen3_5MoeRMSNorm
    .mlp                            Qwen3_5MoeSparseMoeBlock (or dense)

Run (qb1):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/inspect_hf_qwen35_structure.py
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def main():
    print(f"loading {MODEL_ID} (bf16)…")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"top type: {type(m).__name__}")
    print("top children:", [n for n, _ in m.named_children()])
    print(f"m.model type: {type(m.model).__name__}")
    print("m.model children:", [n for n, _ in m.model.named_children()])
    L0 = m.model.layers[0]
    print(f"L0 type: {type(L0).__name__}")
    print("L0 children:", [n for n, _ in L0.named_children()])
    # Print one layer's full module tree (depth 2)
    for n, mod in L0.named_modules():
        depth = n.count(".")
        if depth <= 1 and n:
            print(f"  L0.{n}: {type(mod).__name__}")


if __name__ == "__main__":
    main()
