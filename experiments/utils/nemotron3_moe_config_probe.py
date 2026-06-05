#!/usr/bin/env python3
"""Quick probe of Nemotron MoE config values needed for v0.1.3 router."""
from transformers import AutoConfig

c = AutoConfig.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True,
)
keys = [
    "num_experts_per_tok", "n_routed_experts", "n_group", "topk_group",
    "norm_topk_prob", "routed_scaling_factor",
    "moe_intermediate_size", "moe_shared_expert_intermediate_size",
    "hidden_act",
]
for k in keys:
    v = getattr(c, k, "NOT_SET")
    print(f"  {k}: {v}")
