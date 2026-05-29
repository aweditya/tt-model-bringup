#!/usr/bin/env python3
"""B12.6 — Capture layer-3 attn-only intermediate + layer-3 MoE weights.

Augments B3 (which only saved the full layer-3 output and attention
weights). We now also need:
  - `attn_intermediate`: hidden state AFTER attention + residual_1,
                         BEFORE post_attention_layernorm + MoE
  - layer-3 MoE weights (router, experts gate_up/down, shared, shared_gate)

This lets B12.5 do a real cosine check vs `attn_intermediate`, and a
future B12.7 do the full attn+MoE cosine vs B3's full output.

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91al_qwen36_35b_a3b_layer3_attn_intermediate_hf.py'

Output: `.cache/qb2_35b_moe/b3p_layer3_intermediates.npz`
"""
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
)

SNAPSHOT_DIR = Path(
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)
OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b3p_layer3_intermediates.npz"
LAYER_IDX = 3


def build_key_to_shard(snapshot_dir):
    shards = sorted(snapshot_dir.glob("*.safetensors"))
    out = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


def load_tensor(key_to_shard, key):
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key)


class FakeKVCache:
    """Minimal cache for attention: update returns inputs unchanged."""
    def update(self, k, v, layer_idx, cache_kwargs=None):
        return k, v


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output: {OUT_PATH}")

    print("[1] config + modules…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    text_cfg._attn_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)

    attn = Qwen3_5MoeAttention(text_cfg, layer_idx=LAYER_IDX).eval()
    input_layernorm = Qwen3_5MoeRMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).eval()
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg).eval()
    moe = Qwen3_5MoeSparseMoeBlock(text_cfg).eval()
    post_layernorm = Qwen3_5MoeRMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).eval()

    print("[2] load layer-3 weights…")
    key_to_shard = build_key_to_shard(SNAPSHOT_DIR)
    prefix = f"model.language_model.layers.{LAYER_IDX}."
    full_sd = {k[len(prefix):]: load_tensor(key_to_shard, k)
               for k in key_to_shard if k.startswith(prefix)}
    attn_sd = {k.replace("self_attn.", ""): v for k, v in full_sd.items()
               if k.startswith("self_attn.")}
    moe_sd = {k.replace("mlp.", ""): v for k, v in full_sd.items()
              if k.startswith("mlp.")}
    attn.load_state_dict(attn_sd, strict=True)
    moe.load_state_dict(moe_sd, strict=True)
    input_layernorm.weight.data = full_sd["input_layernorm.weight"]
    post_layernorm.weight.data = full_sd["post_attention_layernorm.weight"]
    print(f"  attn keys loaded: {len(attn_sd)}")
    print(f"  moe keys loaded:  {len(moe_sd)}")

    print("[3] same input as B3 (torch.randn seed=0)…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)
    position_ids = torch.tensor([[3]], dtype=torch.long)

    print("[4] compute manually: residual + attn + residual + moe…")
    with torch.no_grad():
        cos, sin = rotary(hidden_in, position_ids)

        # Attention sub-block
        residual_1 = hidden_in
        h = input_layernorm(hidden_in)
        attn_out, _ = attn(
            hidden_states=h,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=FakeKVCache(),
        )
        attn_intermediate = residual_1 + attn_out  # ← post-attn pre-MoE

        # MoE sub-block
        residual_2 = attn_intermediate
        h2 = post_layernorm(attn_intermediate)
        moe_out = moe(h2)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        final = residual_2 + moe_out

    print(f"  attn_intermediate norm: {attn_intermediate.float().norm().item():.4f}")
    print(f"  final (attn+MoE) norm:  {final.float().norm().item():.4f}")

    print("[5] save npz…")
    np.savez(
        OUT_PATH,
        hidden_in=hidden_in.float().numpy(),
        attn_intermediate=attn_intermediate.float().numpy(),
        final=final.float().numpy(),
        # MoE weights for B12.7 to use
        router_weight=moe.gate.weight.detach().float().numpy(),
        experts_gate_up_proj=moe.experts.gate_up_proj.detach().float().numpy(),
        experts_down_proj=moe.experts.down_proj.detach().float().numpy(),
        shared_gate_proj=moe.shared_expert.gate_proj.weight.detach().float().numpy(),
        shared_up_proj=moe.shared_expert.up_proj.weight.detach().float().numpy(),
        shared_down_proj=moe.shared_expert.down_proj.weight.detach().float().numpy(),
        shared_expert_gate=moe.shared_expert_gate.weight.detach().float().numpy(),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\nB12.6 DONE.")


if __name__ == "__main__":
    main()
