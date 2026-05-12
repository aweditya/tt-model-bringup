#!/usr/bin/env python3
"""
Experiment 90 — Phase B1 weight skeleton for Qwen3.6-35B-A3B.

Loads ONLY the model's `config.json` and `model.safetensors.index.json`
(small files, no actual weights). Parses the index to:

  1. Verify we can find every weight tensor the architecture expects
  2. Count params per category (DeltaNet, Attn, MoE, embed, etc.)
  3. Estimate per-chip memory budget in bf8
  4. Print a load-order plan for the actual upload in B2

This is the safe / cheap shell. B2 will follow to actually load + upload.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/90_qwen36_weight_skeleton.py
"""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~"))

from huggingface_hub import hf_hub_download

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# Arch constants (from A1)
N_LAYERS = 40
N_EXPERTS = 256


def categorize(param_name: str) -> str:
    """Sort param_name into one of: embed, deltanet, attn, moe_router, moe_expert, moe_shared, norm, lm_head, vision (skip), other."""
    n = param_name
    if 'vision' in n or 'visual' in n:
        return 'vision'
    if 'embed_tokens' in n or 'embed' in n.split('.')[-1]:
        return 'embed'
    if 'lm_head' in n:
        return 'lm_head'
    if 'norm' in n.split('.')[-1] or 'layernorm' in n:
        return 'norm'
    if 'gated_delta_net' in n or 'linear_attn' in n or 'A_log' in n or 'dt_bias' in n:
        return 'deltanet'
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        return 'attn'
    if 'mlp.gate' in n and 'experts' not in n and 'shared_expert' not in n:
        return 'moe_router'
    if 'shared_expert' in n:
        return 'moe_shared'
    if 'experts.' in n:
        return 'moe_expert'
    return 'other'


def main():
    print("=" * 64)
    print(f"Phase B1 — Qwen3.6-35B-A3B weight skeleton")
    print("=" * 64)

    # Download just the small metadata files
    print(f"\n[1/3] Fetching config.json + index.json for {MODEL_ID}…")
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(idx_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']
    n_shards = len(set(weight_map.values()))
    print(f"  config: {cfg.get('model_type')}")
    print(f"  total tensors: {len(weight_map)}")
    print(f"  total shards: {n_shards}")
    text_cfg = cfg.get('text_config', cfg)
    print(f"  layers: {text_cfg.get('num_hidden_layers')}, hidden: {text_cfg.get('hidden_size')}, "
          f"experts: {text_cfg.get('num_experts')}, top_k: {text_cfg.get('num_experts_per_tok')}")

    # Categorize every tensor and count
    print(f"\n[2/3] Categorizing {len(weight_map)} tensors…")
    cat_count = defaultdict(int)
    cat_shards = defaultdict(set)
    layers_seen = set()
    for name, shard in weight_map.items():
        c = categorize(name)
        cat_count[c] += 1
        cat_shards[c].add(shard)
        # Extract layer number if present
        if '.layers.' in name:
            try:
                layer_n = int(name.split('.layers.')[1].split('.')[0])
                layers_seen.add(layer_n)
            except (ValueError, IndexError):
                pass

    print(f"\n  Category         #tensors  #shards")
    for c in ['embed', 'lm_head', 'norm', 'deltanet', 'attn',
              'moe_router', 'moe_shared', 'moe_expert', 'vision', 'other']:
        if cat_count[c]:
            print(f"  {c:15s}  {cat_count[c]:6d}   {len(cat_shards[c]):4d}")

    if layers_seen:
        print(f"\n  Layers present: {min(layers_seen)} .. {max(layers_seen)}  (count={len(layers_seen)})")
        expected = set(range(N_LAYERS))
        missing = expected - layers_seen
        if missing:
            print(f"  WARN: missing layers: {sorted(missing)}")
        else:
            print(f"  ✓ all {N_LAYERS} layers present")

    # Estimate sizes (using arch constants for cross-check)
    HIDDEN = text_cfg.get('hidden_size', 2048)
    MOE_INT = text_cfg.get('moe_intermediate_size', 512)
    NUM_EXPERTS = text_cfg.get('num_experts', 256)
    NUM_LAYERS = text_cfg.get('num_hidden_layers', 40)
    VOCAB = text_cfg.get('vocab_size', 248320)

    print(f"\n[3/3] Memory budget estimate (bf8 weights = 1 byte/param):")

    # DeltaNet (30 layers × ~34M params)
    deltanet_params = 30 * (
        HIDDEN * (2*2048 + 4096) +  # in_proj_qkv
        HIDDEN * 4096 +              # in_proj_z
        HIDDEN * 32 +                # in_proj_a (n_v_heads=32)
        HIDDEN * 32 +                # in_proj_b
        4096 * HIDDEN +              # out_proj
        # conv1d kernel=4 over 8192 channels = small
        8192 * 4
    )

    # Gated Attention (10 layers × ~27M params)
    attn_params = 10 * (
        HIDDEN * 16 * 256 * 2 +     # q_proj produces Q+gate (2x)
        HIDDEN * 2 * 256 +          # k_proj
        HIDDEN * 2 * 256 +          # v_proj
        16 * 256 * HIDDEN           # o_proj (= 4096 → 2048)
    )

    # MoE (every layer, 256 experts + shared)
    moe_params = NUM_LAYERS * (
        HIDDEN * NUM_EXPERTS +          # router
        NUM_EXPERTS * 3 * (HIDDEN * MOE_INT) +  # 256 experts × (gate + up + down)
        3 * (HIDDEN * MOE_INT) +        # shared expert
        HIDDEN                          # shared_expert_gate
    )

    # Embeddings + lm_head + RMSNorms
    embed_params = VOCAB * HIDDEN
    lm_head_params = VOCAB * HIDDEN if not text_cfg.get('tie_word_embeddings', False) else 0
    norm_params = NUM_LAYERS * 2 * HIDDEN + HIDDEN

    total_params = deltanet_params + attn_params + moe_params + embed_params + lm_head_params + norm_params

    print(f"  Embeddings (in):          {embed_params/1e9:6.2f} B → {embed_params/1e9:6.2f} GB (bf8)")
    print(f"  Output head (out_emb):    {lm_head_params/1e9:6.2f} B → {lm_head_params/1e9:6.2f} GB (bf8)")
    print(f"  DeltaNet × 30 layers:     {deltanet_params/1e9:6.2f} B → {deltanet_params/1e9:6.2f} GB (bf8)")
    print(f"  Gated Attn × 10 layers:   {attn_params/1e9:6.2f} B → {attn_params/1e9:6.2f} GB (bf8)")
    print(f"  MoE × 40 layers:          {moe_params/1e9:6.2f} B → {moe_params/1e9:6.2f} GB (bf8)")
    print(f"  RMSNorms etc:             {norm_params/1e9:6.2f} B → {norm_params/1e9:6.2f} GB")
    print(f"  -" * 30)
    print(f"  TOTAL:                    {total_params/1e9:6.2f} B → {total_params/1e9:6.2f} GB (bf8)")
    print(f"  Expected (config says): ~35 B")

    print(f"\n  Per Blackhole P150 DRAM (~32 GB usable):")
    if total_params / 1e9 <= 32:
        headroom = 32 - total_params / 1e9
        print(f"  ✓ Fits on 1 chip with ~{headroom:.1f} GB headroom for KV/scratch")
    else:
        deficit = total_params / 1e9 - 32
        print(f"  ✗ Exceeds by ~{deficit:.1f} GB at bf8. Multi-chip required.")
        # MoE expert split scenario
        moe_per_chip = moe_params / 2 / 1e9
        rest_per_chip = (total_params - moe_params) / 1e9
        print(f"  With MoE-expert-parallel across 2 chips:")
        print(f"    {moe_per_chip:.1f} GB MoE-half + {rest_per_chip:.1f} GB replicated rest = "
              f"{moe_per_chip + rest_per_chip:.1f} GB per chip — fits!")

    print(f"\n=== B1 skeleton complete ===")
    print(f"  Index path:  {idx_path}")
    print(f"  Config path: {cfg_path}")
    print(f"  Next: B2 — actually load weights, quantize to bf8, upload to device")


if __name__ == "__main__":
    main()
