#!/usr/bin/env python3
"""Audit Qwen3.6-35B-A3B state dict — discover prefixes, layer types, tensor shapes.

Replaces the ad-hoc inline ssh heredocs used during Phase 0 audit on 2026-05-21.
Run on qb2 (model weights ~67 GB live in ~/.cache/huggingface/hub/):

    ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/qwen36_35b_a3b_state_dict_audit.py'

Output is grouped, deterministic — paste into research notes when discovering
new findings.

Findings logged 2026-05-21 (see research/qwen36_35b_a3b_state_dict_findings_*.md):
  - State-dict prefix is `model.language_model.<layers|embed_tokens|norm>` (multimodal nest)
  - Layer 0 uses `linear_attn.*` (Mamba2-style selective SSM, NOT the 27B GatedDeltaNet)
  - MoE experts FUSED: single [256, ...] tensors for gate_up_proj and down_proj
  - Layer 3 is the first full-attention layer (every 4th per `full_attention_interval=4`)
"""
import os
from pathlib import Path
from safetensors import safe_open

DEFAULT_SNAPSHOT = (
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)


def build_key_to_shard(snapshot_dir: str) -> dict:
    shards = sorted(Path(snapshot_dir).glob("*.safetensors"))
    out = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


def get_shape(key_to_shard, key):
    with safe_open(key_to_shard[key], framework="pt") as f:
        t = f.get_tensor(key)
        return list(t.shape), str(t.dtype)


def dump(label, key_to_shard, predicate):
    print(f"=== {label} ===")
    keys = sorted(k for k in key_to_shard if predicate(k))
    if not keys:
        print("  (no matches)")
    for k in keys:
        shape, dtype = get_shape(key_to_shard, k)
        print(f"  {k}: shape={shape}, dtype={dtype}")
    print()


def main():
    snapshot = os.environ.get("QWEN35B_SNAPSHOT", DEFAULT_SNAPSHOT)
    print(f"snapshot: {snapshot}")
    key_to_shard = build_key_to_shard(snapshot)
    print(f"total keys: {len(key_to_shard)}")
    print()

    dump("Layer 0 DeltaNet (linear_attn)", key_to_shard,
         lambda k: "language_model.layers.0.linear_attn" in k)
    dump("Layer 0 layernorms", key_to_shard,
         lambda k: "language_model.layers.0" in k and "layernorm" in k)
    dump("Layer 0 MoE", key_to_shard,
         lambda k: "language_model.layers.0.mlp" in k)

    # Find first full-attention layer (expect 3 per full_attention_interval=4)
    for L in range(8):
        prefix = f"language_model.layers.{L}.self_attn"
        if any(prefix in k for k in key_to_shard):
            dump(f"Layer {L} self_attn", key_to_shard,
                 lambda k, p=prefix: p in k)
            break

    dump("Top-level (text-only — no visual, no layers, no mtp)", key_to_shard,
         lambda k: "visual" not in k and "layers." not in k and "mtp" not in k)

    dump("MTP head", key_to_shard, lambda k: k.startswith("mtp."))

    # Quick visual-tower size summary (not a full dump — vision tower has hundreds of keys)
    visual_keys = [k for k in key_to_shard if "visual" in k]
    print(f"=== Vision tower: {len(visual_keys)} keys (text-only inference skips this) ===")


if __name__ == "__main__":
    main()
