#!/usr/bin/env python3
"""B4 — Qwen3.6-35B-A3B FOUR-layer chain HF reference (3 linear_attn + 1 attn).

Validates multi-layer composition: hidden flows through layer 0 (DN+MoE)
→ layer 1 (DN+MoE) → layer 2 (DN+MoE) → layer 3 (attn+MoE). Each DN layer
maintains its own conv_state + recurrent_state.

This is the smallest end-to-end test that exercises:
  - DN cache plumbing across 3 sequential DN layers
  - The transition from linear_attention to full_attention
  - Real RoPE position_embeddings (shared across all layers via the same
    rotary module)
  - Residual chain depth

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ab_qwen36_35b_a3b_4layer_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b4_4layer_reference.npz`
"""
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
)


SNAPSHOT_DIR = Path(
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)
OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b4_4layer_reference.npz"
N_LAYERS = 4


def build_key_to_shard(snapshot_dir: Path) -> dict:
    shards = sorted(snapshot_dir.glob("*.safetensors"))
    out = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


def load_tensor(key_to_shard: dict, key: str) -> torch.Tensor:
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key)


class _LayerState:
    def __init__(self, conv_state, recurrent_state):
        self.conv_states = conv_state
        self.recurrent_states = recurrent_state


class MultiLayerFakeCache:
    """Multi-layer duck-typed cache.

    Holds per-layer (conv_state, recurrent_state) for DN layers AND mirrors
    the same interface for attention layers (KV cache not used here since
    we pass past_key_values=None per layer for first-token capture).
    """
    def __init__(self):
        self.layers = {}

    def add_dn_layer(self, layer_idx, dn_module, batch_size=1):
        conv_state = torch.zeros(
            batch_size, dn_module.conv_dim, dn_module.conv_kernel_size,
            dtype=torch.bfloat16,
        )
        recurrent_state = torch.zeros(
            batch_size, dn_module.num_v_heads, dn_module.head_k_dim, dn_module.head_v_dim,
            dtype=torch.float32,
        )
        self.layers[layer_idx] = _LayerState(conv_state, recurrent_state)

    def has_previous_state(self, layer_idx):
        return layer_idx in self.layers

    def update_conv_state(self, new_state, layer_idx):
        self.layers[layer_idx].conv_states = new_state

    def update_recurrent_state(self, new_state, layer_idx):
        self.layers[layer_idx].recurrent_states = new_state

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """KV-cache interface for attention layers.

        For B4 single-token capture we don't need to STORE the KV — just
        return the inputs unchanged so attention can compute against them.
        Real production would persist them.
        """
        return key_states, value_states


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot: {SNAPSHOT_DIR}")
    print(f"output:   {OUT_PATH}")

    print("[1] load config…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    text_cfg._attn_implementation = "eager"
    print(f"  layer_types[:4]={text_cfg.layer_types[:4]}")
    assert text_cfg.layer_types[:4] == ["linear_attention"] * 3 + ["full_attention"], \
        f"unexpected layer pattern: {text_cfg.layer_types[:4]}"

    print("[2] build 4 decoder layers + rotary embed…")
    torch.set_default_dtype(torch.bfloat16)
    layers = [Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=i) for i in range(N_LAYERS)]
    for L in layers:
        L.eval()
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg)
    rotary.eval()
    print(f"  built {len(layers)} layers; layer_types: "
          f"{[L.layer_type for L in layers]}")

    print("[3] load weights for layers 0..3…")
    key_to_shard = build_key_to_shard(SNAPSHOT_DIR)
    for L_idx in range(N_LAYERS):
        prefix = f"model.language_model.layers.{L_idx}."
        state_dict = {}
        for k in sorted(key_to_shard):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = load_tensor(key_to_shard, k)
        missing, unexpected = layers[L_idx].load_state_dict(state_dict, strict=False)
        assert not unexpected, f"layer {L_idx} unexpected: {unexpected}"
        print(f"  layer {L_idx}: {len(state_dict)} tensors loaded (missing: {missing})")

    print("[4] build input + position embeddings + caches…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)  # snapshot for self-test
    hidden = hidden_in.clone()
    position_ids = torch.tensor([[3]], dtype=torch.long)
    print(f"  hidden_in norm: {hidden.float().norm().item():.4f}")
    with torch.no_grad():
        cos, sin = rotary(hidden, position_ids)
    cache = MultiLayerFakeCache()
    for L_idx in range(N_LAYERS):
        if layers[L_idx].layer_type == "linear_attention":
            cache.add_dn_layer(L_idx, layers[L_idx].linear_attn)
    print(f"  cache: DN layers {sorted(cache.layers.keys())}")

    print("[5] chain forward through 4 layers…")
    per_layer_norms = []
    with torch.no_grad():
        for L_idx in range(N_LAYERS):
            layer = layers[L_idx]
            out = layer(
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
            )
            hidden = out[0] if isinstance(out, tuple) else out
            per_layer_norms.append(hidden.float().norm().item())
            print(f"  after layer {L_idx} ({layer.layer_type}): norm={per_layer_norms[-1]:.4f}")
    output = hidden
    print(f"  final output norm: {output.float().norm().item():.4f}")

    print("[6] save npz…")
    # Save input + output + intermediate DN states; skip heavy per-layer weights
    # (those are in B0/B1/B3 npzs). B4 is a composition gate, not a weight dump.
    state_payload = {
        "hidden_in": hidden_in.float().numpy(),
        "output": output.float().numpy(),
        "position_ids": position_ids.numpy(),
        "cos": cos.detach().float().numpy(),
        "sin": sin.detach().float().numpy(),
        "per_layer_output_norms": np.array(per_layer_norms),
    }
    np.savez(OUT_PATH, **state_payload)
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.2f} MB)")

    print("[7] self-test: re-run from fresh state…")
    # Rebuild caches (they were mutated during chain)
    cache2 = MultiLayerFakeCache()
    for L_idx in range(N_LAYERS):
        if layers[L_idx].layer_type == "linear_attention":
            cache2.add_dn_layer(L_idx, layers[L_idx].linear_attn)
    hidden2 = hidden_in.clone()
    with torch.no_grad():
        for L_idx in range(N_LAYERS):
            out = layers[L_idx](
                hidden_states=hidden2,
                position_embeddings=(cos.clone(), sin.clone()),
                attention_mask=None,
                position_ids=position_ids.clone(),
                past_key_values=cache2,
            )
            hidden2 = out[0] if isinstance(out, tuple) else out
    delta = (output - hidden2).float().abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output.float().flatten().unsqueeze(0),
        hidden2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  self-test: max|Δ|={delta:.2e}, cos={cos_sim:.8f}")
    assert delta < 1e-5, f"determinism broken: max|Δ|={delta}"
    assert cos_sim > 0.9999999, f"cos={cos_sim}"
    print("  ✓ deterministic")

    print("\nB4 DONE.")


if __name__ == "__main__":
    main()
