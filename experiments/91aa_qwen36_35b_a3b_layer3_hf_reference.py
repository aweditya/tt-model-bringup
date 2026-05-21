#!/usr/bin/env python3
"""B3 — Qwen3.6-35B-A3B Layer 3 (first FULL_ATTENTION layer) HF reference capture.

Layer 3 swaps `linear_attn` for `self_attn` (gated GQA, 16 Q / 2 KV / head_dim 256,
attn_output_gate=True). MoE block is identical to layer 0 except for trained
weight differences.

Two new wrinkles vs B2:
  1. Real `position_embeddings` from `Qwen3_5MoeTextRotaryEmbedding` (which
     handles MRoPE internally; for text-only input it degenerates to standard
     partial RoPE).
  2. Attention has KV cache concept but we pass `past_key_values=None` for a
     stateless single-token forward (no cache needed for first-token capture).

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91aa_qwen36_35b_a3b_layer3_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b3_layer3_full_reference.npz`
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
OUT_PATH = OUT_DIR / "b3_layer3_full_reference.npz"
LAYER_IDX = 3


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot: {SNAPSHOT_DIR}")
    print(f"output:   {OUT_PATH}")

    print("[1] load config…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    text_cfg._attn_implementation = "eager"
    print(f"  layer_types[{LAYER_IDX}]={text_cfg.layer_types[LAYER_IDX]}")
    assert text_cfg.layer_types[LAYER_IDX] == "full_attention"
    print(f"  num_attention_heads={text_cfg.num_attention_heads}, "
          f"num_key_value_heads={text_cfg.num_key_value_heads}, "
          f"head_dim={text_cfg.head_dim}")
    print(f"  partial_rotary_factor={text_cfg.partial_rotary_factor}, "
          f"rope_theta={text_cfg.rope_parameters['rope_theta']}, "
          f"mrope_section={text_cfg.rope_parameters.get('mrope_section')}")

    print("[2] build single Qwen3_5MoeDecoderLayer (layer 3) and rotary embed…")
    torch.set_default_dtype(torch.bfloat16)
    layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=LAYER_IDX)
    layer.eval()
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg)
    rotary.eval()
    print(f"  layer attributes: {[n for n, _ in layer.named_children()]}")

    print("[3] enumerate shards + load layer-3 weights…")
    key_to_shard = build_key_to_shard(SNAPSHOT_DIR)
    prefix = f"model.language_model.layers.{LAYER_IDX}."
    state_dict = {}
    for k in sorted(key_to_shard):
        if k.startswith(prefix):
            short = k[len(prefix):]
            state_dict[short] = load_tensor(key_to_shard, k)
    print(f"  collected {len(state_dict)} tensors")
    missing, unexpected = layer.load_state_dict(state_dict, strict=False)
    print(f"  missing: {missing}")
    print(f"  unexpected: {unexpected}")
    assert not unexpected, f"unexpected keys: {unexpected}"

    print("[4] build synthetic deterministic input + position_ids…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)
    position_ids = torch.tensor([[3]], dtype=torch.long)  # arbitrary position
    print(f"  hidden_in norm: {hidden_in.float().norm().item():.4f}")
    print(f"  position_ids: {position_ids.tolist()}")

    print("[5] compute position_embeddings via rotary…")
    with torch.no_grad():
        cos, sin = rotary(hidden_in, position_ids)
    print(f"  cos shape: {list(cos.shape)}, dtype={cos.dtype}")
    print(f"  sin shape: {list(sin.shape)}, dtype={sin.dtype}")

    print("[6] HF forward pass through layer 3…")
    forward_kwargs = {
        "hidden_states": hidden_in,
        "position_embeddings": (cos, sin),
        "attention_mask": None,
        "position_ids": position_ids,
        "past_key_values": None,  # stateless single-token forward
    }
    with torch.no_grad():
        out = layer(**forward_kwargs)
    output = out[0] if isinstance(out, tuple) else out
    print(f"  output: shape={list(output.shape)}, dtype={output.dtype}")
    print(f"  output norm: {output.float().norm().item():.4f}")
    print(f"  delta(in, out) norm: {(output - hidden_in).float().norm().item():.4f}")

    print("[7] save npz…")
    np.savez(
        OUT_PATH,
        hidden_in=hidden_in.float().numpy(),
        position_ids=position_ids.numpy(),
        cos=cos.detach().float().numpy(),
        sin=sin.detach().float().numpy(),
        output=output.float().numpy(),
        input_layernorm_weight=layer.input_layernorm.weight.detach().float().numpy(),
        post_attention_layernorm_weight=layer.post_attention_layernorm.weight.detach().float().numpy(),
        # Attention weights for downstream ttnn loading
        q_proj=layer.self_attn.q_proj.weight.detach().float().numpy(),
        k_proj=layer.self_attn.k_proj.weight.detach().float().numpy(),
        v_proj=layer.self_attn.v_proj.weight.detach().float().numpy(),
        o_proj=layer.self_attn.o_proj.weight.detach().float().numpy(),
        q_norm=layer.self_attn.q_norm.weight.detach().float().numpy(),
        k_norm=layer.self_attn.k_norm.weight.detach().float().numpy(),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    print("[8] self-test: re-run from fresh state…")
    with torch.no_grad():
        out2 = layer(
            hidden_states=hidden_in.clone(),
            position_embeddings=(cos.clone(), sin.clone()),
            attention_mask=None,
            position_ids=position_ids.clone(),
            past_key_values=None,
        )
    output2 = out2[0] if isinstance(out2, tuple) else out2
    delta = (output - output2).float().abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output.float().flatten().unsqueeze(0),
        output2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  self-test: max|Δ|={delta:.2e}, cos={cos_sim:.8f}")
    assert delta < 1e-5, f"determinism broken: max|Δ|={delta}"
    assert cos_sim > 0.9999999, f"cos={cos_sim}"
    print("  ✓ deterministic")

    print("\n[9] MRoPE-vs-standard-RoPE probe (for B7/B10 implementation guidance)…")
    # Check: for text-only with all 3 mrope axes carrying the same temporal position,
    # does MRoPE reduce to standard partial RoPE?
    # Standard RoPE uses just inv_freq * position (no axis interleaving).
    base = text_cfg.rope_parameters["rope_theta"]
    partial = text_cfg.partial_rotary_factor
    head_dim = text_cfg.head_dim
    rope_dim = int(head_dim * partial)
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    pos = float(position_ids[0, 0].item())
    freqs_standard = inv_freq * pos  # [rope_dim/2]
    emb_standard = torch.cat((freqs_standard, freqs_standard), dim=-1)
    cos_std = emb_standard.cos().to(torch.bfloat16)
    sin_std = emb_standard.sin().to(torch.bfloat16)
    cos_mrope_flat = cos[0, 0].to(torch.bfloat16)  # cos shape was [bs, seq, head_dim]
    sin_mrope_flat = sin[0, 0].to(torch.bfloat16)
    mrope_vs_std_max = max(
        (cos_mrope_flat - cos_std).abs().max().item(),
        (sin_mrope_flat - sin_std).abs().max().item(),
    )
    print(f"  MRoPE vs standard partial RoPE (text-only, single pos): max|Δ|={mrope_vs_std_max:.2e}")
    if mrope_vs_std_max < 1e-4:
        print("  ✓ MRoPE degenerates to standard partial RoPE for text-only — can reuse 27B RoPE code")
    else:
        print("  ⚠ MRoPE differs from standard partial RoPE — need to implement interleaved-section logic")

    print("\nB3 DONE.")


if __name__ == "__main__":
    main()
