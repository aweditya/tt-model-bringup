#!/usr/bin/env python3
"""B2 — Qwen3.6-35B-A3B Layer 0 FULL decoder layer HF reference capture.

Builds on B0 (`91x_*_dn_hf_reference.py`) and B1 (`91y_*_moe_hf_reference.py`).
Captures the entire `Qwen3_5MoeDecoderLayer.forward()` so we can validate
the composition (input_layernorm → DN → residual → post_attention_layernorm
→ MoE → residual) end-to-end against HF.

Why this matters: B0 + B1 individually pass. B2 confirms our residual order,
the qwen3_5 RMSNorm convention (`output * (1.0 + weight)`), and the input
layernorm tensor wiring are all correct.

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91z_qwen36_35b_a3b_layer0_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b2_layer0_full_reference.npz`
"""
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
)


SNAPSHOT_DIR = Path(
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)
OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b2_layer0_full_reference.npz"
LAYER_IDX = 0


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


class FakeCache:
    def __init__(self, conv_state, recurrent_state, layer_idx):
        self.layers = {layer_idx: _LayerState(conv_state, recurrent_state)}

    def has_previous_state(self, layer_idx):
        return True

    def update_conv_state(self, new_state, layer_idx):
        self.layers[layer_idx].conv_states = new_state

    def update_recurrent_state(self, new_state, layer_idx):
        self.layers[layer_idx].recurrent_states = new_state


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot: {SNAPSHOT_DIR}")
    print(f"output:   {OUT_PATH}")

    print("[1] load config…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    # Inspect layer_types: layer 0 should be "linear_attention"
    print(f"  layer_types[0]={text_cfg.layer_types[0]} (expect linear_attention)")
    print(f"  layer_types[3]={text_cfg.layer_types[3]} (expect full_attention)")
    print(f"  rms_norm_eps={text_cfg.rms_norm_eps}")

    print("[2] build single Qwen3_5MoeDecoderLayer (layer 0)…")
    torch.set_default_dtype(torch.bfloat16)
    layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=LAYER_IDX)
    layer.eval()
    print(f"  layer: {layer.__class__.__name__}")
    # Inspect attribute names that the layer composes
    print(f"  attributes: {[n for n, _ in layer.named_children()]}")

    print("[3] enumerate shards + load all layer-0 weights…")
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

    print("[4] build synthetic deterministic input…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)
    print(f"  hidden_in norm: {hidden_in.float().norm().item():.4f}")

    print("[5] init DN cache state (zeros)…")
    # Layer 0 is linear_attention so we need DN cache.
    dn = layer.linear_attn
    conv_state_init = torch.zeros(B, dn.conv_dim, dn.conv_kernel_size, dtype=torch.bfloat16)
    recurrent_state_init = torch.zeros(
        B, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim, dtype=torch.float32
    )
    cache = FakeCache(conv_state_init.clone(), recurrent_state_init.clone(), LAYER_IDX)

    print("[6] HF forward pass through Qwen3_5MoeDecoderLayer…")
    # DecoderLayer.forward signature varies; inspect to call correctly
    import inspect
    sig = inspect.signature(layer.forward)
    print(f"  forward signature params: {list(sig.parameters.keys())}")

    # Layer 0 is linear_attention so position_embeddings is unused — but the
    # signature still requires it as a positional arg. Pass dummy (None, None).
    forward_kwargs = {
        "hidden_states": hidden_in,
        "position_embeddings": (None, None),
        "attention_mask": None,
        "position_ids": torch.zeros(B, T, dtype=torch.long),
        "past_key_values": cache,
    }
    # Only pass parameters the layer actually accepts.
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if k in sig.parameters}
    print(f"  passing: {list(forward_kwargs.keys())}")

    with torch.no_grad():
        out = layer(**forward_kwargs)

    # output can be a tensor or a tuple — handle both
    if isinstance(out, tuple):
        output = out[0]
        print(f"  output is tuple of len {len(out)}; using out[0]")
    else:
        output = out
    print(f"  output: shape={list(output.shape)}, dtype={output.dtype}")
    print(f"  output norm: {output.float().norm().item():.4f}")
    print(f"  residual check: output should differ from input significantly")
    print(f"  delta(output, input) norm: {(output - hidden_in).float().norm().item():.4f}")

    print("[7] save npz…")
    np.savez(
        OUT_PATH,
        hidden_in=hidden_in.float().numpy(),
        output=output.float().numpy(),
        conv_state_in=conv_state_init.float().numpy(),
        recurrent_state_in=recurrent_state_init.float().numpy(),
        conv_state_out=cache.layers[LAYER_IDX].conv_states.float().numpy(),
        recurrent_state_out=cache.layers[LAYER_IDX].recurrent_states.float().numpy(),
        input_layernorm_weight=layer.input_layernorm.weight.detach().float().numpy(),
        post_attention_layernorm_weight=layer.post_attention_layernorm.weight.detach().float().numpy(),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    print("[8] self-test: re-run from fresh state…")
    cache2 = FakeCache(conv_state_init.clone(), recurrent_state_init.clone(), LAYER_IDX)
    kw2 = dict(forward_kwargs)
    kw2["hidden_states"] = hidden_in.clone()
    kw2["past_key_values"] = cache2
    with torch.no_grad():
        out2 = layer(**kw2)
    output2 = out2[0] if isinstance(out2, tuple) else out2
    delta = (output - output2).float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten().unsqueeze(0),
        output2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  self-test: max|Δ|={delta:.2e}, cos={cos:.8f}")
    assert delta < 1e-5, f"determinism broken: max|Δ|={delta}"
    assert cos > 0.9999999, f"cos={cos}"
    print("  ✓ deterministic")

    print("\nB2 DONE.")


if __name__ == "__main__":
    main()
