#!/usr/bin/env python3
"""B1 — Qwen3.6-35B-A3B Layer 0 MoE block HF reference capture.

Companion to `91x_qwen36_35b_a3b_dn_hf_reference.py`. Same pattern: instantiate
just `Qwen3_5MoeSparseMoeBlock` (router + 256 routed experts + shared expert
+ shared_expert_gate), load layer-0 mlp.* weights, run forward, save npz.

Simpler than B0 — no recurrent state, no in-place conv mutation. Just
pure-functional forward of `[B, T, hidden] → [B, T, hidden]`.

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91y_qwen36_35b_a3b_moe_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b1_moe_layer0_reference.npz`
"""
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)


SNAPSHOT_DIR = Path(
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)
OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b1_moe_layer0_reference.npz"
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot: {SNAPSHOT_DIR}")
    print(f"output:   {OUT_PATH}")

    print("[1] load config…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    print(f"  hidden={text_cfg.hidden_size}, "
          f"num_experts={text_cfg.num_experts}, "
          f"top_k={text_cfg.num_experts_per_tok}, "
          f"moe_inter={text_cfg.moe_intermediate_size}, "
          f"shared_inter={text_cfg.shared_expert_intermediate_size}")

    print("[2] build single Qwen3_5MoeSparseMoeBlock module…")
    torch.set_default_dtype(torch.bfloat16)
    moe = Qwen3_5MoeSparseMoeBlock(text_cfg)
    moe.eval()
    print(f"  module: {moe.__class__.__name__}")

    print("[3] enumerate shards + load layer-0 mlp weights…")
    key_to_shard = build_key_to_shard(SNAPSHOT_DIR)
    prefix = f"model.language_model.layers.{LAYER_IDX}.mlp."
    state_dict = {}
    for k in sorted(key_to_shard):
        if k.startswith(prefix):
            short = k[len(prefix):]
            t = load_tensor(key_to_shard, k)
            state_dict[short] = t
            print(f"  loaded {short}: shape={list(t.shape)}, dtype={t.dtype}")
    missing, unexpected = moe.load_state_dict(state_dict, strict=False)
    print(f"  missing: {missing}")
    print(f"  unexpected: {unexpected}")
    assert not unexpected, f"unexpected keys after load: {unexpected}"

    print("[4] build synthetic deterministic input…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)
    print(f"  hidden_in norm: {hidden_in.float().norm().item():.4f}")

    print("[5] HF forward pass…")
    with torch.no_grad():
        output = moe(hidden_in)
    print(f"  output: shape={list(output.shape)}, dtype={output.dtype}")
    print(f"  output norm: {output.float().norm().item():.4f}")

    print("[6] inspect routing decision (for debug context)…")
    with torch.no_grad():
        flat = hidden_in.view(-1, H)
        _, routing_weights, selected_experts = moe.gate(flat)
    print(f"  selected_experts (top-{text_cfg.num_experts_per_tok}): "
          f"{selected_experts[0].tolist()}")
    print(f"  routing_weights: {routing_weights[0].tolist()}")
    print(f"  weights sum: {routing_weights[0].sum().item():.6f} (should be ~1.0)")

    print("[7] save npz…")
    np.savez(
        OUT_PATH,
        hidden_in=hidden_in.float().numpy(),
        output=output.float().numpy(),
        selected_experts=selected_experts.detach().numpy(),
        routing_weights=routing_weights.detach().float().numpy(),
        # Weights for downstream ttnn loading
        router_weight=moe.gate.weight.detach().float().numpy(),
        experts_gate_up_proj=moe.experts.gate_up_proj.detach().float().numpy(),
        experts_down_proj=moe.experts.down_proj.detach().float().numpy(),
        shared_gate_proj=moe.shared_expert.gate_proj.weight.detach().float().numpy(),
        shared_up_proj=moe.shared_expert.up_proj.weight.detach().float().numpy(),
        shared_down_proj=moe.shared_expert.down_proj.weight.detach().float().numpy(),
        shared_expert_gate=moe.shared_expert_gate.weight.detach().float().numpy(),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    print("[8] self-test: re-run from fresh state, verify same output…")
    with torch.no_grad():
        output2 = moe(hidden_in.clone())
    delta = (output - output2).float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten().unsqueeze(0),
        output2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  self-test: max|Δ|={delta:.2e}, cos={cos:.8f}")
    assert delta < 1e-5, f"determinism broken: max|Δ|={delta}"
    assert cos > 0.9999999, f"cos={cos}"
    print("  ✓ deterministic")

    print("\nB1 DONE.")


if __name__ == "__main__":
    main()
