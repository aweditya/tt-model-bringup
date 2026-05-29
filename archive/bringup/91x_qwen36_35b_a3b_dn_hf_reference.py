#!/usr/bin/env python3
"""B0 — Qwen3.6-35B-A3B Layer 0 DeltaNet HF reference capture.

Per `research/qwen36_35b_a3b_incremental_block_plan_2026_05_21.md` and the
DN correction note, this script:

  1. Instantiates a SINGLE `Qwen3_5MoeGatedDeltaNet` module (not the full
     70GB model) by loading just layer 0's `linear_attn.*` weights from
     the cached safetensors shards.
  2. Runs forward on synthetic deterministic input (numpy RNG seed=0).
  3. Saves (input, weights, conv_state_in, recurrent_state_in, output,
     conv_state_out, recurrent_state_out) to a single npz.

The npz becomes the GOLD REFERENCE for B7 (single-chip ttnn) and B10 (TP
mesh) — those will load identical inputs/weights, run our ttnn
implementation, and cosine-compare against the saved HF output.

User directive 2026-05-21: use HF directly as reference instead of writing
a numpy reimplementation — avoids the class of "I got the math wrong"
bugs by construction.

Run on qb2 (weights cached at ~/.cache/huggingface/hub/...):

    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91x_qwen36_35b_a3b_dn_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b0_dn_layer0_reference.npz`
(per-CLAUDE.md NN#7, project dir not /tmp).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)


SNAPSHOT_DIR = Path(
    "/home/aditya/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0"
)
OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b0_dn_layer0_reference.npz"
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
    """Per-layer cache holder with `conv_states` and `recurrent_states`."""
    def __init__(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor):
        self.conv_states = conv_state
        self.recurrent_states = recurrent_state


class FakeCache:
    """Duck-typed cache satisfying Qwen3_5MoeGatedDeltaNet's contract.

    The DN forward reads `cache_params.has_previous_state(layer_idx)`,
    `cache_params.layers[layer_idx].conv_states`, `.recurrent_states`, and
    calls `update_conv_state(...)` / `update_recurrent_state(...)`. We
    deliberately do NOT inherit from `transformers.cache_utils.Cache`
    because its __init__ signature changed in 5.x and we only need a tiny
    subset of the interface.
    """
    def __init__(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor, layer_idx: int):
        self.layers = {layer_idx: _LayerState(conv_state, recurrent_state)}

    def has_previous_state(self, layer_idx: int) -> bool:
        return True

    def update_conv_state(self, new_state: torch.Tensor, layer_idx: int):
        self.layers[layer_idx].conv_states = new_state

    def update_recurrent_state(self, new_state: torch.Tensor, layer_idx: int):
        self.layers[layer_idx].recurrent_states = new_state


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"snapshot: {SNAPSHOT_DIR}")
    print(f"output:   {OUT_PATH}")

    print("[1] load config…")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg.dtype = torch.bfloat16
    print(f"  hidden={text_cfg.hidden_size}, "
          f"v_heads={text_cfg.linear_num_value_heads}, "
          f"k_heads={text_cfg.linear_num_key_heads}, "
          f"head_dim={text_cfg.linear_value_head_dim}, "
          f"conv_kernel={text_cfg.linear_conv_kernel_dim}")

    print("[2] build single Qwen3_5MoeGatedDeltaNet module…")
    torch.set_default_dtype(torch.bfloat16)
    dn = Qwen3_5MoeGatedDeltaNet(text_cfg, layer_idx=LAYER_IDX)
    dn.eval()
    print(f"  module: {dn.__class__.__name__}")
    print(f"  conv_dim={dn.conv_dim}, key_dim={dn.key_dim}, value_dim={dn.value_dim}")

    print("[3] enumerate shards + load layer-0 linear_attn weights…")
    key_to_shard = build_key_to_shard(SNAPSHOT_DIR)
    prefix = f"model.language_model.layers.{LAYER_IDX}.linear_attn."
    state_dict = {}
    for k in sorted(key_to_shard):
        if k.startswith(prefix):
            short = k[len(prefix):]
            t = load_tensor(key_to_shard, k)
            state_dict[short] = t
            print(f"  loaded {short}: shape={list(t.shape)}, dtype={t.dtype}")
    # The HF module names match what's in the state dict (no remap needed)
    missing, unexpected = dn.load_state_dict(state_dict, strict=False)
    print(f"  missing: {missing}")
    print(f"  unexpected: {unexpected}")
    assert not unexpected, f"unexpected keys after load: {unexpected}"

    print("[4] build synthetic deterministic input…")
    torch.manual_seed(0)
    np.random.seed(0)
    B, T, H = 1, 1, text_cfg.hidden_size
    # Realistic post-layernorm scale: hidden state has elementwise std ~ 1 after RMSNorm.
    # (Earlier scale 0.05 produced output norm ~1e-3 — signal/noise was bad.)
    hidden_in = torch.randn(B, T, H, dtype=torch.bfloat16)

    print("[5] build zero conv_state + recurrent_state for fresh decode…")
    # NOTE: causal_conv1d_update MUTATES conv_state in-place. We snapshot a copy
    # *before* any forward so we can both record the true input state and run
    # the determinism self-test from the same fresh state in step [9].
    conv_state_init = torch.zeros(B, dn.conv_dim, dn.conv_kernel_size, dtype=torch.bfloat16)
    recurrent_state_init = torch.zeros(
        B, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim, dtype=torch.float32
    )
    print(f"  conv_state: {list(conv_state_init.shape)}")
    print(f"  recurrent_state: {list(recurrent_state_init.shape)} (fp32 per mamba_ssm_dtype)")

    print("[6] HF forward pass…")
    # Pass CLONES into the cache so the init tensors stay pristine for self-test.
    cache = FakeCache(conv_state_init.clone(), recurrent_state_init.clone(), LAYER_IDX)
    with torch.no_grad():
        output = dn(hidden_in, cache_params=cache, attention_mask=None)
    print(f"  output: shape={list(output.shape)}, dtype={output.dtype}")
    print(f"  output norm: {output.float().norm().item():.4f}")

    print("[7] capture state-out…")
    conv_state_out = cache.layers[LAYER_IDX].conv_states
    recurrent_state_out = cache.layers[LAYER_IDX].recurrent_states
    print(f"  conv_state_out: {list(conv_state_out.shape)}, "
          f"changed_norm={(conv_state_out - conv_state_init).float().norm().item():.6f}")
    print(f"  recurrent_state_out: {list(recurrent_state_out.shape)}, "
          f"changed_norm={(recurrent_state_out - recurrent_state_init).float().norm().item():.6f}")

    print("[8] save npz…")
    np.savez(
        OUT_PATH,
        hidden_in=hidden_in.float().numpy(),
        conv_state_in=conv_state_init.float().numpy(),
        recurrent_state_in=recurrent_state_init.float().numpy(),
        output=output.float().numpy(),
        conv_state_out=conv_state_out.float().numpy(),
        recurrent_state_out=recurrent_state_out.float().numpy(),
        # Weights for downstream ttnn loading
        in_proj_qkv=dn.in_proj_qkv.weight.detach().float().numpy(),
        in_proj_z=dn.in_proj_z.weight.detach().float().numpy(),
        in_proj_a=dn.in_proj_a.weight.detach().float().numpy(),
        in_proj_b=dn.in_proj_b.weight.detach().float().numpy(),
        conv1d_weight=dn.conv1d.weight.detach().float().numpy(),
        A_log=dn.A_log.detach().float().numpy(),
        dt_bias=dn.dt_bias.detach().float().numpy(),
        norm_weight=dn.norm.weight.detach().float().numpy(),
        out_proj=dn.out_proj.weight.detach().float().numpy(),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    print("[9] self-test: re-run from fresh state, verify same output…")
    # Use the snapshotted init tensors (not the ones the first call mutated).
    cache2 = FakeCache(conv_state_init.clone(), recurrent_state_init.clone(), LAYER_IDX)
    with torch.no_grad():
        output2 = dn(hidden_in.clone(), cache_params=cache2, attention_mask=None)
    delta = (output - output2).float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten().unsqueeze(0),
        output2.float().flatten().unsqueeze(0),
    ).item()
    print(f"  self-test: max|Δ|={delta:.2e}, cos={cos:.8f}")
    assert delta < 1e-5, f"determinism broken: max|Δ|={delta}"
    assert cos > 0.9999999, f"cos={cos}"
    print("  ✓ deterministic")

    print("\nB0 DONE.")


if __name__ == "__main__":
    main()
