#!/usr/bin/env python3
"""
Experiment 91x — HF Qwen3.6-27B prefill oracle at seq_len=128.

Validates the numpy prefill ref from 91w (Phase B.1) against an external
HF transformers ground truth. If we don't match HF here, we have a bug
in our numpy math; no point writing ttnn yet.

Approach (mirrors 91o decode-mode validation pattern):
  - Use the same seed and per-position random input as 91w
  - Forward through HF Qwen3NextDecoderLayer for layer 0 (DeltaNet) and
    layer 3 (full attention) at fp32 on CPU
  - Compare per-position output to 91w's saved .npz
  - Gate: cosine ≥ 0.999 per position; max|Δ| < 1e-4 per element

Run on qb1 (HF weights cached there):
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91x_hf_prefill_oracle_seq128.py
"""
import os
import sys
import json
import inspect
import time

import numpy as np
import torch

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
import transformers

MODEL_ID = "Qwen/Qwen3.6-27B"
NUMPY_REF_PATH = os.path.expanduser(
    "~/tt-xla/.cache/qwen36_27b_prefill_numpy_ref_seq128.npz")
OUT_PATH = os.path.expanduser(
    "~/tt-xla/.cache/qwen36_27b_hf_prefill_oracle_seq128.npz")
SEQ_LEN = 128
SEED = 42


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def per_position_cosine(a, b):
    """a, b: [seq, hidden]. Returns per-position cosine [seq]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = (a * b).sum(axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return dot / (na * nb + 1e-12)


def load_layer_state_for_idx(layer_idx, weight_map):
    """Load all safetensors weights for a given layer idx as a state dict."""
    layer_keys = sorted(k for k in weight_map.keys()
                        if k.startswith(f'model.language_model.layers.{layer_idx}.'))
    state = {}
    shards_needed = sorted(set(weight_map[k] for k in layer_keys))
    for shard in shards_needed:
        shard_path = hf_hub_download(MODEL_ID, shard)
        with safe_open(shard_path, framework="pt") as f:
            for k in layer_keys:
                if weight_map[k] == shard:
                    local_key = k.replace(f'model.language_model.layers.{layer_idx}.', '')
                    state[local_key] = f.get_tensor(k).float()
    return state


def hf_forward_layer(layer, hidden_states, RotaryEmb, text_cfg, seq_len):
    """Run one HF DecoderLayer forward at fp32 CPU; return [seq, hidden]."""
    sig = inspect.signature(layer.forward)
    params = list(sig.parameters.keys())

    forward_kwargs = {}
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq]
    if 'position_ids' in params:
        forward_kwargs['position_ids'] = position_ids
    if 'cache_position' in params:
        forward_kwargs['cache_position'] = torch.arange(seq_len, dtype=torch.long)
    if 'attention_mask' in params:
        # Causal mask gets computed internally if None
        forward_kwargs['attention_mask'] = None
    for k in ('past_key_value', 'past_key_values'):
        if k in params:
            forward_kwargs[k] = None
    if 'use_cache' in params:
        forward_kwargs['use_cache'] = False
    if 'output_attentions' in params:
        forward_kwargs['output_attentions'] = False
    if 'position_embeddings' in params:
        rot = RotaryEmb(config=text_cfg).float().eval()
        with torch.no_grad():
            pe = rot(hidden_states, position_ids)
        forward_kwargs['position_embeddings'] = (pe[0].float(), pe[1].float())

    with torch.no_grad():
        result = layer(hidden_states, **forward_kwargs)
    out = result[0] if isinstance(result, tuple) else result
    out_np = out.float().cpu().numpy()
    if out_np.ndim == 3:
        out_np = out_np[0]  # drop batch
    return out_np


def main():
    print("=" * 64)
    print(f"Experiment 91x — HF prefill oracle (seq_len={SEQ_LEN}) vs 91w numpy ref")
    print("=" * 64)
    print(f"transformers version: {transformers.__version__}")

    # Load HF config + modeling module (auto-detected per 91o pattern)
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, 'text_config', full_cfg)
    print(f"  text_cfg.model_type: {getattr(text_cfg, 'model_type', '?')}")

    config_module_path = type(text_cfg).__module__
    modeling_module_path = config_module_path.replace('configuration_', 'modeling_')
    import importlib
    mqn = importlib.import_module(modeling_module_path)
    print(f"  modeling module: {mqn.__name__}")

    # Find DecoderLayer + RotaryEmbedding
    DecoderLayer = None
    for name in ('Qwen3_5DecoderLayer', 'Qwen3_5MoeDecoderLayer',
                 'Qwen3NextDecoderLayer', 'Qwen3DecoderLayer'):
        if hasattr(mqn, name):
            DecoderLayer = getattr(mqn, name)
            break
    RotaryEmb = None
    for name in ('Qwen3_5RotaryEmbedding', 'Qwen3NextRotaryEmbedding'):
        if hasattr(mqn, name):
            RotaryEmb = getattr(mqn, name)
            break
    print(f"  using DecoderLayer: {DecoderLayer.__name__}")
    print(f"  using RotaryEmb:    {RotaryEmb.__name__}")

    # Same seed + input as 91w
    rng = np.random.default_rng(SEED)
    x_seq_np = (rng.standard_normal((SEQ_LEN, text_cfg.hidden_size))
                .astype(np.float32) * 0.05)
    hidden_states = torch.from_numpy(x_seq_np).unsqueeze(0)  # [1, seq, hidden]
    print(f"  input hidden_states: {tuple(hidden_states.shape)} (mirrors 91w seed)")

    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    # ── Layer 0: DeltaNet ──────────────────────────────────────
    print(f"\n[layer 0: DeltaNet]")
    print(f"  loading layer 0 weights...")
    layer0 = DecoderLayer(text_cfg, layer_idx=0).float().eval()
    state0 = load_layer_state_for_idx(0, weight_map)
    info = layer0.load_state_dict(state0, strict=False)
    if info.missing_keys:
        print(f"  missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"  unexpected keys: {info.unexpected_keys}")

    print(f"  forwarding hidden_states ({SEQ_LEN} positions) through HF layer 0...")
    t0 = time.time()
    hf_layer0_out = hf_forward_layer(layer0, hidden_states, RotaryEmb, text_cfg, SEQ_LEN)
    print(f"  HF layer 0 forward: {time.time()-t0:.1f}s; "
          f"shape={hf_layer0_out.shape}, "
          f"per-position norm mean={np.linalg.norm(hf_layer0_out, axis=-1).mean():.4f}")

    # ── Layer 3: Gated Attention ───────────────────────────────
    # We feed it the layer 0 output (matches what numpy ref does)
    print(f"\n[layer 3: Gated Attention]")
    print(f"  loading layer 3 weights...")
    layer3 = DecoderLayer(text_cfg, layer_idx=3).float().eval()
    state3 = load_layer_state_for_idx(3, weight_map)
    info = layer3.load_state_dict(state3, strict=False)
    if info.missing_keys:
        print(f"  missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"  unexpected keys: {info.unexpected_keys}")

    # HF layer 0 included its MLP. Our numpy ref does the same. So we feed
    # HF layer 0 OUTPUT to HF layer 3, and compare to numpy post_layer3_seq.
    hidden_for_layer3 = torch.from_numpy(hf_layer0_out).unsqueeze(0)
    print(f"  forwarding through HF layer 3 ({SEQ_LEN} positions)...")
    t0 = time.time()
    hf_layer3_out = hf_forward_layer(layer3, hidden_for_layer3, RotaryEmb, text_cfg, SEQ_LEN)
    print(f"  HF layer 3 forward: {time.time()-t0:.1f}s; "
          f"shape={hf_layer3_out.shape}, "
          f"per-position norm mean={np.linalg.norm(hf_layer3_out, axis=-1).mean():.4f}")

    # ── Compare to 91w numpy ref ───────────────────────────────
    print(f"\n[validation]")
    if not os.path.exists(NUMPY_REF_PATH):
        print(f"  ERROR: numpy ref not found at {NUMPY_REF_PATH}")
        print(f"  Run experiments/91w_qwen36_27b_prefill_numpy_ref.py first.")
        sys.exit(1)

    ref = np.load(NUMPY_REF_PATH)
    numpy_layer0 = ref['post_layer0_seq']
    numpy_layer3 = ref['post_layer3_seq']

    def report(name, hf_out, numpy_out):
        print(f"\n  ── {name} ──")
        print(f"    shapes: HF={hf_out.shape}, numpy={numpy_out.shape}")
        print(f"    norms:  HF mean={np.linalg.norm(hf_out, axis=-1).mean():.4f}, "
              f"numpy mean={np.linalg.norm(numpy_out, axis=-1).mean():.4f}")
        pos_cos = per_position_cosine(hf_out, numpy_out)
        max_abs = np.abs(hf_out.astype(np.float64) - numpy_out.astype(np.float64)).max()
        print(f"    per-position cosine: min={pos_cos.min():.6f}, "
              f"median={np.median(pos_cos):.6f}, mean={pos_cos.mean():.6f}")
        print(f"    max|Δ| (any element): {max_abs:.6e}")
        if pos_cos.min() >= 0.999:
            verdict = "PASS (numpy ref math validated against HF)"
        elif pos_cos.min() >= 0.99:
            verdict = "BORDERLINE (small bug; localize per substep)"
        elif pos_cos.min() >= 0.5:
            verdict = "FAIL — major bug, re-derive math"
        else:
            verdict = "FAIL — essentially uncorrelated, severe wiring error"
        print(f"    VERDICT: {verdict}")
        return pos_cos, max_abs

    pos_cos_l0, max_abs_l0 = report("Layer 0 (DeltaNet)", hf_layer0_out, numpy_layer0)
    pos_cos_l3, max_abs_l3 = report("Layer 3 (Gated Attention)", hf_layer3_out, numpy_layer3)

    print(f"\n[save] writing oracle artifacts to {OUT_PATH}")
    np.savez(OUT_PATH,
             input_x_seq=x_seq_np,
             hf_post_layer0=hf_layer0_out,
             hf_post_layer3=hf_layer3_out,
             per_position_cosine_layer0=pos_cos_l0,
             per_position_cosine_layer3=pos_cos_l3,
             max_abs_diff_layer0=max_abs_l0,
             max_abs_diff_layer3=max_abs_l3)

    print(f"\n=== B.1 numpy ref validation complete ===")
    if pos_cos_l0.min() >= 0.999 and pos_cos_l3.min() >= 0.999:
        print(f"  ✓ numpy ref math validated. Proceed to B.2 (ttnn prefill).")
    else:
        print(f"  ✗ numpy ref FAILED validation. Debug math before ttnn work.")
        sys.exit(2)


if __name__ == "__main__":
    main()
