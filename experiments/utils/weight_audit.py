#!/usr/bin/env python3
"""
Permanent utility for auditing safetensors weights of any HF model.

Replaces inline `python -c` audits that violated the no-inline-scripts
non-negotiable. Used during B'9.5 to find the 4 missing weights bug;
keep available for every future model bringup.

Modes:
  --list-layer-keys LAYER_IDX
      Print every safetensors key under .layers.{LAYER_IDX}.
      Use to audit your loader vs what's actually stored.
  --stats KEY [KEY ...]
      Print mean/std/min/max/per-row-norm for the given keys.
      Use to distinguish (1+w) parameterizations (mean ≈ 0) from
      raw weights (mean ≈ 1) for RMSNorm-style layers.
  --norm-key-pattern PATTERN
      Print stats for every key matching the substring pattern.
      Example: --norm-key-pattern norm.weight
  --diff-loader-vs-safetensors LOADER_KEYS_FILE LAYER_IDX
      Read a JSON file listing the safetensors keys your loader requests,
      cross-check against actual safetensors keys for that layer.
      Reports missing keys (in safetensors but not requested) and extra
      keys (requested but not in safetensors).

Examples:
    python experiments/utils/weight_audit.py \
        --model Qwen/Qwen3.6-27B --list-layer-keys 0
    python experiments/utils/weight_audit.py \
        --model Qwen/Qwen3.6-27B --list-layer-keys 3
    python experiments/utils/weight_audit.py \
        --model Qwen/Qwen3.6-27B --norm-key-pattern norm.weight
    python experiments/utils/weight_audit.py \
        --model Qwen/Qwen3.6-27B --stats \
            model.language_model.layers.0.input_layernorm.weight \
            model.language_model.layers.0.linear_attn.norm.weight

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/weight_audit.py [options]
"""
import os, sys, json, argparse
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def load_weight_map(model_id):
    """Load the safetensors index and return the weight_map dict."""
    idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(idx_path) as f:
        return json.load(f)['weight_map']


def list_layer_keys(model_id, layer_idx):
    """Print every safetensors key under layers.{layer_idx}."""
    weight_map = load_weight_map(model_id)
    pat = f"layers.{layer_idx}."
    keys = sorted(k for k in weight_map.keys() if pat in k)
    print(f"=== {model_id} — layers.{layer_idx} keys ===")
    if not keys:
        print(f"  (no keys matching {pat!r})")
        return
    for k in keys:
        print(f"  {k}  →  {weight_map[k]}")
    print(f"  ({len(keys)} keys total)")


def weight_stats(model_id, keys):
    """Load and print statistics for the given safetensors keys."""
    weight_map = load_weight_map(model_id)
    for k in keys:
        if k not in weight_map:
            print(f"  {k}: NOT IN INDEX")
            continue
        path = hf_hub_download(model_id, weight_map[k])
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor(k).float().numpy()
        per_row_norm = (np.linalg.norm(w, axis=-1) if w.ndim >= 2 else None)
        print(f"  {k}")
        print(f"    shape={w.shape}  dtype_in_safetensors=<see fp32 cast>")
        print(f"    mean={w.mean():+.6f}  std={w.std():.6f}")
        print(f"    min={w.min():+.4f}  max={w.max():+.4f}  abs_max={np.abs(w).max():.4f}")
        if per_row_norm is not None and per_row_norm.size > 0:
            print(f"    per-row ‖·‖: mean={per_row_norm.mean():.4f}  "
                  f"std={per_row_norm.std():.4f}  "
                  f"min={per_row_norm.min():.4f}  max={per_row_norm.max():.4f}")
        # Heuristic for Qwen-style RMSNorm parameterization
        if 'norm' in k.lower():
            if abs(w.mean()) < 0.05:
                hint = "Qwen3_5RMSNorm (offset-from-1: load as 1+w)"
            elif 0.7 < w.mean() < 1.3:
                hint = "standard RMSNormGated (load as raw w)"
            else:
                hint = "unclear — inspect manually"
            print(f"    norm hint: {hint}")


def matching_keys(model_id, pattern):
    """All keys matching substring `pattern`."""
    weight_map = load_weight_map(model_id)
    return sorted(k for k in weight_map.keys() if pattern in k)


def diff_loader_vs_safetensors(model_id, layer_idx, requested_keys):
    """Cross-check loader's expected keys against safetensors for one layer."""
    weight_map = load_weight_map(model_id)
    pat = f"layers.{layer_idx}."
    actual = set(k for k in weight_map.keys() if pat in k)
    requested = set(requested_keys)
    missing = actual - requested  # in safetensors but not requested
    extra = requested - actual    # requested but not in safetensors
    print(f"=== Loader vs safetensors for layers.{layer_idx} ===")
    print(f"  loader expects {len(requested)} keys; safetensors has {len(actual)}")
    if missing:
        print(f"  MISSING (in safetensors, not requested by loader):")
        for k in sorted(missing):
            print(f"    {k}")
    if extra:
        print(f"  EXTRA (requested by loader, not in safetensors):")
        for k in sorted(extra):
            print(f"    {k}")
    if not missing and not extra:
        print(f"  ✓ loader and safetensors agree")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--list-layer-keys", type=int, default=None,
                   help="Print all safetensors keys for this layer index")
    p.add_argument("--stats", nargs="+", default=None,
                   help="Print stats for these safetensors keys")
    p.add_argument("--norm-key-pattern", default=None,
                   help="Print stats for keys containing this substring")
    p.add_argument("--diff-loader", nargs=2, metavar=("LOADER_KEYS_JSON", "LAYER_IDX"),
                   default=None,
                   help="Compare loader keys (JSON file) vs safetensors for layer")
    args = p.parse_args()

    if args.list_layer_keys is not None:
        list_layer_keys(args.model, args.list_layer_keys)

    if args.norm_key_pattern:
        keys = matching_keys(args.model, args.norm_key_pattern)
        print(f"=== Keys matching {args.norm_key_pattern!r}: {len(keys)} ===")
        weight_stats(args.model, keys[:30])  # cap to 30
        if len(keys) > 30:
            print(f"  ({len(keys) - 30} more keys not shown; pass explicit --stats)")

    if args.stats:
        print(f"=== Stats for {len(args.stats)} key(s) ===")
        weight_stats(args.model, args.stats)

    if args.diff_loader:
        json_path, layer_idx = args.diff_loader
        with open(json_path) as f:
            requested = json.load(f)
        diff_loader_vs_safetensors(args.model, int(layer_idx), requested)


if __name__ == "__main__":
    main()
