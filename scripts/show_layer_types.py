#!/usr/bin/env python3
"""Print a model's text_config.layer_types pattern. Helper for
correlating per-layer cosine drift (e.g. from
`scripts/inspect_ladder_npz.py`) against attention type
(sliding vs full).

Usage:
    python3 scripts/show_layer_types.py [model-id]
        model-id defaults to google/gemma-4-12B-it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it"
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snap_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snap_dir.exists():
        print(f"FATAL: {snap_dir} not found", file=sys.stderr)
        return 2
    snap = next(snap_dir.iterdir())
    cfg_path = snap / "config.json"
    cfg = json.loads(cfg_path.read_text())
    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg.get("layer_types")
    if layer_types is None:
        print(f"no `layer_types` key in {cfg_path}", file=sys.stderr)
        return 1
    print(f"model_id = {model_id}")
    print(f"n_layers = {len(layer_types)}")
    print(f"unique = {sorted(set(layer_types))}")
    print()
    # One char per layer (s for sliding, F for full/global).
    short = "".join("s" if t == "sliding_attention" else "F"
                    for t in layer_types)
    print(f"pattern (s=sliding, F=full): {short}")
    print()
    # Full enumeration.
    for i, t in enumerate(layer_types):
        print(f"  layer {i:2d}: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
