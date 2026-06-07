#!/usr/bin/env python3
"""Phase 1 v0.1 — weight introspection for google/gemma-4-12b-it-assistant.

Walks the safetensors index for the drafter model and prints every key +
shape + dtype. We need this BEFORE writing the bootstrap function in
`experiments/serve/server_gemma4_12b_assistant_ttnn.py` because the drafter
has a unique structure (pre_projection + post_projection + tied lm_head +
4 Gemma 4 layers) and the HF key prefixes might not match the target's
(`model.language_model.` etc.).

Forks `experiments/utils/nemotron3_weights_introspect.py`.

Run on qb2:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/gemma4_assistant_weights_introspect.py'
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path


MODEL_ID = "google/gemma-4-12b-it-assistant"


def main() -> int:
    t0 = time.time()
    cache_dirname = "models--google--gemma-4-12b-it-assistant"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname / "snapshots"
    if not snapshot_root.exists():
        print(f"FATAL: no HF snapshot at {snapshot_root}. fetch first.", file=sys.stderr)
        return 1

    snap = None
    for cand in snapshot_root.iterdir():
        if not cand.is_dir():
            continue
        if (cand / "model.safetensors.index.json").exists() or \
                any(cand.glob("*.safetensors")):
            snap = cand
            break
    if snap is None:
        print(f"FATAL: no snapshot dir contains weights under {snapshot_root}",
              file=sys.stderr)
        return 1
    print(f"snapshot: {snap}")

    # Read config
    cfg = json.loads((snap / "config.json").read_text())
    print(f"\nTop-level config:")
    for k in ("model_type", "architectures", "backbone_hidden_size",
              "use_ordered_embeddings", "tie_word_embeddings", "dtype"):
        print(f"  {k}: {cfg.get(k)}")
    tc = cfg.get("text_config", {})
    print(f"\ntext_config:")
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "num_global_key_value_heads",
              "head_dim", "intermediate_size", "vocab_size",
              "sliding_window", "rms_norm_eps", "layer_types"):
        print(f"  {k}: {tc.get(k)}")

    # Walk safetensors
    from safetensors import safe_open
    index = snap / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        key_to_shard = {k: str(snap / v) for k, v in idx["weight_map"].items()}
    else:
        sf = next(snap.glob("*.safetensors"))
        with safe_open(sf, framework="pt") as f:
            key_to_shard = {k: str(sf) for k in f.keys()}

    print(f"\n{len(key_to_shard)} keys total")
    print(f"\n--- ALL KEYS (key | shape | dtype) ---")

    # Group keys by prefix for readability.
    by_prefix = defaultdict(list)
    for k in sorted(key_to_shard.keys()):
        # Cluster by top-level prefix (everything before the first '.layers.<N>.' or terminal segment).
        if ".layers." in k:
            prefix = k.split(".layers.")[0] + ".layers.<L>"
            tail = k.split(".layers.")[1]
            # Strip the layer index
            tail_no_idx = ".".join(tail.split(".")[1:])
            by_prefix[prefix + "." + tail_no_idx].append(k)
        else:
            by_prefix[k].append(k)

    # Open each shard once
    open_files = {}
    for shard in set(key_to_shard.values()):
        open_files[shard] = safe_open(shard, framework="pt", device="cpu")

    try:
        # Print top-level (non-layer) keys first.
        print("\n[TOP-LEVEL keys]")
        for k in sorted(key_to_shard.keys()):
            if ".layers." in k:
                continue
            slice_ = open_files[key_to_shard[k]].get_slice(k)
            shape = tuple(slice_.get_shape())
            dt = slice_.get_dtype()
            print(f"  {k}  shape={shape}  dtype={dt}")

        # Per-layer keys: print pattern + show ALL 4 layers for compare.
        print("\n[PER-LAYER patterns — showing L0,L1,L2,L3]")
        # Collect all unique tails
        tails = set()
        for k in key_to_shard.keys():
            if ".layers." not in k:
                continue
            prefix, suffix = k.split(".layers.")
            parts = suffix.split(".", 1)
            if len(parts) == 2:
                tails.add((prefix, parts[1]))
        for prefix, tail in sorted(tails):
            for L in (0, 1, 2, 3):
                key = f"{prefix}.layers.{L}.{tail}"
                if key in key_to_shard:
                    slice_ = open_files[key_to_shard[key]].get_slice(key)
                    shape = tuple(slice_.get_shape())
                    dt = slice_.get_dtype()
                    print(f"  L{L}: {key}  shape={shape}  dtype={dt}")
                else:
                    print(f"  L{L}: {key}  ABSENT")

        # Also: enumerate per-layer tails seen (to catch missing keys e.g. v_proj for full layer)
        print("\n[PER-LAYER tails seen (sorted set across all 4 layers)]")
        seen_tails_by_L = defaultdict(set)
        for k in key_to_shard.keys():
            if ".layers." not in k:
                continue
            prefix, suffix = k.split(".layers.")
            parts = suffix.split(".", 1)
            if len(parts) == 2:
                L = int(parts[0])
                seen_tails_by_L[L].add(parts[1])
        all_tails = sorted(set().union(*seen_tails_by_L.values()))
        print(f"  union: {len(all_tails)} unique tails")
        for tail in all_tails:
            presence = "".join("X" if tail in seen_tails_by_L[L] else "." for L in range(4))
            print(f"  [{presence}] {tail}")

        print(f"\ndone in {time.time()-t0:.1f}s")
    finally:
        for f in open_files.values():
            try:
                f.__exit__(None, None, None)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
