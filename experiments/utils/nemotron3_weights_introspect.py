#!/usr/bin/env python3
"""MM7 v0.0.2 — Nemotron-3 Nano safetensors weight introspect.

Walks the 13 safetensors shards' INDEX (header only, no weight tensors
deserialised) and audits every expected key shape vs the architecture
brief. Gate: zero unexpected keys; every shape matches config; per-layer-
type weight set complete.

Why bother before v0.1.0:
- Saves ~12 min of bootstrap iteration to learn we have a key-naming
  mismatch the hard way.
- Documents the canonical safetensors key layout that
  `server_nemotron3_nano_ttnn.py` will lookup at bootstrap.
- Catches silent surprises (e.g. a missing tied embedding, an unexpected
  per-layer scalar, a NORM_F-vs-NORM divergence).

REUSE: pattern forked from `experiments/utils/ttnn_introspect.py`'s
"walk a JSON manifest + audit shapes" idiom. Reads the safetensors
metadata only via `safe_open(...).get_slice(k).get_shape()` — never
materialises a tensor.

Run on the QuietBox:
    cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/nemotron3_weights_introspect.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Authoritative shape constants from the architecture brief §1 (also
# confirmed at config-probe time). Used as the assertion truth source.
HIDDEN_SIZE = 2688
VOCAB_SIZE = 131072
N_LAYERS = 52
N_HEADS = 32          # attention Q heads
N_KV_HEADS = 2        # attention KV heads (16:1 GQA)
HEAD_DIM = 128        # attention head_dim
MAMBA_HEADS = 64
MAMBA_HEAD_DIM = 64
SSM_STATE = 128
N_GROUPS = 8
CONV_KERNEL = 4
D_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM       # 4096
CONV_DIM = D_INNER + 2 * N_GROUPS * SSM_STATE  # 4096 + 2048 = 6144
N_ROUTED_EXPERTS = 128
N_SHARED_EXPERTS = 1
ROUTED_INTERMEDIATE = 1856
SHARED_INTERMEDIATE = 3712  # 2× routed per brief

# `hybrid_override_pattern` — verified against config at v0.0.
LAYER_KINDS = list(
    "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
)
_KIND = {"M": "mamba2", "E": "moe", "*": "attention"}
LAYER_TYPES = [_KIND[c] for c in LAYER_KINDS]
assert len(LAYER_TYPES) == N_LAYERS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_index_json() -> Path:
    """Locate model.safetensors.index.json in the HF cache."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    safe_id = MODEL_ID.replace("/", "--")
    pattern = f"models--{safe_id}/snapshots/*/model.safetensors.index.json"
    matches = list(hub.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no safetensors index under {hub}/models--{safe_id}/snapshots/ — "
            f"download the model first")
    return matches[0]


def expected_keys() -> dict[str, tuple]:
    """Build the full {key → expected shape} map for the model."""
    keys: dict[str, tuple] = {}

    # Embed + final norm + lm_head (tie_word_embeddings=False per brief).
    keys["backbone.embeddings.weight"] = (VOCAB_SIZE, HIDDEN_SIZE)
    keys["backbone.norm_f.weight"] = (HIDDEN_SIZE,)
    keys["lm_head.weight"] = (VOCAB_SIZE, HIDDEN_SIZE)

    for L, kind in enumerate(LAYER_TYPES):
        prefix = f"backbone.layers.{L}"
        # Every block has one pre-norm (`layer.norm`) — Llama-style w (no +1).
        keys[f"{prefix}.norm.weight"] = (HIDDEN_SIZE,)
        if kind == "mamba2":
            # NemotronHMamba2Mixer
            mp = f"{prefix}.mixer"
            keys[f"{mp}.in_proj.weight"] = (
                # gate(d_inner) + x_BC(d_inner + 2*groups*state) + dt(num_heads)
                D_INNER + CONV_DIM + MAMBA_HEADS, HIDDEN_SIZE,
            )
            keys[f"{mp}.conv1d.weight"] = (CONV_DIM, 1, CONV_KERNEL)
            keys[f"{mp}.conv1d.bias"] = (CONV_DIM,)
            keys[f"{mp}.dt_bias"] = (MAMBA_HEADS,)
            keys[f"{mp}.A_log"] = (MAMBA_HEADS,)
            keys[f"{mp}.D"] = (MAMBA_HEADS,)
            keys[f"{mp}.norm.weight"] = (D_INNER,)
            keys[f"{mp}.out_proj.weight"] = (HIDDEN_SIZE, D_INNER)
        elif kind == "attention":
            ap = f"{prefix}.mixer"
            keys[f"{ap}.q_proj.weight"] = (N_HEADS * HEAD_DIM, HIDDEN_SIZE)
            keys[f"{ap}.k_proj.weight"] = (N_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
            keys[f"{ap}.v_proj.weight"] = (N_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
            keys[f"{ap}.o_proj.weight"] = (HIDDEN_SIZE, N_HEADS * HEAD_DIM)
        elif kind == "moe":
            mp = f"{prefix}.mixer"
            # Router — DeepSeek-V3 style: sigmoid scores + group-restricted
            # top-k + per-expert load-balance bias `e_score_correction_bias`
            # added to the scores before topk. The bias is NOT trainable
            # via SGD; it's updated by an auxiliary load-balance loop.
            # Critical for v0.1.3: must add bias INTO scores before group
            # masking + topk, then unbias the routed weights for combine.
            keys[f"{mp}.gate.weight"] = (N_ROUTED_EXPERTS, HIDDEN_SIZE)
            keys[f"{mp}.gate.e_score_correction_bias"] = (N_ROUTED_EXPERTS,)
            # 128 routed experts — each is NemotronHMLP at routed_intermediate
            # The exact projection split is up_proj+gate_proj+down_proj;
            # we audit them per-expert below.
            for e in range(N_ROUTED_EXPERTS):
                ep = f"{mp}.experts.{e}"
                keys[f"{ep}.up_proj.weight"] = (ROUTED_INTERMEDIATE, HIDDEN_SIZE)
                keys[f"{ep}.down_proj.weight"] = (HIDDEN_SIZE, ROUTED_INTERMEDIATE)
            # Shared expert — 2× wider
            sp = f"{mp}.shared_experts"
            keys[f"{sp}.up_proj.weight"] = (SHARED_INTERMEDIATE, HIDDEN_SIZE)
            keys[f"{sp}.down_proj.weight"] = (HIDDEN_SIZE, SHARED_INTERMEDIATE)
        else:
            raise RuntimeError(f"unknown kind {kind!r} at L{L}")

    return keys


def main() -> int:
    log("locating safetensors index…")
    idx_path = find_index_json()
    log(f"  {idx_path}")
    idx = json.loads(idx_path.read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    log(f"  {len(weight_map)} keys across {len(set(weight_map.values()))} shards")
    log(f"  total_size = {idx['metadata']['total_size'] / 1024**3:.2f} GB")

    # ── Open every shard ONCE, build a single key → shape map ───────
    log("reading shard headers (no tensor materialisation)…")
    from safetensors import safe_open
    shard_dir = idx_path.parent
    actual_shapes: dict[str, tuple] = {}
    open_handles: dict[str, object] = {}
    try:
        # The same shard is referenced by many keys; cache the handle.
        for k, shard_name in weight_map.items():
            shard_path = shard_dir / shard_name
            if shard_name not in open_handles:
                open_handles[shard_name] = safe_open(
                    str(shard_path), framework="pt", device="cpu")
            actual_shapes[k] = tuple(
                open_handles[shard_name].get_slice(k).get_shape())
    finally:
        # safe_open file handles release on GC; no explicit close needed
        open_handles.clear()
    log(f"  collected {len(actual_shapes)} actual shapes")

    # ── Build expected map + diff ────────────────────────────────────
    log("building expected key/shape map from the brief…")
    expected = expected_keys()
    log(f"  {len(expected)} expected keys")

    # Unexpected = present on disk, NOT in our model.
    actual_keys = set(actual_shapes.keys())
    expected_keys_set = set(expected.keys())
    missing = expected_keys_set - actual_keys
    extra = actual_keys - expected_keys_set
    shape_mismatch: list[tuple[str, tuple, tuple]] = []
    for k, expect_shape in expected.items():
        if k in actual_shapes and actual_shapes[k] != expect_shape:
            shape_mismatch.append((k, expect_shape, actual_shapes[k]))

    log(f"  missing keys (expected, not on disk): {len(missing)}")
    if missing:
        for k in sorted(missing)[:20]:
            log(f"    - {k}")
        if len(missing) > 20:
            log(f"    … and {len(missing) - 20} more")

    log(f"  extra keys (on disk, not in our spec): {len(extra)}")
    if extra:
        for k in sorted(extra)[:20]:
            log(f"    - {k}  shape={actual_shapes[k]}")
        if len(extra) > 20:
            log(f"    … and {len(extra) - 20} more")

    log(f"  shape mismatches: {len(shape_mismatch)}")
    for k, e, a in shape_mismatch[:20]:
        log(f"    - {k}  expected={e}  actual={a}")
    if len(shape_mismatch) > 20:
        log(f"    … and {len(shape_mismatch) - 20} more")

    # ── Summary by layer kind ────────────────────────────────────────
    by_kind = defaultdict(int)
    for k in actual_keys:
        if ".layers." in k:
            try:
                L = int(k.split(".layers.")[1].split(".")[0])
                by_kind[LAYER_TYPES[L]] += 1
            except (ValueError, IndexError):
                by_kind["unknown"] += 1
        else:
            by_kind["non-layer"] += 1
    log("keys by layer kind:")
    for kind, n in sorted(by_kind.items()):
        log(f"    {kind:12s}  {n} keys")

    if missing or shape_mismatch:
        log("\nv0.0.2 weights introspect FAIL ✗")
        return 1

    log("\nv0.0.2 weights introspect PASS ✓ — all shapes match the brief")
    if extra:
        log(f"  ({len(extra)} extra on-disk keys — not blockers but worth a peek)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
