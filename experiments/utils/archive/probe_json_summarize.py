#!/usr/bin/env python3
"""Summarize a probe JSON artifact from this project's resident server
endpoints. Replaces the ad-hoc `python3 -c` / heredoc parsing that crept
in during the late-evening conv1d/decay-gate debug arc (per the no-inline-
scripts non-negotiable).

Auto-detects the probe schema by top-level keys, and prints a concise
human-readable summary plus a single-line headline:

- decay_gate_g1_sweep: per-layer max_abs_diff/PCC for decay+beta, aggregate
  min/max/median, all_pass.
- conv1d_split_check (extended): all_match + layout_meta_match + forward_clean,
  plus the forward conv_out max_abs_diff.
- cosine_ladder_compare: per-mode min/median/mean cosine, top-1
  disagreement rate, first disagreement step.

Usage:
    python experiments/utils/probe_json_summarize.py \\
        .cache/qb2_tp_deltanet/owned_decay_gate_g1_sweep_20260519_1039.json

    # Multiple files OK; each is summarized in turn.
    python experiments/utils/probe_json_summarize.py .cache/qb2_tp_deltanet/*.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


def _classify(d: dict[str, Any]) -> str:
    keys = set(d.keys())
    if {"n_layers_swept", "all_pass", "per_layer"} <= keys:
        return "decay_gate_g1_sweep"
    if {"comparisons", "all_match"} <= keys and "forward_check" in keys:
        return "conv1d_split_check_extended"
    if {"comparisons", "all_match"} <= keys:
        return "conv1d_split_check_minimal"
    if {"comparisons", "n_steps", "vocab", "base_mode"} <= keys:
        return "cosine_ladder_compare"
    return "unknown"


def _summarize_decay_gate_g1_sweep(d: dict[str, Any]) -> None:
    per_layer: dict[str, dict[str, float]] = d.get("per_layer", {})
    decay_diffs = [v["decay_max_diff"] for v in per_layer.values()]
    beta_diffs = [v["beta_max_diff"] for v in per_layer.values()]
    decay_pccs = [v["decay_pcc"] for v in per_layer.values()]
    beta_pccs = [v["beta_pcc"] for v in per_layer.values()]
    fails = [k for k, v in per_layer.items() if not v.get("pass", False)]
    n = d.get("n_layers_swept", len(per_layer))
    thresh = d.get("threshold")

    print(f"  schema: decay_gate_g1_sweep  layers={n}  threshold={thresh}")
    print(f"  all_pass: {d.get('all_pass')}")
    if decay_diffs:
        print(f"  decay_max_diff:  min={min(decay_diffs):.6f}  max={max(decay_diffs):.6f}  "
              f"median={statistics.median(decay_diffs):.6f}")
        print(f"  decay_pcc:       min={min(decay_pccs):.8f}  max={max(decay_pccs):.8f}")
    if beta_diffs:
        print(f"  beta_max_diff:   min={min(beta_diffs):.6f}  max={max(beta_diffs):.6f}  "
              f"median={statistics.median(beta_diffs):.6f}")
        print(f"  beta_pcc:        min={min(beta_pccs):.8f}  max={max(beta_pccs):.8f}")
    print(f"  failing layers:  {fails if fails else 'none'}")


def _summarize_conv1d_split_check_minimal(d: dict[str, Any]) -> None:
    print(f"  schema: conv1d_split_check_minimal  layer={d.get('layer_idx')}  "
          f"threshold={d.get('threshold')}")
    print(f"  all_match: {d.get('all_match')}")
    if d.get("diagnosis"):
        print(f"  diagnosis: {d['diagnosis']}")


def _summarize_conv1d_split_check_extended(d: dict[str, Any]) -> None:
    fc = d.get("forward_check", {})
    print(f"  schema: conv1d_split_check_extended  layer={d.get('layer_idx')}  "
          f"threshold={d.get('threshold')}")
    print(f"  all_match: {d.get('all_match')}  "
          f"layout_meta_match: {d.get('layout_meta_match')}  "
          f"forward_clean: {d.get('forward_clean')}")
    if fc:
        print(f"  forward conv_out max_abs_diff: {fc.get('max_abs_diff'):.6f}  "
              f"p99: {fc.get('p99_abs_diff', float('nan')):.6f}  "
              f"num_above_1e_3: {fc.get('num_above_1e_3')}/{(lambda s: s[0] if s else None)(fc.get('shape', []))}")
    if d.get("diagnosis"):
        print(f"  diagnosis: {d['diagnosis']}")


def _summarize_cosine_ladder_compare(d: dict[str, Any]) -> None:
    base = d.get("base_mode")
    M = d.get("n_steps")
    print(f"  schema: cosine_ladder_compare  base={base}  steps={M}  vocab={d.get('vocab')}")
    for mode, c in d.get("comparisons", {}).items():
        disagree = c.get("top1_disagree_count")
        rate = (disagree / M) if M else float("nan")
        print(f"  vs {mode}:  min_cos={c.get('min_cos'):.6f}  "
              f"med_cos={c.get('med_cos'):.6f}  mean_cos={c.get('mean_cos'):.6f}  "
              f"top1_disagree={disagree}/{M} ({rate*100:.2f}%)  "
              f"first_disag=step {c.get('first_disagree_step')}")


def _summarize(path: Path) -> int:
    try:
        text = path.read_text()
    except OSError as e:
        print(f"{path}: read failed: {e}", file=sys.stderr)
        return 1
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"{path}:")
        print(f"  not valid JSON ({e}) — first 200 chars:")
        print(f"  {text[:200]!r}")
        return 1
    if not isinstance(d, dict):
        print(f"{path}: top-level is not a dict ({type(d).__name__})")
        return 1

    schema = _classify(d)
    print(f"{path.name}:")
    if schema == "decay_gate_g1_sweep":
        _summarize_decay_gate_g1_sweep(d)
    elif schema == "conv1d_split_check_extended":
        _summarize_conv1d_split_check_extended(d)
    elif schema == "conv1d_split_check_minimal":
        _summarize_conv1d_split_check_minimal(d)
    elif schema == "cosine_ladder_compare":
        _summarize_cosine_ladder_compare(d)
    else:
        print(f"  schema: unknown — top-level keys: {sorted(d.keys())[:10]}")
        if d.get("error"):
            print(f"  error: {d['error']}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="probe JSON artifact paths")
    args = parser.parse_args(argv)

    rc = 0
    for i, p in enumerate(args.paths):
        if i > 0:
            print()
        if _summarize(p) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
