#!/usr/bin/env python3
"""Print shapes/dtypes for every Gemma 4 12B assistant HF oracle artifact.

Forks `experiments/utils/npz_inspect.py` pattern (a per-array printer). Run
on qb2 ONLY:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/gemma4_assistant_oracle_shapes.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"

ARRAYS = (
    "input_ids",
    "target_h_last",
    "target_h_prev",
    "shared_kv_full_K",
    "shared_kv_full_V",
    "shared_kv_sliding_K",
    "shared_kv_sliding_V",
    "drafter_inputs_embeds",
    "drafter_hidden",
    "drafter_logits",
    "drafter_argmax",
    "drafter_topk_ids",
    "drafter_topk_vals",
)


def main() -> int:
    if not ORACLE_DIR.exists():
        print(f"FATAL: oracle missing at {ORACLE_DIR}", flush=True)
        return 1
    for p in range(5):
        base = ORACLE_DIR / f"prompt_{p}"
        if not base.exists():
            print(f"prompt_{p}: MISSING dir", flush=True)
            continue
        print(f"=== prompt_{p} ===", flush=True)
        for n in ARRAYS:
            f = base / f"{n}.npy"
            if not f.exists():
                print(f"  {n}: MISSING", flush=True)
                continue
            a = np.load(f)
            print(f"  {n}: shape={a.shape} dtype={a.dtype} "
                  f"first={float(a.flat[0]) if a.size else 'empty':.6g}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
