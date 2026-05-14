"""mtp_weight_stats.py - print mean/std of every MTP weight to detect (1+w) vs (w) RMSNorm convention.

For Qwen3_5RMSNorm-style weights (which add 1.0 at load): raw mean ≈ 0 (small offset around 0)
For RMSNormGated-style weights: raw mean ≈ 1 (scale around 1)
For regular projection weights: mean ≈ 0, std ≈ 0.02

This helps detect if any MTP norm weight was mishandled in load_mtp_weights().
"""
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.stdout.reconfigure(line_buffering=True)

MODEL_ID = "Qwen/Qwen3.6-27B"


def main():
    idx = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp"))

    by_shard = {}
    for k in mtp_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    print(f"{'key':60s} {'shape':25s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 130)
    for shard, keys in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k).float().numpy()
                m, s = t.mean(), t.std()
                print(f"{k:60s} {str(list(t.shape)):25s} {m:10.5f} {s:10.5f} {t.min():10.4f} {t.max():10.4f}")


if __name__ == "__main__":
    main()
