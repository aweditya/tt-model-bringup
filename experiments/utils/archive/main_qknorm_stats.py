"""main_qknorm_stats.py - print main-model q_norm/k_norm stats for layer 3 (first full_attention layer).
Confirms convention vs MTP's q_norm/k_norm so we know whether to add 1.0.
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

    # First full_attention layer is layer 3 (layer_types repeat L L L F)
    target_keys = [
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "model.language_model.layers.3.input_layernorm.weight",
        "model.language_model.layers.3.post_attention_layernorm.weight",
        "model.language_model.norm.weight",
    ]
    print(f"{'key':70s} {'shape':15s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 130)
    for k in target_keys:
        if k not in weight_map:
            print(f"{k} NOT FOUND")
            continue
        shard = weight_map[k]
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            t = f.get_tensor(k).float().numpy()
        print(f"{k:70s} {str(list(t.shape)):15s} {t.mean():10.5f} {t.std():10.5f} {t.min():10.4f} {t.max():10.4f}")


if __name__ == "__main__":
    main()
