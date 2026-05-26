"""mtp_inspect_weights.py - dump shape + dtype of every MTP weight in Qwen3.6-27B.

Step 1 of the speculative-decoding probe. Pure host-side, no device opens.

Usage:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/mtp_inspect_weights.py
"""
import json
import sys

from huggingface_hub import hf_hub_download
from safetensors import safe_open


MODEL_ID = "Qwen/Qwen3.6-27B"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp"))
    print(f"MTP weight keys (n={len(mtp_keys)}):")

    by_shard = {}
    for k in mtp_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    total_bytes = 0
    for shard, keys in sorted(by_shard.items()):
        path = hf_hub_download(MODEL_ID, shard)
        print(f"\n  shard {shard}:")
        with safe_open(path, framework="pt") as f:
            for k in keys:
                s = f.get_slice(k)
                shape = list(s.get_shape())
                dtype = s.get_dtype()
                # rough byte count: bf16 = 2 bytes/elt
                elt = 1
                for d in shape:
                    elt *= d
                bytes_ = elt * 2
                total_bytes += bytes_
                shape_str = str(shape)
                print(f"    {k:55s} shape={shape_str:25s} dtype={dtype}")

    print(f"\nTotal MTP size: {total_bytes / 1e6:.1f} MB (assuming bf16, 2 B/elt)")

    # Also probe the text config for context
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    print("\nRelevant config keys:")
    for k in (
        "hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "rms_norm_eps", "partial_rotary_factor",
        "vocab_size", "num_hidden_layers",
        "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
    ):
        if k in text:
            print(f"  {k}: {text[k]}")
        elif k in cfg:
            print(f"  {k}: {cfg[k]}")


if __name__ == "__main__":
    main()
