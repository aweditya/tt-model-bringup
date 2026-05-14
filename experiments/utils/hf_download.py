#!/usr/bin/env python3
"""Download a HuggingFace model snapshot. Avoids inline python -c.

Usage: python experiments/utils/hf_download.py <model_id> [cache_dir]
  e.g. python experiments/utils/hf_download.py Qwen/Qwen3.6-27B ~/.cache/huggingface/hub
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("usage: hf_download.py <model_id> [cache_dir]")
        sys.exit(2)
    model_id = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/huggingface/hub")
    from huggingface_hub import snapshot_download
    print(f"downloading {model_id} to {cache_dir}...")
    path = snapshot_download(model_id, cache_dir=cache_dir)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
