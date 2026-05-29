"""Scrape HuggingFace API for the latest Qwen models.

Filters out quantizations (FP8, GPTQ, AWQ, GGUF, MLX, INT4, INT8) and
mirrors / SAEs. Shows the most recently modified BASE models in three
buckets: coder-specific, dense general, MoE/hybrid.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python pjrt_plugin/scripts/qwen_latest.py

Outputs a markdown-formatted table to stdout. Use this whenever you
want to verify we're chasing the actual latest, not just what we
remember from a week ago.
"""

import json
import sys
import urllib.request
from datetime import datetime

API = "https://huggingface.co/api/models?author=Qwen&sort=lastModified&direction=-1&limit=200"

QUANT_SUFFIXES = (
    "FP8", "GPTQ", "AWQ", "GGUF", "MLX",
    "Int4", "INT4", "Int8", "INT8",
    "W4A16", "BNB", "BitNet",
)
SAE_PREFIX = "SAE-"
# Models with these in the name are interpretability artifacts / not for inference
SKIP_PATTERNS = ("Embedding", "Reranker", "VL-Predict")


def is_base_model(model_id: str) -> bool:
    name = model_id.split("/", 1)[-1]
    if name.startswith(SAE_PREFIX):
        return False
    for suf in QUANT_SUFFIXES:
        if suf in name:
            return False
    for skip in SKIP_PATTERNS:
        if skip in name:
            return False
    return True


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "qwen-latest-scraper/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify(model_id: str) -> str:
    name = model_id.split("/", 1)[-1]
    n = name.lower()
    if "coder" in n:
        return "coder"
    if "vl" in n or "vision" in n:
        return "vision"
    if "math" in n:
        return "math"
    if "audio" in n or "speech" in n:
        return "audio"
    if "webworld" in n:
        return "agent"
    if "-a" in n and "b" in n.lower().split("-a", 1)[-1]:
        return "moe"
    return "general"


def main():
    data = fetch(API)
    rows = []
    for m in data:
        mid = m["id"]
        if not is_base_model(mid):
            continue
        last_mod = m.get("lastModified", "")[:10]  # YYYY-MM-DD
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        kind = classify(mid)
        rows.append({
            "id": mid, "last_modified": last_mod, "kind": kind,
            "downloads": downloads, "likes": likes,
        })

    rows.sort(key=lambda r: r["last_modified"], reverse=True)

    print(f"# Qwen latest models — fetched {datetime.utcnow().isoformat(timespec='seconds')}Z")
    print()
    print(f"Scanned {len(data)} models, kept {len(rows)} after filtering quantizations/SAEs.")
    print()

    for bucket in ("coder", "moe", "general", "vision", "math", "agent", "audio"):
        bucket_rows = [r for r in rows if r["kind"] == bucket][:8]
        if not bucket_rows:
            continue
        print(f"## {bucket.title()} (top 8 by date)")
        print()
        print("| Model ID | Last modified | Downloads | Likes |")
        print("|---|---|---:|---:|")
        for r in bucket_rows:
            print(f"| {r['id']} | {r['last_modified']} | {r['downloads']:,} | {r['likes']:,} |")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
