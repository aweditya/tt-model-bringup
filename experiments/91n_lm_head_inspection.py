#!/usr/bin/env python3
"""
Experiment 91n — Smoke test the lm_head / embedding / config.

After B'9 confirmed the fp32 residual didn't fix the 'FR' fixed-point
and 'Paris' never appears in top-5, the suspect is somewhere upstream
of the residual — most likely the lm_head or embed loading.

This script runs entirely on host (numpy, no device) and answers:

  Q1: What does the config say about tie_word_embeddings?
  Q2: Are lm_head.weight and embed_tokens.weight actually different
      tensors, or aliases of the same memory?
  Q3: lm_head shape, dtype, basic stats. Sanity-check vs typical
      well-trained Linear weight: mean ≈ 0, std ~ 0.01-0.1, range bounded.
  Q4: Does the prompt tokenize sensibly? Print decoded tokens.
  Q5: What does lm_head produce when fed an embedding vector directly?
      For tied-weight models, top-1 should usually be the input token
      itself (self-similarity). For untied, top-1 likely the same token
      or a closely related one — but NOT random junk.
  Q6: What does ' Paris' tokenize to?

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91n_lm_head_inspection.py
"""
import os, sys, json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"


def main():
    print("=" * 64)
    print("Experiment 91n — lm_head / embed / config smoke test")
    print("=" * 64)

    # ---- Q1: config ----
    print("\n[Q1] Loading config.json…")
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg['text_config']

    interesting = ['tie_word_embeddings', 'hidden_size', 'vocab_size',
                   'num_hidden_layers', 'head_dim', 'partial_rotary_factor',
                   'num_attention_heads', 'num_key_value_heads',
                   'linear_num_key_heads', 'linear_num_value_heads',
                   'linear_key_head_dim', 'linear_value_head_dim',
                   'linear_conv_kernel_dim']
    for k in interesting:
        if k in full_cfg:
            print(f"  {k} (root): {full_cfg[k]}")
        if k in text_cfg:
            print(f"  {k} (text): {text_cfg[k]}")
    if 'tie_word_embeddings' not in full_cfg and 'tie_word_embeddings' not in text_cfg:
        print(f"  tie_word_embeddings: NOT PRESENT (default: False per HF transformers)")

    # ---- Q2: safetensors index ----
    print("\n[Q2] safetensors index — where are lm_head and embed_tokens?")
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    lm_head_key = "lm_head.weight"
    embed_key = "model.language_model.embed_tokens.weight"
    print(f"  '{lm_head_key}' → {weight_map.get(lm_head_key, 'NOT FOUND')}")
    print(f"  '{embed_key}' → {weight_map.get(embed_key, 'NOT FOUND')}")
    print(f"  same shard?  {weight_map.get(lm_head_key) == weight_map.get(embed_key)}")

    # ---- Q3: lm_head and embed weight stats ----
    print("\n[Q3] Loading and inspecting weights…")
    lm_head_shard = weight_map[lm_head_key]
    embed_shard = weight_map[embed_key]

    lm_head_path = hf_hub_download(MODEL_ID, lm_head_shard)
    embed_path = hf_hub_download(MODEL_ID, embed_shard)

    with safe_open(lm_head_path, framework="pt") as f:
        lm_head = f.get_tensor(lm_head_key).float().numpy()
    with safe_open(embed_path, framework="pt") as f:
        embed = f.get_tensor(embed_key).float().numpy()

    def stats(name, t):
        print(f"  {name}: shape={t.shape} dtype={t.dtype}")
        print(f"    mean={t.mean():+.6f}  std={t.std():.6f}")
        print(f"    min={t.min():+.4f}  max={t.max():+.4f}")
        print(f"    abs_max={np.abs(t).max():.4f}")
        per_row_norm = np.linalg.norm(t, axis=1)
        print(f"    per-row ‖·‖: mean={per_row_norm.mean():.4f}  "
              f"std={per_row_norm.std():.4f}  "
              f"min={per_row_norm.min():.4f}  max={per_row_norm.max():.4f}")
        nz = (t == 0).mean()
        print(f"    %zero={nz*100:.4f}%")

    stats("lm_head", lm_head)
    stats("embed",   embed)

    # ---- Are lm_head and embed.T related? ----
    print("\n[Q3b] Relationship between lm_head and embed:")
    # HF convention: both stored as [vocab, hidden]
    print(f"  shapes match (both [vocab, hidden])?  {lm_head.shape == embed.shape}")
    if lm_head.shape == embed.shape:
        # Element-wise identity check (cheap)
        identical = np.array_equal(lm_head, embed)
        print(f"  np.array_equal(lm_head, embed)?  {identical}")
        # Correlation: cosine similarity per row, averaged
        sample_rows = np.random.choice(lm_head.shape[0], size=min(100, lm_head.shape[0]),
                                        replace=False)
        cos_per_row = []
        for r in sample_rows:
            a = lm_head[r]
            b = embed[r]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            cos_per_row.append(np.dot(a, b) / denom)
        cos_per_row = np.array(cos_per_row)
        print(f"  cosine per row (100 sampled): "
              f"mean={cos_per_row.mean():+.4f}  std={cos_per_row.std():.4f}  "
              f"min={cos_per_row.min():+.4f}  max={cos_per_row.max():+.4f}")
        # Max abs diff per row
        diffs = np.abs(lm_head[sample_rows] - embed[sample_rows]).max(axis=1)
        print(f"  max|Δ| per row (100 sampled): "
              f"mean={diffs.mean():.6f}  max={diffs.max():.6f}")

    # ---- Q4: tokenize "The capital of France is" ----
    print("\n[Q4] Tokenizing the prompt…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The capital of France is"
    ids = tok.encode(prompt)
    print(f"  prompt: {prompt!r}")
    print(f"  ids: {ids}")
    for i, tid in enumerate(ids):
        print(f"    [{i}] {tid:6d}  =  {tok.decode([tid])!r}")
    print(f"  decoded full: {tok.decode(ids)!r}")

    # ---- Q5: sanity decode with various inputs ----
    print("\n[Q5] Sanity decode through lm_head only…")
    # Feed embed[X] for a few token IDs. For tied weights this should produce
    # top-1 = X (self-similarity). For untied, top-1 should be at least sensible.
    test_tokens = ids + [10191]  # prompt tokens + 'FR'
    # Add 'Paris' if it tokenizes to a single token
    paris_ids = tok.encode(" Paris", add_special_tokens=False)
    print(f"  ' Paris' tokenizes to: {paris_ids}  decoded: {[tok.decode([t]) for t in paris_ids]}")
    if len(paris_ids) == 1:
        test_tokens.append(paris_ids[0])

    print(f"\n  For each test token, compute embed[X] @ lm_head.T and check top-5:")
    print(f"  (HF convention: lm_head is [vocab, hidden], so we use lm_head.T as [hidden, vocab])")
    for tid in test_tokens:
        e = embed[tid]                                    # [hidden]
        # logits via untied projection: e @ lm_head.T = e @ [hidden, vocab] = [vocab]
        logits = e @ lm_head.T                             # using HF's stored layout
        top5_idx = np.argsort(logits)[::-1][:5]
        print(f"\n  embed[{tid}] = {tok.decode([tid])!r}  (‖e‖={np.linalg.norm(e):.3f})")
        print(f"  → e @ lm_head.T top-5:")
        for r, i in enumerate(top5_idx):
            mark = " ← MATCHES INPUT" if i == tid else ""
            print(f"      rank {r+1}: token {i:6d}  {tok.decode([int(i)])!r:>16s}  "
                  f"logit={logits[i]:.3f}{mark}")

    # ---- Q6: what's the lm_head ROW for our junk tokens? ----
    print("\n[Q6] lm_head row stats for our junk tokens:")
    for tid, name in [(10191, "FR"), (88871, "jadi"), (209092, "itata"),
                       (76901, "_texts"), (61151, "illac")]:
        row = lm_head[tid]
        print(f"  token {tid:6d} ({name:>10s}): "
              f"‖row‖={np.linalg.norm(row):.3f}  "
              f"max|·|={np.abs(row).max():.3f}  "
              f"mean={row.mean():+.4f}  std={row.std():.4f}")
    # For comparison: 'Paris'
    if len(paris_ids) == 1:
        ptid = paris_ids[0]
        row = lm_head[ptid]
        print(f"  token {ptid:6d} ({'Paris':>10s}): "
              f"‖row‖={np.linalg.norm(row):.3f}  "
              f"max|·|={np.abs(row).max():.3f}  "
              f"mean={row.mean():+.4f}  std={row.std():.4f}")
    # Average row norm for reference
    all_norms = np.linalg.norm(lm_head, axis=1)
    print(f"\n  all rows: mean‖row‖={all_norms.mean():.3f}  "
          f"median={np.median(all_norms):.3f}  "
          f"max={all_norms.max():.3f}  min={all_norms.min():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
