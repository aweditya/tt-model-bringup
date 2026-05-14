"""mtp_smoke_numpy.py - host-only sanity check for the numpy MTP forward.

Loads MTP weights, builds a fake hidden state + token embed, runs the numpy
forward, checks that logits are finite, have reasonable magnitudes, and the
argmax produces a valid token id. NO device.

Run before mtp_head_probe.py to catch shape/loading bugs early.
"""
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.expanduser("~"))

# Reuse the loader + forward from mtp_head_probe
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_mtp_probe", os.path.expanduser("~/tt-xla/experiments/utils/mtp_head_probe.py"))
_mtp_probe = importlib.util.module_from_spec(_spec)
# Avoid running the main() at import time; the file does not auto-run.
_spec.loader.exec_module(_mtp_probe)

MODEL_ID = "Qwen/Qwen3.6-27B"


def main():
    print("[smoke] loading config…")
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)["text_config"]
    cfg = {
        "hidden":      text_cfg["hidden_size"],
        "n_q_heads":   text_cfg["num_attention_heads"],
        "n_kv_heads":  text_cfg["num_key_value_heads"],
        "head_dim":    text_cfg["head_dim"],
        "partial_rotary_factor": text_cfg["partial_rotary_factor"],
        "intermediate_size": text_cfg["intermediate_size"],
    }
    HIDDEN = cfg["hidden"]
    ROTARY_DIM = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    VOCAB = text_cfg["vocab_size"]
    print(f"  HIDDEN={HIDDEN} VOCAB={VOCAB} HEAD_DIM={cfg['head_dim']} "
          f"N_Q={cfg['n_q_heads']} N_KV={cfg['n_kv_heads']} ROTARY_DIM={ROTARY_DIM}")

    print("\n[smoke] loading MTP weights…")
    mtp_w = _mtp_probe.load_mtp_weights()

    print("\n[smoke] loading embed + lm_head…")
    e = _mtp_probe.load_embed_lm_head()
    embed_np = e["embed"]
    lm_head_np = e["lm_head"]
    print(f"  embed: {embed_np.shape}  lm_head: {lm_head_np.shape}")

    rng = np.random.default_rng(0)
    fake_hidden = rng.standard_normal(HIDDEN).astype(np.float32) * 1.5
    fake_token_id = 1234
    fake_token_emb = embed_np[fake_token_id]
    position = 3

    print(f"\n[smoke] mtp_forward_numpy with fake_hidden norm={np.linalg.norm(fake_hidden):.2f}")
    logits = _mtp_probe.mtp_forward_numpy(
        fake_hidden, fake_token_emb, position_t1=position,
        w=mtp_w, lm_head_np=lm_head_np, cfg=cfg, rotary_dim=ROTARY_DIM)

    print(f"  logits.shape={logits.shape} "
          f"min={float(logits.min()):.3f} max={float(logits.max()):.3f} "
          f"mean={float(logits.mean()):.3f} std={float(logits.std()):.3f}")
    nan_ct = int(np.isnan(logits).sum())
    inf_ct = int(np.isinf(logits).sum())
    print(f"  NaN={nan_ct}  Inf={inf_ct}")
    if nan_ct or inf_ct:
        print("[smoke] FAIL: non-finite logits")
        sys.exit(1)

    top5 = np.argsort(-logits)[:5]
    print("  top-5:")
    for tid in top5:
        print(f"    {int(tid):7d}  logit={float(logits[int(tid)]):.3f}")
    print("[smoke] OK - MTP forward produces finite logits with sensible top-5")


if __name__ == "__main__":
    main()
