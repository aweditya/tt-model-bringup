"""mtp_smoke_hf_hidden.py - sanity check MTP head against cached HF hidden states.

The cached file ~/.cache/hf_per_layer_hidden_states.npz contains 65 hidden_states
arrays from a CPU HF forward of the prompt "The capital of France is".

Per HuggingFace convention for transformers' output_hidden_states=True:
  hidden_states is a tuple of length (num_layers + 1).
  hidden_states[0] = output of the embedding layer
  hidden_states[i] for i in 1..N = output of decoder_layer_{i-1} (residual stream)
  Some models append the post-final-norm state; some don't.

For Qwen3.6-27B (Qwen3NextModel) the last layer output is hidden_states[64] in
the cache (we have indices 0..64 = 65 entries). Looking at the std spread:
  hidden_0 (embed)   std=0.014
  ...
  hidden_63          std=5.03
  hidden_64          std=1.98  <- low std => this is POST-final-RMSNorm
The drop from 5.03 to 1.98 is the final RMSNorm dividing by ~rms(hidden_63).
So:
  hidden_t_for_MTP = hidden_states[63][position]  (pre-final-norm, post-layer-64)

We feed (hidden_states[63][t], embed(token_{t+1})) and report MTP's top-5
predictions for each token_{t+2}. Tokens are "The capital of France is" =>
ids [tok.encode("The capital of France is")] -> something like [The, capital, of, France, is].
For t=0..2 we test predictions for tokens at positions 2,3,4 (which are
'of', 'France', 'is'). These are dictionary tokens with strong prior context;
if MTP is loaded correctly its top-K should rank the correct token highly.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/mtp_smoke_hf_hidden.py
"""
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.expanduser("~"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_mtp_probe", os.path.expanduser("~/tt-xla/experiments/utils/mtp_head_probe.py"))
_mtp_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtp_probe)

MODEL_ID = "Qwen/Qwen3.6-27B"
CACHE = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")


def main():
    print(f"[smoke-hf] loading cached HF hidden states from {CACHE}")
    data = np.load(CACHE)
    print(f"  keys count = {len(data.files)}")
    prompt_ids = data["prompt_ids"].tolist()
    print(f"  prompt_ids = {prompt_ids}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  decoded = {[tok.decode([t]) for t in prompt_ids]}")

    # The last pre-norm layer output is hidden_64 - 1 = hidden_63 (heuristic above)
    # But let's also probe hidden_64 to confirm.
    h63 = data["hidden_63"][0]  # [seq, hidden]
    h64 = data["hidden_64"][0]
    print(f"  hidden_63 shape={h63.shape}  std={h63.std():.3f}")
    print(f"  hidden_64 shape={h64.shape}  std={h64.std():.3f}")

    # Load weights
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
    ROTARY_DIM = int(cfg["head_dim"] * cfg["partial_rotary_factor"])

    mtp_w = _mtp_probe.load_mtp_weights()
    e = _mtp_probe.load_embed_lm_head()
    embed_np = e["embed"]
    lm_head_np = e["lm_head"]

    # Test BOTH hypotheses: feed hidden_63 (pre-norm) vs hidden_64 (post-norm)
    # The DeepSeek reference and the existence of pre_fc_norm_hidden in Qwen3.6 MTP
    # strongly suggest hidden_63 (pre-final-norm) is the input. We'll verify by
    # which one gives stronger top-1/top-5 rankings for the actual next-next tokens.
    P = len(prompt_ids)
    print(f"\nFor each t in 0..{P-3}, predict token at position t+2:")
    print(f"  feeding (hidden_states[?][t], embed(prompt_ids[t+1]))  =>  argmax should be prompt_ids[t+2]")
    print(f"  prompt = {' '.join(tok.decode([t]) for t in prompt_ids)!r}")

    for hidden_key, h_arr in [("hidden_63 (pre-norm hypothesis)", h63),
                                ("hidden_64 (post-norm hypothesis)", h64)]:
        print(f"\n  === Using {hidden_key} ===")
        for t in range(P - 2):
            tok_t1 = prompt_ids[t + 1]
            tok_t2_actual = prompt_ids[t + 2]
            tok_t1_emb = embed_np[tok_t1]
            logits = _mtp_probe.mtp_forward_numpy(
                h_arr[t], tok_t1_emb, position_t1=t + 1,
                w=mtp_w, lm_head_np=lm_head_np, cfg=cfg, rotary_dim=ROTARY_DIM)
            # Compute rank of the actual t+2 token
            target_logit = logits[tok_t2_actual]
            rank = int((logits > target_logit).sum() + 1)
            top5 = np.argsort(-logits)[:5]
            top5_str = " | ".join(f"{tok.decode([int(i)])!r}" for i in top5)
            print(f"    t={t}  tok_t1={tok.decode([tok_t1])!r:>15s}  "
                  f"actual_t2={tok.decode([tok_t2_actual])!r:>15s}  "
                  f"rank={rank}/{logits.size}  "
                  f"top5=[{top5_str}]")


if __name__ == "__main__":
    main()
