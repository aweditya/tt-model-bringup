#!/usr/bin/env python3
"""Phase 1 v0.0 — HF oracle generator for google/gemma-4-12b-it-assistant.

Captures the artifacts our ttnn drafter bringup ladder needs to validate
against. Pattern forks experiments/utils/hf_reference_35b.py.

The drafter takes (inputs_embeds, shared_kv_states) — NOT raw token IDs.
inputs_embeds is the target Gemma 4 12B IT's projected hidden state at
2 positions, concatenated. shared_kv_states are the target's last
sliding + last full attention layer KV. So generating the oracle
requires running BOTH target + drafter (on CPU here; ~5 min total).

Output: .cache/hf_oracle_gemma4_12b_assistant/
  meta.json         — prompt_ids, shapes, drafter config
  target_h2last.npy — [B=1, 2, 3840] target last hidden at 2 positions
  shared_kv_*.npy   — sliding + full KV tensors
  drafter_logits.npy   — [B, 1, 262144] drafter output logits
  drafter_hidden.npy   — [B, 1, 3840] drafter post_projection output
  drafter_argmax.npy   — [B, 1] top-1 token id
  drafter_topk.npy     — [B, 1, 8] top-8 tokens (for partial-match tolerance)

Per [[needle-prompt-shape-not-precision]] we use BASE-style prompts
(no chat-template wrapping) for the oracle so we exercise the actual
drafter forward, not IT conversational habits.

Run via:
  python experiments/utils/hf_oracle_gemma4_assistant.py
or via the launcher script `scripts/run_gemma4_assistant_oracle.sh`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"

TARGET_ID = "google/gemma-4-12B-it"
DRAFTER_ID = "google/gemma-4-12b-it-assistant"

# 5 simple BASE prompts — short to keep CPU inference tractable.
PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "Python is a high-level programming language that",
    "The Pythagorean theorem states that",
    "Quantum entanglement is a phenomenon where",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {OUT_DIR}")

    log("importing transformers…")
    from transformers import (
        AutoTokenizer,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedAssistantForCausalLM,
    )

    log("loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(TARGET_ID)

    log(f"loading target {TARGET_ID} (bf16, CPU, ~24 GB)…")
    t0 = time.time()
    target = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        TARGET_ID,
        torch_dtype=torch.bfloat16,
    )
    target.eval()
    log(f"  target loaded in {time.time() - t0:.0f}s")

    log(f"loading drafter {DRAFTER_ID} (bf16, CPU, ~760 MB)…")
    t0 = time.time()
    drafter = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        DRAFTER_ID,
        torch_dtype=torch.bfloat16,
    )
    drafter.eval()
    log(f"  drafter loaded in {time.time() - t0:.0f}s")

    # Drafter introspection
    log(f"drafter config:")
    log(f"  backbone_hidden_size = {drafter.config.backbone_hidden_size}")
    log(f"  use_ordered_embeddings = {drafter.config.use_ordered_embeddings}")
    tc = drafter.config.get_text_config()
    log(f"  hidden_size = {tc.hidden_size}")
    log(f"  num_hidden_layers = {tc.num_hidden_layers}")
    log(f"  num_attention_heads = {tc.num_attention_heads}")
    log(f"  num_key_value_heads = {tc.num_key_value_heads}")
    log(f"  layer_types = {tc.layer_types}")

    per_prompt = []
    for i, prompt in enumerate(PROMPTS):
        log("")
        log(f"=== prompt {i}: {prompt!r} ===")
        t_p = time.time()

        input_ids = tok(prompt, return_tensors="pt").input_ids
        log(f"  input_ids shape {tuple(input_ids.shape)} ({input_ids.shape[1]} tokens)")

        # --- TARGET FORWARD ---
        # We need: last hidden state, last sliding + last full attn KV.
        log(f"  running target forward (this is slow on CPU)…")
        with torch.no_grad():
            target_out = target.model.language_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        log(f"    target hidden_states tuple len: {len(target_out.hidden_states)}")
        target_h = target_out.hidden_states[-1]  # [B, L, 3840]
        log(f"    target last hidden shape: {tuple(target_h.shape)}")

        # past_key_values: list of (K, V) per layer
        pkv = target_out.past_key_values
        log(f"    target past_key_values type: {type(pkv).__name__}")
        # Extract the last sliding + last full KV layers.
        # Gemma 4 12B has 48 layers; layer_types known via config.
        target_layer_types = target.config.text_config.layer_types
        last_full_idx = max(
            i for i, t in enumerate(target_layer_types) if t == "full_attention"
        )
        last_sliding_idx = max(
            i for i, t in enumerate(target_layer_types)
            if t == "sliding_attention"
        )
        log(f"    last full attn layer idx: {last_full_idx}")
        log(f"    last sliding attn layer idx: {last_sliding_idx}")

        # past_key_values is a DynamicCache in transformers 5.10+. Try the
        # legacy tuple API first, fall back to .layers[i] container API.
        if hasattr(pkv, "to_legacy_cache"):
            legacy = pkv.to_legacy_cache()
            kv_full_K, kv_full_V = legacy[last_full_idx]
            kv_sliding_K, kv_sliding_V = legacy[last_sliding_idx]
        else:
            kv_full_K = pkv.layers[last_full_idx].keys
            kv_full_V = pkv.layers[last_full_idx].values
            kv_sliding_K = pkv.layers[last_sliding_idx].keys
            kv_sliding_V = pkv.layers[last_sliding_idx].values
        log(f"    full KV shapes: K={tuple(kv_full_K.shape)} V={tuple(kv_full_V.shape)}")
        log(f"    sliding KV shapes: K={tuple(kv_sliding_K.shape)} V={tuple(kv_sliding_V.shape)}")

        # --- DRAFTER FORWARD ---
        # drafter inputs_embeds = concat(target_h[:, -1], target_h[:, -2])
        # When prompt has only 1 token, duplicate the same hidden state.
        if target_h.shape[1] >= 2:
            h_last = target_h[:, -1:, :]
            h_prev = target_h[:, -2:-1, :]
        else:
            h_last = target_h[:, -1:, :]
            h_prev = h_last.clone()
        drafter_inputs_embeds = torch.cat([h_prev, h_last], dim=-1)  # [B, 1, 2*3840=7680]
        log(f"    drafter inputs_embeds shape: {tuple(drafter_inputs_embeds.shape)}")

        shared_kv_states = {
            "sliding_attention": (kv_sliding_K, kv_sliding_V),
            "full_attention": (kv_full_K, kv_full_V),
        }

        log(f"  running drafter forward…")
        with torch.no_grad():
            drafter_out = drafter(
                inputs_embeds=drafter_inputs_embeds,
                shared_kv_states=shared_kv_states,
                use_cache=False,
                return_dict=True,
            )
        drafter_logits = drafter_out.logits         # [B, 1, 262144]
        drafter_hidden = drafter_out.last_hidden_state  # [B, 1, 3840]
        log(f"    drafter logits shape: {tuple(drafter_logits.shape)}")
        log(f"    drafter last_hidden shape: {tuple(drafter_hidden.shape)}")

        # Top-K analysis
        topk_vals, topk_ids = torch.topk(
            drafter_logits.float(), k=8, dim=-1,
        )
        argmax = drafter_logits.argmax(dim=-1)
        log(f"    drafter argmax: {argmax.tolist()} → "
            f"{tok.decode(argmax.flatten().tolist())!r}")
        log(f"    top-8 ids: {topk_ids.flatten().tolist()}")
        log(f"    top-8 text: {[tok.decode([t]) for t in topk_ids.flatten().tolist()]}")

        # --- SAVE ---
        prompt_dir = OUT_DIR / f"prompt_{i}"
        prompt_dir.mkdir(exist_ok=True)
        np.save(prompt_dir / "input_ids.npy", input_ids.cpu().numpy())
        np.save(
            prompt_dir / "target_h_last.npy",
            h_last.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "target_h_prev.npy",
            h_prev.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_inputs_embeds.npy",
            drafter_inputs_embeds.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "shared_kv_full_K.npy",
            kv_full_K.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "shared_kv_full_V.npy",
            kv_full_V.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "shared_kv_sliding_K.npy",
            kv_sliding_K.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "shared_kv_sliding_V.npy",
            kv_sliding_V.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_logits.npy",
            drafter_logits.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_hidden.npy",
            drafter_hidden.float().cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_argmax.npy",
            argmax.cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_topk_ids.npy",
            topk_ids.cpu().numpy(),
        )
        np.save(
            prompt_dir / "drafter_topk_vals.npy",
            topk_vals.cpu().numpy(),
        )
        log(f"  saved → {prompt_dir}")

        per_prompt.append({
            "i": i,
            "prompt": prompt,
            "input_ids": input_ids.flatten().tolist(),
            "argmax": int(argmax.flatten().item()),
            "argmax_text": tok.decode(argmax.flatten().tolist()),
            "top8": topk_ids.flatten().tolist(),
            "top8_text": [tok.decode([t]) for t in topk_ids.flatten().tolist()],
            "time_s": time.time() - t_p,
        })

    # Save meta
    meta = {
        "target_id": TARGET_ID,
        "drafter_id": DRAFTER_ID,
        "drafter_config": {
            "backbone_hidden_size": int(drafter.config.backbone_hidden_size),
            "use_ordered_embeddings": bool(drafter.config.use_ordered_embeddings),
            "hidden_size": int(drafter.config.get_text_config().hidden_size),
            "num_hidden_layers": int(drafter.config.get_text_config().num_hidden_layers),
            "num_attention_heads": int(drafter.config.get_text_config().num_attention_heads),
            "num_key_value_heads": int(drafter.config.get_text_config().num_key_value_heads),
            "layer_types": list(drafter.config.get_text_config().layer_types),
            "sliding_window": int(drafter.config.get_text_config().sliding_window),
            "intermediate_size": int(drafter.config.get_text_config().intermediate_size),
            "vocab_size": int(drafter.config.get_text_config().vocab_size),
        },
        "target_layer_types": list(target.config.text_config.layer_types),
        "target_last_full_idx": int(last_full_idx),
        "target_last_sliding_idx": int(last_sliding_idx),
        "prompts": per_prompt,
        "saved_at": int(time.time()),
        "transformers_version": __import__("transformers").__version__,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log("")
    log(f"DONE — oracle saved to {OUT_DIR}")
    log(f"meta.json + {len(PROMPTS)} prompt subdirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
