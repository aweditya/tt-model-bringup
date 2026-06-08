#!/usr/bin/env python3
"""HF oracle helper v2 — corrected drafter inputs construction (R-2).

R-1 finding (research/gemma4_drafter_parallel_design.md): the v1 oracle
(hf_oracle_gemma4_assistant.py) used the WRONG inputs_embeds construction:
  WRONG: drafter_inputs_embeds = torch.cat([target_h_prev, target_h_last], dim=-1)

HF's official `AssistedCandidateGeneratorGemma4` (transformers/generation/
candidate_generator.py:1376-1404) uses:
  for _ in range(K):
      last_token_embedding = target_model_input_embeddings(last_token_id)
      inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)
      outputs = assistant_model(inputs_embeds=inputs_embeds, shared_kv_states=...)
      last_token_id = outputs.logits.argmax(dim=-1)
      last_hidden_state = outputs.last_hidden_state

This v2 helper implements the correct loop and saves the full K-step
trajectory: argmax, hidden, logits, inputs_embeds per round, per prompt.

ALSO runs HF's actual `target.generate(input_ids, assistant_model=drafter, ...)`
on the same prompts to compare our K-step oracle against HF's actual
spec-dec output.

Outputs to .cache/hf_oracle_gemma4_12b_assistant_v2/:
  prompt_{i}/
    input_ids.npy
    target_embed_table.npy            (vocab × hidden, replicated; small lookup)
    target_h_last.npy                  (target's hidden at last prompt position)
    shared_kv_sliding_K.npy, shared_kv_sliding_V.npy
    shared_kv_full_K.npy, shared_kv_full_V.npy
    K{k}/round_{r}/                    (k = lookahead depth, r = 0..k-1)
      inputs_embeds.npy
      drafter_logits.npy
      drafter_argmax.npy
      drafter_hidden.npy
    hf_assisted_output_tokens.npy      (real HF spec-dec output, K-agnostic)

CPU only, ~5-10 min per prompt for K=5. Run remote:
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \\
      experiments/utils/hf_oracle_gemma4_assistant_v2.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
TARGET_ID = "google/gemma-4-12B-it"  # IT — paired with the drafter
DRAFTER_ID = "google/gemma-4-12b-it-assistant"
K_VALUES = [3, 5, 7]
PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "Python is a programming language that",
    "The largest planet in our solar system is",
    "Quantum entanglement occurs when",
]
HF_ASSIST_MAX_NEW_TOKENS = 16  # HF spec-dec rollout length


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    from transformers import (
        AutoTokenizer,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedAssistantForCausalLM,
    )

    log(f"loading target {TARGET_ID} (bf16, CPU, ~24 GB)…")
    t0 = time.time()
    target = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        TARGET_ID, torch_dtype=torch.bfloat16,
    )
    target.eval()
    log(f"  target loaded in {time.time()-t0:.0f}s")

    log(f"loading drafter {DRAFTER_ID} (bf16, CPU, ~760 MB)…")
    t0 = time.time()
    drafter = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        DRAFTER_ID, torch_dtype=torch.bfloat16,
    )
    drafter.eval()
    log(f"  drafter loaded in {time.time()-t0:.0f}s")

    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    log(f"tokenizer: {tok.__class__.__name__}")

    # Target's input embeddings (used as prev half of inputs_embeds per HF).
    target_embed_table = target.get_input_embeddings()
    log(f"target_embed_table type: {type(target_embed_table).__name__}, "
        f"weight shape: {tuple(target_embed_table.weight.shape)}")

    # Layer type indices for shared_kv_states extraction.
    target_layer_types = target.config.text_config.layer_types
    last_full_idx = max(i for i, t in enumerate(target_layer_types) if t == "full_attention")
    last_sliding_idx = max(i for i, t in enumerate(target_layer_types) if t == "sliding_attention")
    log(f"target layer types: last_full={last_full_idx}, last_sliding={last_sliding_idx}")

    for prompt_i, prompt in enumerate(PROMPTS):
        log("")
        log(f"━━━━━━ prompt {prompt_i}: {prompt!r} ━━━━━━")
        t_p = time.time()

        input_ids = tok(prompt, return_tensors="pt").input_ids
        L = int(input_ids.shape[1])
        log(f"  input_ids shape {tuple(input_ids.shape)} ({L} tokens)")

        prompt_dir = OUT_DIR / f"prompt_{prompt_i}"
        prompt_dir.mkdir(exist_ok=True)
        np.save(prompt_dir / "input_ids.npy", input_ids.cpu().numpy())

        # ── 1. Target forward to get shared_kv_states + last hidden ──
        log("  running target forward…")
        t_t = time.time()
        with torch.no_grad():
            target_out = target.model.language_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        log(f"    target forward took {time.time()-t_t:.1f}s")
        target_h = target_out.hidden_states[-1]  # [B, L, 3840]
        log(f"    target last-layer hidden: {tuple(target_h.shape)}")

        # Extract last sliding + last full KV layers.
        pkv = target_out.past_key_values
        if hasattr(pkv, "to_legacy_cache"):
            legacy = pkv.to_legacy_cache()
            kv_full_K, kv_full_V = legacy[last_full_idx]
            kv_sliding_K, kv_sliding_V = legacy[last_sliding_idx]
        else:
            kv_full_K = pkv.layers[last_full_idx].keys
            kv_full_V = pkv.layers[last_full_idx].values
            kv_sliding_K = pkv.layers[last_sliding_idx].keys
            kv_sliding_V = pkv.layers[last_sliding_idx].values
        log(f"    KV shapes: sliding K={tuple(kv_sliding_K.shape)} "
            f"V={tuple(kv_sliding_V.shape)}; "
            f"full K={tuple(kv_full_K.shape)} V={tuple(kv_full_V.shape)}")

        # Save target artifacts.
        np.save(prompt_dir / "target_h_last.npy",
                target_h[:, -1:, :].detach().float().cpu().numpy())
        np.save(prompt_dir / "shared_kv_sliding_K.npy",
                kv_sliding_K.detach().float().cpu().numpy())
        np.save(prompt_dir / "shared_kv_sliding_V.npy",
                kv_sliding_V.detach().float().cpu().numpy())
        np.save(prompt_dir / "shared_kv_full_K.npy", kv_full_K.detach().float().cpu().numpy())
        np.save(prompt_dir / "shared_kv_full_V.npy", kv_full_V.detach().float().cpu().numpy())

        shared_kv_states = {
            "sliding_attention": (kv_sliding_K, kv_sliding_V),
            "full_attention": (kv_full_K, kv_full_V),
        }

        # ── 2. K-step drafter trajectory per K_VALUES ──
        # For each K, run the autoregressive loop. The K=3 trajectory's first
        # 3 rounds should be IDENTICAL to K=5's first 3 rounds (no rolling
        # state difference) — verify after by comparing argmaxes.

        for k_val in K_VALUES:
            log(f"  K={k_val} drafter trajectory:")
            k_dir = prompt_dir / f"K{k_val}"
            k_dir.mkdir(exist_ok=True)

            # Initialize per HF: last_token_id = input_ids[:, -1:],
            # last_hidden_state = target's hidden at LAST position.
            last_token_id = input_ids[:, -1:]  # [B, 1]
            last_hidden_state = target_h[:, -1:, :]  # [B, 1, hidden]

            for round_r in range(k_val):
                # inputs_embeds = concat(target_embed(last_token_id), last_hidden)
                last_token_emb = target_embed_table(last_token_id)  # [B, 1, hidden]
                inputs_embeds = torch.cat(
                    [last_token_emb, last_hidden_state], dim=-1
                ).to(torch.bfloat16)  # [B, 1, 2*hidden]

                with torch.no_grad():
                    drafter_out = drafter(
                        inputs_embeds=inputs_embeds,
                        shared_kv_states=shared_kv_states,
                        use_cache=False,
                        return_dict=True,
                    )
                drafter_logits = drafter_out.logits  # [B, 1, vocab]
                drafter_hidden = drafter_out.last_hidden_state  # [B, 1, hidden]
                argmax = drafter_logits.argmax(dim=-1)  # [B, 1]

                # Save this round's artifacts.
                round_dir = k_dir / f"round_{round_r}"
                round_dir.mkdir(exist_ok=True)
                np.save(round_dir / "inputs_embeds.npy",
                        inputs_embeds.detach().float().cpu().numpy())
                np.save(round_dir / "drafter_logits.npy",
                        drafter_logits.detach().float().cpu().numpy())
                np.save(round_dir / "drafter_argmax.npy",
                        argmax.cpu().numpy())
                np.save(round_dir / "drafter_hidden.npy",
                        drafter_hidden.detach().float().cpu().numpy())

                tok_int = int(argmax.flatten()[0])
                log(f"    round {round_r}: argmax={tok_int} "
                    f"({tok.decode([tok_int])!r})")

                # Update for next round.
                last_token_id = argmax  # drafter's prediction becomes next "last_token"
                last_hidden_state = drafter_hidden  # drafter's hidden becomes next "last_hidden"

        # ── 3. HF actual generate() with assistant_model — ground truth ──
        log(f"  running HF target.generate(assistant_model=drafter, max_new_tokens={HF_ASSIST_MAX_NEW_TOKENS})…")
        t_g = time.time()
        try:
            with torch.no_grad():
                # The HF docs example. This is the canonical spec-dec invocation.
                gen_out = target.generate(
                    input_ids,
                    assistant_model=drafter,
                    max_new_tokens=HF_ASSIST_MAX_NEW_TOKENS,
                    do_sample=False,
                )
            gen_tokens = gen_out[0, L:].tolist()  # just the new tokens
            np.save(prompt_dir / "hf_assisted_output_tokens.npy",
                    np.array(gen_tokens, dtype=np.int64))
            log(f"    HF assisted output ({len(gen_tokens)} tok in {time.time()-t_g:.1f}s): "
                f"{gen_tokens}")
            log(f"    text: {tok.decode(gen_tokens)!r}")
        except Exception as e:
            log(f"    ✗ HF generate failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

        log(f"  prompt {prompt_i} total {time.time()-t_p:.1f}s")

    log("")
    log(f"DONE — artifacts at {OUT_DIR}")


if __name__ == "__main__":
    main()
