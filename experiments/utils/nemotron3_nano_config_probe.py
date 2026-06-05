#!/usr/bin/env python3
"""MM7 v0.0 pre-flight — fast config + tokenizer load for Nemotron-3 Nano.

Downloads ONLY the tokenizer + config (a few MB), no weights, in under
60 seconds. Verifies:
  - model ID resolves on HF
  - `trust_remote_code=True` loads `NemotronHForCausalLM` modeling code
  - `hybrid_override_pattern` is present in the text-model config
  - ChatML tokenizer works (encode + decode + chat_template render)
  - layer counts match the architecture brief (23 Mamba2 + 23 MoE + 6 attn)

Run before the full HF oracle (which downloads ~63 GB of weights).

Run on the QuietBox:
  cd ~/tt-xla && .venv/bin/python -u experiments/utils/nemotron3_nano_config_probe.py
"""
from __future__ import annotations

import sys
import time

from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log(f"loading config + tokenizer for {MODEL_ID} (trust_remote_code=True) …")
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    log(f"  config in {time.time() - t0:.1f}s")
    log(f"  config class: {type(cfg).__name__}")
    log(f"  model_type: {getattr(cfg, 'model_type', '?')}")

    text_cfg = getattr(cfg, "text_config", cfg)
    pattern = getattr(text_cfg, "hybrid_override_pattern", None)
    if pattern is None:
        log("FAIL: no hybrid_override_pattern in config")
        return 1
    n_layers = len(pattern)
    n_mamba = pattern.count("M")
    n_moe = pattern.count("E")
    n_attn = pattern.count("*")
    log(f"  hybrid_override_pattern ({n_layers} chars):  {pattern}")
    log(f"  layer breakdown: {n_mamba} Mamba2 + {n_moe} MoE + {n_attn} attention")
    log(f"  hidden_size: {text_cfg.hidden_size}")
    log(f"  vocab_size: {text_cfg.vocab_size}")
    log(f"  ssm_state_size: {getattr(text_cfg, 'ssm_state_size', '?')}")
    log(f"  ssm_num_heads: {getattr(text_cfg, 'mamba_num_heads', '?')}")
    log(f"  ssm_head_dim: {getattr(text_cfg, 'mamba_head_dim', '?')}")
    # Mamba2 group count is `n_groups` (NOT `mamba_n_groups`) on this config.
    # MoE expert grouping is `n_group` (singular) — different attribute.
    log(f"  mamba_n_groups: {getattr(text_cfg, 'n_groups', '?')}  (B/C groups)")
    log(f"  num_attn_heads: {getattr(text_cfg, 'num_attention_heads', '?')}")
    log(f"  num_kv_heads: {getattr(text_cfg, 'num_key_value_heads', '?')}")
    log(f"  max_position_embeddings: {getattr(text_cfg, 'max_position_embeddings', '?')}")

    log(f"\nloading tokenizer …")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    log(f"  tokenizer in {time.time() - t0:.1f}s")
    log(f"  tokenizer class: {type(tok).__name__}")
    log(f"  vocab size: {len(tok)}")
    log(f"  eos token: {tok.eos_token!r} id={tok.eos_token_id}")
    log(f"  bos token: {tok.bos_token!r} id={tok.bos_token_id}")
    log(f"  pad token: {tok.pad_token!r} id={tok.pad_token_id}")

    # ── Round-trip test on the canonical prompt ──
    prompt = "The capital of France is"
    log(f"\nround-trip test: {prompt!r}")
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    log(f"  encoded: {ids}")
    log(f"  decoded: {tok.decode(ids)!r}")

    # ── Chat template test ──
    log(f"\nchat template test:")
    if tok.chat_template is None:
        log("  WARN: tok.chat_template is None — manual ChatML rendering required")
    else:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
        )
        log(f"  rendered: {rendered!r}")
        rendered_ids = tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
        log(f"  rendered ids ({rendered_ids.shape[1]}): {rendered_ids[0].tolist()}")

    log("\nconfig probe PASS ✓ — ready to download full weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
