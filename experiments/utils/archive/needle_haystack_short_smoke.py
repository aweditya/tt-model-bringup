#!/usr/bin/env python3
"""Sanity smoke: send a SHORT no-think chat prompt to the server via
chat=False (raw text) and confirm the model produces a coherent answer
to a simple question.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from experiments.utils.needle_haystack_probe import stream_generate_long
from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

# === Test: pre-rendered no-think prompt, chat=False ===
msgs = [{"role": "user", "content": "What is the capital of France?"}]
rendered_no_think = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False,
                                              enable_thinking=False)
print("=" * 60)
print("Test: chat=False with pre-rendered no-think prompt, plain greedy")
print(f"rendered: {rendered_no_think!r}")
final = stream_generate_long(rendered_no_think, max_pos=256, max_tokens=30,
                               dry_multiplier=0.0, repetition_penalty=1.0,
                               chat=False)
print(f"  generated: {final.get('generated_text','')!r}")
print(f"  n_prompt_tokens: {final.get('n_prompt_tokens')}")
print(f"  prefill_ms: {final.get('prefill_ms')}")
print()

# Repeat with DRY+rp
print("=" * 60)
print("Test: same prompt, with DRY=0.8 rp=1.1")
final = stream_generate_long(rendered_no_think, max_pos=256, max_tokens=30,
                               dry_multiplier=0.8, repetition_penalty=1.1,
                               chat=False)
print(f"  generated: {final.get('generated_text','')!r}")
