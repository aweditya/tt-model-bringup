#!/usr/bin/env python3
"""Check whether Qwen3.6-27B chat template supports enable_thinking=False."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
msgs = [{"role": "user", "content": "What is 2+2?"}]
for flag in (None, True, False):
    print(f"=== enable_thinking={flag} ===")
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if flag is not None:
        kwargs["enable_thinking"] = flag
    try:
        rendered = tok.apply_chat_template(msgs, **kwargs)
        print(rendered[-200:])
    except Exception as e:
        print(f"ERROR: {e}")
    print()
