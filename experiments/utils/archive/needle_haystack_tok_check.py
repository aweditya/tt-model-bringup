#!/usr/bin/env python3
"""Check whether tok.encode() on a chat-rendered string preserves the
special tokens (<|im_start|>, <|im_end|>, etc.) or splits them into literal
text. This determines whether sending chat=False with a pre-rendered
prompt works at all.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

# Get id of <|im_start|>
im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_start|> id = {im_start_id}")
print(f"<|im_end|>   id = {im_end_id}")

# Render chat
msgs = [{"role": "user", "content": "hi"}]
rendered = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
print(f"\nrendered ({len(rendered)} chars):")
print(repr(rendered))

# Encode with default tok.encode (as the server does for chat=False)
ids_default = tok.encode(rendered)
print(f"\ntok.encode(rendered) → {len(ids_default)} tokens")
print(f"first 10 ids: {ids_default[:10]}")
print(f"first 10 decoded: {[tok.decode([i]) for i in ids_default[:10]]}")
print(f"\ncontains <|im_start|> id ({im_start_id})? {im_start_id in ids_default}")
print(f"contains <|im_end|> id   ({im_end_id})? {im_end_id in ids_default}")

# Now tokenize via apply_chat_template (correct path)
ids_correct = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True, enable_thinking=False)
if hasattr(ids_correct, "keys") and "input_ids" in ids_correct:
    ids_correct = ids_correct["input_ids"]
if isinstance(ids_correct, list) and ids_correct and isinstance(ids_correct[0], list):
    ids_correct = ids_correct[0]
print(f"\napply_chat_template(tokenize=True) → {len(ids_correct)} tokens")
print(f"first 10 ids: {list(ids_correct)[:10]}")
