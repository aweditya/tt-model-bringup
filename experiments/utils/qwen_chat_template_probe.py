"""qwen_chat_template_probe.py — verify Qwen3.6 chat template tokenization.

No device needed. Loads the tokenizer for Qwen/Qwen3.6-27B and prints:
  1. eos_token / eos_token_id / im_end / endoftext token IDs
  2. raw tokenize of "Implement a JSON parser combinator in Rust"
  3. chat-templated tokenize of same prompt (no system) — apply_chat_template
  4. chat-templated tokenize WITH system prompt
  5. decode of each to show structure visible

Run via: ssh qb1 'cd ~/tt-xla && .venv/bin/python -m experiments.utils.qwen_chat_template_probe'
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"
PROMPT = "Implement a JSON parser combinator in Rust"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"=== {MODEL_ID} tokenizer ===")
    print(f"  eos_token = {tok.eos_token!r}  id={tok.eos_token_id}")
    print(f"  bos_token = {tok.bos_token!r}  id={tok.bos_token_id}")
    for tok_str in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        print(f"  {tok_str} -> {tid}")
    print()

    raw_ids = tok.encode(PROMPT)
    print(f"=== RAW tokenize: {PROMPT!r} ===")
    print(f"  ids ({len(raw_ids)}): {raw_ids}")
    print(f"  decoded: {tok.decode(raw_ids)!r}")
    print()

    chat_no_sys = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
    )
    print(f"=== CHAT (no system) tokenize ===")
    print(f"  ids ({len(chat_no_sys)}): {chat_no_sys}")
    print(f"  decoded: {tok.decode(chat_no_sys)!r}")
    print()

    chat_with_sys = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
    )
    print(f"=== CHAT (with system) tokenize ===")
    print(f"  ids ({len(chat_with_sys)}): {chat_with_sys}")
    print(f"  decoded: {tok.decode(chat_with_sys)!r}")


if __name__ == "__main__":
    main()
