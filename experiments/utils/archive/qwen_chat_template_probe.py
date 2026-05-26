"""qwen_chat_template_probe.py — verify Qwen3.6 chat template tokenization.

Validates the _encode_prompt path for the persistent server: handles
BatchEncoding (Qwen3.6) and flat-list returns, coerces to list[int].
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"
PROMPT = "Implement a JSON parser combinator in Rust"


def encode_via_server_logic(tok, prompt: str, chat: bool, system: str = ""):
    """Mirror of _encode_prompt in experiments/serve/server.py — exact copy
    of the chat branch + coercion. Run this to verify before restarting the
    real server."""
    if chat:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(out, "keys") and "input_ids" in out:
            ids_raw = out["input_ids"]
        else:
            ids_raw = out
        try:
            import torch as _torch
            if isinstance(ids_raw, _torch.Tensor):
                ids_raw = ids_raw.tolist()
        except ImportError:
            pass
        try:
            import numpy as _np
            if isinstance(ids_raw, _np.ndarray):
                ids_raw = ids_raw.tolist()
        except ImportError:
            pass
        if (isinstance(ids_raw, list) and len(ids_raw) > 0
                and isinstance(ids_raw[0], (list, tuple))):
            ids_raw = ids_raw[0]
        return [int(t) for t in ids_raw]
    else:
        return [int(t) for t in tok.encode(prompt)]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = encode_via_server_logic(tok, PROMPT, chat=False)
    chat = encode_via_server_logic(tok, PROMPT, chat=True)
    chat_sys = encode_via_server_logic(tok, PROMPT, chat=True,
                                         system="You are a helpful assistant.")
    print(f"raw ({len(raw)}):       {raw}")
    print(f"  all ints: {all(isinstance(x, int) for x in raw)}")
    print(f"chat ({len(chat)}):      {chat}")
    print(f"  all ints: {all(isinstance(x, int) for x in chat)}")
    print(f"chat+sys ({len(chat_sys)}): {chat_sys}")
    print(f"  all ints: {all(isinstance(x, int) for x in chat_sys)}")
    print()
    print(f"chat decoded: {tok.decode(chat)!r}")
    print(f"chat+sys decoded: {tok.decode(chat_sys)!r}")


if __name__ == "__main__":
    main()
