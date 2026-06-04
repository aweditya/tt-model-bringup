"""Dump Qwen3.6's chat template source + render turn-1 vs turn-2 side-by-side.

The chat template is a Jinja string on the tokenizer. Looking at it directly
beats peeling tokenization-mismatch onions one at a time.

Run on qb1:
  cd ~/tt-xla && .venv/bin/python experiments/cb/isolate/chat_template_inspect.py
"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

# --- 1. Dump the raw Jinja chat template ---
print("=" * 72)
print("CHAT TEMPLATE (Jinja source)")
print("=" * 72)
template_src = tok.chat_template
if template_src is None:
    print("!!! tokenizer.chat_template is None — checking alt sources")
    # Some Qwen models put it in tokenizer_config.json under chat_template
else:
    print(template_src)
print("=" * 72)
print(f"\ntemplate length: {len(template_src) if template_src else 0} chars\n")

# --- 2. Inspect special-token classification ---
print("=" * 72)
print("SPECIAL TOKENS")
print("=" * 72)
print(f"eos_token_id: {tok.eos_token_id} ({tok.decode([tok.eos_token_id])!r})")
print(f"all_special_ids: {tok.all_special_ids}")
print(f"all_special_tokens: {tok.all_special_tokens}")
print("added_tokens_decoder (selected):")
for tid, t in (tok.added_tokens_decoder or {}).items():
    if hasattr(t, 'content'):
        content = t.content
    else:
        content = str(t)
    if any(k in content for k in ("think", "im_", "endof", "pad")):
        special = getattr(t, "special", "?")
        print(f"  {tid}: {content!r} (special={special})")
print()

# Is <think> a "special" token? Does skip_special_tokens=True strip it?
print(f"Encoding '<think>': {tok.encode('<think>', add_special_tokens=False)}")
print(f"Decoding [248068, 271, 248069, 271] with skip_special=False: "
      f"{tok.decode([248068, 271, 248069, 271], skip_special_tokens=False)!r}")
print(f"Decoding [248068, 271, 248069, 271] with skip_special=True:  "
      f"{tok.decode([248068, 271, 248069, 271], skip_special_tokens=True)!r}")
print()

# --- 3. Render turn-1 vs turn-2 side by side ---
sys_msg = {"role": "system",
           "content": "You are a concise, helpful assistant. Reply in 1-2 sentences."}
user_1 = {"role": "user",
          "content": "What is the capital of France, and what is one famous landmark there?"}
# Use the EXACT response the live server emitted in the last smoke test:
assistant_1_content = (
    "<think>\n\n</think>\n\n"
    "The capital of France is Paris, and one of its most famous landmarks "
    "is the Eiffel Tower."
)
assistant_1 = {"role": "assistant", "content": assistant_1_content}
user_2 = {"role": "user", "content": "And what about Germany?"}

# Show rendered TEXT (before tokenization) for both turns.
print("=" * 72)
print("TURN 1 prompt (raw rendered text)")
print("=" * 72)
turn1_text_default = tok.apply_chat_template(
    [sys_msg, user_1], tokenize=False, add_generation_prompt=True)
print(repr(turn1_text_default))
print()
print("with enable_thinking=False:")
turn1_text_nothink = tok.apply_chat_template(
    [sys_msg, user_1], tokenize=False, add_generation_prompt=True,
    enable_thinking=False, preserve_thinking=True)
print(repr(turn1_text_nothink))
print()

print("=" * 72)
print("TURN 2 prompt (raw rendered text)")
print("=" * 72)
turn2_text_default = tok.apply_chat_template(
    [sys_msg, user_1, assistant_1, user_2],
    tokenize=False, add_generation_prompt=True)
print(repr(turn2_text_default))
print()
print("with enable_thinking=False:")
turn2_text_nothink = tok.apply_chat_template(
    [sys_msg, user_1, assistant_1, user_2],
    tokenize=False, add_generation_prompt=True,
    enable_thinking=False, preserve_thinking=True)
print(repr(turn2_text_nothink))
print()

# --- 4. Tokenize both and find exact divergence ---
print("=" * 72)
print("TOKEN-LEVEL DIFF")
print("=" * 72)
# Cached for turn 1 = patched(turn1_text) + model_gen
# Where model_gen = encode(assistant_1_content) (model emitted these tokens) + [EOS]
_THINK_SUFFIX = "<think>\n\n</think>\n\n"
def _strip_trailing_think(text):
    if text.endswith(_THINK_SUFFIX):
        return text[:-len(_THINK_SUFFIX)]
    return text
turn1_patched_text = _strip_trailing_think(turn1_text_nothink)
turn1_patched_ids = tok.encode(turn1_patched_text)
gen_ids = tok.encode(assistant_1_content, add_special_tokens=False) + [tok.eos_token_id]
cached = turn1_patched_ids + gen_ids
print(f"turn 1 patched prompt len: {len(turn1_patched_ids)}")
print(f"gen (encoded response + EOS) len: {len(gen_ids)}")
print(f"cached total len: {len(cached)}")

turn2_patched_text = _strip_trailing_think(turn2_text_nothink)
turn2_patched_ids = tok.encode(turn2_patched_text)
print(f"turn 2 patched prompt len: {len(turn2_patched_ids)}")

# longest common prefix
n_match = 0
limit = min(len(cached), len(turn2_patched_ids))
while n_match < limit and cached[n_match] == turn2_patched_ids[n_match]:
    n_match += 1
print(f"\nlongest common prefix: {n_match} / {len(cached)} cached")
if n_match < len(cached):
    print(f"\n!!! DIVERGENCE at position {n_match}")
    lo, hi = max(0, n_match - 5), min(limit, n_match + 6)
    print(f"  cached  [{lo}..{hi}]: {cached[lo:hi]}")
    print(f"  turn 2  [{lo}..{hi}]: {turn2_patched_ids[lo:hi]}")
    print(f"  cached  detok: {tok.decode(cached[lo:hi])!r}")
    print(f"  turn 2  detok: {tok.decode(turn2_patched_ids[lo:hi])!r}")
else:
    print("✓ FULL PREFIX MATCH")
