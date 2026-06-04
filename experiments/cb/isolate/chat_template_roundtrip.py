"""Probe: does turn-2's chat-template tokenization match what we cached from turn-1?

Mimics exactly what cb_api does: tokenize via chat template, generate response,
re-tokenize turn 2's prompt (with the response text as assistant content), and
compare the first N tokens to the "what we cached" sequence (turn-1 prompt + gen).

Run on qb1:
  cd ~/tt-xla && .venv/bin/python experiments/cb/isolate/chat_template_roundtrip.py
"""


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
sys_msg = {"role": "system",
           "content": "You are a concise, helpful assistant. Reply in 1-2 sentences."}
user_1 = {"role": "user",
          "content": "What is the capital of France, and what is one famous landmark there?"}

# Turn 1: produces this 46-token prompt.
turn1_prompt_text = tok.apply_chat_template([sys_msg, user_1], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
# Patch: strip the empty <think>\n\n</think>\n\n block that Qwen3.6's chat
# template injects at the active assistant prompt but NOT in past assistant
# messages. Without this fix, turn 2's tokenization (which renders the past
# assistant message via the template, no thinking block) won't match cached.
turn1_prompt_text = turn1_prompt_text.replace("<think>\n\n</think>\n\n", "")
turn1_prompt_ids = tok.encode(turn1_prompt_text)
print(f"turn 1 prompt (stripped <think>): {len(turn1_prompt_ids)} tokens")

# Simulate the model's response from the actual smoke test (200 tokens via max_tokens cap).
# The smoke test turn 1 response (from real run) — paste exact text:
turn1_response_text = (
    "Thinking Process:\n\n1.  **Identify the core questions:** The user is "
    "asking for the capital of France and one famous landmark located there.\n"
    "2.  **Retrieve knowledge:**\n    *   Capital of France: Paris.\n"
    "    *   Famous landmark in Paris: Eiffel Tower, Louvre Museum, "
    "Notre-Dame, Arc de Triomphe, etc.\n"
    "3.  **Formulate the answer:** Combine the two pieces of information "
    "into a concise sentence.\n"
    "    *   Draft 1: The capital of France is Paris, and a famous landmark "
    "there is the Eiffel Tower.\n"
    "4.  **Check constraints:** Concise, helpful, 1-2 sentences. Draft 1 is "
    "one sentence. Perfect.\n"
    "5.  **Final Polish:** \"The capital of France is Paris, and one of its "
    "most famous landmarks is the Eiffel Tower.\" (Slightly more natural "
    "flow). Keep it simple. \"The capital of France is Paris,"
)
re_tokenized_response = tok.encode(turn1_response_text, add_special_tokens=False)
print(f"re-tokenized response: {len(re_tokenized_response)} tokens")
print(f"first 5: {re_tokenized_response[:5]}")
print(f"last 5:  {re_tokenized_response[-5:]}")

# "Cached" tokens_so_far at end of turn 1: prompt + gen.
# We don't have the model's actual gen tokens here, but we know the COUNT was
# 200 (max_tokens cap). For this probe, we assume the model produced text
# that re-tokenizes identically (a clean BPE roundtrip would).
# So "cached" = turn1_prompt_ids + re_tokenized_response (200 tokens of text)
cached_tokens_so_far = turn1_prompt_ids + re_tokenized_response
print(f"cached len = {len(cached_tokens_so_far)}")

# Turn 2: chat template renders the FULL conversation including assistant_1.
assistant_1 = {"role": "assistant", "content": turn1_response_text}
user_2 = {"role": "user", "content": "And what about Germany?"}
turn2_prompt_text = tok.apply_chat_template(
    [sys_msg, user_1, assistant_1, user_2], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
turn2_prompt_text = turn2_prompt_text.replace("<think>\n\n</think>\n\n", "")
turn2_prompt_ids = tok.encode(turn2_prompt_text)
print(f"\nturn 2 prompt: {len(turn2_prompt_ids)} tokens")

# Compare: how long is the longest matching prefix?
n_match = 0
while n_match < len(cached_tokens_so_far) and n_match < len(turn2_prompt_ids):
    if cached_tokens_so_far[n_match] != turn2_prompt_ids[n_match]:
        break
    n_match += 1
print(f"\nlongest common prefix: {n_match} / {len(cached_tokens_so_far)} cached")

if n_match < len(cached_tokens_so_far):
    print(f"\n!!! DIVERGENCE at position {n_match}")
    lo = max(0, n_match - 5)
    hi = min(len(cached_tokens_so_far), n_match + 6)
    print(f"  cached  [{lo}..{hi}]: {cached_tokens_so_far[lo:hi]}")
    print(f"  turn 2  [{lo}..{hi}]: {turn2_prompt_ids[lo:hi]}")
    print(f"  cached  detok: {tok.decode(cached_tokens_so_far[lo:hi])!r}")
    print(f"  turn 2  detok: {tok.decode(turn2_prompt_ids[lo:hi])!r}")
else:
    print("\n✓ full prefix match — cache hit would fire")

# Additional: dump the "join point" — where assistant_1 ends in turn 2.
print(f"\nturn 2 last 10 tokens at position {len(cached_tokens_so_far)-5}..{len(cached_tokens_so_far)+5}:")
lo = max(0, len(cached_tokens_so_far) - 5)
hi = min(len(turn2_prompt_ids), len(cached_tokens_so_far) + 6)
for i in range(lo, hi):
    marker = "  <-- cached end" if i == len(cached_tokens_so_far) - 1 else ""
    print(f"  turn2[{i}] = {turn2_prompt_ids[i]} ({tok.decode([turn2_prompt_ids[i]])!r}){marker}")
