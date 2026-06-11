# Gemma 4 prefix-cache miss — chat template analysis (2026-06-04)

Sub-agent investigation timed out mid-fix but reached a concrete root-cause hypothesis. Capturing here so the next session can implement.

## Symptom

`stress_multiturn_http.py` against TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it TT_CB_PREFIX_CACHE=1 → `cb_prefix_cache_hits_total = 0, cb_prefix_cache_misses_total = 60`. Wall time scales linearly in prompt_t — no PC benefit.

Qwen3.6-27B with the same probe hits PC on turn 2 (6.3× wall speedup). So the matcher and infrastructure work; only the chat-template / tokenisation round-trip is broken for Gemma 4.

## Root-cause hypothesis

1. **`| trim` filter in Gemma's chat template strips trailing whitespace from past-assistant `content`**. The Jinja for Gemma 4 IT renders past assistant turns as:
   ```
   <start_of_turn>model
   {content | trim}<end_of_turn>
   ```
   If the model originally generated `"…answer.\n"` and stopped at `<end_of_turn>`, the stored `tokens_so_far` includes the trailing `\n` token. On turn N+1, the client sends `assistant.content = decode(gen_ids, skip_special_tokens=True)` which gives `"…answer.\n"`. The template trims to `"…answer."` and re-tokenises — producing FEWER tokens than the stored sequence. Prefix match fails by ≥1 token.

2. **Possible BOS double-token at the render boundary**. Gemma's tokenizer may prepend `<bos>` (id=2) via `add_special_tokens=True` when re-encoding the rendered string. If the chat template ALSO emits `<bos>` as part of the rendered prefix, we get TWO `<bos>` tokens in the encoded prompt. As long as this is consistent across turns it shouldn't break PC, but worth verifying with the validator script.

3. **`skip_special_tokens=True` at decode time** (`cb_api.py:212` per sub-agent): when we decode `gen_ids` for the client, special tokens like `<end_of_turn>` are stripped from the returned text. The client cannot re-include them when sending turn N+1. So the round-trip path is `gen_ids → text (special-stripped) → client → messages → render (template re-injects role markers and trim's the content)`. The `| trim` is the lossy step we cannot reverse.

## Proposed fix (next session)

**Option A (preferred — eliminates the round-trip entirely)**: have `_messages_to_prompt` return TOKEN IDs directly via `apply_chat_template(..., tokenize=True)` instead of returning a string then re-encoding. The tokenisation happens INSIDE the template renderer, with the trim applied before tokenisation — which IS what we'd want for a stable cache key. Then plumb the token list straight into `engine.submit(prompt=tokens, ...)`. cb_api would change to:
```python
prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                 enable_thinking=False, preserve_thinking=True,
                                 tools=body.get("tools"))
```
(`apply_chat_template` accepts `tokenize=True/False`.)

**Side effect**: this also fixes silent BOS-doubling bugs and any other invisible whitespace divergence between the template renderer and the encoder. It's the single source of truth for tokenization, period.

**Option B (additive — for backward compat)**: keep `_messages_to_prompt → str`, but at request completion, store `tokens_so_far = decode(gen_ids) → re-encode_via_template_round_trip(prompt_messages + assistant_text)`. This costs one extra tokenisation pass per request. Hairy.

**Option C (pragmatic — strip what the template will strip)**: at request completion, decode `gen_ids[:-1]` (excluding EOS) → text → `text.rstrip()` → re-tokenize. Store `prompt + re-tokenize(text.rstrip()) + [eos_id]`. Matches what the template will produce on the next turn IF the template's `| trim` only strips trailing whitespace (the common case). Doesn't help if Jinja does anything more exotic.

Recommendation: **ship Option A**. One source of truth, fixes both this bug and any future BOS/whitespace gotcha. Risk: existing callers (validators, dev probes) that pass `prompt` as a string need to be updated.

## Validation gate

Write `experiments/cb/validate/gm4_pc_token_match.py` that:
1. Loads `tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")`.
2. Builds `messages_1 = [{"role": "user", "content": "hi"}]`. Tokenises via the fix.
3. Simulates a 5-token assistant response: `gen_ids = [10, 20, 30, 40, 106]` (where 106 is `<end_of_turn>`).
4. Builds `messages_2 = messages_1 + [assistant_1_round_trip, user_2]` where the round-trip is the cb_api flow.
5. Asserts `tokenize(messages_2)[:len(messages_1_tokens) + len(gen_ids)] == messages_1_tokens + gen_ids`.

If this passes, the live HTTP PC will hit. If not, narrow down which token diverges and patch accordingly.

## Files involved

- `experiments/serve/openai_endpoint.py:35-90` — `_messages_to_prompt` (current impl returns string).
- `experiments/serve/cb_api.py:212` — `decode(gen_ids, skip_special_tokens=True)` for the client-facing text. Don't change this; the client wants clean text.
- `experiments/serve/cb_api.py:219` — `_messages_to_prompt(tok, messages, tools=...)` call. After the fix this changes from `→ str` to `→ list[int]`.
- `experiments/serve/cb_engine.py` — verify `engine.submit(prompt=...)` accepts a list-of-ints prompt (it should — `r['prompt']` is already `list[int]`).

## Hand-off

After applying Option A:
1. Run `experiments/cb/validate/gm4_pc_token_match.py` locally (no GPU needed).
2. Deploy + restart serve_cb at TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it TT_CB_SLOTS=32 TT_CB_PREFIX_CACHE=1.
3. Run `scripts/stress_multiturn_http.py` and confirm `cb_prefix_cache_hits_total ≥ 1`.

Expected outcome: ~6× wall-time speedup on turn 2 of a 3-turn chat, matching the 27B result `[[cb-api-clobbered-27b-owned-gdn]]` baseline.
