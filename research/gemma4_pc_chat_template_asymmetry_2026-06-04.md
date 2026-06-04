# Gemma 4 IT chat template — second asymmetry discovered (2026-06-04)

Local validation via `experiments/cb/validate/pc_token_match.py` revealed
a SECOND chat-template asymmetry, upstream of the `| trim` issue, that
keeps gm4 PC at 0 hits.

## What the validator caught

```
google/gemma-4-12B-it: 0/3 cases PASS
MISMATCH at index 14 / 36
  prompt_1 length: 18
  canonical gen length: 18 (incl EOS)
  prompt_2 length: 53

  tokens_so_far[around]:
    [12] 4368 'model'
    [13]  107 '\n'
    [14]  101 '<channel|>'   ← here

  prompt_2[around]:
    [12] 4368 'model'
    [13]  107 '\n'
    [14] 3883 'Group'         ← past-assistant content starts here, no <channel|>
```

The divergence is at **index 14 — inside `prompt_1`**, not inside the
generated tokens. The `| trim` fix in `cb_scheduler._finish` (commit
`8aeeb53`) doesn't help because the prompts already differ before we
even concatenate `canonical_gen`.

## Root cause

Gemma 4 IT's chat template adds a `<channel|>` token (id 101) AFTER
`<start_of_turn>model\n` when `add_generation_prompt=True`. This is a
generation-time-only marker that the model never emits and the
past-assistant render on turn N+1 does NOT include. Concretely:

```
apply_chat_template([user_1], add_generation_prompt=True, tokenize=True)
  → [..., '<start_of_turn>', 'model', '\n', '<channel|>']   (length 18)

apply_chat_template([user_1, asst_1, user_2],
                    add_generation_prompt=True, tokenize=True)
  → [..., '<start_of_turn>', 'model', '\n', 'Group', 'ed', ...]
```

Notice: the past-assistant boundary is `<start_of_turn>model\n` →
straight into content, with NO `<channel|>`. The active-prompt has
`<start_of_turn>model\n<channel|>` instead. Asymmetric by 1 token, but
the cache-equality contract requires byte-exact match.

This is the same class of asymmetry as Qwen3.6's
`<think>\n\n</think>\n\n` active-prompt suffix (`feedback_qwen36_preserve_thinking.md`)
— both insert tokens at the assistant-open boundary that only appear in
the active-prompt render.

For Qwen3.6 we worked around it by stripping the `<think>` suffix in
`_messages_to_prompt`. The analogous fix for Gemma would be to strip the
`<channel|>` token (id 101) from the END of the rendered prompt after
`add_generation_prompt=True`.

## Fix (proposed, not yet shipped)

In `experiments/serve/openai_endpoint.py:_messages_to_prompt`, after the
existing Qwen `<think>` strip, add a Gemma 4 equivalent:

```python
# Gemma 4 IT: active-assistant prompt ends with a <channel|> token
# (id 101). Past-assistant renders on subsequent turns do not include
# it; strip from the trailing position to make prompt_1 a prefix of
# prompt_2. (Same asymmetry-strip pattern as Qwen3.6's <think> suffix.)
gemma_channel_id = 101  # convert via tokenizer.convert_tokens_to_ids("<channel|>")
if ids and ids[-1] == gemma_channel_id:
    ids = ids[:-1]
```

Better: discover the suffix dynamically by diffing
`apply_chat_template(..., add_generation_prompt=True)` vs
`add_generation_prompt=False`. The trailing tokens that ONLY appear in
the `True` variant are the active-prompt suffix; strip those. That makes
the fix backend-agnostic and survives template updates.

**But** stripping has a knock-on effect: the engine processed the full
`prompt_1 + gen` through its KV cache, including the stripped tokens.
On turn N+1 we'd reclaim the slot at `cur_pos = len(prompt_1) - 1` (one
short of where the KV actually advanced) and re-process those 2 positions
in the new prompt. Cheap (2 decode steps) and correct.

## After the strip, the `| trim` canonicalise still matters

The two fixes compose:
1. Strip the active-prompt suffix (`<channel|>` for Gemma 4 IT) from
   `prompt_1` BEFORE storing.
2. Canonicalise `gen` (decode → rstrip → re-encode + EOS) in `_finish`
   to match what the next-turn's `| trim` produces.

Both are necessary and neither is sufficient.

## Validation gate

`experiments/cb/validate/pc_token_match.py` is the regression test. After
applying both fixes:

```
google/gemma-4-12B-it: 3/3 PASS expected
```

That's the gate before re-deploying.

## Why we're documenting and parking

The fix is well-understood but requires:
- Editing `_messages_to_prompt` to handle backend-specific suffix stripping
- Possibly also adjusting `cb_engine.submit` or `cb_scheduler._admit_from_cache`
  to handle the `cur_pos = len(prompt) - SUFFIX_LEN` reclaim mismatch
- Server restart + multi-turn test to verify

The validator is cheap to rerun and pinpoints the exact divergence
position, so this is shovel-ready when the server frees up.

## Related

- `research/gemma4_pc_chat_template_analysis_2026-06-04.md` — the
  original analysis (focused on `| trim`; missed this upstream asymmetry).
- `research/tokenizer_chat_template_reference.md` — needs an addendum to
  capture the "dynamic suffix detection" pattern as the universal answer.
- `feedback_qwen36_preserve_thinking.md` — the Qwen3.6 precedent for
  active-prompt asymmetry stripping.
