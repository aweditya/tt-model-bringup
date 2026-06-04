# Tokenizer + chat-template reference

**Read this before touching `_messages_to_prompt`, `_finish`, `live_slot_store`, or anything that round-trips text through the tokenizer.** Every bug in this file has cost the project at least one debugging session. Do/don't checklist at the bottom.

Authoritative sources inline. Verifiable behaviour is given as one-line Python reproducers — anything not reproducible in <60s on a tokenizer-only checkout is folklore.

---

## 1. The lifecycle of a multi-turn chat (and where it goes wrong)

A single OpenAI-style chat completion looks like this on our server:

1. **Client → server**: `messages = [user_1, asst_1, user_2]` as a JSON payload.
2. **Server prompt build** (`openai_endpoint._messages_to_prompt`): `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, preserve_thinking=True)` → `list[int]` (after normalising the `BatchEncoding` / `dict` / `list` return shape).
3. **Server generate** (`cb_api._complete`): the engine produces `gen_ids: list[int]`. The client is sent `tok.decode(gen_ids, skip_special_tokens=True)`.
4. **Server finish** (`cb_scheduler._finish`): `tokens_so_far = list(prompt) + list(gen)` is stored in the `LiveSlotStore` (`live_slot_store.py:49`).
5. **Client → server, turn N+1**: client appends the decoded text as `assistant_1.content`, sends `messages_2 = [user_1, asst_1, user_2]`.
6. **Server prompt build (again)**: `apply_chat_template(messages_2, ...)` → new `prompt_2: list[int]`.
7. **Prefix-cache match** (`live_slot_store.find_longest_match`): exact prefix equality — `prompt_2[:n] == entry.tokens_so_far` (`live_slot_store.py:81`). Hit → reclaim slot at `cur_pos = n`. Miss → re-prefill.

The cache key is **byte-exact token equality**. Step 4 (what we *stored*) must equal step 6 (what we *rebuild*) for the first `len(tokens_so_far)` tokens, or the cache misses silently and the round trip pays full prefill.

Empirically that equality holds for Qwen3.6 (turn-2 hit + 1.96× speedup, commit `2cad663`) and currently fails for Gemma 4 IT (`stress_multiturn_http.py` → 0 hits / 60 misses, reported in `feedback_prefix_cache_multiturn_miss_2026-06-04.md`). Sections 3–5 below explain why, in detail.

---

## 2. Where round-trip lossiness hides

These six failure modes are model-independent. Any new backend must be audited for every one.

### 2.1 `skip_special_tokens=True` at decode

`cb_api.py:206` decodes assistant text as `tok.decode(gen_ids, skip_special_tokens=True)` — correct for the client (no stray `<|im_end|>` / `<end_of_turn>` markers in the chat bubble) but the client cannot losslessly round-trip the assistant's exit token. The server stores `tokens_so_far` with the EOS still in it (`cb_scheduler.py:497`); the template re-emits the same EOS as the turn separator, so storing-with-EOS works **provided the template's EOS placement is symmetric with the model's emission**. It is for Qwen3.6 (`<|im_end|>` id 151645). It is not for templates that `| trim` content before re-emitting EOS (Gemma 4).

### 2.2 Jinja `| trim` on past assistant content

Both Qwen3.6 and Gemma 4 chat templates apply `| trim` to past assistant `content`. From the Qwen3.6 template (`https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/chat_template.jinja`, confirmed via WebFetch 2026-06-04):

```jinja
<|im_start|>assistant
{{ content | trim }}{{ tool_calls }}
<|im_end|>
```

Qwen3.6 rarely emits trailing whitespace before `<|im_end|>`, so the trim is usually a no-op and `prompt + gen` matches. Gemma 4 frequently terminates with `…answer.\n` before `<end_of_turn>`, the `\n` is its own token, and `| trim` strips it on rebuild. Result: `tokens_so_far` has one extra token vs `prompt_2[:n]`, exact match fails, PC misses. Active Gemma 4 bug (`research/gemma4_pc_chat_template_analysis_2026-06-04.md`). `| trim` is the single Jinja filter most likely to break PC — grep for it on every new template.

### 2.3 BPE / BPE-boundary effects

BPE tokenizers like Qwen2Tokenizer treat `" hello"` (leading space) and `"hello"` as different token sequences. If you ever mix the token-space path (`apply_chat_template(tokenize=True)`) with the string-space path (`apply_chat_template(tokenize=False)` then `tok.encode(...)`), boundary mismatches at message joins will diverge by 1+ tokens and PC will miss. This is the main reason `_messages_to_prompt` returns `list[int]` directly, never a string.

### 2.4 BOS doubling

HF docs (https://huggingface.co/docs/transformers/main/en/chat_templating, "Some tokenizers add special `<bos>` and `<eos>` tokens..."):

> Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you tokenize later to avoid duplicating these tokens. This isn't an issue if you use `apply_chat_template(tokenize=True)`, which means it's usually the safer option!

In practice for our codebase:

- `apply_chat_template(tokenize=True)` is canonical. Returns ids with the template's own BOS, no doubling.
- `apply_chat_template(tokenize=False)` then `tok.encode(rendered)` — the string already has `<bos>` from the template, and `encode` adds another. We never do this; if you ever feel the need to, pass `add_special_tokens=False` to `encode`. **Gemma in particular emits `<bos>` (id 2) from the template; double-BOS will be invisible in `decode` output but breaks PC on turn 1.**

### 2.5 `add_generation_prompt=True/False` changes byte count

Turn-1 prompt build uses `add_generation_prompt=True` — the template appends an assistant-prompt suffix (`<|im_start|>assistant\n<think>\n\n</think>\n\n` for Qwen3.6, `<start_of_turn>model\n` for Gemma 4). On turn 2 the suffix is appended AFTER the new user turn — not at the position of the turn-1 cache entry. `tokens_so_far = prompt_1 + gen_1` is the prefix the matcher needs, with the assistant-prompt suffix STRIPPED from `prompt_1` (`openai_endpoint.py:77-80`). Without the strip, `prompt_1` would have trailing "empty think" tokens that the model never actually generates, and PC will miss by the suffix length.

### 2.6 `{%- ... -%}` whitespace stripping inside Jinja

Jinja block markers `{%-` and `-%}` strip leading/trailing whitespace from the surrounding text. We don't author chat templates, but two "identical-looking" templates from different model cards can produce different token streams if one uses the strip dashes and the other doesn't. Vet via `apply_chat_template(msgs, tokenize=False)` then `repr()` — invisible whitespace shows up as `' '` or `'\n'`.

---

## 3. Per-tokenizer gotchas in this codebase

### 3.1 Qwen3.6 (Qwen2Tokenizer / PreTrainedTokenizerFast)

- **Return type from `apply_chat_template(tokenize=True)`**: bare `list[int]`. No `input_ids` wrapper.
- **`<think>` is NOT a special token**: `tok.added_tokens_decoder[248068]` is `AddedToken('<think>', ..., special=False)`. `skip_special_tokens=True` does NOT strip `<think>` / `</think>` from client text. Upstream design — Qwen3.6 round-trips reasoning via `<think>` markers and the template extracts them on next turn (`if '</think>' in content` branch).
- **Active-prompt suffix**: with `enable_thinking=False`, template appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. The trailing `<think>\n\n</think>\n\n` is an empty block the model never emits. `_messages_to_prompt` strips it in token space (`openai_endpoint.py:77-80`). Don't use `string.replace(...)` — it kills legitimate empty-think blocks in past messages.
- **Past-assistant rendering**: template drops `<think>...</think>` from history by default ("rolling checkpoint"). Pass `preserve_thinking=True` so historical reasoning is re-wrapped — without it, turn-N+1 lacks tokens turn N's slot generated, PC misses.
- **EOS**: `<|im_end|>` = 151645. Chat EOS and turn separator. Model emits it; template re-emits it between turns. Keep it in `tokens_so_far` (`cb_scheduler.py:490-495`).
- **No `| trim` divergence in practice**: template has `{{ content | trim }}` but Qwen rarely emits trailing whitespace before `<|im_end|>`. PC works with `tokens_so_far = prompt + gen`.
- **Vocab**: 248320 (`feedback_vocab_sharded_lm_head_result.md`).
- **Tools mode**: "rolling checkpoint" overrides `preserve_thinking` for assistant turns before the last non-tool-call user turn. Re-validate PC if tools are added.

One-liner repro of the active-prompt suffix:
```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
ids = t.apply_chat_template([{"role":"user","content":"hi"}], tokenize=True,
                              add_generation_prompt=True, enable_thinking=False)
print(t.decode(ids[-8:]))  # ends with: <|im_start|>assistant\n<think>\n\n</think>\n\n
```

### 3.2 Gemma 4 IT (GemmaTokenizer / GemmaTokenizerFast)

- **Return type**: `BatchEncoding` (a `UserDict` subclass with `input_ids` — NOT a plain dict, NOT `list[int]`). Does NOT pass `isinstance(x, dict)` on all transformers versions. Correct test: `hasattr(x, "input_ids")` OR `hasattr(x, "keys") and "input_ids" in x` (`feedback_long_context_chat_template_behavior_not_drift.md`). The normaliser at `openai_endpoint.py:71-74` covers both shapes.
- **`| trim` on past assistant content**:
  ```jinja
  <start_of_turn>model
  {{ message['content'] | trim }}<end_of_turn>
  ```
  Active PC-miss source. If `gen_ids` ends in tokens for `"\n"`, `tokens_so_far` has them but the rebuild won't.
- **Multi-EOS**: generation_config lists `eos_token_id: [1, 106, 50]` — `1=<eos>` (corpus), `106=<end_of_turn>` (dialog), `50=<unused50>`. Server collects all three into a `frozenset` (`cb_api.py:301-323`); `cb_engine` matches any. We override `tokenizer.eos_token = "<end_of_turn>"` (`server_gemma4_unified_ttnn.py:354-356`) so dialog stops on 106, but the EOS *set* still includes all three.
- **BOS**: Gemma emits `<bos>` (id 2) from the template. Don't `tok.encode(rendered_string)` — `add_special_tokens=True` doubles the BOS. Always go through `apply_chat_template(tokenize=True)`.
- **Base vs IT**: base variant ships no chat template (`server_gemma4_unified_ttnn.py:336-346` installs a minimal one). For base, `<end_of_turn>` is multi-token text → EOS won't fire reliably. PC validation is IT-only.
- **`tokens_so_far = prompt + gen` is WRONG for Gemma**. Required canonicalisation (`research/gemma4_pc_chat_template_analysis_2026-06-04.md` Option C):
  ```python
  decoded = tok.decode(gen, skip_special_tokens=True).rstrip()
  canonical_gen = tok.encode(decoded, add_special_tokens=False) + [eot_id]
  tokens_so_far = list(prompt) + canonical_gen
  ```
  Mirrors the `| trim` the template applies on next turn. Option A (token-direct prompt build) is already in place; the missing half is finish-side canonicalisation.

One-liner repro of the trim divergence:
```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")  # placeholder if 4 IT unavailable
msgs = [{"role":"user","content":"hi"}, {"role":"assistant","content":"hi.\n"}]
a = t.apply_chat_template(msgs, tokenize=True)
msgs[-1]["content"] = "hi."  # what | trim produces
b = t.apply_chat_template(msgs, tokenize=True)
assert a != b, "trim is observable in token space"
```

---

## 4. The single-source-of-truth pattern

There is one supported path from `messages` to token ids and it is `apply_chat_template(tokenize=True)`. Period.

```python
# CORRECT — single source of truth
kw = dict(tokenize=True, add_generation_prompt=True,
          enable_thinking=False, preserve_thinking=True)  # silently ignored by non-Qwen
if tools: kw["tools"] = tools
raw = tokenizer.apply_chat_template(messages, **kw)
# Normalise the return shape — different transformers versions / tokenizers
# return list[int] (Qwen2Tokenizer), BatchEncoding (GemmaTokenizer), or dict.
if isinstance(raw, dict) or hasattr(raw, "input_ids"):
    ids = list(raw["input_ids"])
else:
    ids = list(raw)
```

```python
# WRONG — every variant of this is a bug factory
rendered = tokenizer.apply_chat_template(messages, tokenize=False, ...)
ids = tokenizer.encode(rendered)               # double-BOS for Gemma
ids = tokenizer.encode(rendered, add_special_tokens=False)  # boundary effects at joins
```

Why the round-trip is the bug factory: a Jinja template can apply arbitrary transformations (`| trim`, `| safe`, conditional whitespace) to content **before tokenization**. `tokenize=True` runs those transformations inside the renderer and tokenizes the result — one tokenization. `tokenize=False` then `encode` rebuilds tokens from a renderer-mutated string, and the encoder may add `<bos>`, split BPE differently at boundaries, etc. — two tokenizations with a black box between them.

This is also why we never hand-construct chat strings and POST them to `/v1/completions` to "avoid the template" — that forces us to own the special-token layout of every model forever. The chat template is upstream-maintained; we ride future fixes for free (`research/vllm_chat_template_handling.md` §5).

---

## 5. The `tokens_so_far` canonical form

For multi-turn PC to work, the equality the matcher checks is:

```
apply_chat_template([user_1, assistant_1_text_decoded, user_2], tokenize=True)[:n]
    == tokens_so_far_stored_at_finish
```

where `n = len(tokens_so_far)`. There are two regimes, and the right canonicalisation depends on the template:

**Regime A — `| trim` is a no-op on model output** (Qwen3.6):

```python
tokens_so_far = list(prompt) + list(gen)  # gen includes trailing EOS
```

Works because (a) `| trim` doesn't change typical Qwen output, (b) `gen` ends in `<|im_end|>` which the template also emits as the turn separator, (c) `<think>` blocks round-trip through `content` and `preserve_thinking=True` re-wraps them on next render.

**Regime B — `| trim` is observable** (Gemma 4):

```python
decoded = tok.decode(gen, skip_special_tokens=True).rstrip()
canonical_gen = tok.encode(decoded, add_special_tokens=False) + [eot_id]
tokens_so_far = list(prompt) + canonical_gen
```

Pre-apply the trim the template will apply on rebuild, then re-add the EOS the template emits at the turn boundary.

**Don't** strip the trailing EOS from `tokens_so_far`. Both regimes require it — the template re-emits it as the turn separator, and the slot's DN+KV state has already processed that token. Comment block at `cb_scheduler.py:486-495` documents this for Qwen3.6.

**Don't** loosen `min_match_tokens` to 0 to hide a mismatch (warning in `feedback_prefix_cache_multiturn_miss_2026-06-04.md`). That just hides the bug.

---

## 6. Validation harness

The single test that would have caught both the Qwen3.6 initial bug AND the active Gemma 4 bug — write it for every new backend before shipping PC.

Suggested path: `experiments/cb/validate/pc_token_match.py` (sibling of the existing `experiments/cb/validate/cb35_v0_chat.py` and friends).

```python
"""PC token-match invariant — no GPU, runs in seconds locally.

For each (model_id, expected_regime) tuple:
1. Load tokenizer.
2. Build messages_1 = [user_1].
3. prompt_1_ids = apply_chat_template(messages_1, tokenize=True, ...).
4. Simulate 10 gen tokens INCLUDING the EOS the model would emit.
5. Apply the regime's _finish canonicalisation -> tokens_so_far.
6. Rebuild messages_2 = [user_1, assistant_1_decoded, user_2].
7. prompt_2_ids = apply_chat_template(messages_2, tokenize=True, ...).
8. Assert prompt_2_ids[:len(tokens_so_far)] == tokens_so_far.

Failure mode: print the first divergent index and decode(both[i-3:i+3]).
"""
```

Run before any deploy that changes:
- the tokenizer (model swap, HF version bump),
- `_messages_to_prompt` or `_finish` canonicalisation,
- `apply_chat_template` kwargs (`enable_thinking`, `preserve_thinking`, `tools`, etc.),
- the chat template itself (custom override or upstream HF revision pin).

This is the same invariant the upstream mlx-lm benchmark measures (`research/vllm_chat_template_handling.md` §3), and it is what `cache_salt` and other isolation mechanisms provably **cannot** enforce. It has to live in our rendering layer.

---

## 7. Tokenizer-version sensitivity

`apply_chat_template`'s return type shifts across transformers versions:

- ≥ 4.34: `tokenize=False` returns `str`; `tokenize=True` returns `list[int]` for most tokenizers, `BatchEncoding` for some (Gemma is the canonical offender).
- ≥ 4.43 added `continue_final_message` for prefill-style responses. Don't combine with `add_generation_prompt`.

The normaliser at `openai_endpoint.py:71-74` handles `list`, `dict`, and `BatchEncoding`. Never assume the return shape. Pin the tokenizer (and ideally the model card revision) in CI — a silent upstream template change is indistinguishable from a code bug from the matcher's perspective.

---

## 8. Do / don't checklist

**Do**:

- Use `apply_chat_template(tokenize=True)` and pass tokens straight to the engine. Single source of truth.
- Normalise the return shape: `list[int]` vs `BatchEncoding` vs `dict` — handle all three.
- Pass `enable_thinking=False, preserve_thinking=True` to `apply_chat_template`. Both are silently ignored by templates that don't define them — safe by default.
- Strip the Qwen3.6 trailing `<think>\n\n</think>\n\n` block in **token space**, never via `string.replace`.
- Include the trailing EOS in `tokens_so_far`. The template re-emits it as the turn separator.
- Run the `pc_token_match.py` validator on every new backend, every tokenizer version bump, and every change to `_messages_to_prompt` or `_finish`.
- For templates that apply `| trim` to past assistant content (Gemma 4): canonicalise `gen` by `decode → rstrip → re-encode → re-add EOS` before storing as `tokens_so_far`.
- Grep new chat templates for `| trim`, `| safe`, `{%-`, `-%}`, and any role-conditional `if` branches. These are the silent killers.
- Cite the tokenizer + transformers version in any PR that touches this code.

**Don't**:

- Don't `tokenizer.encode(apply_chat_template(messages, tokenize=False))`. The string round-trip will silently produce different ids than the token-direct path.
- Don't `string.replace("<think>\n\n</think>\n\n", "")` to strip the Qwen3.6 suffix. It kills legitimate past-message empty-think blocks.
- Don't strip EOS from `tokens_so_far`. The template re-emits it at the turn boundary; stripping breaks PC by 1 token.
- Don't strip BOS from `tokens_so_far`. Same reason for templates that emit BOS.
- Don't lower `min_match_tokens` in `LiveSlotStore` to "loosen" a mismatch. That hides the bug instead of fixing it.
- Don't construct chat strings by hand and POST them to `/v1/completions` to "bypass the template". You'll own the special-token layout of every model forever.
- Don't change `cb_api.py`'s `skip_special_tokens=True` at decode. The client wants clean text; the round-trip is fixed at `_finish`, not at the API boundary.
- Don't assume `apply_chat_template(tokenize=True)` returns the same Python type across tokenizers or across transformers versions.
- Don't merge a PC change without running `pc_token_match.py`. If it doesn't exist for the backend yet, write it first.

---

## References

In-repo:
- `experiments/serve/openai_endpoint.py:32-81` — `_messages_to_prompt` reference.
- `experiments/serve/cb_api.py:170-222, 285-360` — prompt → engine, client decode, multi-EOS frozenset assembly.
- `experiments/serve/cb_scheduler.py:481-500` — `_finish` and `tokens_so_far` storage; comment explains why EOS stays in.
- `experiments/serve/live_slot_store.py:57-85` — byte-exact prefix match.
- `experiments/serve/server_gemma4_unified_ttnn.py:320-360` — Gemma base/IT template install, multi-EOS.
- `research/gemma4_pc_chat_template_analysis_2026-06-04.md` — active Gemma 4 PC miss diagnosis, Options A/B/C.
- `research/vllm_chat_template_handling.md` — vLLM/SGLang/TGI comparison; upstream prior art.
- `research/27b_prefix_caching_plan.md` — slot-level PC design.

External:
- https://huggingface.co/docs/transformers/main/en/chat_templating — canonical API + BOS-doubling warning.
- https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/chat_template.jinja — verified `preserve_thinking` conditional.
- https://huggingface.co/blog/qwen-3-chat-template-deep-dive — rolling-checkpoint design rationale.
- `QwenLM/Qwen3#1826` — mlx-lm 90× benchmark on the same `enable_thinking=False` PC fix class.

---

*Last updated 2026-06-04. If you change `_messages_to_prompt` or `_finish` without updating this file, you owe the next engineer a debugging session.*
