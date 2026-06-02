# How vLLM / SGLang / TGI handle Qwen3.6 chat-template asymmetry, and what it means for our APC fix

**Context.** Our slot-level prefix-caching server is being built around Qwen3.6-27B. We are hitting the known Qwen3-family pathology where the Jinja chat template tokenizes a multi-turn conversation differently depending on whether an assistant message is "the active one" (`add_generation_prompt=True`) or "in history". Because `<think>` is *not* a special token in the Qwen3.6 tokenizer (`skip_special_tokens=True` does not strip it), the literal `<think>...</think>` that the model emits at turn N becomes part of the prefix that turn N+1's renderer is supposed to reproduce — but by default it does not. We claim that turning on `preserve_thinking=True` in `chat_template_kwargs` plus a trailing-only strip of the active prompt's empty `<think>...</think>` block gets a 69/69 token-level prefix match in isolation. This doc audits how the field handles the same problem.

## 1. vLLM proper

**vLLM does NOT do any per-turn special-casing.** The OpenAI-compatible chat endpoint resolves a single dict of `chat_template_kwargs` per request and passes it straight to `tokenizer.apply_chat_template` for every turn. See `vllm/entrypoints/openai/chat_completion/serving.py` (downloaded from `main`, 1467 lines), `_effective_chat_template_kwargs` at line 178-188 — it merges the request's `chat_template_kwargs` over `default_chat_template_kwargs` (set via the CLI flag `--default-chat-template-kwargs`) and that's it. The same merged dict is then handed to the renderer (line 246) and to the reasoning parser (line 244-247). There is no "render the active turn one way, the history another way" code path. Whatever asymmetry exists is purely a property of the Jinja template the tokenizer ships.

**vLLM does not pass `preserve_thinking` by default.** The closest configurable surface is `--default-chat-template-kwargs '{"preserve_thinking": true}'`, which is the same plumbing as for `enable_thinking`. Confirmed in the Qwen3.6 model card itself (`https://huggingface.co/Qwen/Qwen3.6-27B`), which shows the exact form `extra_body={"chat_template_kwargs": {"preserve_thinking": True}}` and notes: *"this can improve KV cache utilization, optimizing inference efficiency in both thinking and non-thinking modes."* That is the upstream-blessed answer to our problem.

**vLLM has no Qwen3-aware chat-template patch in `main`.** Code searches across `vllm-project/vllm` for `preserve_thinking` return zero hits; for `apply_hf_chat_template` zero hits (the function was reorganized into `vllm/entrypoints/renderer.py` / chat_completion/serving.py and now just calls `tokenizer.apply_chat_template`). The vLLM "Qwen3.5 & Qwen3.6 Usage Guide" recipe page only documents the `--reasoning-parser qwen3` and `--default-chat-template-kwargs` flags; it does not ship a fixed template.

**Cache salting is the only other relevant lever.** vLLM RFC #16016 added a top-level `cache_salt` field that gets injected into the hash of the first block. It is for *isolation* (multi-tenant), not for *normalization* — it cannot rescue a broken-prefix situation, it can only make different sessions provably not share blocks.

Sources:
- `vllm/entrypoints/openai/chat_completion/serving.py:178-247` (downloaded 2026-06-01)
- https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- https://github.com/vllm-project/vllm/issues/16016 (cache_salt RFC)

## 2. Tenstorrent vLLM fork and SGLang

**Tenstorrent vLLM fork (`tenstorrent/vllm`, branch `dev`):** the fork tracks upstream — branches `ben/qwen3-rebased`, `qwen3`, `cleonidouTT/qwen35-27b-model-readiness`, `ssanjay/qwen3-vl-support-rebased` exist but a code search for `preserve_thinking` and for Qwen3-specific chat-template edits returns nothing. The README directs hardware users to a "TT vLLM plugin on the dev branch" but the entrypoints layer is upstream. **Conclusion: the TT fork does not patch this.** If we want it patched, we own it.

**SGLang:** SGLang has its own conversation rendering at `python/sglang/srt/parser/conversation.py` (1198 lines), but it is only used for legacy `SeparatorStyle` paths (e.g. `QWEN2_VL_EMBED`, `QWEN2_AUDIO`). For Qwen3.6 it defers to the HuggingFace Jinja template just like vLLM does. SGLang's own Qwen3.6 cookbook (`docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`) advertises "Thinking Preservation: New option to retain reasoning context from historical messages" as a Qwen3.6 feature, not an SGLang feature — same `chat_template_kwargs` knob is exposed. No SGLang-specific template patch.

Sources:
- `gh api repos/tenstorrent/vllm/branches` (queried 2026-06-01)
- `python/sglang/srt/parser/conversation.py` (downloaded 2026-06-01) — Qwen3.6 not in `SeparatorStyle` enum
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` in `sgl-project/sglang` (only mentions `chat_template_kwargs`)

## 3. Qwen3 ecosystem

This is the most informative section.

**The Qwen3.6 model card (HF) is the authoritative source on `preserve_thinking`.** Verbatim from `Qwen/Qwen3.6-27B`: *"By default, only thinking blocks from the latest user message are retained. Qwen3.6 has been trained to preserve and leverage thinking traces from historical messages. This can be enabled via the `preserve_thinking` option."* The same card includes the exact `extra_body={"chat_template_kwargs": {"preserve_thinking": True}}` snippet and notes that it improves KV cache utilization.

**The Jinja template proves it.** From `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/chat_template.jinja`, the assistant-history branch is:

```jinja
{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

So `preserve_thinking=True` is exactly the override that makes *every* historical assistant turn render with its `<think>...</think>` block — provided `reasoning_content` is populated, which is parsed from the message either via `message.reasoning_content` (a string field) or by extracting from `<think>` tags in `content`. **Our finding aligns with the design intent**, not a workaround.

**Known prior art on the asymmetry bug — all of these confirm our diagnosis.**

- `QwenLM/Qwen3.6/issues/131` ("chat template emits empty historical `<think>` blocks, causing prompt drift and cache invalidation", reporter `latent-variable`, opened 2026-04-09, **closed**): proposes adding `and reasoning_content` to the conditional so empty `<think></think>` is not emitted in history when there *is* no reasoning content. Independent llama.cpp validation cited in the thread.
- `QwenLM/Qwen3/issues/1826` ("Chat template breaks KV-cache reuse when enable_thinking=false", opened 2026-03-03, **open**): same class of bug for the `enable_thinking=false` path — generation prompt inserts `<think>\n\n</think>` but historical turns don't. Proposed fix adds an `elif enable_thinking is defined and enable_thinking is false` branch. mlx-lm benchmark in the thread: **~90× speedup** for follow-up turns once cache hits become possible (36 s → 0.4 s, 99.4% reuse).
- `QwenLM/Qwen3/issues/1831` ("21-fix chat template for Qwen 3.5"): laundry list, includes fix #12 "enable_thinking applied to in-context assistant turns" — same bucket. **Closed as not planned** — upstream has not adopted these patches.
- `allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix` (community repo, ~71 commits): ships a patched `qwen3.5-enhanced.jinja` and tells operators to launch vLLM with `--chat-template qwen3.5-enhanced.jinja`. Does not touch `preserve_thinking` directly.
- HF blog `qwen-3-chat-template-deep-dive` ("The 4 Things Qwen-3's Chat Template Teaches Us") describes the "rolling checkpoint" design: in vanilla Qwen3 the template walks messages in reverse to find the last non-tool-call user turn (`last_query_index`) and only retains `<think>` for assistant messages *after* that index. Qwen3.6 then adds `preserve_thinking` as the explicit override.
- `vllm-project/vllm` issues for Qwen3.6: #40696 "Prefix caching completely ineffective for Mamba-hybrid models (Qwen3.5) when prompt < block_size (528 tokens)" and #36493 "hit rate of prefix caching in Qwen3.5 35BA3B is very low, always less than 0.1%" — these are **block-size and hybrid-cache issues, separate from the chat template asymmetry**, but they will compound with it. Worth knowing.

Sources:
- https://huggingface.co/Qwen/Qwen3.6-27B (model card)
- https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/chat_template.jinja
- https://github.com/QwenLM/Qwen3.6/issues/131
- https://github.com/QwenLM/Qwen3/issues/1826
- https://github.com/QwenLM/Qwen3/issues/1831
- https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix
- https://huggingface.co/blog/qwen-3-chat-template-deep-dive
- https://github.com/vllm-project/vllm/issues/40696, /36493

## 4. Alternative architectural patterns

- **Session-id / `conversation_id`–keyed cache.** Anthropic's public API offers explicit `cache_control` markers in messages. vLLM's analogue is `cache_salt` (RFC #16016), but it does *not* skip tokenization or fix mismatched prefixes — it only segments hashes. **No production server we found uses a `conversation_id` to bypass re-tokenization**; everyone re-applies the chat template every turn and relies on tokenized-prefix matching. The architectural reason is that requests are stateless in OpenAI-protocol servers; introducing a sticky session ID would break load-balancing and routing assumptions.
- **Bypass the chat template entirely.** This is the path taken by every framework that doesn't want to be at the mercy of upstream Jinja: pass tokenized prompt IDs straight to `/v1/completions` (not `/v1/chat/completions`) and append the previous turn's exact output IDs locally. We already do something like this internally in `server_tp.py`. The downside is the client has to know the special-token layout of every model it serves.
- **TGI (Hugging Face).** TGI exposes chat completions and applies the model's Jinja template. There is no documented Qwen3-specific patch and no `preserve_thinking`-equivalent server flag. Prefix-caching guidance is generic. TGI does not solve this either.
- **llama.cpp / mlx-lm.** Both rely on the same Jinja template and have been the loudest reporters of the bug (see Qwen3 #1826). llama.cpp users have shipped a `--chat-template` override file as the standard mitigation.

## 5. Recommendation

Our `preserve_thinking=True + trailing strip` fix is **the upstream-blessed path** for Qwen3.6 specifically. It is not a hack — the Qwen3.6 model card documents `preserve_thinking` exactly for this purpose, and the Jinja code (`(preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index)`) is built to make it work. The fact that we needed an extra trailing-strip of the active prompt's empty `<think>\n\n</think>` is a separate, well-known issue (Qwen3 #1826) and our strip is the same surgical move that mlx-lm users showed gave a 90× speedup.

**Robustness expectations.** Two Qwen3.6 template quirks we should be ready for:
1. `reasoning_content` must be populated on past assistant messages. If the client only round-trips `content` (with literal `<think>...</think>` text inside), the template *will* still parse the `<think>` block out — the Jinja extracts `reasoning_content` from `content` when the field is absent (`if '</think>' in content` branch). Our trailing-strip needs to be aware that the response content contains both reasoning and final answer separated by `</think>`. Test path: send turn 1, capture the model's raw text output verbatim, send turn 2 with `messages[-2] = {"role":"assistant","content":<raw>}` and check token-level prefix equality.
2. When tools are active, Qwen3.6 (like Qwen3) has a "rolling checkpoint" that **overrides** `preserve_thinking` for messages before the last non-tool-call user turn. If we are doing pure chat without tools, this is moot; if we add tool calls later, re-validate the 69/69 result.

**Long-term posture.** Keep using the chat template with `preserve_thinking=True` plus our trailing strip. Two reasons:
1. The bypass-the-template path forces us to track Qwen's special-token layout, the `<|im_start|>` separators, and any future template change ourselves — i.e. we'd take ownership of a moving target for a one-time benefit.
2. Upstream is actively maintaining this surface. Qwen3.6 added `preserve_thinking` *between* Qwen3 and Qwen3.6 in direct response to issues #131 / #1826. Sticking with the template means we ride future fixes for free.

One concrete follow-up: lift the trailing-strip logic into a single helper, gate it on `model_id.startswith("Qwen/Qwen3.6")`, and add a unit test that asserts `tokenize(render(messages_N)) == tokenize(render(messages_{N+1}))[:len(N)]` for a 3-turn fixture. That is the same invariant the mlx-lm benchmark measures, and it is what `cache_salt` provably *cannot* enforce — so it has to live in our rendering layer.

Sources (consolidated):
- https://github.com/vllm-project/vllm — `vllm/entrypoints/openai/chat_completion/serving.py:178-247`
- https://github.com/tenstorrent/vllm — branches and code search (no Qwen3 chat-template patch)
- https://github.com/sgl-project/sglang — `python/sglang/srt/parser/conversation.py`, Qwen3.6 cookbook
- https://huggingface.co/Qwen/Qwen3.6-27B (model card, `preserve_thinking` documented)
- https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/chat_template.jinja (the actual conditional)
- https://github.com/QwenLM/Qwen3.6/issues/131
- https://github.com/QwenLM/Qwen3/issues/1826 (mlx-lm 90× benchmark)
- https://github.com/QwenLM/Qwen3/issues/1831
- https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix
- https://huggingface.co/blog/qwen-3-chat-template-deep-dive
- https://github.com/vllm-project/vllm/issues/16016 (cache_salt RFC)
- https://github.com/vllm-project/vllm/issues/40696, /36493 (orthogonal Qwen3 APC bugs)
- https://github.com/open-webui/open-webui/discussions/23895 (preserve_thinking + reasoning_content round-trip issue)
