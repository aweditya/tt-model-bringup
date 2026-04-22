# Long-Form Text Generation: Why Models Degenerate and How to Fix It

Research compiled for TT-XLA project (CS440LX). We're running transformer models on
Tenstorrent Blackhole hardware and seeing 3B models degenerate after ~30 tokens of
greedy decoding. This document covers why that happens and what production systems
actually do about it.

---

## 1. Why Small Models Degenerate During Long-Form Generation

### The Repetition/Degeneration Problem

Autoregressive language models generate text one token at a time, conditioning on all
previously generated tokens. This creates a self-reinforcing feedback loop: once a
high-probability token is selected, it enters the context and *increases* the
probability of being selected again. The result is repetitive, degenerate text —
loops of the same phrase or word repeated indefinitely.

This is not a bug in any specific model. It's a fundamental property of maximum
likelihood decoding applied to autoregressive models. The seminal paper "The Curious
Case of Neural Text Degeneration" (Holtzman et al., 2020) demonstrated that human
text occupies a surprisingly *low*-probability region of the model's distribution.
Maximizing probability produces text that is statistically likely but humanly
unnatural.

### Why Greedy Decoding Specifically Causes This

Greedy decoding selects the single highest-probability token at every step. This is
the worst possible strategy for open-ended generation because:

1. **No exploration**: The model never considers alternative continuations. Once it
   enters a high-probability attractor (like "I don't know what to say"), it stays
   there.
2. **Self-amplification**: Common words/phrases ("the", "of", "I") have naturally
   high probabilities. Greedy selection biases toward them, which biases the context
   toward patterns containing them, which further increases their probability.
3. **No recovery mechanism**: Unlike sampling, there's no randomness to escape a
   repetitive loop once entered.

**Key insight**: Greedy decoding is fine for short, constrained outputs (single-token
classification, short factual answers). It is *catastrophically bad* for open-ended
generation beyond ~10-30 tokens.

### How Model Size Affects Coherence Length

Within a model family, larger models degenerate less:

- **< 1B parameters**: Degenerate almost immediately under greedy decoding. Limited
  capacity means the learned distribution is "peakier" — fewer viable continuations.
- **1B-3B parameters**: Can produce coherent greedy output for ~10-30 tokens before
  looping. Enough capacity for basic patterns but not enough to maintain long-range
  coherence.
- **7B parameters**: Substantially better. Can sometimes produce coherent greedy
  output for paragraphs, though still degenerates on longer sequences.
- **13B+ parameters**: Greedy decoding remains problematic but the degeneration
  onset is much later. The distribution is smoother and more "spread out."

Research on the "repeat curse" (arXiv:2504.14218) shows that distilled/smaller models
loop far more than their teacher models, even when the teacher rarely loops. This
points to **imperfect learning** as a primary cause — smaller models don't fully
capture the distributional nuances needed to avoid repetitive attractors.

**Bottom line for our 3B model**: Degeneration at ~30 tokens under greedy decoding is
*completely expected behavior*. The fix is not a better model — it's better decoding.

---

## 2. Sampling Strategies That Actually Work in Production

### Temperature Scaling

**What it does**: Divides logits by temperature T before softmax. T < 1 sharpens the
distribution (more deterministic), T > 1 flattens it (more random).

```
p_i = softmax(logit_i / T)
```

**Production values**:
- T = 0.0: Equivalent to greedy decoding (argmax)
- T = 0.3-0.5: Conservative, good for factual Q&A
- T = 0.7: General-purpose sweet spot (used by many chatbots)
- T = 0.8: Ollama's default
- T = 1.0: Model's raw distribution (vLLM default)
- T > 1.2: Too random for most uses, incoherent output

**Implementation**: Single division on the logits tensor. Trivial to implement.

### Top-k Sampling

**What it does**: Keeps only the k highest-probability tokens, redistributes
probability mass among them.

**Production values**:
- k = 40: Common default (used in early GPT-2 demos)
- k = 50: Another common choice
- k = 0 or disabled: Let other methods (top-p) handle truncation

**Drawback**: Fixed k is too rigid. For a confident prediction (one token at 95%),
k=40 introduces 39 near-zero-probability distractors. For an uncertain prediction
(flat distribution), k=40 might cut off viable options.

### Top-p (Nucleus) Sampling

**What it does**: Keeps the smallest set of tokens whose cumulative probability
exceeds p. This *dynamically* adjusts the number of candidates based on model
confidence.

```python
sorted_probs = sort_descending(probs)
cumsum = cumulative_sum(sorted_probs)
mask = cumsum <= p  # keep tokens until we hit p
# sample from masked distribution
```

**Production values**:
- p = 0.9: Most common default (OpenAI, many services)
- p = 0.95: Slightly more diverse
- p = 1.0: No filtering (vLLM default)

**Standard practice**: Use temperature + top-p together, but don't combine top-p
with top-k (they serve similar purposes, and the interaction is confusing).

### Repetition Penalty (HuggingFace/CTRL-style)

**What it does**: For each token that has already appeared in the context, modifies
its logit:
- If logit > 0: divide by penalty
- If logit < 0: multiply by penalty

Both operations reduce the token's probability.

```python
for token_id in previously_seen_tokens:
    if logits[token_id] > 0:
        logits[token_id] /= repetition_penalty
    else:
        logits[token_id] *= repetition_penalty
```

**Production values**:
- 1.0: No penalty (default in vLLM and HuggingFace)
- 1.1: Ollama's default — mild but effective
- 1.1-1.3: Recommended range for most models
- 1.5+: Aggressive, can cause incoherent output (model avoids necessary words)

**Implementation note**: This scans all previously generated token IDs. The "window"
can be limited (Ollama uses `repeat_last_n=64` by default — only penalizes tokens
from the last 64 positions).

### Min-p Sampling (Newer Technique — ICLR 2025)

**What it does**: Sets a dynamic probability floor relative to the top token's
probability. Any token with probability < (min_p * top_token_probability) is
discarded.

```python
top_prob = max(probs)
threshold = min_p * top_prob
mask = probs >= threshold
# sample from masked distribution
```

**Why it's better than top-p**: When the model is confident (top token at 90%),
min-p aggressively prunes (only tokens above 4.5% survive with min_p=0.05). When the
model is uncertain (top token at 10%), min-p is lenient (tokens above 0.5% survive).
Top-p can't do this — it always keeps tokens summing to the same cumulative mass.

**Production values**:
- min_p = 0.05-0.1: Recommended range
- Combine with temperature = 0.7-1.0

**Status**: Adopted in HuggingFace Transformers, vLLM, llama.cpp, Ollama. This is
the emerging best practice for open-source model serving as of 2026.

### Frequency and Presence Penalties (OpenAI-style)

These are *additive* penalties applied to logits, unlike HuggingFace's multiplicative
repetition penalty.

**Frequency penalty**: Subtracts `alpha_freq * count(token)` from the logit. Scales
with how often a token has appeared.

**Presence penalty**: Subtracts `alpha_pres * 1(token has appeared)` from the logit.
Binary — applied once regardless of count.

```python
logits[token_id] -= frequency_penalty * token_count[token_id]
logits[token_id] -= presence_penalty * (1 if token_count[token_id] > 0 else 0)
```

**Production values** (OpenAI API):
- Default: 0.0 for both (no penalty)
- Range: -2.0 to 2.0
- Reasonable values: 0.1 to 1.0 for either
- Frequency penalty: Prevents exact word repetition
- Presence penalty: Encourages topic diversity (won't revisit mentioned concepts)

### What Parameters Do People Actually Use?

| Service | Temperature | Top-p | Repetition/Penalties | Other |
|---------|-------------|-------|---------------------|-------|
| **vLLM** (defaults) | 1.0 | 1.0 | rep_penalty=1.0 | Essentially raw model output |
| **Ollama** (defaults) | 0.8 | 0.9 | repeat_penalty=1.1, last_n=64 | Most opinionated defaults |
| **TGI** (defaults) | None (greedy) | None | None | do_sample=false by default |
| **OpenAI API** | 1.0 | 1.0 | freq=0, pres=0 | Users expected to configure |
| **llama.cpp community** | 0.7-1.0 | disabled | min_p=0.05-0.1 | Temp + min-p is the meta |

**The practical recommendation for our 3B model on Blackhole**:

```
temperature = 0.7
min_p = 0.05          # or top_p = 0.9 if min-p is hard to implement
repetition_penalty = 1.1
repeat_last_n = 64
```

This should eliminate the degeneration we're seeing at ~30 tokens. Start with
temperature alone — even T=0.7 will dramatically improve output over greedy.

---

## 3. Architectural Techniques for Better Long-Form Generation

### Sliding Window Attention

Not directly a generation-quality technique, but relevant: models like Mistral use
sliding window attention (window size 4096) to handle long contexts efficiently.
This doesn't fix degeneration but enables longer context windows without quadratic
memory growth, which helps the model "see" more of its previous output.

### Repeated N-gram Blocking

**What it does**: Hard-blocks any token that would create a repeated n-gram. Sets the
logit of the offending token to negative infinity.

```python
# no_repeat_ngram_size = 3
# If "the cat sat" already appeared, block any token that would
# create "the cat sat" again when preceded by "the cat"
```

**When to use it**: Primarily used with beam search for summarization (BART uses this
as standard). For sampling-based generation, soft penalties (repetition_penalty) are
usually better because hard blocking can prevent legitimate repeated phrases
("thank you", "on the other hand").

**Production values**: no_repeat_ngram_size = 3 or 4 is standard.

### Contrastive Search

**What it does**: At each step, selects a token that balances two objectives:
1. High model confidence (probability)
2. Low similarity to previously generated tokens (in the model's hidden representation space)

```
score(v) = (1 - alpha) * p(v) - alpha * max_similarity(v, context_tokens)
```

**Key parameter**: alpha controls the tradeoff. alpha = 0 is greedy, alpha = 1
ignores probability entirely.

**Results**: Achieves human-level text quality on 12/16 evaluated languages in the
original paper (Su et al., 2022), without any additional training. Available in
HuggingFace Transformers via `model.generate(penalty_alpha=0.6, top_k=4)`.

**Drawback**: Requires access to hidden states at each step, which adds latency.
May be impractical for our hardware setup where we want minimal round-trips.

### Beam Search vs Sampling

**Beam search**: Maintains k candidate sequences, expands each, keeps top-k at each
step. Good for constrained tasks (translation, summarization) where there's a
"correct" output. *Terrible* for open-ended generation — produces even more
repetitive text than greedy decoding because it optimizes for joint probability.

**Sampling**: Better for open-ended generation. The consensus since ~2020 is clear:
**use sampling (not beam search) for open-ended text generation**.

**When to use beam search**: Translation, summarization, constrained outputs. Always
pair with n-gram blocking.

### Do Instruct-Tuned Models Degenerate Less?

**Yes, significantly.** Instruct-tuned and RLHF'd models degenerate less than base
models for several reasons:

1. **Training signal against repetition**: Human raters in RLHF penalize repetitive
   outputs. The reward model learns to downweight repetitive text, and the policy
   learns to avoid it.
2. **Instruction following creates structure**: The model learns to produce structured
   responses (intro, body, conclusion) rather than unconstrained continuation, which
   naturally reduces the chance of entering repetitive loops.
3. **Shorter expected outputs**: Instruct models tend to produce concise answers and
   stop, rather than generating indefinitely.

**However**: Instruct tuning doesn't eliminate the problem. Small instruct models
(1B-3B) still degenerate under greedy decoding, just later than their base
counterparts. Sampling is still necessary.

**Recommendation**: Always use instruct/chat variants of models for our use case.
Qwen-3B-Instruct >> Qwen-3B-Base for generation quality.

---

## 4. What Model Sizes Are Needed for Different Tasks?

These are rough guidelines based on community experience and benchmarks. "Minimum
viable" means the model can produce *usable* output with proper sampling, not that
it matches GPT-4.

### Short Q&A (1-3 sentences)
- **Minimum viable**: 1B-3B with instruct tuning
- **Comfortable**: 3B-7B
- **Notes**: Even small models handle factual retrieval and short answers well.
  This is where our 3B Qwen shines.

### Multi-Paragraph Explanations (100-500 tokens)
- **Minimum viable**: 3B with good sampling (temp=0.7, rep_penalty=1.1)
- **Comfortable**: 7B
- **Notes**: 3B can do this but needs sampling to avoid degeneration. Without
  sampling, expect failure at ~30 tokens (exactly what we're seeing). 7B is the
  practical sweet spot.

### Creative Writing (500+ tokens)
- **Minimum viable**: 7B instruct
- **Comfortable**: 13B+
- **Notes**: Creative writing requires maintaining plot, character consistency, and
  variety — capabilities that scale with model size. 3B models produce generic,
  repetitive creative text even with good sampling.

### Code Generation
- **Minimum viable**: 3B (code-specialized models like Qwen-Coder)
- **Comfortable**: 7B-13B
- **Notes**: Code is more structured than natural language, so smaller models handle
  it better. Code-specialized 3B models can generate working functions. Complex
  multi-file generation needs 13B+.

### Summary Table

| Task | Min. Size | Sweet Spot | Notes |
|------|-----------|------------|-------|
| Short Q&A | 1B | 3B | Works even with greedy |
| Multi-paragraph | 3B* | 7B | *Requires sampling |
| Creative writing | 7B | 13B | Size matters a lot here |
| Code generation | 3B (specialized) | 7B | Code models punch above weight |
| Complex reasoning | 7B | 13B-30B | Chain-of-thought helps |

---

## 5. Production Best Practices

### Default Parameters by Framework

**Ollama** (most conservative/user-friendly defaults):
```
temperature: 0.8
top_k: 40
top_p: 0.9
repeat_penalty: 1.1
repeat_last_n: 64
num_predict: 128  (max tokens)
```

**vLLM** (neutral defaults — expects user configuration):
```
temperature: 1.0
top_p: 1.0
top_k: -1 (disabled)
repetition_penalty: 1.0
max_tokens: 16
min_p: 0.0
```

**TGI** (greedy by default):
```
do_sample: false
max_new_tokens: 672
temperature: None (greedy)
top_p: None
repetition_penalty: None
```

**OpenAI API** (neutral defaults):
```
temperature: 1.0
top_p: 1.0
frequency_penalty: 0.0
presence_penalty: 0.0
max_tokens: model-dependent
```

### Stopping Criteria

Production systems use multiple stopping conditions simultaneously:

1. **EOS token**: The model's trained end-of-sequence token. Primary signal.
2. **Max tokens**: Hard limit to prevent runaway generation. Set as a safety net.
   - For chat: 256-2048 tokens typical
   - For long-form: 2048-4096 tokens
3. **Stop sequences**: User-specified strings that halt generation. Common choices:
   - `"\n\nHuman:"` or `"\n\nUser:"` for chat formats
   - `"```"` to end code blocks
   - `"\n\n"` for single-paragraph responses
   - `"<|endoftext|>"`, `"<|im_end|>"` for chat templates
4. **Repetition detection**: Some systems detect when the model is looping and
   forcibly stop. This is a last resort — better to prevent loops with sampling.

### How People Evaluate Generation Quality

1. **Perplexity**: Measures how "surprised" the model is by text. Lower = more
   predictable. Useful for comparing models, not for evaluating individual outputs.
2. **MAUVE score**: Measures distributional similarity between generated text and
   human text. The gold standard for evaluating open-ended generation.
3. **Repetition metrics**: Count repeated n-grams in generated text. High rep-3
   (fraction of repeated trigrams) indicates degeneration.
4. **Human evaluation**: Still the gold standard. Rate for fluency, coherence,
   relevance, and interestingness.
5. **Cosine similarity to reference**: What we've been using for correctness
   validation. Good for verifying model behavior, not for evaluating generation
   quality.

### Quick-Start Recipe for Our Blackhole Setup

For our Qwen 3B model running on Tenstorrent Blackhole, here's the minimum viable
sampling implementation:

**Priority 1 — Temperature (implement first)**:
```python
# After getting logits from the model
logits = logits / temperature  # temperature = 0.7

# Convert to probabilities
probs = softmax(logits)

# Sample from the distribution instead of argmax
next_token = multinomial_sample(probs)
```

This alone should eliminate the ~30 token degeneration. It's one line of math on
the logits tensor.

**Priority 2 — Top-p filtering**:
```python
sorted_probs, sorted_indices = sort(probs, descending=True)
cumulative_probs = cumsum(sorted_probs)
# Remove tokens with cumulative probability above the threshold
mask = cumulative_probs > top_p  # top_p = 0.9
# Shift mask right so the first token above threshold is kept
mask[..., 1:] = mask[..., :-1].clone()
mask[..., 0] = False
sorted_probs[mask] = 0.0
# Renormalize and sample
probs = sorted_probs / sorted_probs.sum()
next_token = multinomial_sample(probs)
```

**Priority 3 — Repetition penalty**:
```python
# Before temperature scaling
for token_id in generated_token_ids[-64:]:  # last 64 tokens
    if logits[token_id] > 0:
        logits[token_id] /= 1.1  # repetition_penalty = 1.1
    else:
        logits[token_id] *= 1.1
```

**Where does this run?** The sampling logic runs on CPU (or could run on device).
It only touches the final logits vector (vocab_size floats). The expensive part
(the transformer forward pass) stays on the Blackhole device. Sampling adds
negligible overhead.

---

## Key Takeaways

1. **Our 3B model degenerating at ~30 tokens with greedy decoding is expected.**
   This is not a bug in our implementation or hardware. Every 3B model does this.

2. **Temperature sampling is the single most important fix.** Just dividing logits
   by 0.7 before softmax and sampling (instead of argmax) will dramatically improve
   output quality.

3. **The production stack is: temperature (0.7) + top-p (0.9) OR min-p (0.05) +
   repetition penalty (1.1).** This is what Ollama, llama.cpp, and most local LLM
   tools converge on.

4. **For multi-paragraph coherent output from a 3B model, sampling is mandatory.**
   With proper sampling, 3B instruct models can produce usable multi-paragraph text.
   Without it, they can't.

5. **7B is the sweet spot for general-purpose generation quality.** If we can run
   Qwen-7B on Blackhole, it would substantially improve output quality for longer
   generation tasks. But fixing sampling on 3B comes first — it's lower effort and
   higher impact.

---

## Sources

- [Understanding Text Degeneration During Decoding (Medium)](https://medium.com/@ikim1994914/understanding-the-modern-llm-part-5-understanding-text-degeneration-during-decoding-and-methods-966a4d33e9c8)
- [Understanding the Repeat Curse in LLMs (arXiv)](https://arxiv.org/html/2504.14218v1)
- [Greedy Decoding (Aussie AI)](https://www.aussieai.com/research/greedy-decoding)
- [Decoding Strategies: How LLMs Choose The Next Word (AssemblyAI)](https://www.assemblyai.com/blog/decoding-strategies-how-llms-choose-the-next-word)
- [LLM Sampling Parameters Explained (Let's Data Science)](https://letsdatascience.com/blog/llm-sampling-temperature-top-k-top-p-and-min-p-explained)
- [LLM Temperature, Top-P, Top-K Guide (Amit Ray)](https://amitray.com/llm-parameters-temperature-top-p-top-k-guide/)
- [Min-p Sampling for LLMs (Thoughtworks)](https://www.thoughtworks.com/insights/blog/generative-ai/Min-p-sampling-for-LLMs)
- [Min-p Sampling for Creative and Coherent LLM Outputs (ICLR 2025, arXiv)](https://arxiv.org/abs/2407.01082)
- [vLLM Sampling Parameters](https://docs.vllm.ai/en/v0.8.4/api/inference_params.html)
- [Ollama Modelfile Reference](https://docs.ollama.com/modelfile)
- [OpenAI Frequency and Presence Penalties (Community)](https://community.openai.com/t/difference-between-frequency-and-presence-penalties/2777)
- [HuggingFace Repetition Penalty (Forum)](https://discuss.huggingface.co/t/transformers-repetition-penalty-parameter/43638)
- [Contrastive Search for Neural Text Generation (arXiv)](https://arxiv.org/abs/2210.14140)
- [Generating Human-level Text with Contrastive Search (HuggingFace Blog)](https://huggingface.co/blog/introducing-csearch)
- [Repetition Penalties: Preventing Loops (Brenndoerfer)](https://mbrenndoerfer.com/writing/repetition-penalties-language-model-generation)
- [How to Generate Text (HuggingFace Blog)](https://huggingface.co/blog/how-to-generate)
- [LLM Model Parameters Guide (Local AI Zone)](https://local-ai-zone.github.io/guides/what-is-ai-model-3b-7b-30b-parameters-guide-2025.html)
- [LLM Model Size Comparison (Label Your Data)](https://labelyourdata.com/articles/llm-fine-tuning/llm-model-size)
- [Vendor-Recommended LLM Parameter Quick Reference (Muxup)](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference)
- [LLM Sampling Parameters Guide (smcleod.net)](https://smcleod.net/2025/04/llm-sampling-parameters-guide/)
- [TGI Default Parameters (GitHub Issue #2978)](https://github.com/huggingface/text-generation-inference/issues/2978)
