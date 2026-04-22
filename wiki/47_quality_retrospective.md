# Wiki 47: Quality Retrospective — What's Real, What's Not

## The Honest Assessment

### What's Proven (with evidence)

| Claim | Evidence |
|-------|----------|
| Implementation is correct | 10/10 token match vs numpy float32 reference (Llama-1B), 20/20 match, cosine 0.998-0.9998 |
| Short Q&A works perfectly | "The capital of France is Paris." — greedy, deterministic, correct |
| Factual explanations work | "Quantum computing is a type of computer..." — 22 tokens, coherent, stops at EOS |
| Structured output works | Numbered lists with markdown formatting (exercise benefits) |
| Precision is high | Cosine 0.999765 for 3B (28 layers × 3072 hidden through bf16) |

### What Doesn't Work Yet

| Issue | Root Cause | Not a TT-NN bug |
|-------|-----------|-----------------|
| Long creative text degenerates after ~30 tokens | 1B/3B model capacity limit | Confirmed: numpy reference produces same tokens |
| Haiku quality is mediocre ("Waves on the waves") | Small model, not poetry-specialized | Expected for 3B |
| Exercise list truncates ("type 2. Improved mental health.") | Context length + model capacity | Gets confused mid-sentence |

### What We Should Be Realistic About

1. **Token/sec numbers are only meaningful if the text is readable.** 140 tok/sec of garbage is worse than 33 tok/sec of coherent text. The 3B-Instruct model is our first *genuinely usable* deployment.

2. **3B models have limits.** They're great for classification, extraction, short Q&A, and simple instruction following. They're not great for multi-paragraph essays, creative writing, or complex reasoning. This is a model limitation, not hardware.

3. **For long-form coherent text at 3B scale**, we need either:
   - Better models (SmolLM3-3B, Phi-4-mini may be stronger)
   - Larger models (8B would require bf8 quantization to fit)
   - Or accept that short answers are the use case at this scale

## Reproducibility Checklist

For every experiment to be reproducible:
- [ ] Model ID is explicit (e.g., `unsloth/Llama-3.2-3B-Instruct`)
- [ ] Architecture constants are hardcoded, not derived from config
- [ ] Exact prompt and chat template are in the code
- [ ] Numpy reference produces verifiable ground truth
- [ ] Token-by-token comparison shows exact match count
- [ ] Random seed is fixed for sampled generation (currently NOT fixed — should add)
- [ ] tt-metal version is documented (should add `ttnn.__version__` printing)

## What Quality Experiments Taught Us

1. **Chat template precision matters.** A single extra BOS token (from `add_special_tokens=True`) completely breaks instruction following. The model saw garbled prompts and produced garbled output.

2. **Cosine similarity alone is insufficient.** Cosine 0.99 at prefill doesn't guarantee correct decode. Token-by-token comparison over multiple steps is the real test.

3. **Base models vs instruct models are fundamentally different.** Our first 4 models (base) always produced degenerate text — this was expected but made it hard to tell if precision was the issue. Should have started with instruct models.

4. **Model size vs decode quality is not linear.** 1B-Instruct degenerates at ~30 tokens, 3B-Instruct sustains ~40-100 tokens for factual content. 8B would likely sustain 200+.

## Updated Quality Status

| Model | Cosine | Token Match | Short Q&A | Long-form | Status |
|-------|--------|-------------|-----------|-----------|--------|
| Qwen2.5-0.5B (base) | N/A | N/A | N/A | Degenerate | Base model, expected |
| Qwen2.5-0.5B-Instruct | 0.999 | Top-1 True | Weak | Degenerate | Too small |
| Llama-3.2-1B (base) | N/A | N/A | N/A | Degenerate | Base model, expected |
| Llama-3.2-1B-Instruct | 0.998 | 20/20 | Perfect | Degenerates ~30 tok | Working but limited |
| Llama-3.2-3B (base) | N/A | N/A | N/A | Degenerate | Base model, expected |
| **Llama-3.2-3B-Instruct** | **0.9998** | **10/10** | **Perfect** | **OK for factual** | **First usable model** |
| Qwen3-0.6B | N/A | Pending | Pending | Pending | Needs validation |
| SmolLM3-3B | N/A | Pending | Pending | Pending | Needs validation |

## Next Steps (Priority)

1. **Fix reproducibility**: Add `np.random.seed()`, print `ttnn.__version__`, document tt-metal version
2. **Validate remaining models**: Qwen3-0.6B-Instruct, SmolLM3-3B-Instruct (if available)
3. **Research**: What 3B models are best at sustained generation? (Phi-4-mini? SmolLM3?)
4. **Long-form quality**: Try repetition penalty, nucleus sampling (top-p), and longer prompts with explicit instructions to write more
