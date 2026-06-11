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
- [x] Random seed is fixed for sampled generation (`np.random.seed(42)` since exp 72)
- [x] tt-metal version is documented (`ttnn.__version__` printed since exp 75)

## What Quality Experiments Taught Us

1. **Chat template precision matters.** A single extra BOS token (from `add_special_tokens=True`) completely breaks instruction following. The model saw garbled prompts and produced garbled output.

2. **Cosine similarity alone is insufficient.** Cosine 0.99 at prefill doesn't guarantee correct decode. Token-by-token comparison over multiple steps is the real test.

3. **Base models vs instruct models are fundamentally different.** Our first 4 models (base) always produced degenerate text — this was expected but made it hard to tell if precision was the issue. Should have started with instruct models.

4. **Model size vs decode quality is not linear.** 1B-Instruct degenerates at ~30 tokens, 3B-Instruct sustains ~40-100 tokens for factual content. 8B with greedy degenerates at ~50 tokens, but with production sampling (temp=0.7 + min_p=0.05 + rep=1.1) produces coherent 70-token factual responses with EOS. However, creative/long-form still degenerates — **investigating whether this is a precision bug or model behavior** (exp 76).

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

## Sampling Strategy Results (Exp 72)

Tested 5 strategies on Llama-3.2-3B-Instruct creative writing prompt:

| Strategy | Tokens Before Degeneration | Total Tokens | Quality |
|----------|---------------------------|--------------|---------|
| Greedy | ~20 | 43 (EOS) | Good start, collapses |
| Top-k (temp=0.6) | ~20 | 26 (EOS) | Similar |
| Top-p (p=0.9) | ~20 | 32 (EOS) | Similar |
| Top-p + RepPenalty 1.2 | ~15 | 23 (EOS) | Stops even earlier |
| Top-p + RepPenalty 1.5 | ~20 | 118 | Forced diversity = incoherent |

**Conclusion: Sampling strategies cannot fix the fundamental model capacity limit.**
The 3B model has enough knowledge to start well but runs out of coherence after ~20-30 tokens regardless of how we sample. Aggressive repetition penalty forces longer output but the content is nonsensical.

## What 3B Models ARE Good For

1. Short Q&A ("The capital of France is Paris.")
2. Classification and extraction
3. Code snippets and simple functions
4. 1-2 sentence summaries
5. Structured output (markdown headers, numbered lists — the structure is right, content degenerates)

## What Needs 8B+

1. Multi-paragraph explanations
2. Creative writing (stories, poetry)
3. Detailed technical descriptions
4. Sustained reasoning chains

## Next Steps (Priority)

1. ~~Fix reproducibility~~: Done (seed=42, ttnn version printing) in exp 72
2. **SmolLM3-3B-Instruct**: May be better at sustained generation than Llama-3.2-3B
3. **Research 8B on Blackhole**: bf8 weights = ~8GB, but KV cache needs space too
4. **Validate remaining base models**: Qwen3, SmolLM3 cosine checks
5. **Accept 3B limits**: Focus on use cases where short answers are sufficient
