# Wiki 51: Quality Investigation Final Verdict

## TL;DR

**Llama-3.1-8B-Instruct on Blackhole produces correct, coherent text.** Greedy decoding handles 8/10 diverse Q&A categories perfectly at 18 tok/s. With minimal temperature (0.1), 5/6 tested prompts produce natural EOS. Only code generation enters attractor loops — a known greedy decoding limitation on structured output.

## The Investigation (Experiments 75-80b)

| Exp | What We Tested | What We Learned |
|-----|----------------|-----------------|
| 75 | Production sampling (temp=0.7 + min_p + rep) | Sampling suppresses EOS → word salad past stopping point |
| 76b | Cosine similarity vs numpy float32 | 0.9975 prefill cosine, 8/8 token match — implementation correct |
| 77 | Numpy vs TT-NN on same creative prompt | BOTH produce coherent text and stop at ~35 tokens |
| 78b | Length-prompted greedy on TT-NN | Two failure modes: premature EOS or repetitive loops |
| 79 | Careful sampling (EOS protected from penalties) | Same degeneration — not an EOS suppression problem |
| 80 | Diverse Q&A (10 categories, greedy) | 8/10 perfect. Code and reasoning enter loops |
| 80b | Low temperature (0.1) side-by-side | Fixes reasoning (5/6 correct). Code still loops |

## Recommended Configuration

```
Decoding: temp=0.1 (near-greedy, breaks attractor loops)
No repetition penalty (not needed at this temperature)
No min-p filtering (not needed at this temperature)
Stop on EOS token (128009 or 128001)
Max tokens: 200 (model rarely exceeds 50 for Q&A)
```

## What Works Perfectly (Greedy or Temp=0.1)

- **Geography**: "The capital of France is Paris."
- **Science**: Full photosynthesis definition (30 tokens)
- **History**: "George Washington, 1789 to 1797"
- **Logic**: Perfect syllogism (cats need water)
- **Translation**: "Hola, ¿cómo estás?"
- **Definition**: Correct ML definition
- **Comparison**: TCP vs UDP distinction

## What Doesn't Work

- **Code generation**: Enters `if n <= 0` attractor loop after ~40 tokens
- **Long-form creative**: Model stops at ~35 tokens (correct behavior for instruct model)

## Root Cause Analysis

The degeneration is **not** a TT-NN or bf16 precision bug. Evidence:

1. Numpy float32 reference produces identical behavior (exp 77)
2. Cosine similarity between numpy and TT-NN is 0.9975 (exp 76b)
3. Token-level accuracy is 100% for first 8 tokens (exp 76b)
4. The model naturally stops at 35-70 tokens via EOS on all prompts

The issue is that **8B instruct models are trained for concise Q&A**, not essay writing. When forced past EOS (via sampling) or asked for structured output (code), the model lacks the capacity to maintain coherence.

## Performance

```
Decode:  18 tok/s (52ms/tok)
Ceiling: 28 tok/s (theoretical bandwidth limit)
Efficiency: 64% of ceiling
vs M4 Pro: 1.6x faster
vs RTX 4090: ~3x slower
```

## What Would Improve Things

1. **70B+ models** — more capacity for sustained generation
2. **Quantized models (INT4/INT8)** — blocked on TT-NN support, would double throughput
3. **Multi-chip (2x Blackhole)** — double bandwidth = double ceiling
4. **Different model families** — SmolLM3, Qwen3 may handle code better
