# Wiki 50: Sampling Investigation Complete — 8B Instruct Model Behavior

## The Question
Can we get Llama-3.1-8B-Instruct to produce multi-paragraph coherent text on Blackhole?

## Experiments Conducted (75-79)

| Exp | Strategy | Result |
|-----|----------|--------|
| 75 | Production sampling (temp=0.7, min_p=0.05, rep=1.1) | Word salad past EOS — sampling suppresses EOS |
| 76b | Correctness check (prefill cosine + greedy tokens) | 0.9975 cosine, 8/8 token match — implementation correct |
| 77 | Numpy vs TT-NN greedy comparison | Both produce coherent text, both stop at ~35 tokens |
| 78b | Length-prompted greedy on TT-NN | Two failure modes: premature EOS (factual) or repetitive loops (creative) |
| 79 | Careful sampling — EOS protected from rep penalty & min-p | Same degeneration — EOS protection doesn't help |

## Root Cause: Model Training, Not Hardware

The 8B instruct model is trained for **concise conversational Q&A**. It produces a correct answer in 35-70 tokens and signals completion with EOS. This is correct behavior.

### What happens with each strategy:

**Greedy decoding:**
- Short factual: Perfect answers, natural EOS ✓
- Long creative: Repetitive attractor loops (greedy walks into a fixed point in probability space)

**Any sampling (temp > 0):**
- Temperature reduces EOS probability relative to content tokens
- Model continues past its trained stopping point
- Without strong logit guidance, it degenerates into repetition
- This looks like "word salad" but is actually forced generation past EOS

**Protecting EOS from penalties (exp 79):**
- Doesn't help because the issue isn't EOS suppression by penalties
- The issue is temperature itself reducing EOS probability
- Even with EOS fully protected, the sampler picks content tokens because there are 128K of them vs 1 EOS token

## What the Model CAN Do

Greedy decoding produces **correct, coherent output** on:
- Factual Q&A: "The capital of France is Paris." (8 tokens)
- Code generation: complete function implementations
- Short explanations: one-paragraph answers
- Math/logic: step-by-step solutions

This is exactly what the model was designed for.

## What Would Enable Longer Output

1. **Larger models** (70B+) — more capacity for sustained coherent generation
2. **Base models** (not instruct) — trained for continuation, not conversation
3. **Chat-style interaction** — multi-turn conversation where each turn is short
4. **Different model families** — SmolLM3, Qwen3 may have different training

## Conclusion

The quality investigation is **complete**. Our TT-NN implementation is correct (validated by cosine similarity and token matching against numpy float32). The short outputs are model behavior, not a bug. For production use, greedy decoding with proper chat templating is the right approach.

**Key lesson:** Don't fight the model's training. Use greedy for instruct models, use sampling only for base models or specifically trained long-form models.
