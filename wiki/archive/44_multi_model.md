# Wiki 44: Multi-Model Results — 4 Models on Blackhole

## Question
How well does our TT-NN infrastructure generalize across different model architectures?

## Answer: 4 models running, near-linear parameter scaling

| Model | Params | Layers | Hidden | Q/KV Heads | head_dim | ms/tok | tok/sec |
|-------|--------|--------|--------|------------|----------|--------|---------|
| Qwen2.5-0.5B | 0.5B | 24 | 896 | 14/2 | 64 | 7.1 | 140 |
| Qwen3-0.6B | 0.6B | 28 | 1024 | 16/8 | 128 | 13.2 | 76 |
| Llama-3.2-1B | 1.24B | 16 | 2048 | 32/8 | 64 | 12.8 | 78 |
| Llama-3.2-3B | 3.2B | 28 | 3072 | 24/8 | 128 | 29.7 | 34 |

## Key Findings

### 1. Speed depends on architecture, not just parameter count

Qwen3-0.6B (0.6B params) is *slower* than Llama-3.2-1B (1.24B params) despite having half the parameters. The bottleneck is:
- **head_dim=128** doubles the rotation matrix matmul size
- **8 KV heads** requires split SDPA (2× SDPA calls per layer)
- **28 layers** (vs 16 for Llama) means more sequential ops

### 2. SDPA flash decode has a power-of-2 KV head bug

The `sdpa_flash_decode` kernel on Blackhole only compiles when KV heads is a power of 2, but also fails when the total (Q heads × KV heads) creates too much register pressure.

**Working**: 8Q/2KV, 16Q/4KV, 14Q/2KV
**Failing**: 32Q/8KV, 24Q/6KV, 14Q/3KV

**Workaround**: Split into groups with 4 KV heads each:
- 32Q/8KV → 2×(16Q/4KV)  
- 24Q/8KV → 2×(12Q/4KV)
- 16Q/8KV → 2×(8Q/4KV)

### 3. Architecture-specific adaptations needed

| Feature | Qwen2.5 | Qwen3 | Llama |
|---------|---------|-------|-------|
| Biases | Q/K/V | None | None |
| RoPE format | Half | Half | Interleaved |
| QK-Norm | No | Yes | No |
| Split SDPA | No (2 KV) | Yes (8 KV) | Yes (8 KV) |
| head_dim | 64 | 128 | 64/128 |
| Tied embeddings | No | Yes | Yes |

### 4. Near-linear parameter scaling

Comparing Llama-3.2-1B → 3B:
- 2.5× parameters → 2.3× latency increase
- The sub-linear scaling suggests larger matmuls are more efficient on the 120-core mesh

### 5. Continuous batching works

Exp 65 proved continuous batching with 8 batch slots:
- 1,042 tok/sec decode (matches static batch)
- 24 diverse prompts served through slot cycling
- Position=-1 skips SDPA compute (zero overhead for empty slots)

## Performance Timeline Update

```
exp 41:     582ms/tok    1.7 tok/s    Qwen2.5 full recompute
exp 60:     7.1ms/tok  140.4 tok/s    Qwen2.5 traced + native RoPE  ← BEST SINGLE
exp 59:    13.2ms/step 4,867 tok/s    Qwen2.5 batch=64              ← PEAK AGGREGATE
exp 64:    12.8ms/tok   78.2 tok/s    Llama-3.2-1B
exp 65:     1,042 tok/s decode       Continuous batching (8 slots)
exp 66:    13.2ms/tok   76.0 tok/s    Qwen3-0.6B (QK-Norm)
exp 67:    29.7ms/tok   33.7 tok/s    Llama-3.2-3B                  ← LARGEST MODEL
```

## What's Next

1. **bf8 weights on larger models** — 3B at bf8 = 3.2 GB instead of 6.4 GB
2. **SmolLM3-3B** — NoPE layers (skip RoPE every 4th layer)
3. **Phi-4-mini (3.8B)** — fractional RoPE (75% of head_dim)
4. **Batch scaling on 3B** — test throughput scaling at larger model size
