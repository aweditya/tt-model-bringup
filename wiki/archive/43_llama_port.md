# Wiki 43: Llama-3.2-1B Port — Architecture Generality

## Question
Can our TT-NN infrastructure run a different model architecture with zero new ops?

## Answer: Yes — 78 tok/sec on Blackhole

Llama-3.2-1B has significantly different architecture from Qwen2.5-0.5B but uses the exact same TT-NN operations.

### Architecture Comparison

| Property | Qwen2.5-0.5B | Llama-3.2-1B |
|----------|-------------|--------------|
| Parameters | ~0.5B | ~1.24B |
| Layers | 24 | 16 |
| Hidden dim | 896 | 2048 |
| Q heads | 14 | 32 |
| KV heads | 2 | 8 |
| GQA ratio | 7:1 | 4:1 |
| Head dim | 64 | 64 |
| Intermediate | 4864 | 8192 |
| RoPE format | Half | Interleaved |
| Biases | Q/K/V have bias | No biases |
| rope_theta | 1,000,000 | 500,000 |
| Vocab | 151,936 | 128,256 |

### Bug: sdpa_flash_decode Kernel Compilation

The `scaled_dot_product_attention_decode` kernel on Blackhole only compiles when the KV head count is a power of 2. With 8 KV heads, the SFPU compiler runs out of vector registers in the exponential function.

**Tested:**
- 8Q/2KV: OK
- 16Q/4KV: OK
- 14Q/3KV: FAIL
- 20Q/5KV: FAIL
- 24Q/6KV: FAIL
- 28Q/7KV: FAIL
- 32Q/8KV: FAIL

**Workaround:** Split into two groups of (16Q, 4KV), run SDPA on each, concat results. This works because 4 is a power of 2.

Note: the regular `scaled_dot_product_attention` (prefill) handles GQA natively without splitting.

### Performance

```
Upload:     10.3s (1.24B params, bf16)
Prefill:    152ms (6 tokens)
Decode:     12.8ms/tok = 78.2 tok/sec
```

Proportional scaling: Llama is ~2.5x more params, gets ~1.8x slower (12.8ms vs 7.1ms). The less-than-linear scaling suggests wider matmuls are more efficient.

### Key Code Changes

1. **No biases**: `has_bias` flag auto-detected from weights, skip `ttnn.add` for bias
2. **Interleaved RoPE**: Rotation matrix swaps adjacent pairs `(x[2i], x[2i+1])` instead of half-halves
3. **Split SDPA**: Two `scaled_dot_product_attention_decode` calls per layer, concatenated
4. **Tokenizer**: `PreTrainedTokenizerFast(tokenizer_file=...)` avoids AutoTokenizer → torchvision chain

### Implications

Our TT-NN inference infrastructure is model-agnostic for standard decoder-only transformers. Porting a new model requires only:
1. Change config constants (layers, hidden, heads, etc.)
2. Handle bias presence/absence
3. Set correct RoPE format (half vs interleaved)
4. Work around SDPA head count limitations if needed
