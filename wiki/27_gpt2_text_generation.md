# Wiki 27: GPT-2 Text Generation on Blackhole

## Q: Can Blackhole generate coherent text?

**A:** Yes. GPT-2 small (124M params) generates fluent English via greedy autoregressive decode:

| Prompt | Generated Continuation |
|--------|----------------------|
| "The meaning of life is" | "not the same as the meaning of death." |
| "Once upon a time, in a land far away," | "there was a man who had been a slave, and he had been a slave to a man who" |
| "Artificial intelligence will" | "be able to do things like search for and find people, and to find people who are in danger" |

Top-1 predictions match JAX CPU reference. The text is grammatically correct and semantically coherent.

## Q: How fast is generation?

**A:** Per-token latency breakdown:

| Phase | Time |
|-------|------|
| Weight upload (one-time) | 393 ms |
| First token (cold) | 182 ms |
| Subsequent tokens | 110-140 ms |
| 20-token generation | ~2.5 sec |

The bottleneck is CPU round-trips — each token requires:
1. LayerNorm on CPU (2 per layer × 12 layers = 24 round-trips)
2. QKV split on CPU (1 per layer = 12 round-trips)  
3. Head concat on CPU (1 per layer = 12 round-trips)
4. GELU on CPU (1 per layer = 12 round-trips)

That's ~60 host↔device transfers per token. Eliminating these would bring latency down to the device compute time (~5-10ms per token based on our traced transformer benchmarks).

## Q: What's the architecture of the generation pipeline?

**A:**

```
For each new token:
  1. Embed: token_ids → wte[ids] + wpe[:len] → (1, T, 768)
  2. For each of 12 layers:
     a. LayerNorm (CPU)
     b. QKV matmul (device: ttnn.matmul)
     c. QKV split + reshape (CPU → 4D tensors)
     d. Causal attention (device: ttnn.transformer.scaled_dot_product_attention)
     e. Head concat (CPU)
     f. Output projection + residual (device: ttnn.matmul + ttnn.add)
     g. LayerNorm (CPU)
     h. MLP matmul (device: ttnn.matmul)
     i. GELU (CPU)
     j. MLP projection + residual (device: ttnn.matmul + ttnn.add)
  3. Final LayerNorm (CPU)
  4. Project to vocab: hidden @ wte.T (CPU numpy matmul)
  5. argmax → next token
```

On-device ops: matmul, add, FlashAttention-2
CPU fallback ops: LayerNorm, GELU, QKV split, head concat, vocab projection

## Q: What would make this fast?

**A:** Three levels of optimization:

1. **Move LayerNorm + GELU to device** — `ttnn.layer_norm` and `ttnn.gelu` exist. This eliminates ~48 of the 60 CPU round-trips per token.

2. **Use ttnn.transformer.split_query_key_value_and_split_heads** — Native QKV split eliminates another 12 round-trips. With `concatenate_heads`, we're down to just the initial embedding and final logit projection on CPU.

3. **Trace capture** — Once all ops are on-device, trace the full 12-layer forward pass. Replay with `ttnn.execute_trace` for each token. Expected latency: ~5-10ms/token based on our traced benchmarks (Experiment 25 showed 4.6ms for 12 traced layers at smaller model size).

Target: **~10ms/token** → 100 tokens/sec, competitive with GPU inference for this model size.

---

*Experiment 31, run on Blackhole device 0. GPT-2 small from HuggingFace, greedy decode.*
