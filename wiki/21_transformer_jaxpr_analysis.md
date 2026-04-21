# 21: Transformer Jaxpr Analysis — What JAX Actually Compiles

## The Experiment

We defined a single-head transformer encoder block in pure JAX (attention + FFN + 2x layer norm + residuals) and traced it to Jaxpr to understand exactly what the IR looks like and what our backend needs to support.

## The Jaxpr

**Q: How many Jaxpr equations does a single transformer block produce?**

A: **56 equations** using **17 unique primitives**. Here's the breakdown:

| Primitive | Count | What it does |
|-----------|-------|-------------|
| `dot_general` | 8 | Q/K/V projections (3), attention scores (1), context (1), output proj (1), FFN layers (2) |
| `broadcast_in_dim` | 8 | Expanding (32,) → (32,1) for layer norm, (64,) → (1,64) for scale/bias |
| `div` | 6 | Attention scaling (÷√d), mean computation (÷64), layer norm (÷variance) |
| `reduce_sum` | 4 | Two means in layer norm (2x2) |
| `sub` | 4 | Centering in layer norm, subtracting max for softmax stability |
| `add` | 4 | Residual connections (2), bias additions (2) |
| `mul` | 2 | Scale parameter in layer norm |
| `sqrt` | 2 | Layer norm denominator |
| `integer_pow` | 2 | Variance computation: (x - mean)^2 |
| `convert_element_type` | 1 | sqrt(64) result cast |
| `exp` | 1 | Softmax numerator |
| `reduce_max` | 1 | Softmax stability |
| `max` | 1 | Softmax stability (max with -inf) |
| `stop_gradient` | 1 | Softmax max is detached from gradient |
| `transpose` | 1 | k.T for attention scores |
| `custom_jvp_call` | 1 | Wraps relu (with gradient rule) |
| `pjit` | 1 | Inside custom_jvp_call, wraps the actual relu → max(x, 0) |

**Q: Are there any surprising patterns?**

Several:

1. **Softmax decomposes into 6 ops**: reduce_max → max(-inf, ...) → stop_gradient → broadcast → sub → exp → reduce_sum → broadcast → div. JAX doesn't use a fused softmax primitive — it's all elementwise + reductions.

2. **Layer norm is expensive**: Each layer norm is ~10 equations (two reduce_sums for mean, integer_pow for variance, sqrt, broadcasts for keepdims, mul/div for normalization). Two layer norms account for ~20 of the 56 equations.

3. **No reshape needed**: Our transformer avoids reshaping because it's single-head. Multi-head attention would need reshape + transpose to split/merge heads, adding `reshape` to the primitive set.

4. **`custom_jvp_call` wraps relu**: JAX wraps `jax.nn.relu` in a custom JVP rule (for gradient computation). Inside it, there's a `pjit` which wraps `max(x, 0)`. Our interpreter handles this by recursively interpreting the sub-jaxpr.

5. **`stop_gradient` in softmax**: JAX detaches the max value in softmax from the gradient tape. Our interpreter treats this as identity (pass-through), which is correct for forward-only inference.

## Implications for the Backend

**Q: What does this tell us about building a real JAX backend?**

1. **Op count is manageable**: A transformer — the foundation of modern AI — only needs 17 primitives. Our registry already has 20. The Jaxpr IR is surprisingly compact.

2. **Matmul dominates compute**: 8 of 56 equations are dot_general, but they account for >99% of the FLOPs. Everything else is cheap elementwise/reduction ops.

3. **Broadcast is frequent**: 8 broadcast_in_dim ops, all doing shape manipulation for layer norm and softmax. These are currently our biggest performance bottleneck (CPU round-trip). A fused layer norm TT-NN op would eliminate most of these.

4. **Fused ops would help enormously**: TT-NN has `ttnn.softmax` and could potentially do fused layer norm. If we pattern-matched these sequences in the Jaxpr and replaced them with single TT-NN calls, we'd eliminate ~20 equations and their broadcast overhead.

5. **Multi-head attention adds reshape**: For real transformers, we'd need reshape to split (batch, seq, d_model) → (batch, heads, seq, d_head). This is just a metadata operation but needs careful tile alignment.

## The Full Jaxpr (Annotated)

```
{ lambda ; x:f32[32,64] w_q:f32[64,64] w_k:f32[64,64] w_v:f32[64,64]
          w_o:f32[64,64] w1:f32[64,256] w2:f32[256,64]
          g1:f32[64] b1:f32[64] g2:f32[64] b2:f32[64]. let

    # === Self-attention ===
    q = dot_general(x, w_q)           # Q projection
    k = dot_general(x, w_k)           # K projection  
    v = dot_general(x, w_v)           # V projection
    k_t = transpose(k)                # K^T
    scores = dot_general(q, k_t)      # QK^T
    scale = sqrt(64.0)                # √d_model
    scores = div(scores, scale)       # QK^T / √d

    # === Softmax (decomposed) ===
    max_s = reduce_max(scores)        # max per row
    max_s = max(-inf, max_s)          # clamp
    max_s = broadcast(max_s)          # (32,) → (32,1)
    max_s = stop_gradient(max_s)      # detach from grad
    shifted = sub(scores, max_s)      # numerical stability
    exp_s = exp(shifted)              # e^(s - max)
    sum_s = reduce_sum(exp_s)         # Σe^(s - max)
    sum_s = broadcast(sum_s)          # (32,) → (32,1)
    attn = div(exp_s, sum_s)          # softmax output

    # === Context + residual ===
    context = dot_general(attn, v)    # attention output
    proj = dot_general(context, w_o)  # output projection
    h = add(x, proj)                  # residual connection

    # === Layer norm 1 (10 ops) ===
    sum1 = reduce_sum(h)
    sum1 = broadcast(sum1)            # (32,) → (32,1)
    mean1 = div(sum1, 64.0)
    centered = sub(h, mean1)
    sq = integer_pow(centered, 2)     # (x - μ)²
    var_sum = reduce_sum(sq)
    var_sum = broadcast(var_sum)
    var = div(var_sum, 64.0)
    # g1 * (x - μ) / √(var + ε) + b1
    ...

    # === FFN ===
    ff = dot_general(h_normed, w1)    # first linear
    ff = relu(ff)                     # via custom_jvp_call → pjit → max(x, 0)
    ff = dot_general(ff, w2)          # second linear
    h2 = add(h_normed, ff)            # residual

    # === Layer norm 2 (10 ops, same pattern) ===
    ...

  in (output,) }
```

## Key Takeaway

The gap between "JAX transformer" and "hardware execution" is exactly 56 flat equations. No control flow, no loops, no dynamic shapes. This is why Jaxpr is such a good target for hardware backends — **the entire transformer forward pass is a straight-line program**.
