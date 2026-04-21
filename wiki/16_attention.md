# Self-Attention on Blackhole

## Q: Can we implement transformer self-attention using TT-NN primitives on Blackhole?

**A: Yes.** Both single-head and multi-head attention work correctly using matmul, transpose, scalar multiply, and softmax. The full attention pipeline runs in ~0.2ms eager and ~0.07ms traced at seq_len=128.

## Results

### Test 1: Single-Head Self-Attention

Parameters: seq_len=128, d_model=256, d_k=64.

Pipeline: `Q,K,V = x @ Wq, x @ Wk, x @ Wv` then `softmax(Q @ K^T / sqrt(d_k)) @ V`

| Metric | Value |
|--------|-------|
| Max abs error vs PyTorch | 0.0344 |
| Mean abs error | 0.0036 |
| Attention weight row sums | 0.964 - 1.005 (expect 1.0) |

Errors are consistent with bf16 precision through a chain of matmul + softmax operations. The attention weights sum to approximately 1.0, confirming softmax works correctly in the pipeline.

**Key discovery:** `ttnn.multiply(tensor, tensor)` fails with "Invalid subtile broadcast type" when one tensor is a scalar (1x1). Must use `ttnn.multiply(tensor, float_scalar)` instead for scaling.

### Test 2: Multi-Head Attention (4 heads)

Parameters: 4 heads, d_model=256, d_k=64 per head, with output projection.

| Head | Max Error |
|------|-----------|
| Head 0 | 0.0273 |
| Head 1 | 0.0242 |
| Head 2 | 0.0398 |
| Head 3 | 0.0407 |
| **Full MHA output** | **0.0282 max, 0.0039 mean** |

**Strategy:** Compute each head independently in a loop (avoiding 4D reshape complexity on TT-NN), read back per-head results, concatenate in PyTorch, then upload for output projection. This works but is not optimal -- a proper batched implementation would keep everything on device.

### Test 3: Latency vs Sequence Length

| Seq Len | Latency (ms) | TFLOPS |
|---------|-------------|--------|
| 64 | 0.197 | 0.037 |
| 128 | 0.189 | 0.089 |
| 256 | 0.193 | 0.218 |
| 512 | 0.202 | 0.582 |
| 1024 | 0.216 | 1.710 |

**Key insight:** Latency is nearly flat (~0.19-0.22ms) across 16x range of sequence lengths. This means dispatch overhead dominates at small sizes, and compute only starts to matter at seq_len=1024. At seq_len=1024 we hit 1.71 TFLOPS, still well below Blackhole's theoretical peak, but the quadratic scaling of attention (O(n^2) in the score matmul) is starting to show meaningful compute.

### Test 4: Trace Capture Speedup

| Mode | Latency | Speedup |
|------|---------|---------|
| Eager | 0.200 ms | 1.0x |
| Traced | 0.070 ms | **2.86x** |

Trace capture removes 0.130ms of dispatch overhead. With ~8 ops in the attention pipeline, that's ~16us/op dispatch cost -- consistent with the ~21us/op measured in previous experiments.

## TT-NN Ops Used for Attention

| Op | Purpose |
|----|---------|
| `ttnn.matmul(a, b)` | Q/K/V projections, score computation, output |
| `ttnn.transpose(t, -2, -1)` | K^T for attention scores |
| `ttnn.multiply(t, float)` | Scale by 1/sqrt(d_k) -- must use float scalar, not tensor |
| `ttnn.softmax(t, dim=-1)` | Attention weight normalization |

## What This Means for JAX Backend

Self-attention is the core of transformer models, and it works on Blackhole with good accuracy. The building blocks (matmul, transpose, softmax, elementwise ops) are all functional. Key challenges for a real transformer:

1. **4D tensor reshaping** -- multi-head attention ideally needs `(batch, heads, seq, d_k)` reshaping, which we avoided by computing heads in a loop
2. **Dispatch overhead** -- at small sizes, Python dispatch dominates; trace capture provides 2.9x speedup
3. **Memory bandwidth** -- at seq_len=1024, the score matrix is 1024x1024 which starts to stress memory; longer sequences will need careful memory management
