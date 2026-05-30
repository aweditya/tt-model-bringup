# Advanced Optimization Techniques for Tenstorrent Blackhole Inference

Research document covering the next wave of optimizations beyond our current
Qwen2.5-0.5B pipeline (132 tok/sec single-sequence, 4,819 tok/sec batch=64
on Blackhole P150).

---

## 1. Speculative Decoding

### 1.1 Core Mechanism

Speculative decoding (Leviathan et al. 2022, Chen et al. 2023) exploits a
fundamental asymmetry: **verifying N tokens in parallel is cheaper than
generating N tokens sequentially.**

The algorithm:

1. A small/fast **draft model** autoregressively generates K candidate tokens.
2. The large **target model** runs a single forward pass on all K candidates
   simultaneously (this is essentially a prefill-style pass on the K-token
   sequence).
3. A **rejection sampling** step compares the draft model's probability
   distribution against the target model's at each position. Tokens are
   accepted left-to-right until the first rejection.
4. On rejection at position i, a corrected token is sampled from an adjusted
   distribution (target - draft, renormalized). All tokens after position i
   are discarded.
5. Repeat from step 1 with the accepted prefix.

The key theorem: **speculative decoding produces the exact same distribution
as the target model.** This is not an approximation -- the rejection sampling
mathematically guarantees distributional equivalence.

### 1.2 Acceptance Rate and Speedup

The acceptance rate alpha depends on how well the draft model approximates
the target:

- **alpha = P(draft token accepted)** -- typically 0.7-0.9 for well-matched
  draft/target pairs.
- Expected tokens per step = 1/(1-alpha) when generating K >> 1/(1-alpha)
  candidates. More precisely, with K candidates: E[accepted] = (1 - alpha^(K+1)) / (1 - alpha).
- With alpha=0.8 and K=5: E[accepted] ~= 4.0 tokens per verification step.
- With alpha=0.7 and K=4: E[accepted] ~= 3.0 tokens per verification step.

**Practical speedup** = (E[accepted tokens per step]) / (cost_ratio), where
cost_ratio = (time for draft K tokens + time for target verification) / (time
for target single token).

If the draft model is 10x faster than the target, and verification of K
tokens costs roughly the same as one target forward pass (due to parallelism),
the speedup can approach E[accepted] -- i.e., 2-4x wall-clock improvement.

### 1.3 Variants

**Standard speculative decoding (Leviathan/Chen):**
- Separate draft model (e.g., 68M draft for a 7B target).
- Requires loading two models in memory.
- Draft model quality directly determines acceptance rate.

**Self-speculative decoding (Draft & Verify, Zhang et al. 2023):**
- The target model generates drafts itself using early exit or layer skipping.
- Approach: use the first N layers of the target model as the "draft."
- No additional model needed; but acceptance rate may be lower.
- Bayling variant: skip every other layer for draft, use all layers for verify.

**Medusa (Cai et al. 2024):**
- Adds multiple lightweight "heads" to the target model, each predicting a
  future token position (head 1 predicts t+1, head 2 predicts t+2, etc.).
- Heads are small MLPs (2 layers, same hidden dim) trained on the target
  model's hidden states.
- Generates a **tree** of candidate continuations and verifies the entire tree
  in one forward pass using tree attention (a modified causal mask).
- Medusa-1: frozen backbone, train only heads. Medusa-2: fine-tune backbone
  too (better acceptance rate).
- Typical speedup: 2.2-3.6x with 5 Medusa heads.
- **No separate model required** -- the heads add < 1% parameters.

**EAGLE (Li et al. 2024):**
- An autoregressive draft head that takes the target model's hidden states
  as input and generates draft tokens via a lightweight transformer layer.
- Key insight: predicting features (hidden states) is easier than predicting
  tokens directly. EAGLE predicts the next feature, then projects to vocab.
- Speedup: 2.5-3.8x, consistently outperforms Medusa.
- EAGLE-2 adds a confidence-aware dynamic draft tree -- expands branches
  with high confidence, prunes low-confidence ones.

**Lookahead decoding (Fu et al. 2024):**
- Uses Jacobi iteration to solve the autoregressive decoding fixed-point
  problem in parallel.
- Maintains N-gram pools and trajectory history to generate candidates.
- No draft model needed, no training required.
- More effective for long sequences where N-gram patterns repeat.
- Typical speedup: 1.5-2.5x (lower than EAGLE/Medusa but zero setup cost).

**Staged speculative decoding:**
- Chain of draft models: tiny -> small -> medium -> target.
- Each stage verifies and filters candidates from the previous stage.
- Useful when there's a large capability gap between draft and target.

### 1.4 Application to Our Setup

**Scenario A: Use Qwen2.5-0.5B as draft for a larger model (e.g., Qwen2.5-7B)**

This is the textbook use case. The 0.5B model generates K=4-5 candidates,
and the 7B model verifies. Questions:
- Can we fit both models on a single Blackhole P150? The 0.5B uses ~1GB, the
  7B would use ~14GB in bf16. Blackhole P150 has 24GB DRAM -- so yes, both
  fit in memory.
- The 0.5B runs at 132 tok/sec (7.6ms/tok). If the 7B runs at ~15 tok/sec
  (rough estimate from TT reference numbers), then generating 5 draft tokens
  takes 5 * 7.6ms = 38ms, and verification takes ~67ms (one 7B forward pass
  on 5 tokens). Total: ~105ms for ~3.5 accepted tokens = ~33 tok/sec
  effective for the 7B model vs ~15 tok/sec baseline. That would be a ~2.2x
  speedup.
- **This is the most impactful use case** -- it makes a 7B model feasible on
  a single P150 at reasonable throughput.

**Scenario B: Tiny draft for the 0.5B itself**

Less obvious benefit. Options:
- Medusa heads on Qwen2.5-0.5B: adds ~5M params (5 heads * 2 layers * 896 dim).
  Would need training data (distillation from the 0.5B). Could boost from
  132 to ~300-400 tok/sec if 2-3 tokens accepted per step.
- Self-speculative with the 0.5B (skip layers): use first 12 layers as draft,
  all 24 for verify. Draft at ~264 tok/sec, verify at ~132 tok/sec. With
  alpha=0.6: E[accepted] ~= 2.0, cost = (12/24 * 7.6ms * 4) + 7.6ms = 22.8ms
  for ~2 tokens = ~88 tok/sec. **Worse than baseline** because the draft is
  not fast enough relative to verification.
- **Verdict: not worth it for the 0.5B alone.** Speculative decoding shines
  when draft is 5-10x faster than target. Within a single small model, the
  ratio is only 1.5-2x.

**Scenario C: Speculative decoding with batch decode**

This is where it gets interesting. At batch=64, we achieve 4,819 tok/sec
(13.3ms per batch step). If speculative decoding could increase tokens
accepted per step to 3, that's ~14,400 tok/sec theoretical. But the
verification step processes K*batch_size tokens, which may exceed memory.
Practical batch+speculative likely requires batch=16-32 with K=3-4.

### 1.5 Key Implementation Challenges on Blackhole

1. **KV cache management:** Speculative tokens that get rejected must have
   their KV cache entries rolled back. Our paged_update_cache supports
   per-sequence position tracking, which helps.
2. **Tree attention masks:** For Medusa/EAGLE tree verification, we need
   custom attention masks. Our SDPA currently uses standard causal masking.
3. **Two-model memory:** Both draft and target weights plus KV caches for
   both must fit in DRAM. Memory pressure is the binding constraint.
4. **Trace compatibility:** Our trace-captured decode assumes a fixed
   computation graph. Speculative decoding has variable-length verification
   -- may need multiple traces for different K values.

---

## 2. Quantization Techniques

### 2.1 The Landscape

**Post-Training Quantization (PTQ)** methods -- no retraining needed:

| Method | Year | Bit width | Approach | Key idea |
|--------|------|-----------|----------|----------|
| GPTQ | 2022 | 3-4 bit | Weight-only | Layer-wise Hessian-based quantization; solves optimal rounding via OBQ |
| AWQ | 2023 | 4 bit | Weight-only | Activation-aware: protect salient weight channels (those multiplied by large activations) |
| SqueezeLLM | 2024 | 3-4 bit | Weight-only | Sensitivity-based non-uniform quantization + sparse outlier storage |
| QuIP# | 2024 | 2-4 bit | Weight-only | Incoherence processing via random Hadamard rotations; enables sub-4-bit without fine-tuning |
| AQLM | 2024 | 2 bit | Weight-only | Additive quantization: approximate each weight vector as a sum of codebook entries |

**Quantization-Aware Training (QAT)** methods -- require fine-tuning:

| Method | Key idea |
|--------|----------|
| QLoRA | 4-bit NormalFloat base weights + LoRA adapters trained in bf16 |
| GGUF (llama.cpp) | Mixed 2-6 bit per-layer quantization with importance weighting |
| BitNet | 1.58-bit (ternary: -1, 0, +1) weights from scratch -- not post-training |

### 2.2 How Each Method Works

**GPTQ (Generative Pre-trained Transformer Quantization):**
- Quantizes weights layer-by-layer, minimizing the squared error of each
  layer's output using a small calibration dataset (~128 samples).
- Uses the Hessian matrix H = 2*X^T*X (where X is the calibration input) to
  determine which weights to quantize first (those with smallest Hessian
  diagonal entries, meaning least sensitive).
- After quantizing each weight, adjusts remaining unquantized weights to
  compensate for the rounding error (Optimal Brain Quantization).
- Achieves near-lossless 4-bit at 7B+ scale; some degradation at 3-bit.
- Standard in the ecosystem: supported by AutoGPTQ, transformers, vLLM.

**AWQ (Activation-Aware Weight Quantization):**
- Observation: 1% of weight channels carry disproportionate importance
  because they multiply with large activation magnitudes.
- Instead of mixed precision (which hardware hates), AWQ applies per-channel
  scaling: multiply salient weights by s, divide corresponding activations by s.
  This shifts the quantization range to protect important channels.
- Mathematically: Q(w * s) * (x / s) approximates w * x better than Q(w) * x
  for well-chosen s.
- Optimal s is found by minimizing layer output error on calibration data.
- Typically 0.1-0.5 perplexity better than GPTQ at 4-bit for the same speed.

**SqueezeLLM:**
- Identifies weight outliers that cause disproportionate quantization error.
- Two components: (1) non-uniform quantization using k-means clustering of
  weight values, (2) sparse storage of extreme outliers in a separate
  CSR-format matrix.
- At inference: dense_output + sparse_outlier_output = full output.
- Better quality than GPTQ/AWQ at 3-bit, but the sparse component adds
  complexity.

**QuIP# (Quantization with Incoherence Processing):**
- Key insight: quantization error is worst when weight matrices have
  "coherent" structure (large entries concentrated in few rows/columns).
- Applies random orthogonal rotations (Hadamard transforms) to make weights
  incoherent before quantization, then reverses the rotation at inference.
- Enables 2-bit quantization with reasonable quality -- a regime where
  GPTQ/AWQ collapse.
- The Hadamard transforms are O(n log n) and add minimal inference overhead.

**AQLM (Additive Quantization for Language Models):**
- Represents each weight vector as a sum of M entries from learned codebooks.
- With M=2 codebooks of 2^8 entries: each weight vector stored as two 8-bit
  indices = effectively 2 bits per parameter.
- Inference uses codebook lookups + additions instead of multiplications.
- State-of-the-art at 2-bit; matches GPTQ 4-bit quality with 2x compression.
- Downside: requires custom GPU kernels for the codebook lookup; not trivially
  portable to TT-NN.

### 2.3 How bfloat8_b Compares

TT-NN natively supports `bfloat8_b` -- Tenstorrent's custom 8-bit
brain float format. This is a block-scaled format:

- 8 bits per element: 1 sign + 5 exponent + 2 mantissa (approximate).
- A shared block exponent per group of values (the "b" in bfloat8_b).
- The block scaling gives better dynamic range than a flat 8-bit format.

Comparison with standard quantization:

| Format | Bits | Type | Dynamic range | Quality |
|--------|------|------|---------------|---------|
| bfloat16 | 16 | Float | High | Baseline |
| bfloat8_b | 8 | Block float | Good (shared exponent) | ~0.5-1.0 perplexity loss |
| INT8 (GPTQ-style) | 8 | Integer | Limited | ~0.2-0.5 perplexity loss (calibrated) |
| INT4 (GPTQ/AWQ) | 4 | Integer | Very limited | ~0.5-1.5 perplexity loss (calibrated) |

**Key difference:** bfloat8_b is a floating-point format (preserves relative
precision across magnitudes), while INT8/INT4 are fixed-point (uniform
quantization bins). For weights with wide value distributions, float formats
are naturally better without calibration. But calibrated integer methods
(GPTQ, AWQ) can outperform uncalibrated float8 because they optimally
allocate the quantization bins.

**Practical tradeoff for our pipeline:**

bfloat8_b is the low-hanging fruit because:
1. TT-NN supports it natively -- zero implementation effort.
2. 2x memory reduction vs bf16 (490M params: 980MB -> 490MB).
3. 2x potential throughput improvement (if memory-bandwidth-bound).
4. No calibration dataset needed.

But we established (wiki 38) that at decode sizes, DRAM bandwidth is NOT
the bottleneck -- compute is. So bfloat8_b weight compression may not
speed up single-sequence decode. It WOULD help at larger batch sizes
where weight reads from DRAM start to matter, and it would help fit
larger models (e.g., Qwen2.5-7B) in memory.

**Our current precision findings:**
- SDPA softmax is precision-critical (cosine drops from 0.999 to 0.985 in bf16).
- Matmuls are NOT precision-critical (0.999998 cosine in bf16).
- This suggests: weights in bfloat8_b for matmuls, activations in bf16,
  and SDPA in HiFi4+fp32_dest_acc.

### 2.4 Mixed-Precision Strategy for Blackhole

Based on our precision analysis (wiki 33, experiments 44-46e):

| Operation | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| Linear projections (Q/K/V/O) | bf16 weights, HiFi4 | **bf8 weights, bf16 activations, HiFi4** | Matmuls are insensitive (0.999998 cosine) |
| MLP gate/up/down | bf16 weights, HiFi4 | **bf8 weights, bf16 activations, HiFi4** | Same -- matmuls tolerant |
| SDPA softmax | bf16, HiFi4+fp32_dest | **Keep bf16+fp32_dest** | THE precision-critical op |
| RMSNorm | bf16, HiFi4 | bf16, HiFi4 | Already fine (0.9999 cosine) |
| Embedding | bf16 | bf16 | Small, precision matters for first layer |
| LM head | bf16 | bf16 | Final projection, precision matters for token selection |

This mixed strategy would give ~1.8x memory reduction (most params are in
linear layers) while maintaining the precision we fought hard for.

### 2.5 Recent Methods: TurboQuant and Beyond

**TurboQuant (2024):** Combines block-wise quantization with adaptive
precision allocation. Assigns more bits to sensitive blocks (identified by
Fisher information) and fewer bits to insensitive ones. Achieves near-
lossless 3.5 average bits. Potentially compatible with bfloat8_b if we
implement block-level bit allocation.

**FP8 (E4M3/E5M2) standardization:** NVIDIA H100 and AMD MI300 natively
support FP8. Our bfloat8_b is Tenstorrent's variant. The ecosystem is
converging on 8-bit floats as the default inference format.

**SpinQuant (Meta, 2024):** Learned rotation matrices (similar to QuIP# but
optimized end-to-end) that make weights more quantization-friendly. Achieves
4-bit with near-zero perplexity loss even at 1B scale.

**FLUTE (2024):** Flexible look-up table engine for mixed-precision
quantization. Uses lookup tables to dequantize on-the-fly, supporting
arbitrary bit widths (2-8) with minimal overhead.

---

## 3. vLLM-style Continuous Batching and Paged Attention

### 3.1 PagedAttention

PagedAttention (Kwon et al. 2023, vLLM) applies **virtual memory concepts**
to KV cache management:

**The problem:** In standard inference serving, each request pre-allocates a
contiguous KV cache for the maximum sequence length. With a 2048-token limit
and 1000 concurrent requests, most memory is wasted on unused positions
(average utilization ~30-50%).

**The solution:** Divide the KV cache into fixed-size **pages** (blocks of
16-256 positions). A page table maps logical sequence positions to physical
memory blocks. When a sequence grows, allocate a new page. When a sequence
finishes, free its pages for reuse.

```
Logical view (per sequence):
  [page_0] -> [page_1] -> [page_2] -> ... -> [page_N]

Physical memory:
  [block_7] [block_2] [block_15] ... (non-contiguous)

Page table:
  seq_0: [7, 2, 15, ...]
  seq_1: [3, 11, 8, ...]
```

Benefits:
- Near-zero memory waste (only the last page of each sequence may be partial).
- Memory sharing: for beam search or parallel sampling, sequences that share
  a prefix can share pages (copy-on-write).
- Dynamic memory: sequences grow one page at a time, no pre-allocation.

The attention kernel is modified to follow page table indirection when
reading K/V values. This adds one level of pointer chasing but the overhead
is minimal compared to the memory savings.

### 3.2 How Close Is Our paged_update_cache?

Our current implementation uses `ttnn.paged_update_cache` with
`update_idxs_tensor` (a per-sequence position vector). Let's compare:

| Feature | vLLM PagedAttention | Our Implementation |
|---------|--------------------|--------------------|
| Non-contiguous pages | Yes (page table indirection) | **No** -- contiguous per-sequence cache |
| Per-sequence position tracking | Yes | **Yes** (update_idxs_tensor) |
| Dynamic allocation | Yes (page-level malloc/free) | **No** -- pre-allocated for max_seq_len |
| Memory sharing (CoW) | Yes (beam search, prefix caching) | **No** |
| Page size configurable | Yes (16-256) | N/A (no paging) |
| Batch support | Yes | **Yes** (batch dimension in cache) |
| KV cache layout | Paged blocks | **Contiguous per sequence** (N, 2, 256, 64) |

**Verdict:** Our implementation has the per-sequence position tracking needed
for independent sequence management, but lacks the actual paging (non-
contiguous allocation). For our current scale (batch=64, max_seq=256,
cache=96MB total), the memory waste is manageable. Paging would matter for:
- Larger max_seq_len (2048+): wasted memory per sequence grows linearly.
- Larger batch (256+): total memory becomes binding constraint.
- Beam search: where prefix sharing saves N-1 copies of shared history.

The TT-NN API `paged_update_cache` has "paged" in the name, suggesting
Tenstorrent envisions paging support, but our usage is effectively
contiguous. Investigating whether tt-metal supports actual page table
indirection in the SDPA kernel would be valuable.

### 3.3 Continuous Batching vs Static Batching

**Static batching** (what we do now):
- All sequences in a batch start and end together.
- If sequence 3 hits EOS at token 20 but others run to token 100, sequence 3's
  slot is wasted for 80 tokens.
- Throughput: bounded by the longest sequence.

**Continuous batching** (Orca, Yu et al. 2022):
- A scheduler maintains a **waiting queue** of incoming requests and an
  **active batch** of in-progress sequences.
- At each step:
  1. Remove finished sequences (hit EOS or max length).
  2. Admit new sequences from the waiting queue into freed slots.
  3. Run one decode step on the active batch.
- Each sequence is independently tracked (position, KV cache, state).
- Throughput: near-optimal utilization because slots are never wasted.

```
Time step 1: [seq_A(pos=5), seq_B(pos=3), seq_C(pos=12), ___empty___]
             seq_C finishes (EOS)
Time step 2: [seq_A(pos=6), seq_B(pos=4), seq_D(pos=0*), ___empty___]
             *seq_D enters, does prefill
Time step 3: [seq_A(pos=7), seq_B(pos=5), seq_D(pos=1), seq_E(pos=0*)]
```

### 3.4 Continuous Batching Scheduler for Our Pipeline

A minimal scheduler for our setup:

```
State:
  active_batch: array of batch_size slots, each {seq_id, position, token_ids, kv_cache_slot}
  waiting_queue: FIFO of pending requests
  free_slots: set of available batch positions

Loop:
  1. For each active slot:
     - If sequence hit EOS or max_length:
       - Mark slot as free
       - Return generated text to requester
       - Zero out KV cache for this slot (or just overwrite on reuse)

  2. For each free slot, if waiting_queue is non-empty:
     - Dequeue next request
     - Run PREFILL for this sequence (outside the batch trace)
     - Fill its KV cache entries
     - Set its position in update_idxs_tensor
     - Assign to slot

  3. Run one DECODE step on the full batch
     - update_idxs_tensor has per-sequence positions (some may be 0 for
       just-prefilled sequences)
     - Extract per-sequence logits
     - Sample/argmax per sequence
     - Update positions

  4. Yield control / check for new requests
```

**Challenge: Prefill in the middle of serving.** When a new sequence enters,
it needs a prefill pass (process all prompt tokens at once). This conflicts
with the decode trace, which assumes single-token-per-sequence. Options:
- **Chunked prefill:** Process the new sequence's prefill in chunks of 1
  token each, interleaved with decode steps. Slow but simple.
- **Separate prefill trace:** Maintain a second trace for prefill, run it
  on the new sequence before adding to the decode batch.
- **Prefill batching:** Accumulate new sequences and batch their prefills
  together periodically.

**Our advantage:** The `update_idxs_tensor` already supports per-sequence
positions, and our KV cache layout (N, 2, 256, 64) has independent slots.
The main missing piece is the scheduler logic and handling prefill injection.

---

## 4. StableHLO Lowering

### 4.1 How Models Become StableHLO

The pipeline from user code to StableHLO:

```
Python (JAX/PyTorch/TF)
  |
  v
Framework IR (Jaxpr / torch.export / tf.function)
  |
  v
StableHLO (the interchange format)
  |
  v
XLA HLO (internal XLA representation)
  |
  v
Target-specific code (CUDA PTX, CPU machine code, TPU ops)
```

For JAX specifically:
1. `jax.jit(f)` traces `f` to produce a **Jaxpr** (JAX expression).
2. The Jaxpr is lowered to **StableHLO** via `jax.jit(f).lower(args)`.
3. StableHLO is passed to the PJRT plugin, which either:
   a. Hands it to XLA for compilation (CUDA, TPU path), or
   b. Interprets it directly (the jax-mps/applejax approach), or
   c. Compiles it with a custom backend compiler.

You can inspect the StableHLO:
```python
lowered = jax.jit(f).lower(x)
print(lowered.as_text())  # StableHLO text format
```

### 4.2 What a Transformer Looks Like in StableHLO

A single attention layer lowers to approximately:

```mlir
// Q/K/V projections (three dot_general ops)
%q = stablehlo.dot_general %x, %wq, contracting_dims = [[-1], [0]]
%k = stablehlo.dot_general %x, %wk, contracting_dims = [[-1], [0]]
%v = stablehlo.dot_general %x, %wv, contracting_dims = [[-1], [0]]

// Reshape for multi-head: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
%q_r = stablehlo.reshape %q : (1, T, 896) -> (1, T, 14, 64)
%q_t = stablehlo.transpose %q_r, dims = [0, 2, 1, 3]

// Attention scores: Q @ K^T / sqrt(d)
%kt = stablehlo.transpose %k_r, dims = [0, 1, 3, 2]
%scores = stablehlo.dot_general %q_t, %kt  // batched matmul
%scale = stablehlo.constant 0.125  // 1/sqrt(64)
%scaled = stablehlo.multiply %scores, %scale

// Causal mask
%mask = stablehlo.compare %iota_row, %iota_col, GE
%masked = stablehlo.select %mask, %scaled, %neg_inf

// Softmax (decomposed into primitives)
%max = stablehlo.reduce_max %masked, dims=[3]
%shifted = stablehlo.subtract %masked, %max_broadcast
%exp = stablehlo.exponential %shifted
%sum = stablehlo.reduce_sum %exp, dims=[3]
%attn = stablehlo.divide %exp, %sum_broadcast

// Attention output: attn @ V
%out = stablehlo.dot_general %attn, %v_t

// Output projection
%proj = stablehlo.dot_general %out_reshaped, %wo
```

Key StableHLO ops for transformers:
- `dot_general` -- all matmuls (projections, attention, MLP)
- `reshape`, `transpose` -- head splitting/merging
- `exponential`, `reduce_max`, `reduce_sum`, `divide` -- softmax
- `select` -- masking
- `multiply`, `add` -- scaling, residual connections
- `rsqrt` -- RMSNorm
- `slice`, `dynamic_slice` -- KV cache operations
- `custom_call` -- fused operations (Flash Attention)

### 4.3 StableHLO -> TT-NN Compiler: Feasibility

**Option A: Direct StableHLO interpretation (like jax-mps/applejax)**

Walk the StableHLO module op by op, dispatching each to a TT-NN call.
This is essentially what our Jaxpr interpreter already does, but at a
different IR level.

Mapping difficulty:

| StableHLO op | TT-NN equivalent | Difficulty |
|-------------|------------------|------------|
| dot_general | ttnn.matmul | Easy (need to handle batched dims) |
| add, multiply, subtract, divide | ttnn.add, mul, sub, div | Easy |
| exponential, log, tanh, rsqrt | ttnn.exp, log, tanh, rsqrt | Easy |
| reshape | ttnn.reshape | Medium (tile alignment) |
| transpose | ttnn.transpose / ttnn.permute | Medium |
| broadcast_in_dim | ttnn.repeat | Medium (see experiment 21) |
| reduce_max, reduce_sum | ttnn.max, ttnn.sum | Medium (dim handling) |
| slice, dynamic_slice | ttnn.slice | Hard (tile boundaries) |
| gather, scatter | No direct equivalent | Hard |
| while, if, case | No equivalent | Very hard (control flow) |
| sort | No direct equivalent | Hard |
| iota | Must construct manually | Medium |
| convolution | ttnn.conv2d | Medium |
| custom_call | Case-by-case | Variable |

**Coverage estimate:** ~70% of ops needed for transformer inference are
straightforward. The hard 30% (gather, scatter, control flow, dynamic
slicing) are mostly needed for training or advanced sampling -- not for
the core forward pass.

**Option B: Pattern-matched fusion compiler**

Instead of 1:1 op mapping, recognize patterns and emit fused TT-NN calls:

```
Pattern: dot_general + bias_add -> ttnn.linear
Pattern: reduce_max + subtract + exp + reduce_sum + divide -> ttnn.softmax
Pattern: dot_general + dot_general + transpose + softmax + dot_general -> ttnn.transformer.scaled_dot_product_attention
Pattern: rsqrt + multiply -> ttnn.rms_norm
Pattern: silu + multiply -> ttnn.silu (with gate)
```

This is how real compilers (XLA, TVM, TensorRT) achieve performance --
fused kernels are 2-10x faster than composing elementwise ops.

**Option C: MLIR-based compilation pipeline**

Build a proper MLIR pass pipeline:
```
StableHLO -> (canonicalize) -> (pattern match) -> TT-NN dialect -> (lower) -> TT-NN API calls
```

This is the most "correct" approach but requires significant MLIR
infrastructure. Tenstorrent's own `tt-mlir` project is building exactly
this (StableHLO -> TTIR -> TTNN -> metal).

### 4.4 Relationship Between StableHLO and Our Jaxpr Interpreter

```
JAX code
  |
  |-- jax.make_jaxpr() --> Jaxpr  --> [Our interpreter] --> TT-NN
  |
  |-- jax.jit().lower() --> StableHLO --> [PJRT plugin] --> TT-NN
```

**Jaxpr is higher-level than StableHLO.** Key differences:

| Aspect | Jaxpr | StableHLO |
|--------|-------|-----------|
| Level | Python-level IR | MLIR-level IR |
| Primitives | ~100 JAX primitives | ~120 StableHLO ops |
| Control flow | `cond`, `while_loop`, `scan` as higher-order primitives | Lowered to `if`, `while` regions |
| Custom ops | `custom_jvp_call`, `pjit` (sub-jaxprs) | `custom_call` (opaque) |
| Fusion | None | XLA may pre-fuse |
| Composites | Higher-order (carries sub-jaxprs) | Flat (no nesting beyond regions) |
| Access | `jax.make_jaxpr(f)(x)` | `jax.jit(f).lower(x).as_text()` |

**Our Jaxpr interpreter is the right starting point** for a research
project because:
1. It's pure Python -- easy to modify and debug.
2. Jaxpr is closer to the user's mental model.
3. We can intercept at a high level (e.g., recognize `custom_jvp_call`
   wrapping an attention function and emit a single SDPA call).

**A StableHLO compiler would be the production path** because:
1. It's the standard interchange format -- works with PyTorch (via
   torch-xla), TensorFlow, and any MLIR-based frontend.
2. PJRT plugin integration means `jax.devices()` returns TT devices
   natively.
3. Optimization passes are composable in MLIR.

**Migration path:** Jaxpr interpreter (now) -> StableHLO interpreter
(next) -> MLIR-based compiler (production). Each step increases
generality and performance at the cost of implementation complexity.

### 4.5 Tenstorrent's tt-mlir Project

Tenstorrent is building their own compiler:
```
StableHLO -> TTIR (Tenstorrent IR) -> TTNN dialect -> TT-Metal dialect -> binary
```

This is the long-term "official" path. Our Jaxpr interpreter is a
complementary approach: faster to iterate on, better for understanding
the problem, and serves as a reference implementation. If/when tt-mlir
matures, it would replace our interpreter for production use.

---

## 5. Mixed Precision Strategies

### 5.1 Which Operations Are Precision-Sensitive?

From our experimental evidence (experiments 43-46e) and the literature:

**Precision-critical ops (keep bf16 or higher):**

| Operation | Why sensitive | Our evidence |
|-----------|-------------|--------------|
| Softmax (in SDPA) | exp() and normalization amplify rounding | Cosine drops to 0.985 in bf16; HiFi4+fp32 restores 0.999+ |
| LayerNorm / RMSNorm | Variance computation involves subtraction of similar values (catastrophic cancellation) | 0.9999 in bf16 for us, but risky at lower precision |
| Embedding lookup | First-layer inputs define the signal; errors here propagate through all layers | Not tested, but theoretically sensitive |
| Final LM head projection | Small logit differences determine token selection; bf8 rounding could flip top-1 | Not tested quantitatively |
| Loss computation | Training only -- gradients are precision-sensitive | N/A for inference |

**Precision-tolerant ops (can use bf8 or lower):**

| Operation | Why tolerant | Our evidence |
|-----------|-------------|--------------|
| Q/K/V projections | Large matmuls with many accumulations average out rounding errors | 0.999998 cosine in bf16; bf8 likely safe |
| MLP gate/up/down projections | Same reasoning as Q/K/V | Same evidence |
| Elementwise activations (SiLU, ReLU) | Monotonic functions -- rounding shifts the curve slightly but doesn't change shape | 0.9999 cosine |
| Residual addition | Adding a small update to a large residual; the update can tolerate rounding | Verified indirectly |

### 5.2 Weight bf8 + Activation bf16

This is the sweet-spot mixed-precision strategy:

**Weights in bfloat8_b:**
- Weights are static (loaded once, used millions of times).
- The quantization error in weights is a fixed bias, not noise --
  it doesn't compound across tokens (each token sees the same weights).
- With block scaling (bfloat8_b), outlier weights are handled by
  the shared exponent within each block.

**Activations in bfloat16:**
- Activations change every token and flow through residual connections.
- Errors in activations compound across layers (we saw this: per-layer
  error accumulates through 24 layers).
- bf16 gives enough mantissa bits (7) to survive 24 layers with
  residual connections.

**Accumulation in fp32 (already doing this):**
- `fp32_dest_acc_en=True` in our compute kernel config.
- Matmul: bf8 weight * bf16 activation, accumulated in fp32, output in bf16.
- This is exactly the pattern that NVIDIA's FP8 training uses (Micikevicius
  et al. 2022).

**Implementation on TT-NN:**

```python
# Convert weights to bfloat8_b on upload
weight_tt = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

# Activations stay in bfloat16
x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Matmul with mixed types
output = ttnn.matmul(x_tt, weight_tt, compute_kernel_config=hifi4_config)
# output is bf16 (wider type), accumulated in fp32
```

The key question: does TT-NN's matmul correctly handle bf16 * bf8 with
fp32 accumulation? This needs experimental validation. If the mixed-type
matmul isn't supported, we'd need bf8 for both operands (less ideal).

### 5.3 Activation-Aware Quantization for Blackhole

Combining AWQ-style insights with our hardware:

1. **Profile activation magnitudes** across calibration data for each layer.
2. **Identify salient channels** (top 1% by activation magnitude).
3. **Apply per-channel scaling** before quantizing weights to bf8.
4. **Store scales** as a small bf16 vector per layer (896 elements = 1.75KB).
5. At inference: `output = matmul(x, Q(w * s)) * (1/s)` where the final
   scaling is a cheap elementwise multiply.

This would push bf8 quality closer to bf16 while keeping the memory and
bandwidth benefits. The overhead is one extra elementwise multiply per
matmul -- negligible.

### 5.4 Precision Budget

Thinking about precision as a budget across the 24-layer network:

```
Per-layer error budget: cosine > 0.9998 (to achieve > 0.99 final)
  24 layers * (1 - 0.9998) = 0.0048 total error budget

Current (all bf16 + HiFi4): per-layer ~0.9995 -> final 0.998
  Headroom: 0.9998 - 0.9995 = 0.0003 per layer

With bf8 weights: per-layer ~0.999 (estimated) -> final ~0.976
  This EXCEEDS our error budget!
  Need: activation-aware scaling or keep first/last layers in bf16.
```

**Hybrid strategy:**
- Layers 0-1 and 22-23 (first and last): bf16 weights (precision anchors)
- Layers 2-21 (middle 20): bf8 weights with activation-aware scaling
- All SDPA: bf16 activations + HiFi4 + fp32_dest_acc (non-negotiable)

This preserves the precision anchors at network boundaries while getting
~80% of the bf8 memory savings.

---

## 6. Synthesis: Prioritized Optimization Roadmap

Ranking by (expected impact) * (feasibility) / (implementation effort):

### Tier 1: Low-hanging fruit (days of work)

**1. bfloat8_b weight quantization (selective layers)**
- Expected impact: ~1.5x memory reduction, enabling larger models or larger batch.
- Feasibility: High (TT-NN native support).
- Effort: Small (change dtype on weight upload, validate precision).
- Risk: Precision regression -- must validate per-layer cosine.
- First experiment: Upload one layer's weights as bf8, measure cosine.

**2. Continuous batching scheduler**
- Expected impact: Near-100% throughput utilization in serving scenarios.
- Feasibility: High (we have all the building blocks).
- Effort: Medium (Python scheduler logic, prefill integration).
- Risk: Prefill latency spikes when new sequences enter.
- First experiment: Simple round-robin scheduler with fixed batch=8.

### Tier 2: Medium effort, high impact (weeks of work)

**3. Speculative decoding (0.5B draft for 7B target)**
- Expected impact: 2-3x speedup for a 7B model on single P150.
- Feasibility: Medium (need to port 7B model first).
- Effort: Large (port larger model, implement rejection sampling, KV rollback).
- Risk: Acceptance rate may be low if 0.5B and 7B have different behaviors.
- Prerequisite: Working Qwen2.5-7B inference on Blackhole.

**4. StableHLO interpreter (generalize from Jaxpr)**
- Expected impact: Support any JAX model without manual porting.
- Feasibility: Medium (we know 70% of ops map cleanly).
- Effort: Large (C++ PJRT plugin, op coverage, testing).
- Risk: Performance may be poor without fusion passes.
- First step: Enumerate StableHLO ops emitted by Qwen, confirm TT-NN coverage.

### Tier 3: Research frontier (month+ of work)

**5. Medusa heads for 0.5B**
- Expected impact: 2-3x single-sequence throughput (132 -> ~300 tok/sec).
- Feasibility: Low-medium (need training infrastructure on TT hardware).
- Effort: Large (train heads, implement tree attention, tree verification).
- Risk: 0.5B may be too small for Medusa heads to learn useful patterns.

**6. Full PJRT plugin (StableHLO compiler)**
- Expected impact: Drop-in JAX backend (`JAX_PLATFORMS=tt`).
- Feasibility: Medium-low (significant infrastructure).
- Effort: Very large (MLIR passes, pattern matching, full op coverage).
- Risk: tt-mlir may overtake us; but our work informs their design.

**7. Paged KV cache (true virtual memory)**
- Expected impact: 3-5x more concurrent sequences for fixed memory.
- Feasibility: Depends on tt-metal kernel support for page table indirection.
- Effort: Large (custom kernels or tt-metal modifications).
- Risk: May require upstream changes to tt-metal.

---

## 7. Key Numbers to Remember

| Metric | Value | Source |
|--------|-------|--------|
| Blackhole P150 DRAM | 24 GB | Hardware spec |
| Blackhole P150 cores | 110 Tensix | Hardware spec |
| Blackhole DRAM bandwidth | ~200 GB/s | Measured |
| Qwen2.5-0.5B params | 490M (980 MB bf16) | Our measurement |
| Qwen2.5-0.5B bf8 (est.) | 490M (490 MB bf8) | Estimated |
| Qwen2.5-7B params (est.) | 7.6B (~15 GB bf16) | Public info |
| Our decode latency (batch=1) | 7.6 ms | Experiment 53e |
| Our throughput (batch=1) | 132 tok/sec | Experiment 53e |
| Our throughput (batch=64) | 4,819 tok/sec | Experiment 56 |
| SDPA precision threshold | HiFi4 + fp32_dest_acc required | Experiment 44-46e |
| Matmul precision | 0.999998 cosine in bf16 | Experiment 44 |
| KV cache per sequence | 1.5 MB (24 layers) | Calculated |
| Max batch (current) | ~110 (one core per sequence) | Hardware limit |
| Speculative decoding alpha (typical) | 0.7-0.9 | Literature |
| Medusa speedup (typical) | 2.2-3.6x | Cai et al. 2024 |
| EAGLE speedup (typical) | 2.5-3.8x | Li et al. 2024 |

---

*Research compiled for CS440LX TT-XLA project. All claims grounded in
published papers, our experimental data (experiments 41-56), or
first-principles analysis of the Blackhole P150 architecture.*
