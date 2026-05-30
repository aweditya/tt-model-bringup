# Mixed Precision Inference Strategies for Tenstorrent Blackhole

## 1. Data Formats on Blackhole

### bfloat16 (Brain Floating Point 16)

Standard format: 1 sign bit, 8 exponent bits, 7 mantissa bits. This is the native compute format of the Tensix matrix engine — the 372 TFLOPS spec is for bfloat16. Every cycle, the FPU can issue one bfloat16 fused multiply-accumulate per matrix engine lane.

Key property: same exponent range as fp32 (8 bits), so no overflow/underflow issues. But only 7 bits of mantissa means ~0.8% relative precision per operation. For matmuls this compounds across the reduction dimension K — a K=896 dot product accumulates ~sqrt(896) ~ 30x the single-op error, giving ~1-2% relative error in the worst case. Our measurements confirm this: individual matmuls show 0.999998 cosine (essentially lossless).

### bfloat8_b (Block Float 8)

Tenstorrent's custom block floating point format. The "_b" suffix denotes "block" — this is NOT IEEE FP8 (E4M3 or E5M2). The format works as follows:

- Values are grouped into blocks (one block per tile row or column, typically 32 elements)
- Each block shares a single exponent (the maximum exponent in the block)
- Each element stores an 8-bit mantissa relative to the shared exponent
- Effective precision: ~7-8 bits of mantissa, but the shared exponent means elements with smaller magnitude in the block lose precision (their leading bits are zero relative to the shared exponent)

This is fundamentally different from IEEE FP8:
- IEEE E4M3: 4 exponent bits, 3 mantissa bits, per-element — very low precision but wide dynamic range
- bfloat8_b: 8 mantissa bits shared-exponent — high precision for values near the block maximum, degraded for outliers

Measured throughput: 1.08-1.25x faster than bfloat16 depending on size (41.6 vs 38.3 TFLOPS at 896x4864). The speedup comes from reduced memory bandwidth: 8 bits/element vs 16 bits/element halves the data movement, and at model-relevant sizes matmul is often bandwidth-bound rather than compute-bound.

Measured accuracy: 9.5% mean relative error on random 256x256 matmul (vs 7.4% for bf16). The error is modest because the shared exponent preserves the dominant magnitudes well. The problematic case is blocks with high dynamic range — one large outlier forces a large shared exponent, causing small values in the same block to lose precision.

### bfloat4_b (Block Float 4)

Same block floating point scheme but with 4-bit mantissas per element. This exists in TT-NN and we have measured it:

- Throughput: 233.2 TFLOPS (1.32x faster than bf16)
- Accuracy: 56.7% mean relative error — extremely lossy for random data
- Only 16 representable mantissa values per element (4 bits)
- Useful only with quantization-aware training or very aggressive calibration

The speed gain is real (1.32x) but the quality loss is catastrophic without careful quantization. This is the Blackhole equivalent of INT4/NF4 — potentially usable for weight-only quantization with calibration, but not for activations.

### fp32 Accumulation (fp32_dest_acc_en)

The Tensix architecture has a destination accumulator register (DEST) that can operate in either bf16 or fp32 mode. When `fp32_dest_acc_en=True`:

- The matrix engine still computes individual bf16 multiply operations
- But partial sums are accumulated in a 32-bit register instead of rounding back to bf16 after each add
- This eliminates the per-step rounding error in the reduction dimension
- The final result is rounded to bf16 only once, when written back to memory

At the hardware level, the DEST register file in each Tensix core has configurable width. In fp32 mode, it uses twice the register space per element, halving the number of tiles that can be in-flight simultaneously. This is why fp32 accumulation has a throughput cost — though in practice we measured the cost as negligible for our model sizes (the bottleneck was memory bandwidth, not compute).

The `MathFidelity` parameter controls a separate dimension:
- `LoFi`: Truncates input mantissas before multiply (fastest, lowest precision)
- `HiFi2`: Preserves some mantissa bits
- `HiFi3`: Preserves more mantissa bits  
- `HiFi4`: Full mantissa preservation (slowest, highest precision)

Our winning config — `HiFi4 + fp32_dest_acc_en=True` — gives maximum precision at both the multiply and accumulate stages. On our model this achieved 0.998 cosine vs float32 reference with no measurable throughput penalty.

### INT8 and INT4 on Tensix

The Tensix matrix engine is primarily designed for floating-point computation. The situation for integer formats:

- **INT8 matmul:** TT-NN does not expose a general-purpose INT8 matmul in the public API as of early 2026. The FPU lanes are bf16-native. However, the SFPU (Special Function Processing Unit) in each Tensix core can perform integer operations, and there is internal work on INT8 support via the packed math pipeline.
- **INT4:** No direct hardware support for INT4 matmul. The bfloat4_b format is the closest analog — it provides 4-bit quantized values but through the block floating point path, not integer arithmetic.
- **INT32 accumulation for INT8:** The standard pattern (INT8 multiply, INT32 accumulate) is not the native path. Tenstorrent's approach is block floating point rather than pure integer quantization.

The practical implication: on Blackhole, quantization strategies should target bfloat8_b and bfloat4_b rather than INT8/INT4. The block floating point formats are what the hardware accelerates.


## 2. Mixed Precision Strategies for LLMs

### Precision Sensitivity by Operation

Our experiments on Qwen2.5-0.5B provide direct measurements:

**Precision-sensitive ops (need bf16 or higher):**

| Operation | Cosine vs fp32 ref | Why sensitive |
|-----------|-------------------|---------------|
| Softmax (inside SDPA) | 0.985 in bf16, 0.998 with HiFi4+fp32 acc | Exponentiation + normalization amplifies rounding; causal masking creates extreme dynamic range |
| Embedding lookup | N/A (discrete, should be lossless) | Embeddings define the model's vocabulary representation; quantization here shifts all downstream computation |
| Final logit projection | Needs full bf16 | Small differences in logits change token selection; this is the model's "decision boundary" |
| LayerNorm / RMSNorm | 0.9999 in bf16 | The variance computation is a reduction — relatively safe, but quantized inputs would degrade it |

**Precision-insensitive ops (candidates for bf8 or lower):**

| Operation | Cosine vs fp32 ref | Why insensitive |
|-----------|-------------------|-----------------|
| Q/K/V projection matmuls | 0.999998 | Linear transforms; errors are small and don't compound within a single op |
| Output projection matmul | ~0.9999 | Same — linear, no nonlinearity to amplify error |
| Gate/Up projection (MLP) | ~0.9999 | Large matmuls with well-distributed weight values |
| Down projection (MLP) | ~0.9999 | Same |
| SiLU activation | ~0.9999 | Smooth nonlinearity, not precision-sensitive |
| Element-wise multiply (SwiGLU) | ~0.9999 | Simple multiply, errors stay proportional |

The pattern is clear: **nonlinear operations with extreme dynamic range (softmax, exp) are precision-sensitive; linear operations (matmul, add) are not.** This is a universal finding across LLM architectures, not specific to Blackhole.

### The Precision Budget Concept

Think of precision as a budget. The total error in the output is bounded by the sum of per-operation errors, weighted by their amplification through subsequent layers. The optimal strategy spends precision where the error amplification is highest:

1. **Softmax / attention scores:** Highest amplification. The softmax output determines the weighting of all value vectors. A 1.5% error here (our measured 0.985 cosine) propagates through all subsequent computation and compounds across 24 layers to give 0.956 final cosine.

2. **Embedding / unembedding:** Medium amplification. Embedding errors shift the entire representation; unembedding errors directly affect token probabilities.

3. **MLP matmuls:** Low amplification. These are the largest ops by FLOPS (896x4864 gate/up projections dominate compute), but the residual connection limits how much a single matmul's error can affect the final output. A 0.0002% error (0.999998 cosine) per matmul is negligible even after 24 layers.

4. **Residual additions:** Essentially zero amplification for precision purposes.

The actionable insight: we can quantize the MLP matmul weights (which dominate both compute and memory) aggressively while keeping attention in full precision, and the quality impact should be minimal.

### Standard Mixed Precision Schemes

**W16A16 (our current config):**
- Weights: bfloat16, Activations: bfloat16
- With HiFi4 + fp32 accumulation
- Cosine: 0.998, throughput: 132 tok/sec (batch=1, traced)
- This is our quality baseline

**W8A16 (weight-only quantization to bf8):**
- Weights: bfloat8_b, Activations: bfloat16
- The matmul hardware can accept mixed dtypes (bf8 weights x bf16 activations)
- Expected ~1.1x speedup on large matmuls (bandwidth-limited ops benefit from halved weight reads)
- Quality: the shared-exponent format means outlier weights lose precision, but weight distributions in trained models are typically well-behaved (near-Gaussian)
- This is the lowest-hanging fruit for Blackhole

**W4A16 (weight-only quantization to bf4):**
- Weights: bfloat4_b, Activations: bfloat16
- Expected ~1.3x speedup
- Quality: 56.7% mean relative error on random data is alarming, but calibrated quantization (groupwise scaling) can recover most quality
- Requires per-channel or per-group calibration to handle weight magnitude variation
- This is the aggressive option — maximum speed, real quality risk

**W8A8 (both quantized):**
- Weights: bfloat8_b, Activations: bfloat8_b
- The entire data path is 8-bit — maximum bandwidth savings
- Risk: activation outliers are harder to handle than weight outliers (activations are data-dependent)
- LLMs are known to have activation outliers in specific channels (the "outlier features" phenomenon discovered by Dettmers et al.)
- Not recommended without SmoothQuant-style outlier mitigation


## 3. Quantization for Blackhole: Practical Considerations

### TT-NN Mixed-Format Matmul Support

TT-NN's `ttnn.matmul` accepts tensors with different dtypes for the two inputs. The hardware handles the conversion internally:

```python
# This works — bf8 weights, bf16 activations
weight_bf8 = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
activation_bf16 = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
output = ttnn.matmul(activation_bf16, weight_bf8)  # Output is bf16
```

The matrix engine up-converts the bf8 input to bf16 before the multiply, then accumulates in either bf16 or fp32 (depending on config). The speedup comes from reduced DRAM bandwidth for reading the bf8 weight tensor, not from faster compute.

Key questions for experiments:
- Does mixed bf8/bf16 matmul work with HiFi4 + fp32_dest_acc?
- Is the output dtype always bf16 when inputs are mixed?
- Does the program cache hash include input dtypes? (If not, changing dtypes mid-run could use a stale compiled kernel)

### Asymmetric Layer Quantization

Different layers have different sensitivity to quantization. For Qwen2.5-0.5B specifically:

- **First layer (layer 0):** Directly transforms embeddings. Quantization error here propagates through all 24 subsequent layers. Should keep full precision.
- **Middle layers (1-20):** The residual stream has stabilized. These are the safest to quantize — each layer contributes only ~0.0005 cosine error at bf16, so even bf8 error would compound slowly.
- **Later layers (21-23):** Our measurements show layer 21 is a "tipping point" where accumulated error tips into a qualitatively different regime. These should stay at higher precision as a safety margin.
- **LM head (final projection):** Maps hidden states to 151,936 logits. The weight matrix is huge and quantization here directly affects token probabilities. However, because it is applied once (not 24x through residual connections), moderate quantization (bf8) is likely safe.

A practical asymmetric strategy:
```
Layer 0:     bf16 weights (protect embedding transform)
Layers 1-20: bf8 weights  (safe zone, maximum speedup)
Layers 21-23: bf16 weights (protect tipping-point layers)
LM head:     bf8 weights  (large matrix, applied once)
```

This quantizes ~85% of model weights to bf8 while protecting the sensitive edges.

### Block Floating Point and Weight Distributions

The bfloat8_b format's quality depends heavily on the per-block dynamic range. For a block of 32 weights:

- If all weights are similar magnitude (e.g., all near 0.01): shared exponent works well, all 8 mantissa bits are useful
- If one weight is 10x larger than the rest: the shared exponent is set by the outlier, and the smaller weights lose 3-4 bits of effective precision

Neural network weight matrices tend to have near-Gaussian distributions (post-training), which is favorable for block FP — most values in a block are similar magnitude. But:
- Outlier channels exist in transformer weights (the "emergent features" phenomenon)
- Bias terms can have very different magnitudes from weight matrix entries
- The block structure must align with the tile layout (32x32 tiles on Blackhole)

The tile alignment is actually favorable: TT-NN's TILE_LAYOUT already groups 32 consecutive elements, and bfloat8_b's block structure operates on these same groups. There is no misalignment overhead.

### Converting Weights to bf8

The conversion happens at `ttnn.from_torch(tensor, dtype=ttnn.bfloat8_b)` — TT-NN handles the block exponent extraction and mantissa quantization internally. For offline (pre-computed) quantization:

1. Compute per-block scale factors (max absolute value per block of 32)
2. Quantize mantissas relative to block scale
3. Store as bf8 tensors on device
4. Matmul with bf16 activations proceeds normally

The question is whether this naive max-based scaling is sufficient, or whether calibration-based scaling (using representative input data to set scales) improves quality. This is what SmoothQuant and AWQ address.


## 4. Quality vs Speed Trade-offs: Concrete Scenarios

### Scenario A: Current config (baseline)

```
All weights: bf16
All activations: bf16
Compute: HiFi4 + fp32_dest_acc on ALL ops
```

- Quality: 0.998 final cosine, correct top-1 predictions
- Speed: 132 tok/sec (batch=1, traced), 4819 tok/sec (batch=64)
- Memory: 490M params x 2 bytes = 980 MB weights

### Scenario B: Selective fp32 accumulation removal

```
SDPA: HiFi4 + fp32_dest_acc (keep full precision)
MLP matmuls: HiFi4 only (no fp32_dest_acc)
Q/K/V projections: HiFi4 only (no fp32_dest_acc)
```

- Expected quality: Still 0.997+ (individual matmuls showed 0.999998 cosine — the fp32 acc is not needed for them)
- Expected speed: Marginal improvement (fp32 acc cost is small, dominated by bandwidth)
- Risk: The kernel config state leak bug means we CANNOT mix configs between SDPA and matmuls. If we use HiFi4+fp32 on SDPA and HiFi4-only on matmuls, the directional leak (HiFi4+fp32 -> HiFi4-no-fp32) might corrupt the matmuls.
- **Verdict: Not worth the risk.** The uniform config is both safer and nearly free.

### Scenario C: bf8 weights for MLP only

```
Q/K/V/O projections: bf16 weights (attention path)
Gate/Up/Down projections: bf8 weights (MLP path)
SDPA: HiFi4 + fp32_dest_acc
Activations: bf16 everywhere
```

- Expected quality: 0.995+ (MLP matmuls already show 0.9999 cosine in bf16; bf8 adds ~0.1% error per matmul, and there are 3 MLP matmuls per layer x 24 layers = 72 matmuls, but residual connections bound the compounding)
- Expected speed: ~5-8% faster (MLP matmuls are ~60% of total compute; bf8 gives ~10% speedup on those; 0.6 x 0.1 = 6%)
- Memory: Gate+Up+Down = 3 x 896 x 4864 x 2 bytes = ~26 MB at bf16, ~13 MB at bf8, savings = 13 MB per layer x 24 = 312 MB. Total weight memory: 980 - 312 = 668 MB
- **Verdict: Best first experiment.** Low risk, meaningful memory savings, measurable speed gain.

### Scenario D: bf8 weights everywhere except SDPA

```
All projection weights: bf8
SDPA: bf16 Q/K/V/scores (bf16 activations preserved through attention)
All compute: HiFi4 + fp32_dest_acc
```

- Expected quality: 0.990-0.995 (adding bf8 to Q/K/V projections introduces error before the precision-sensitive softmax)
- Expected speed: ~10% faster overall
- Memory: All weights at bf8 = 490 MB (half of bf16)
- Risk: Q/K/V precision errors feed into softmax, which is the known amplifier. This could compound to drop below 0.99.
- **Verdict: Second experiment after Scenario C. Need to measure carefully.**

### Scenario E: bf4 weights for MLP, bf8 for attention projections

```
Q/K/V/O: bf8 weights
Gate/Up/Down: bf4 weights
SDPA: HiFi4 + fp32_dest_acc
Activations: bf16 everywhere
```

- Expected quality: 0.95-0.98 (bf4 is very lossy without calibration)
- Expected speed: ~15% faster
- Memory: ~400 MB total weights
- Risk: High. bf4 without calibration gave 56.7% error on random data. Even with well-behaved weight distributions, this will degrade noticeably.
- **Verdict: Only with calibrated quantization (GPTQ/AWQ). Not a simple dtype swap.**

### Cosine vs Perplexity as Quality Metrics

**Cosine similarity** (what we measure):
- Pro: Fast, deterministic, single-forward-pass
- Pro: Catches systematic errors (wrong scale, wrong direction)
- Con: Doesn't directly measure generation quality
- Con: A cosine of 0.998 could mean "perfect text" or "subtly wrong probability distribution" — it depends on where the 0.2% error falls relative to decision boundaries

**Perplexity** (standard LLM quality metric):
- Pro: Directly measures prediction quality on held-out text
- Pro: Sensitive to the errors that matter — wrong token probabilities
- Con: Requires a dataset (e.g., WikiText-2 or C4)
- Con: Slower — requires many forward passes
- Con: Perplexity differences are hard to interpret in absolute terms (is 15.2 vs 15.4 acceptable?)

**Recommendation:** Use cosine for rapid iteration during development (it catches catastrophic regressions in seconds). Use perplexity as the final validation metric once a configuration looks promising. A meaningful perplexity test on Qwen2.5-0.5B would process ~500-1000 tokens of WikiText-2 and compare perplexity between bf16 and quantized configurations. The baseline bf16 perplexity for Qwen2.5-0.5B is approximately 18-20 on WikiText-2 (small model); a <5% perplexity increase from quantization is generally considered acceptable.


## 5. Advanced Quantization Techniques

### SmoothQuant (Xiao et al., 2023)

**Core idea:** LLM activations have outlier channels where values are 100x larger than typical values. These outliers make activation quantization difficult. SmoothQuant migrates the quantization difficulty from activations to weights by applying a mathematically equivalent per-channel scaling:

```
Y = (X * diag(s)^{-1}) @ (diag(s) * W)  =  X_smooth @ W_smooth
```

where `s` is a per-channel smoothing factor derived from calibration data.

**Interaction with Blackhole:**
- SmoothQuant was designed for INT8 quantization. On Blackhole, the analog is bfloat8_b.
- The per-channel scaling can be folded into the weight matrix offline (no runtime cost).
- The smoothed activations have lower dynamic range, which is favorable for bfloat8_b's block floating point format (smaller per-block dynamic range = fewer wasted mantissa bits).
- The smoothed weights have higher dynamic range, but weight quantization is less sensitive because weights are static and can use per-channel scaling.
- **Compatibility with tile layout:** The per-channel scaling operates along the hidden dimension (896 for Qwen), which aligns with the matmul reduction axis. The smoothing factors can be precomputed and folded into the weight matrix before converting to bf8 on device.
- **Verdict:** Directly applicable. Would enable W8A8 (both weights and activations in bf8) with minimal quality loss. The offline weight transformation is free, and the activation smoothing is a per-channel multiply (negligible cost).

### GPTQ (Frantar et al., 2023)

**Core idea:** Optimal brain quantization (OBQ) applied layer-by-layer. For each layer, GPTQ finds the quantized weight values that minimize the output reconstruction error, using second-order (Hessian) information. It processes columns of the weight matrix sequentially, quantizing each column and adjusting remaining columns to compensate for the quantization error.

**Interaction with Blackhole:**
- GPTQ produces quantized weights offline — the inference path is unchanged
- The output is a quantized weight matrix + optional scale/zero-point per group
- For bfloat8_b: GPTQ would determine the optimal 8-bit mantissa values given the shared block exponent constraint. This is a novel setting — standard GPTQ assumes uniform per-group quantization, but bfloat8_b has a shared exponent within each block.
- For bfloat4_b: This is where GPTQ would be most impactful. The naive bf4 error (56.7%) is catastrophic, but GPTQ's optimal column-wise adjustment could recover significant quality.
- **Tile-based computation consideration:** GPTQ processes weight matrices column by column. The 32x32 tile layout means quantization groups of 32 elements (one tile row) are natural. A GPTQ implementation for Blackhole should use group_size=32 to align with tile boundaries.
- **Verdict:** High value for bf4 quantization. For bf8, the marginal benefit over naive quantization may be small (bf8 is already reasonably accurate). Implementation requires a calibration dataset and ~10 minutes of compute per layer on CPU.

### AWQ (Lin et al., 2024)

**Core idea:** Not all weights are equally important. AWQ identifies the "salient" weight channels — those that correspond to large activation magnitudes — and protects them during quantization by applying per-channel scaling. Similar to SmoothQuant but specifically for weight-only quantization.

```
Quantize(W * s) / s  instead of  Quantize(W)
```

where `s` is chosen to minimize quantization error on the important channels.

**Interaction with Blackhole:**
- AWQ is weight-only, so it maps directly to our W8A16 or W4A16 scenarios
- The per-channel scaling factors `s` are absorbed into the weight matrix offline
- For bfloat8_b: AWQ would adjust the weight magnitudes so that within each 32-element block, the shared exponent is set by the truly important weights rather than random outliers
- This is an excellent fit for block floating point — AWQ essentially optimizes the weight distribution for minimum block FP quantization error
- **Verdict:** Best technique for W4A16 on Blackhole. The per-channel scaling aligns with bfloat4_b's block structure, and the activation awareness means we protect the channels that matter most.

### SpQR (Dettmers et al., 2023)

**Core idea:** Sparse-Quantized Representation. Most weights can be quantized to very low precision (3-4 bits), but a small fraction (~1-5%) of weights are outliers that cause disproportionate quantization error. SpQR stores these outlier weights in higher precision (16-bit) while quantizing the rest aggressively.

```
W = W_quantized (low-bit, dense) + W_outlier (high-bit, sparse)
```

**Interaction with Blackhole:**
- The sparse outlier component is problematic for tile-based computation. Tensix operates on dense 32x32 tiles — a sparse matrix would either need to be padded (wasting bandwidth) or processed through a separate sparse kernel path.
- TT-NN does not currently expose sparse matmul operations for general use
- The dense quantized component maps well to bf4 or bf8
- **Verdict:** Architecturally mismatched with Blackhole's tile engine. The overhead of handling sparse outliers on a dense tile processor likely negates the quality benefit. AWQ or GPTQ are better fits.

### Summary: Technique Compatibility with Blackhole

| Technique | Target format | Blackhole fit | Implementation effort | Expected quality gain |
|-----------|--------------|---------------|----------------------|----------------------|
| SmoothQuant | W8A8 (bf8/bf8) | Excellent | Low (offline weight transform) | Enables W8A8 that otherwise fails |
| GPTQ | W4A16 (bf4/bf16) | Good | Medium (calibration pipeline) | Critical for bf4 quality |
| AWQ | W4A16 or W8A16 | Excellent | Low (per-channel scaling) | Best for block FP formats |
| SpQR | W3A16 (sparse+dense) | Poor | High (sparse kernel needed) | N/A — wrong hardware fit |
| Naive quantization | W8A16 (bf8/bf16) | Trivial | Zero (just change dtype) | Sufficient for bf8 |


## 6. Practical Experiments to Run

### Experiment Series: Precision/Speed Characterization

Each experiment builds on the previous, enabling ablation at each step.

**Experiment A: bf8 Weight Matmul Validation (single layer)**

Goal: Measure cosine impact of bf8 weights on individual matmuls.

```
For each matmul in layer 0 (Q, K, V, O, gate, up, down):
  1. Convert weight to bfloat8_b
  2. Run matmul with bf16 activation input
  3. Measure cosine vs bf16-weight reference
  4. Measure wall-clock time
```

Expected results: cosine ~0.999 per matmul (bf8 adds ~0.1% error), wall clock ~10% faster for large matmuls. This establishes the per-op error budget for bf8 weights.

**Experiment B: bf8 MLP Weights Full Model (24 layers)**

Goal: Measure end-to-end quality with bf8 on the MLP path only (Scenario C).

```
For layers 0-23:
  Q/K/V/O weights: bf16
  Gate/Up/Down weights: bf8
  All compute: HiFi4 + fp32_dest_acc
Measure: per-layer cosine, final logit cosine, top-1 match, generated text quality
```

This is the safest quantization experiment — MLP matmuls showed 0.9999 cosine individually, so bf8 should stay above 0.99 even after 72 MLP matmuls.

**Experiment C: bf8 All Weights (except embeddings)**

Goal: Measure end-to-end quality with bf8 everywhere except embedding lookup.

```
All projection weights: bf8
Embedding: bf16 (lookup table, not quantizable)
LM head: bf8
All compute: HiFi4 + fp32_dest_acc
```

This tests the Q/K/V path in bf8, which feeds into the precision-sensitive softmax. If cosine stays above 0.99, we can use bf8 everywhere.

**Experiment D: HiFi4 vs HiFi2 with bf8 Weights**

Goal: Determine if lower math fidelity is safe when using bf8 weights (since the weights are already approximate, maybe HiFi4's extra precision is wasted).

```
Config sweep:
  1. bf8 weights + HiFi4 + fp32_acc
  2. bf8 weights + HiFi2 + fp32_acc
  3. bf8 weights + HiFi4 + no fp32_acc
  4. bf8 weights + HiFi2 + no fp32_acc
Measure: cosine, throughput
```

If bf8+HiFi2 gives similar quality to bf8+HiFi4, we can stack the speedups.

**Experiment E: Per-Layer Quantization Sensitivity**

Goal: Find which layers are most sensitive to bf8 quantization.

```
For i in 0..23:
  Quantize ONLY layer i's weights to bf8 (all others bf16)
  Run full 24-layer forward
  Measure final cosine
  
This gives a 24-point sensitivity curve showing which layers
are "safe" and which are "hot" for quantization.
```

This directly informs the asymmetric quantization strategy — layers with high sensitivity stay at bf16.

**Experiment F: Throughput Measurement at Different Precision Configs**

Goal: Quantify the actual speedup from bf8 weights in the full traced decode loop.

```
Configs to benchmark:
  1. bf16 weights, HiFi4+fp32 (current baseline)
  2. bf8 MLP weights, HiFi4+fp32
  3. bf8 all weights, HiFi4+fp32
  4. bf8 all weights, HiFi2+fp32
  5. bf8 all weights, HiFi2, no fp32 acc
  
Measure: traced decode ms/tok, batch=1 and batch=8
```

This separates the throughput gains from quality impacts, letting us build a Pareto curve.

**Experiment G: Perplexity Validation**

Goal: Validate the best configuration from experiments A-F using perplexity.

```
1. Encode 1000 tokens of WikiText-2 with tokenizer
2. Run sliding-window perplexity (window=128, stride=64)
3. Compare perplexity: bf16 baseline vs best bf8 config
4. Acceptance criterion: <5% perplexity increase
```

This is the final quality gate before adopting a quantization config for production.

**Experiment H: SmoothQuant Calibration**

Goal: Test whether SmoothQuant enables W8A8 (bf8 weights AND activations).

```
Calibration phase:
  1. Run 100 tokens through the bf16 model
  2. Record per-channel activation statistics (max abs value per channel)
  3. Compute smoothing factors: s = max_act^alpha / max_weight^(1-alpha), alpha=0.5
  4. Apply smoothing to weights offline: W_smooth = diag(s) * W
  5. Convert W_smooth to bf8
  
Inference phase:
  Activations: bf8 (after dividing by s — done as part of preceding layernorm)
  Weights: bf8 (pre-smoothed)
  Measure: cosine, perplexity, throughput
```

If W8A8 works, the entire data path is 8-bit — maximum bandwidth savings and maximum speed.


## 7. Recommended Execution Order

1. **Experiment A** (single layer bf8 matmul) — 30 minutes. Establishes per-op error bounds. If bf8 matmul cosine is below 0.999, we know quantization needs calibration.

2. **Experiment B** (bf8 MLP full model) — 1 hour. The safest full-model test. If this works (cosine >0.995), we have a confirmed speedup with trivial implementation.

3. **Experiment F** (throughput benchmarks) — 1 hour. Quantifies the actual speed gains. If bf8 MLP gives <3% speedup, the complexity isn't worth it.

4. **Experiment E** (per-layer sensitivity) — 2 hours. Informs asymmetric quantization. Only needed if Experiment C (bf8 everywhere) shows quality degradation.

5. **Experiment C** (bf8 all weights) — 1 hour. Tests the aggressive config. If it passes, we skip the per-layer analysis.

6. **Experiment G** (perplexity validation) — 1 hour. Final quality gate on the winning config.

7. **Experiment H** (SmoothQuant) — 3 hours. Only if W8A16 results are promising and we want to push further to W8A8.

8. **Experiment D** (HiFi sweep) — 1 hour. Fine-tuning after the main quantization config is chosen.


## 8. Key Open Questions

1. **Does the kernel config state leak affect mixed-dtype matmuls?** If changing a tensor's dtype triggers a different compiled kernel, the leak bug could resurface. Must test explicitly.

2. **What is the actual memory bandwidth utilization during decode?** If decode is compute-bound (not bandwidth-bound), bf8 weights won't help speed — only memory. If bandwidth-bound, bf8 gives proportional speedup.

3. **Does trace capture work with mixed-dtype weight tensors?** The trace records the op graph including tensor metadata. If bf8 and bf16 tensors coexist in a trace, does replay handle them correctly?

4. **Can we dynamically switch precision per layer within a trace?** If not, asymmetric quantization requires separate traces or a single uniform config.

5. **What is the bf8 weight conversion overhead?** If we convert bf16 weights to bf8 at load time, how much does this add to model initialization? (Likely negligible — one-time cost.)

6. **Does bfloat8_b interact with the fp32_dest_acc flag?** When one input is bf8 and accumulation is fp32, is the up-conversion bf8->bf16->fp32 or bf8->fp32 direct? This affects both precision and throughput.

---

*Research compiled from: experiments 10, 44-46e, wiki pages 10, 33, 36, 39. Blackhole P150 measurements on Qwen2.5-0.5B (490M params). April 2026.*
