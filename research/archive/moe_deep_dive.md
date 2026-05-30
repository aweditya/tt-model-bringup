# Mixture of Experts (MoE) on Tenstorrent Blackhole: A Deep Technical Dive

## 1. MoE Fundamentals

### What is a Mixture of Experts layer?

In a standard transformer, every token passes through the same MLP (feed-forward network) in every layer. The MLP typically accounts for ~2/3 of total model parameters and ~2/3 of FLOPs per forward pass. A Mixture of Experts layer replaces this single monolithic MLP with N parallel "expert" MLPs plus a lightweight **router** (also called a gate) that decides which expert(s) each token should use.

The key insight: you can scale model *capacity* (total parameters) without proportionally scaling *compute* (FLOPs per token). An MoE model with 8 experts, each the size of the original MLP, has 8x more MLP parameters but uses only 1-2x the compute per token (since only 1-2 experts are activated per token).

**Standard transformer layer:**
```
Token -> Attention -> LayerNorm -> MLP -> LayerNorm -> Output
                                  ^^^^
                                  ALL tokens go through the SAME MLP
```

**MoE transformer layer:**
```
Token -> Attention -> LayerNorm -> Router -> Expert_i(token) -> Combine -> LayerNorm -> Output
                                  ^^^^^^    ^^^^^^^^^^^^^^^^
                                  Picks k    Only k of N experts
                                  experts    are activated per token
```

### The Router/Gating Mechanism

The router is a learned linear projection from the hidden dimension to the number of experts:

```
gate_logits = x @ W_gate    # [batch*seq, hidden] @ [hidden, num_experts] -> [batch*seq, num_experts]
gate_probs  = softmax(gate_logits)
top_k_probs, top_k_indices = topk(gate_probs, k)
```

This produces, for each token, the indices of the top-k experts to route to and the corresponding probability weights. The final output is a weighted sum of the selected experts' outputs:

```
output = sum(top_k_probs[i] * Expert_i(token) for i in top_k_indices)
```

### Routing Variants

**Top-k routing (most common):**
- Top-1: Each token goes to exactly 1 expert. Used in Switch Transformer. Minimal compute but lower quality.
- Top-2: Each token goes to 2 experts, outputs are weighted-summed. Used in Mixtral, GShard. Standard choice.
- Top-k (k>2): Diminishing returns. DeepSeek-V2 uses top-6 out of 160 fine-grained experts.

**Expert Choice routing (inverse routing):**
- Instead of each *token* choosing its experts, each *expert* chooses its top-C tokens.
- Guarantees perfect load balance (each expert processes exactly C tokens).
- Downside: some tokens may be processed by 0 experts ("dropped") and some by many.
- Used in some Google research models (Expert Choice paper, Zhou et al. 2022).

**Soft routing / Soft MoE:**
- Instead of hard assignment (token goes to expert or doesn't), compute a soft weighted combination.
- All experts process all tokens but with different weights.
- Eliminates the discrete routing decision but loses the compute savings.
- Not practical for inference efficiency -- defeats the purpose of conditional computation.

### Expert Capacity and Load Balancing

A fundamental problem with learned routing: the router can collapse, sending all tokens to the same expert ("winner-take-all"). This wastes the other experts and creates a compute bottleneck.

**Capacity factor:** Each expert has a buffer of size `capacity = (tokens / num_experts) * capacity_factor`. If more tokens are routed to an expert than its capacity, excess tokens are *dropped* (their MLP output is zero, they pass through via the residual connection). Typical capacity factors: 1.0-1.5.

**Auxiliary load-balancing loss:** Most MoE models add a loss term that penalizes uneven expert utilization:
```
L_balance = alpha * num_experts * sum(fraction_routed_i * mean_gate_prob_i)
```
This encourages the router to spread tokens across experts. Alpha is typically 0.01-0.1.

**Token dropping vs no dropping:** Dropping tokens is acceptable during training (residual connection preserves information) but problematic during inference (lost quality). Modern MoE inference implementations typically do NOT drop tokens -- they process all routed tokens, accepting load imbalance.

### Shared Experts vs Specialized Experts

DeepSeek-MoE introduced the concept of **shared experts** -- a subset of experts that process ALL tokens regardless of routing. The motivation: some knowledge is universal (syntax, common patterns) and shouldn't be duplicated across specialized experts.

```
output = shared_expert(token) + sum(top_k_probs[i] * routed_expert_i(token))
```

This is architecturally equivalent to keeping a "base MLP" and adding routed experts on top. DeepSeek-V2/V3 use 2 shared experts + 6 routed (out of 160 total fine-grained) experts.


## 2. Key MoE Models

### Switch Transformer (Google, 2022)

The foundational modern MoE paper. Key design decisions:
- **Top-1 routing**: Each token goes to exactly 1 expert. Simplest possible routing.
- **Capacity factor**: 1.0-1.25. Tokens exceeding capacity are dropped.
- **Auxiliary loss**: Simple load-balancing loss.
- **Scale**: Up to 1.6T parameters with 2048 experts.
- **Training**: Expert parallelism across TPU pods (each expert on a different TPU core).
- **Key insight**: Top-1 routing works surprisingly well, and simplicity enables scaling to huge expert counts.

### GLaM (Google, 2022)

- 1.2T parameters, 64 experts per layer, top-2 routing.
- Only every other transformer layer is MoE (alternating dense/MoE).
- Demonstrated that MoE can match GPT-3 quality at 1/3 the training compute.

### ST-MoE (Google, 2022)

- Systematic study of MoE training instability (the "router z-loss").
- Introduced router z-loss: penalizes large router logits to prevent overflow and improve stability.
- Top-2 routing with capacity factor 1.25.
- 269B parameters with 32 experts.

### Mixtral 8x7B / 8x22B (Mistral AI, 2024)

The model that made MoE mainstream for open-source.

**Architecture details:**
- Base: Mistral 7B architecture (32 layers, 4096 hidden dim, GQA)
- 8 experts per MoE layer, top-2 routing
- Every MLP layer is replaced with an MoE layer (all 32 layers)
- Total parameters: ~46.7B (8x7B is a misnomer -- the non-MLP parameters are shared)
- Active parameters per token: ~12.9B (2 experts activated + attention + embeddings)
- Router: simple linear layer, no auxiliary load-balancing loss during inference

**Memory vs compute trade-off:**
- Must store all 46.7B parameters in memory (same as a dense 46.7B model)
- But only activates 12.9B parameters per token (compute of a ~13B model)
- Inference: memory-bandwidth-bound for small batch sizes, compute-bound for large batch sizes

**Expert structure (per expert per layer):**
- Gate projection: [4096, 14336] -- up-projection
- Up projection: [4096, 14336] -- parallel up-projection for SwiGLU
- Down projection: [14336, 4096] -- down-projection
- Total per expert per layer: 3 * 4096 * 14336 = ~176M params
- Total MLP params: 8 * 176M * 32 layers = ~45B (the bulk of the model)

**Mixtral 8x22B:**
- Larger base model (22B-class per expert), same 8-expert top-2 structure.
- Total: ~141B parameters, ~39B active per token.

### DeepSeek-MoE / DeepSeek-V2 / DeepSeek-V3 (DeepSeek AI, 2024-2025)

The most architecturally innovative MoE family.

**DeepSeek-MoE (2024):**
- Introduced "fine-grained experts": instead of 8 large experts, use 64 small experts.
- Each fine-grained expert is 1/8 the size of a standard expert.
- Top-6 out of 64 routing (same total compute as top-2 out of 8, but more combinatorial flexibility).
- Key insight: more experts = exponentially more possible expert combinations = better specialization.

**DeepSeek-V2 (2024):**
- 236B total parameters, 21B active per token.
- **Multi-head Latent Attention (MLA):** Compresses KV cache via low-rank projection (not MoE-related but critical for efficiency).
- **MoE structure:** 2 shared experts + 160 routed experts, top-6 routing.
- Each routed expert: hidden_dim -> intermediate (smaller) -> hidden_dim.
- Shared experts always fire, ensuring baseline capability.
- Device-limited expert selection: constrain routing to experts on the same device to reduce cross-device communication.

**DeepSeek-V3 (2025):**
- 671B total parameters, 37B active per token.
- 256 routed experts per layer + 1 shared expert, top-8 routing.
- **Auxiliary-loss-free load balancing:** Instead of a loss term, uses a bias term added to router logits that is adjusted dynamically based on expert load. This avoids the representation-degradation issue of auxiliary losses.
- **Multi-token prediction (MTP):** Uses future token prediction as an auxiliary training objective.
- Training cost: ~2.7M H800 GPU hours (remarkably efficient for the model size).

### DBRX (Databricks, 2024)

- 132B total parameters, 36B active per token.
- 16 experts per layer, top-4 routing.
- **Fine-grained experts:** Each expert is relatively small, more experts selected per token.
- Uses a "megablock" implementation for efficient batched expert computation.
- GQA attention (8 KV heads, 32 query heads).
- Reportedly outperformed Mixtral 8x7B and matched LLaMA-2 70B at lower inference cost.

### Qwen-MoE / Qwen2.5-MoE (Alibaba, 2024-2025)

**Qwen1.5-MoE-A2.7B:**
- 14.3B total parameters, 2.7B active.
- 64 experts per layer, top-4 routing + 4 shared experts.
- Fine-grained expert strategy similar to DeepSeek-MoE.

**Qwen2.5-MoE:**
- Not officially released as of early 2025 (Qwen2.5 exists in dense variants: 0.5B to 72B).
- The MoE line continued with Qwen3-MoE models in 2025.

**Qwen3-235B-A22B (2025):**
- 235B total, 22B active per token.
- 128 experts per layer, top-8 routing.
- Thinking/non-thinking modes (can toggle chain-of-thought).

### Snowflake Arctic (Snowflake, 2024)

- **480B total parameters, 17B active per token.**
- Extreme ratio of total-to-active parameters.
- 128 experts per layer, top-2 routing.
- Dense base: 10B parameter transformer (attention + shared dense MLP).
- MoE residual: 128 experts, each relatively small (~3.7B MLP params per expert per layer).
- Key insight: the dense base does heavy lifting, MoE adds specialization on top.
- Designed for enterprise workloads (SQL, code generation).

### Summary Table

| Model | Total Params | Active Params | Experts/Layer | Top-k | Shared Experts |
|-------|-------------|--------------|---------------|-------|----------------|
| Switch-C | 1.6T | ~1.6B | 2048 | 1 | No |
| GLaM | 1.2T | ~97B | 64 | 2 | No |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | No |
| Mixtral 8x22B | 141B | 39B | 8 | 2 | No |
| DeepSeek-V2 | 236B | 21B | 160+2 shared | 6 | Yes (2) |
| DeepSeek-V3 | 671B | 37B | 256+1 shared | 8 | Yes (1) |
| DBRX | 132B | 36B | 16 | 4 | No |
| Qwen1.5-MoE | 14.3B | 2.7B | 64+4 shared | 4 | Yes (4) |
| Snowflake Arctic | 480B | 17B | 128 | 2 | No (dense base) |


## 3. MoE on Tenstorrent Blackhole Hardware

This is where things get interesting. The Blackhole P150's architecture has properties that are both uniquely advantageous and uniquely challenging for MoE.

### 3.1 The Core-to-Expert Mapping Hypothesis

**Hypothesis:** With 110 Tensix cores and 8 experts (Mixtral-style), we could assign ~13 cores per expert, with each group of cores holding one expert's weights in L1 SRAM.

**Memory analysis:**

Each Tensix core has 1.5 MB L1 SRAM. But this is NOT all available for weights -- we need space for:
- Kernel binaries (~tens of KB)
- Circular buffer space (~100-200 KB for double-buffered tiles)
- Intermediate activations
- Usable for weights: conservatively ~1.0 MB per core

With 13 cores per expert: 13 * 1.0 MB = **~13 MB per expert in L1**.

A Mixtral expert's weights per layer:
- Gate: 4096 * 14336 * 2 bytes (BF16) = 112 MB
- Up: 4096 * 14336 * 2 = 112 MB
- Down: 14336 * 4096 * 2 = 112 MB
- Total: **336 MB per expert per layer**

**Verdict: L1 cannot hold a full Mixtral expert.** 13 MB << 336 MB. Not even close. L1 is 26x too small.

**What COULD fit in L1?**
- 13 MB per expert allows for weight *tiles* to be streamed, not statically resident.
- A fine-grained expert (DeepSeek-style, 1/8 to 1/16 the size): 336/16 = ~21 MB -- still doesn't fit, but closer.
- A very small MoE model where each expert is ~10-13 MB: e.g., hidden=512, intermediate=2048, 3 matrices at BF16 = 3 * 512 * 2048 * 2 = ~6 MB. This WOULD fit.

**The realistic picture:** Expert weights live in DRAM (32 GB total). The question is whether the routing/dispatch pattern can efficiently stream the RIGHT expert's weights to the RIGHT cores.

### 3.2 DRAM Capacity for MoE Models

Total DRAM on our P150: **32 GB**.

| Model | Total Params | BF16 Size | Fits in 32 GB? |
|-------|-------------|-----------|----------------|
| Qwen1.5-MoE-A2.7B | 14.3B | ~28.6 GB | Barely (tight) |
| Mixtral 8x7B | 46.7B | ~93.4 GB | No |
| DBRX | 132B | ~264 GB | No |
| DeepSeek-V2 | 236B | ~472 GB | No |

At BF16, only very small MoE models fit. At INT8/FP8:

| Model | Total Params | FP8 Size | Fits in 32 GB? |
|-------|-------------|----------|----------------|
| Qwen1.5-MoE-A2.7B | 14.3B | ~14.3 GB | Yes |
| Mixtral 8x7B | 46.7B | ~46.7 GB | No |

**Conclusion:** On a single Blackhole P150, we're limited to small MoE models (~14B total parameters at FP8, ~7B at BF16). This points toward:
1. Qwen1.5-MoE-A2.7B as a realistic first target
2. Multi-chip setups for larger models (Tenstorrent's Galaxy architecture supports this)
3. Custom small MoE architectures designed for the hardware

### 3.3 The Routing Decision: On-Device or Host?

The router is a simple operation: linear projection + top-k. Concretely:
```
gate_logits = hidden_states @ W_gate     # [batch*seq, hidden] @ [hidden, num_experts]
gate_probs = softmax(gate_logits, dim=-1)
top_k_values, top_k_indices = topk(gate_probs, k)
```

**Can we do this on-device?**

The linear projection and softmax are standard TT-NN ops -- no issue. The `topk` operation is the question.

TT-NN has `ttnn.topk` -- this op does exist. For a small number of experts (8-64), the topk is over a very short dimension and is trivially fast. The router computation itself is negligible compared to the expert MLPs.

**The real issue: what happens AFTER routing?**

After topk, we have per-token expert assignments: token 0 goes to experts [2, 5], token 1 goes to experts [0, 3], etc. We need to:
1. **Scatter** tokens to their assigned experts (group tokens by expert)
2. **Execute** each expert's MLP on its assigned tokens
3. **Gather** results back and weight-combine them

This scatter/gather pattern is where Tenstorrent's architecture becomes both interesting and challenging.

### 3.4 The Scatter/Gather Problem on Tenstorrent

**On a GPU:** MoE scatter/gather is implemented via:
- Permutation indices computed on-device
- `torch.index_select` / scatter operations to reorder tokens
- Batched matrix multiply with expert dimension (Megablocks approach)
- Or: padding each expert's token batch to a fixed size and masking

**On Tenstorrent:** The NoC provides flexible point-to-point and multicast data movement between cores. In principle, we could:
1. Run the router on a few cores
2. Use NoC DMA to send each token's hidden state to the core(s) assigned to its expert
3. Each group of cores runs its expert MLP
4. Results are gathered back via NoC

But this requires **data-dependent NoC routing**, which is fundamentally at odds with TT-NN's programming model. TT-NN operations have *statically defined* data movement patterns -- the reader/writer kernels are compiled with predetermined source/destination addresses. The routing decision changes which data goes where, requiring different DMA patterns per forward pass.

**Three possible approaches:**

**Approach A: Pad-and-Mask (simplest, wasteful)**
- Run ALL experts on ALL tokens
- Mask out results for non-selected experts
- Zero compute savings -- defeats the purpose of MoE for compute
- But: static computation graph, works with traces
- Total compute: N_experts * (cost of one expert) per token -- worse than dense!

**Approach B: Pre-computed Permutation (practical)**
- Compute routing on host (or on device then read back to host)
- Permute tokens into expert-grouped batches on host
- Send each expert batch to device as a separate matmul
- Gather results on host, weight-combine, send back to device
- Works today with TT-NN as-is
- Cost: host round-trips for routing decisions (kills latency)

**Approach C: On-device Dynamic Dispatch (ideal, requires custom kernels)**
- Write custom Metalium kernels that:
  1. Run the router on a designated "control core"
  2. Use the router output to program NoC DMA transfers, sending tokens to expert-assigned cores
  3. Each expert's core group runs its MLP on received tokens
  4. Results are gathered back via NoC
- This is essentially building a **software-defined data-dependent switch** on the NoC
- Feasible in principle (RISC-V cores are programmable, NoC is flexible)
- Not possible with TT-NN high-level ops -- requires TT-Metalium kernel development
- Breaks trace capture (dynamic data movement)

**Approach D: Batched Grouped GEMM (the Megablocks approach)**
- Instead of scatter/gather, reformulate MoE as a **block-sparse matrix multiply**
- Concatenate all expert weights into one large matrix
- Use token-to-expert assignments to define which blocks of the matrix to compute
- This is the Megablocks / grouped GEMM approach (Gale et al., 2023)
- TT-NN would need a grouped/block-sparse matmul op -- unclear if this exists
- On Tenstorrent, this maps naturally to assigning different core groups to different block regions of the weight matrix

### 3.5 Trace Capture and Dynamic Routing: The Fundamental Tension

Our experiments showed that trace capture gives 2-3x speedup by eliminating dispatch overhead. But traces require **static computation graphs** -- all tensor shapes, memory addresses, and data movement patterns are baked in at capture time.

MoE routing is inherently dynamic: different tokens go to different experts, so the computation pattern changes every forward pass.

**Ways to handle this:**

1. **Fixed-shape per-expert batches with masking:** Allocate a fixed-size buffer per expert. Pad shorter batches, mask computation for padding tokens. Shapes are static, computation is wasted on padding.

2. **Maximum-capacity traces:** Pre-allocate expert buffers at maximum capacity (all tokens routed to one expert). Waste memory but maintain static shapes. This is how some CUDA MoE implementations handle CUDA Graphs.

3. **Multiple traces for common routing patterns:** If routing is predictable (e.g., position-dependent), pre-capture traces for common patterns. Unlikely to work for learned routers.

4. **Abandon traces for MoE layers:** Use eager dispatch for MoE layers, traces for everything else. The MoE dispatch overhead becomes the cost of dynamism.

5. **Host-side routing with traced experts:** Do routing on host, then call pre-traced individual expert computations. Each expert is a separate static trace. The dispatch cost is per-expert-invocation, not per-op.

Option 5 is likely the most practical near-term approach: pre-capture a trace for each expert's MLP, do routing on host (or device + readback), then dispatch the relevant expert traces.

### 3.6 NoC as a Token Router

The Blackhole NoC is a 2D torus with per-hop latency of ~9ns. NoC multicast is natively supported -- one core can send data to a rectangular grid of destination cores in a single operation.

For MoE, the NoC multicast maps naturally to broadcasting tokens:
- After routing, each token needs to reach 2 expert groups (for top-2 routing)
- The "control core" (or host) could issue multicast sends: token_0 to cores [0-12] (expert 0) and cores [26-38] (expert 2)
- Each expert group has the token data in L1 and can begin computation

The challenge: these multicast destinations are data-dependent. Standard TT-NN ops use fixed multicast patterns. Custom Metalium kernels would need to construct multicast commands based on router output.

**NoC bandwidth consideration:**
- Each direction: 32 bytes per cycle per hop
- At 1 GHz: 32 GB/s per direction per NoC
- With 2 NoCs: 64 GB/s total
- For a token with hidden_dim=4096, BF16: 8 KB per token
- At 64 GB/s: can route ~8M tokens/sec across the chip
- This is NOT the bottleneck -- the expert computation (matmuls) dominates

### 3.7 Expert Parallelism: A Natural Fit

The multi-core architecture of Tenstorrent is arguably a better fit for expert parallelism than GPUs:

**On GPUs:**
- Expert parallelism requires placing different experts on different GPUs (or GPU clusters)
- Cross-GPU communication for token routing (NVLink/InfiniBand)
- Within a GPU, experts share the same SM array -- can't truly isolate experts to different SMs
- Megablocks/grouped GEMM works around this by reformulating as block-sparse compute

**On Tenstorrent:**
- Cores are physically separate processors with local SRAM
- We CAN assign specific core groups to specific experts
- Each core group independently streams its expert's weights from DRAM and computes
- No contention between expert groups (separate L1, separate compute units)
- The NoC provides the interconnect for token routing between groups

This maps to a **physical partitioning** of the chip:

```
Blackhole P150 (110 cores, 11x10 grid):

Expert 0: cores [0,0]-[2,4]  (15 cores)   Expert 4: cores [6,0]-[8,4]  (15 cores)
Expert 1: cores [0,5]-[2,9]  (15 cores)   Expert 5: cores [6,5]-[8,9]  (15 cores)
Expert 2: cores [3,0]-[5,4]  (15 cores)   Expert 6: cores [9,0]-[10,4] (10 cores)
Expert 3: cores [3,5]-[5,9]  (15 cores)   Expert 7: cores [9,5]-[10,9] (10 cores)

Router: could run on any small subset, or on the host
Shared expert: could use a dedicated core group, or time-share
```

With 8 experts and 110 cores: ~13 cores per expert. For top-2 routing, 26 cores are active per token, achieving ~24% utilization per token. But across a batch of tokens with diverse routing, ALL cores stay busy -- different tokens activate different expert groups simultaneously.

**This is the key advantage:** Unlike dense models where all cores process the same computation, MoE naturally distributes different work to different cores. With sufficient batch size, we get near-100% core utilization while each individual token only pays the compute cost of 2 experts.

### 3.8 DRAM Bandwidth: The Real Bottleneck

For inference (especially at small batch sizes), MoE models are **memory-bandwidth bound**, not compute bound. Each token needs to read 2 experts' full weights from DRAM.

Our Blackhole P150 specs:
- DRAM bandwidth: ~448-512 GB/s (across 8 GDDR6 channels)
- BF16 expert weights per layer (Mixtral-scale): 336 MB per expert
- For top-2: 672 MB weight reads per layer
- At 512 GB/s: 672 MB / 512 GB/s = **1.3 ms per layer just for weight loading**
- 32 layers: 32 * 1.3 = **41.6 ms per token** -- only ~24 tokens/sec

Compare to a dense 13B model (same active compute):
- ~26 GB total weights
- 26 GB / 512 GB/s = 50.8 ms for full model -- **but weights are read ONCE and reused**
- Per layer: ~0.81 GB / 512 GB/s = 1.6 ms

Wait -- the MoE model doesn't actually read ALL expert weights. It only reads the 2 selected experts per layer. But with 8 different experts stored in DRAM, the memory access pattern becomes scattered:

- Dense model: sequential read through weight memory
- MoE model: jump between expert weight regions based on routing

**Possible optimization: weight caching in L1**

If tokens in the same batch route to the same expert (likely!), we can amortize weight loading:
1. Load expert_i weights from DRAM to the core group's L1 (streaming through tiles)
2. Process ALL tokens routed to expert_i through those weights
3. Move to the next expert

This is essentially **batched expert execution**: group all tokens by expert, then process each expert group. This converts the random-access pattern to sequential access and maximizes DRAM bandwidth utilization.

With batch size B and top-2 routing across 8 experts:
- Expected tokens per expert: B * 2 / 8 = B/4
- Each expert's weights are loaded once, amortized across B/4 tokens
- At B=32: 8 tokens per expert -- reasonable amortization
- At B=128: 32 tokens per expert -- excellent amortization

### 3.9 Comparison: MoE on Tenstorrent vs GPU

| Aspect | GPU (A100/H100) | Blackhole P150 |
|--------|-----------------|----------------|
| Expert placement | All experts in HBM, share SMs | Experts in DRAM, can assign core groups |
| Token routing | Permutation ops in global memory | NoC transfers between core groups |
| Expert compute | Block-sparse GEMM (Megablocks) | Per-group matmul with streamed weights |
| Memory bandwidth | 2-3 TB/s (HBM3) | 448-512 GB/s (GDDR6) |
| Total memory | 80-96 GB | 32 GB |
| Core isolation | SMs time-shared | Cores physically partitioned |
| Dynamic dispatch | CUDA dynamic parallelism | Custom Metalium kernels needed |
| Multi-chip | NVLink (900 GB/s) | Ethernet (800 Gb/s per port) |

**Tenstorrent advantages:**
- Physical core partitioning eliminates expert compute interference
- NoC provides flexible, programmable data routing between core groups
- Per-core L1 SRAM can cache frequently-used expert weight tiles
- Explicit data movement model gives full control over expert scheduling

**Tenstorrent disadvantages:**
- 4-6x lower memory bandwidth than H100 (GDDR6 vs HBM3)
- 2-3x less total memory (32 GB vs 80 GB)
- No existing MoE-optimized ops in TT-NN
- Dynamic routing breaks trace capture
- No grouped GEMM / block-sparse matmul primitive (would need to be built)


## 4. Implementation Challenges

### 4.1 Dynamic Routing vs Static Computation Graphs

This is the single biggest challenge. Our entire inference pipeline relies on trace capture for performance (3.23x speedup). MoE routing is inherently dynamic.

**Resolution strategy:** Hybrid static/dynamic pipeline.

```
STATIC (traced):
  Attention -> LayerNorm -> Router Linear -> Softmax

DYNAMIC (eager or host-orchestrated):
  TopK -> Token grouping -> Expert dispatch

STATIC (traced, one trace per expert):
  Expert_i MLP (fixed max-batch-size, padded)

STATIC (traced):
  Weighted combine -> Residual -> LayerNorm
```

The dynamic portion is limited to topk + token grouping, which is cheap. The expensive expert MLPs are each captured as static traces.

**Cost of dynamism:** The non-traced routing + dispatch adds maybe 50-200us per layer (host round-trip for routing decision + expert trace selection). For 32 layers, that is 1.6-6.4 ms of overhead -- acceptable if the expert compute dominates.

### 4.2 Load Balancing Across Cores

With top-2 routing and 8 experts, the expected load is uniform (each expert gets batch_size/4 tokens on average). But variance exists:
- Worst case: all tokens route to the same 2 experts (2 groups of 13 cores overloaded, 6 groups idle)
- In practice: learned routers produce moderately balanced loads, especially with auxiliary loss

**On Tenstorrent:** Load imbalance means some core groups finish early and idle while others are still computing. Since cores are physically independent, there's no "work stealing" -- a core running expert 3 can't help with expert 7's overflow.

**Mitigations:**
1. Use expert capacity limits with token dropping (quality trade-off)
2. Pad all expert batches to the same size (compute waste)
3. Accept imbalance and let pipeline overlap hide it (next layer's routing starts while current layer's straggler finishes)

### 4.3 Expert Parallelism vs Data Parallelism vs Tensor Parallelism

For MoE models, there are three parallelism strategies that interact:

**Expert parallelism (EP):** Different experts on different devices/core groups.
- Natural for Tenstorrent: each core group = one expert.
- Single-device EP: partition 110 cores into 8 groups.
- Multi-device EP: one expert per chip (or multiple experts per chip).

**Tensor parallelism (TP):** Split each expert's weights across multiple cores.
- Each expert's matmul is distributed across its core group (we already do this for dense matmuls).
- With 13 cores per expert: the expert's [4096, 14336] matmul is tiled across 13 cores.

**Data parallelism (DP):** Replicate experts, each replica handles different tokens.
- Less natural for MoE since different tokens need different experts.
- Useful for large batch sizes where a single expert group can't keep up.

**For single-chip Blackhole:** EP + TP is the natural combination.
- EP at the macro level: 8 expert groups
- TP within each group: 13 cores collectively compute the expert's matmul
- DP not needed at single-chip scale

### 4.4 Memory Requirements: Quantification

For Qwen1.5-MoE-A2.7B (our most realistic target):
- 14.3B total parameters at FP8 = **~14.3 GB** (fits in 32 GB DRAM)
- 2.7B active parameters per token at FP8 = ~2.7 GB weight reads per forward pass
- At 512 GB/s: 2.7 GB / 512 = **5.3 ms per forward** (theoretical bandwidth-limited floor)
- That's ~190 tokens/sec upper bound from bandwidth alone

Structure:
- 24 transformer layers
- Each MoE layer: 64 experts + 4 shared experts, top-4 routing
- Each expert (fine-grained): small MLP, ~50M params per expert per layer
- Shared expert: ~200M params per layer
- Attention: standard GQA

Expert weight per layer: 64 * 50M * 1 byte (FP8) = **3.2 GB per MoE layer**
But only 4 experts activated: 4 * 50M * 1 byte = **200 MB per layer per token**
Plus shared experts: 4 * 200M * 1 byte = **800 MB per layer**
Total weight reads per layer per token: ~1 GB
For 24 layers: ~24 GB weight reads per forward pass

This means we're essentially reading most of DRAM per forward pass -- bandwidth-bound at ~21 ms per token (~48 tokens/sec).

For comparison, a dense 2.7B model at FP8:
- 2.7 GB total weights
- 2.7 GB / 512 GB/s = 5.3 ms per forward
- ~190 tokens/sec

**The MoE penalty:** Although MoE uses the same active compute as a 2.7B dense model, it reads ~9x more weight data from DRAM (all expert weights, even unused ones, are interleaved in DRAM). Wait -- no. We only read the ACTIVATED experts' weights. The key question is whether TT-NN can selectively read just the activated expert weight tiles from DRAM.

**Corrected analysis for Qwen1.5-MoE:**
- Per layer: read 4 activated experts (~200 MB) + 4 shared experts (~800 MB) + attention weights
- This is roughly equivalent to reading the active 2.7B params
- So the bandwidth cost is similar to a 2.7B dense model IF we only read the activated weights
- The overhead is: routing decision + token scatter/gather + any extra DRAM reads for non-contiguous expert weights

### 4.5 Multi-Chip MoE

Tenstorrent's architecture is designed for multi-chip scaling:
- Each chip has 4 QSFP-DD 800G ports
- Standard Ethernet interconnect (no proprietary fabric)
- Galaxy: 32+ chips

For Mixtral 8x7B (46.7B params):
- At FP8: ~46.7 GB total -- needs 2 chips (32 GB each)
- Could place 4 experts per chip
- Token routing between chips uses Ethernet (~100 GB/s per port)
- A token routed to an expert on another chip: send 8 KB hidden state over Ethernet → expert compute → send 8 KB result back
- Ethernet latency: ~few microseconds per transfer
- Totally feasible for multi-chip MoE

For DeepSeek-V3 (671B params):
- At FP8: ~671 GB -- needs ~21 chips
- 256 experts spread across ~21 chips (~12 experts per chip)
- The multi-hop routing becomes more complex
- This is where Tenstorrent's Ethernet-based scaling has an advantage: standard networking, no proprietary interconnect lock-in


## 5. TT-NN Operations Needed for MoE

### 5.1 Router

```python
# All of these exist in TT-NN today:
gate_logits = ttnn.linear(hidden_states, W_gate)     # [B*S, hidden] @ [hidden, N_experts]
gate_probs = ttnn.softmax(gate_logits, dim=-1)       # [B*S, N_experts]
top_k_vals, top_k_idx = ttnn.topk(gate_probs, k)    # [B*S, k], [B*S, k]
```

The router is straightforward. The question is whether `ttnn.topk` returns results on-device (it should) and whether we can use those results for subsequent dispatch decisions without a host round-trip.

### 5.2 Token Scatter (Grouping Tokens by Expert)

This is the missing primitive. We need to:
1. Take a tensor of token hidden states [B*S, hidden]
2. Take expert assignments [B*S, k]
3. Produce per-expert token batches: expert_i gets the tokens assigned to it

**Options with existing TT-NN ops:**

- `ttnn.embedding` or `ttnn.gather`: Could be used for permutation, but the indices are data-dependent.
- `ttnn.where` / masking: Create per-expert masks and multiply. Works but wastes compute (applies mask for all experts to all tokens).
- Manual: Read routing indices back to host, construct per-expert batches, send to device.

**What's really needed:** A scatter/permute op that takes data-dependent indices and reorders tokens. This is equivalent to `torch.index_select` or `scatter` with a per-element index tensor.

### 5.3 Per-Expert MLP

Once tokens are grouped by expert, each expert runs a standard MLP:

```python
# SwiGLU expert (Mixtral-style):
gate = ttnn.linear(x, W_gate_i)        # [n_tokens, hidden] @ [hidden, intermediate]
up = ttnn.linear(x, W_up_i)            # [n_tokens, hidden] @ [hidden, intermediate]
gate_activated = ttnn.silu(gate)        # [n_tokens, intermediate]
hidden = gate_activated * up            # [n_tokens, intermediate]
output = ttnn.linear(hidden, W_down_i)  # [n_tokens, intermediate] @ [intermediate, hidden]
```

This is standard TT-NN -- just linear layers and activations. The challenge is that each expert has different weights and potentially different batch sizes.

**Batched execution strategy:**
- If all experts have the same architecture (same shapes), we can pad all expert batches to the same size and run a single "batched matmul" with an expert batch dimension.
- TT-NN matmul supports batch dimensions, so: `ttnn.matmul(expert_inputs, expert_weights)` where both have shape [N_experts, max_tokens, hidden] and [N_experts, hidden, intermediate].
- This wastes compute on padding but maintains static shapes for trace capture.

### 5.4 Weighted Combine

After expert computation:
```python
# expert_outputs: [B*S, k, hidden] (k expert outputs per token)
# top_k_vals: [B*S, k] (routing weights)

# Weighted sum:
combined = ttnn.sum(expert_outputs * top_k_vals.unsqueeze(-1), dim=1)  # [B*S, hidden]
```

Standard TT-NN ops: multiply + reduce sum. No issues.

### 5.5 Summary: Op Coverage

| Component | TT-NN Op | Status |
|-----------|----------|--------|
| Router linear | `ttnn.linear` | Exists |
| Router softmax | `ttnn.softmax` | Exists |
| Router topk | `ttnn.topk` | Exists |
| Token scatter | `ttnn.index_select` / custom | Gap -- needs investigation |
| Expert MLP (linear) | `ttnn.linear` / `ttnn.matmul` | Exists |
| Expert MLP (activation) | `ttnn.silu`, `ttnn.gelu`, etc. | Exists |
| Token gather | `ttnn.index_select` / custom | Gap -- needs investigation |
| Weighted combine | `ttnn.mul` + `ttnn.sum` | Exists |
| Batched/grouped matmul | `ttnn.matmul` with batch dim | Exists (need to verify grouped variant) |


## 6. Practical Roadmap for MoE on Blackhole

### Phase 1: Validate Feasibility (Host-Orchestrated MoE)

1. Pick a small MoE model: **Qwen1.5-MoE-A2.7B** (14.3B params, fits at FP8)
2. Implement the dense parts (attention, layernorm, embeddings) using our existing TT-NN interpreter
3. For MoE layers:
   - Run router on device (linear + softmax + topk)
   - Read back routing indices to host
   - On host: group tokens by expert, pad batches
   - Send per-expert batches to device, run expert MLPs
   - Read back expert outputs, combine on host
4. Measure end-to-end latency, identify bottlenecks

**Expected bottleneck:** Host round-trips for routing decisions. Each MoE layer requires device→host→device round-trip (~0.5-1 ms based on our transfer benchmarks).

### Phase 2: On-Device Routing (Eliminate Host Round-Trips)

1. Keep routing decision on device
2. Use padded fixed-size expert batches (enables static shapes)
3. All expert MLPs run in parallel on different core groups
4. Weighted combine on device
5. Capture the full MoE layer as a single trace (with padding overhead)

**Key question:** Can we do the token-to-expert permutation on device? Investigate `ttnn.index_select`, `ttnn.gather`, and scatter ops.

### Phase 3: Optimized MoE (Custom Kernels)

1. Implement grouped GEMM / block-sparse matmul as a custom Metalium kernel
2. Expert-parallel execution with dynamic token routing via NoC
3. Weight sharding: place each expert's weights in its core group's L1 (if model is small enough) or use L1 as a streaming cache
4. Fused router + scatter + expert + gather as a single custom op

### Phase 4: Multi-Chip MoE

1. Expert parallelism across chips (different experts on different Blackhole cards)
2. Token routing via Ethernet
3. DeepSeek-V3 style "device-limited routing" to minimize cross-chip traffic


## 7. Key Open Questions

1. **Does `ttnn.topk` work reliably on Blackhole?** We haven't tested it. Need to verify output format (indices + values) and whether it's traceable.

2. **Does TT-NN have a grouped matmul / batched matmul with heterogeneous batch sizes?** Standard batched matmul assumes same batch size for all elements. MoE needs different numbers of tokens per expert.

3. **Can we use `ttnn.index_select` or similar for data-dependent token permutation?** This is the critical scatter/gather primitive.

4. **What is the real DRAM bandwidth utilization for non-contiguous weight reads?** Experts' weights are stored at different DRAM addresses. If the DRAM controller can't efficiently handle scattered reads, the effective bandwidth may be much lower than theoretical.

5. **Can we assign specific core groups to specific operations within a single TT-NN program?** The 2D multicast matmul uses a `compute_with_storage_grid_size` parameter, but can we run two matmuls simultaneously on disjoint core grids?

6. **What is the overhead of running multiple small matmuls (one per expert) vs one large matmul?** If dispatch overhead is ~25us per op, 8 expert dispatches = 200us overhead. A single batched matmul would be ~25us. This 8x dispatch overhead reduction motivates the grouped GEMM approach.

7. **Is there a TT-NN op for conditional execution?** Something like "run this subgraph only if this condition is true" would be ideal for MoE but is unlikely to exist in the current API.


## 8. Why MoE Could Be a Great Fit (and Why It Might Not)

### The Bull Case

1. **Physical core partitioning is natural for expert parallelism.** No other architecture gives you this kind of isolation. Each expert group runs independently with its own L1 SRAM, compute pipeline, and data movement.

2. **The NoC is a programmable interconnect.** Token routing across core groups maps to NoC multicast/unicast operations. The 2D torus topology means any core can reach any other core in a few hops.

3. **Batch-level parallelism is the sweet spot.** With sufficient batch size, all core groups stay busy even though individual tokens only activate 2 experts. This is exactly the compute model Tenstorrent is designed for.

4. **Multi-chip scaling via Ethernet.** MoE's expert parallelism maps cleanly to multi-chip setups, and Tenstorrent's Ethernet interconnect handles cross-chip routing.

5. **Active parameter count is low.** MoE models activate 2-8x fewer parameters per token than their total size suggests. On bandwidth-limited hardware, this means higher tokens/sec than a dense model of equivalent quality.

### The Bear Case

1. **Dynamic routing breaks trace capture.** Our biggest performance optimization (3x speedup) relies on static computation graphs. MoE's data-dependent routing is fundamentally dynamic.

2. **GDDR6 bandwidth is the bottleneck.** At 448-512 GB/s, weight streaming limits throughput. MoE makes this worse by requiring reads from multiple non-contiguous expert weight regions. HBM-based GPUs have 4-6x more bandwidth.

3. **No existing MoE primitives in TT-NN.** We'd need to build scatter/gather, potentially grouped GEMM, and expert-parallel dispatch from scratch.

4. **32 GB DRAM limits model choice.** Most interesting MoE models (Mixtral, DeepSeek, DBRX) don't fit. We're limited to small MoE models on a single chip.

5. **Small-batch inference is memory-bandwidth bound regardless.** For the latency-sensitive single-token-generation case (the most common inference scenario), MoE offers no compute advantage because you're always waiting on DRAM.

### Net Assessment

MoE on Tenstorrent Blackhole is **architecturally promising but practically challenging in the near term.** The core-to-expert mapping is elegant, the NoC provides the routing fabric, and multi-chip scaling is natural. But the immediate blockers -- dynamic routing vs traces, missing TT-NN ops, GDDR6 bandwidth limits, and 32 GB memory ceiling -- mean that a naive implementation would likely be slower than a dense model of equivalent active parameter count.

The path forward is:
1. Start with host-orchestrated MoE on a small model (Qwen1.5-MoE) to validate correctness
2. Measure where time is spent (routing overhead vs expert compute vs data movement)
3. Incrementally move routing and dispatch on-device
4. Invest in custom Metalium kernels only if the architectural advantages prove real in benchmarks

---

*Research conducted 2026-04-22 for Stanford CS440LX tt-xla project.*
*Sources: Switch Transformer (Fedus et al. 2022), GShard (Lepikhin et al. 2021), Mixtral technical report (Mistral AI 2024), DeepSeek-V2/V3 technical reports (DeepSeek AI 2024-2025), DBRX announcement (Databricks 2024), Megablocks (Gale et al. 2023), Expert Choice (Zhou et al. 2022), Corsix Tenstorrent blog series, our experimental data from wiki entries 08/09/12/17/25.*
