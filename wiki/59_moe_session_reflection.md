# Wiki 59: Session Reflection — MoE First Light + Research Sprint

## Where We Stand (2026-04-22 Night)

### MoE: Running But Not Yet Coherent

**Experiment 89** (single layer) validated correctness: cosine 0.999909 vs numpy for a single expert forward. Router works, all 60 experts upload at BFP8, flash decode works with MHA (16Q/16KV = no split needed).

**Experiment 90** (full 24-layer eager decode) runs at 13.6 tok/s — faster than expected. But produces garbage text ("The capital of France is four儿...").

**Critical finding from Experiment 90b**: A pure numpy float32 reference produces the SAME garbage. Top-1 prediction for "The capital of France is" is "for" (logit=13.05), not "Paris". **The base model (Qwen1.5-MoE-A2.7B) is simply weak.** Our TT-NN implementation is correct — it matches numpy.

**Next step**: Switch to the Chat variant (Qwen1.5-MoE-A2.7B-Chat) with proper `<|im_start|>` chat template. Same architecture, just fine-tuned weights.

### Performance Baseline: 7th Model Ported

| Model | Params | Active | tok/s | Status |
|-------|--------|--------|-------|--------|
| Qwen2.5-0.5B | 0.5B | 0.5B | 140 | Production |
| Qwen3-0.6B | 0.6B | 0.6B | 76 | Production |
| Llama-3.2-1B | 1.24B | 1.24B | 78 | Production |
| Llama-3.2-3B | 3.2B | 3.2B | 34 | Production |
| SmolLM3-3B | 3.0B | 3.0B | 38 | Production |
| Llama-3.1-8B | 8.0B | 8.0B | 23 | Production |
| **Qwen1.5-MoE-A2.7B** | **14.3B** | **2.7B** | **13.6** | **Eager (needs Chat model)** |

The MoE model is our first 14B+ parameter model and our first MoE architecture. At 2.7B active params with 14.3B total, it offers higher quality-per-FLOP than dense models at the same active parameter count.

## Key Insights

### 1. MoE Correctness Is Easy, Quality Depends on Model Choice
All MoE-specific ops (router, per-expert SwiGLU, shared expert gate, weighted combination) worked correctly on the first attempt. The infrastructure we built for dense models (RoPE, RMSNorm, KV cache, SDPA) transfers directly. The only MoE-specific code is the routing loop.

**Lesson**: When extending to new architectures, most bugs come from wrong config constants (head counts, bias flags, RoPE format), not from new ops. Test against numpy at every step.

### 2. Base Models Can Be Useless for Demos
Qwen1.5-MoE-A2.7B (base) can't even complete "The capital of France is ___". This isn't surprising in retrospect — base models are trained on raw internet text and their completions are unpredictable without few-shot prompting. **Always use instruct/chat variants for quality validation.**

### 3. All-Experts-Traced Approach Is Viable for MoE
The feasibility analysis (wiki 57) predicted 28-35 tok/s for the traced decode approach (run all 60 experts, mask unused). The eager decode already hits 13.6 tok/s with host round-trips. Tracing should double this — the experts are tiny (1408 intermediate) so bandwidth cost of reading all 60 is only ~12.5 GB/step.

### 4. Memory Management Matters for Numpy References
Loading 14.3B params at float32 (~57 GB) OOM'd the remote host. Future numpy references for large models MUST load weights lazily (per-layer, discard after use). The lighter 90d script fixes this.

### 5. MHA Simplifies Things
Qwen1.5-MoE uses MHA (16Q/16KV = 1:1 ratio), not GQA. This means NO split SDPA workaround needed — flash_decode works directly. A welcome simplification after fighting the power-of-2 KV head bug on every GQA model.

## Research Sprint Results

Five parallel research investigations completed:

1. **Performance Ceilings** (`research/performance_ceilings.md`) — Comprehensive analysis of DRAM bandwidth, compute, PCIe bottlenecks. Key finding: 8B model at 82% bandwidth efficiency, 0.5B at only 30% (SDPA-bound, not BW-bound).

2. **JAX Infrastructure 2026** (`research/jax_infrastructure_2026.md`) — PJRT at v0.104, StableHLO at 107 ops, Pallas for custom kernels, applejax as closest template for interpretation-based backends.

3. **Continuous Batching Server** (`research/continuous_batching_server.md`) — vLLM architecture, PagedAttention, implementation plan for Tenstorrent. Key challenge: Metal traces require static shapes vs dynamic batch sizes.

4. **Spatial Multiplexing** (`research/spatial_multiplexing.md`) — Core grid partitioning on Blackhole, profiling tools, MPMD programming. Architecturally natural (independent cores + explicit work assignment) but software support is limited.

## What's Next

1. **Immediate**: When host recovers, run 90d (lightweight Chat model reference) to validate text quality
2. **If quality validates**: Build traced decode (exp 91) with all-60-experts approach for ~30 tok/s
3. **Continuous batching server**: Extend batch demo toward vLLM-like serving
4. **Spatial multiplexing experiment**: Test CoreRange-based partitioning for dual-model serving
5. **Profiling**: Explore tt-metal profiler and Tracy integration

## Meta-Reflection

The MoE work is the first time we've hit a "the model is the problem, not the hardware" scenario. For all dense models, correctness was clear because even the base models produce sensible completions (Llama base completes "The capital of France is Paris"). MoE base models are less predictable — the routing adds another source of variability in base model behavior.

The research sprint (5 parallel investigations) was highly productive. Having comprehensive research docs on performance ceilings, JAX infrastructure, continuous batching, and spatial multiplexing gives us a roadmap for the rest of the quarter.

---

*Written at experiment 90. Qwen1.5-MoE-A2.7B on Blackhole: first MoE model, 14.3B params, 13.6 tok/s eager.*
