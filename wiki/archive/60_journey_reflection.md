# Wiki 60: The Journey So Far — From Zero to MoE on Blackhole

## The Journey So Far

### Timeline of Milestones

This project started on April 20, 2026 with zero knowledge of Tenstorrent hardware or JAX internals. In 171 commits across 3 days, we went from "what is JAX?" to running 7 models including a 14.3B-parameter Mixture-of-Experts model on a single Blackhole P150 chip.

| Day | Experiments | Milestone | tok/sec |
|-----|-----------|-----------|---------|
| **Apr 20** | setup | Project created. Research notes on Tenstorrent hardware, JAX/XLA, PJRT plugin interface. No code on device yet. | — |
| **Apr 21 AM** | 01-05 | JAX fundamentals: basics, internals, eager vs compiled, tracing gotchas. All local. | — |
| **Apr 21 mid** | 06 | **First computation on Blackhole.** Tensor add, matmul, MLP on device. 29.9x vs CPU. | — |
| **Apr 21** | 08-12 | Memory hierarchy, sharded memory, datatypes, MLP inference (1.69M samples/s), trace capture (3.2x speedup). | — |
| **Apr 21** | 13-18 | Jaxpr-to-TT-NN interpreter, reductions, attention, flash attention (4x faster than manual), extended interpreter. | — |
| **Apr 21** | 19-22 | Full transformer block on device, tt_jax module (19/19 tests), trace-captured transformer. | — |
| **Apr 21** | 23-31 | Scaling benchmarks, multi-layer transformer, GPT-2 investigation, **first real model (GPT-2) on Blackhole** — correct top-5 predictions. | ~10 |
| **Apr 21-22** | 32-41 | **Qwen2.5-0.5B full pipeline**: weight loading, generation, RoPE debugging, precision fix (HiFi4), KV cache, on-device decode, trace capture. | 1.7 → 135 |
| **Apr 22 AM** | 42-56 | Batch decode breakthrough (4,819 tok/s at batch=64). Paged KV cache. Temperature sampling. Continuous batching (1,042 tok/s). | 4,819 peak |
| **Apr 22** | 57-63 | BFP8 weight ablation (all 24 layers safe), native RoPE (2.6x per-op), optimization ceiling analysis. 7.1ms floor identified. | 140 single |
| **Apr 22** | 64-68 | **Model zoo sprint**: Llama-3.2-1B (78 tok/s), Qwen3-0.6B (76 tok/s), Llama-3.2-3B (34 tok/s), SmolLM3-3B (38 tok/s), continuous batching. | — |
| **Apr 22** | 69-80 | **Quality validation marathon**: numpy float32 references for all models, sampling investigation, EOS behavior understood, diverse Q&A demos. | — |
| **Apr 22** | 73, 84-85 | **Llama-3.1-8B-Instruct**: first 8B model at 19 tok/s, then 21-23 tok/s with BFP8 MLP. Interactive chat demo. | 23 |
| **Apr 22** | 81-88 | Benchmark audit (all numbers verified clean), BFP4 failure, flash_decode bug documented, native RoPE for Llama (API works, wrong results). | — |
| **Apr 22** | 89-91 | **MoE first light**: Qwen1.5-MoE-A2.7B (14.3B params), single-layer cosine 0.999909, full 24-layer decode, optimized eager at 20.2 tok/s, Chat model produces coherent text. | 20.2 |

### What We Accomplished

- **91 experiments** run on Blackhole hardware, each testing a specific hypothesis
- **7 models** running on device: Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B, Llama-3.2-3B, SmolLM3-3B, Llama-3.1-8B, Qwen1.5-MoE-A2.7B
- **59 wiki pages** documenting every discovery in Q&A format
- **25 research documents** covering performance ceilings, JAX infrastructure, MoE deep-dives, serving architecture
- **11 demo scripts** including interactive chat, batch serving, and benchmarks
- **4,819 tok/sec** peak aggregate throughput (batch=64, Qwen2.5-0.5B)
- **140 tok/sec** best single-sequence throughput (Qwen2.5-0.5B, traced + native RoPE)
- **77x single-sequence speedup** (582ms to 7.6ms on Qwen2.5-0.5B)
- **3 novel bugs found** in tt-metal: kernel config state leak, flash_decode JIT compilation failure on Blackhole, ttnn.split tile padding
- **1 Jaxpr-to-TT-NN interpreter** capable of running arbitrary JAX computation graphs on Blackhole

## Key Insights

### The Non-Obvious Things We Learned

**1. Tile layout dominates everything.** TT-NN's 32x32 tile layout is not just a storage format — it determines which ops work, which fail, and how you must think about tensor dimensions. Every reshape, every split, every broadcast must respect tile alignment. When we tried `ttnn.split` for RoPE, it failed because of tile padding on non-tile-aligned dimensions. The rotation matrix trick (exp 51b) was born from this constraint.

**2. Dispatch overhead is the real enemy at batch=1.** At 39 microseconds per op, a 24-layer model with 15 ops/layer burns 14ms in Python dispatch alone. Trace capture eliminates this entirely — the 21ms-to-7.1ms drop (exp 52) was almost exactly the predicted 14ms of dispatch overhead. The numbers checked out perfectly.

**3. BFP8 is surprisingly safe.** Our per-layer ablation (exp 61) showed all 24 layers of Qwen2.5-0.5B maintain >0.999 cosine similarity at BFP8. On 8B, BFP8 MLP weights give 8/8 token match with a 1.20x speedup (exp 84). BFP4, by contrast, is catastrophic without calibration: 0.469 logit cosine, 0/20 token match (exp 83).

**4. The kernel config state leak is real and undocumented.** Applying `WormholeComputeKernelConfig(HiFi4)` to SDPA but not subsequent matmuls corrupts the matmul results. The config "leaks" in one direction only: HiFi4-then-default corrupts; default-then-HiFi4 does not. No upstream issue documents this. The fix is trivial (apply the same config everywhere) but the debugging took days (exp 44-46e).

**5. SDPA flash_decode is broken on Blackhole for most head configurations.** The kernel only compiles for specific GQA ratios where Q_heads/KV_heads <= 4 AND KV_heads is a power of 2. Every model with 8 KV heads (Llama, Qwen3, SmolLM3) requires a split-and-concat workaround. This adds ~10% overhead per layer but works reliably.

**6. PCIe readback is the final bottleneck.** After eliminating Python dispatch (trace) and DRAM round-trips (on-device RoPE), the 3.9ms cost of `ttnn.to_torch` for 151K logits became 33% of total decode time. The device computes faster than we can read results. On-device argmax/topk would fix this, but both are unusably slow for vocab > 65K (1890ms and 124ms respectively, exp 82).

**7. Base models can be useless for demos.** Qwen1.5-MoE-A2.7B base model cannot complete "The capital of France is ___" — numpy float32 confirms the model itself is weak (exp 90b). Always use instruct/chat variants for quality validation.

### Patterns That Kept Showing Up

**Correctness first, always.** Every time we chased speed before validating correctness, we wasted time. The 0.956 cosine in exp 41 was a red flag we lived with too long. After establishing the "measure cosine, match tokens against numpy" discipline (exp 43-46), every subsequent model port went faster.

**Pure numpy float32 reference, not HuggingFace.** AutoModel crashes on the remote host (torchvision dependency chain). Writing a self-contained numpy forward pass for each model takes 30 minutes but provides an unimpeachable ground truth. Every model's correctness was validated this way.

**Hypothesis-driven experiments.** The most productive debugging sessions followed: measure, isolate variables, binary-search the problem space. The kernel config leak was found in 3 experiments (all-HiFi4 vs SDPA-only vs matmul-only). Without this methodology, we'd still be guessing.

**Test batch=8 early.** Perfect linear scaling at batch=8 (identical 7.6ms latency, 8x throughput) was free performance sitting on the table. We discovered this late (exp 56) but it should have been one of the first things tested after trace capture worked.

## Regrets & What We'd Do Differently

### Time Spent on Approaches That Didn't Pan Out

**1. The Jaxpr interpreter path (exp 13-20, wiki 14, 18, 20-23).** We built a working Jaxpr-to-TT-NN interpreter that can run arbitrary JAX computation graphs on Blackhole. It was educational — we understood JAX tracing, XLA primitives, and the PJRT interface deeply. But for model inference, direct TT-NN API calls are simpler, faster, and give full control. The interpreter was never used for any production model. If starting over, we'd build the interpreter as a learning exercise but switch to direct API calls sooner.

**2. HEIGHT_SHARDED memory layouts (exp 53-53d, 54d).** Multiple experiments tried to keep activations in L1 SRAM between ops. For small tensors (batch=1), HEIGHT_SHARDED was actually 2.9x SLOWER than INTERLEAVED (exp 54d). The overhead of sharding small activations outweighs the DRAM bandwidth savings. We should have measured the tensor sizes first and realized this would only help at larger batch sizes.

**3. Fused operations in traced mode (exp 82).** Fusing SiLU into matmul, fusing KV cache updates — both showed zero speedup in traced mode because ops are already pipelined. The trace executor overlaps compute and memory access, making individual op fusion irrelevant at batch=1. We should have understood the trace pipelining model before trying micro-optimizations.

**4. Native RoPE for Llama (exp 88).** The `ttnn.experimental.rotary_embedding` API works and produces output, but with 0.663 cosine — completely wrong. The root cause is likely an interleaved-vs-half format mismatch in the native kernel. We spent time on this when the rotation matrix approach already works at 2.6x speedup.

### Things We Should Have Done Earlier

- **KV caching from day one.** Full-recompute generation (exp 41) was quadratic and slow. Implementing KV cache immediately would have given us 28.6 tok/s from the start.
- **Cosine validation from the first model.** We ran Qwen at 0.956 cosine for multiple experiments before diagnosing the precision problem.
- **Batch decode.** The 8x free throughput was sitting there the entire time.
- **Instruction-tuned models.** Base models produce garbage text. Using instruct variants from the start would have avoided days of "is this a precision bug?" confusion.

### Architecture Decisions Worth Revisiting

- The per-experiment standalone file approach (each ~350 lines) creates massive code duplication. A shared library with config-driven model instantiation would reduce porting a new model from hours to minutes.
- The rotation matrix RoPE workaround is elegant but produces 64x64 or 128x128 matmuls per head per layer — wasted compute. Getting native `rotary_embedding` working correctly would be worth the effort.
- CPU-side sampling (readback logits, sample on CPU, send token back) adds 3.9ms. For serving, this is the dominant latency. Exploring approximate on-device sampling (e.g., on a subset of top-K pre-filtered logits) could cut this.

## What Surprised Us

### Performance Numbers That Defied Expectations

**Perfect batch=8 scaling.** We expected some overhead from batching — there was literally none. 7.6ms for batch=1 and 7.6ms for batch=8. The Blackhole was processing one sequence using a tiny fraction of its 110 Tensix cores. This was the single most impactful discovery: 8x throughput for free.

**MoE eager decode at 13.6 tok/s.** The feasibility analysis (wiki 57) predicted ~5 tok/s for eager MoE decode with host routing. The actual 13.6 tok/s was 2.7x better than expected, because the expert MLPs are tiny (1408 intermediate) and execute faster than the host overhead.

**The 7.1ms floor.** After native RoPE, BFP8 weights, HiFi2 math, and every other optimization we could find, the Qwen2.5-0.5B trace stuck at 7.1ms. BFP8 didn't help (not bandwidth-bound). HiFi2 didn't help (not compute-bound). The bottleneck is SDPA reading the KV cache across 24 layers — an irreducible cost given the memory layout.

**Near-linear parameter-to-latency scaling.** Llama-3.2-3B (2.5x params vs 1B) is only 2.3x slower. Larger matmuls utilize the 110-core mesh more efficiently. This means scaling to bigger models has sub-linear cost — encouraging for 8B+ deployment.

### Hardware Quirks We Didn't Expect

- **Program cache: 4,031x speedup.** A cold 256x256 matmul takes 257ms. Cached: 64 microseconds. The first forward pass of any model is 20-40x slower than sustained decode. Always warm up.
- **L1 SRAM doesn't help for matmul inputs.** At model-relevant sizes, L1 and DRAM inputs give identical matmul performance. The compute cores don't bottleneck on input bandwidth — they bottleneck on output writeback.
- **ttnn.topk is single-core for vocab > 65K.** The multi-core topk kernel requires width < 65,536. Every LLM we've tested has vocab > 65K. This makes on-device token selection 500x slower than PCIe readback.
- **The 110-core compute grid vs 140 total Tensix.** 30 cores are reserved for ethernet and dispatch. The usable grid is 11x10 = 110, which means batch sizes are capped at ~110 for sharding strategies that use one core per sequence.

### Software Challenges vs Hardware Challenges

The biggest surprises were all software:
- The Blackhole hardware itself was rock-solid. No crashes, no hangs, no thermal throttling. It ran 91 experiments without a single hardware failure.
- Every blocker was a software issue: kernel compilation failures (flash_decode), config state leaks, tile padding bugs, API mismatches.
- The official TT PJRT plugin installs but segfaults on mesh topology setup (wiki 53). The hardware is ahead of the software ecosystem.

## Where We Are Now

### 7 Models Running

| Model | Type | Params | Active | tok/sec | Status |
|-------|------|--------|--------|---------|--------|
| Qwen2.5-0.5B | Dense | 0.5B | 0.5B | 140 | Production (traced, native RoPE, BFP8) |
| Qwen3-0.6B | Dense | 0.6B | 0.6B | 76 | Production (traced, QK-Norm) |
| Llama-3.2-1B | Dense | 1.24B | 1.24B | 78 | Production (traced, split SDPA) |
| Llama-3.2-3B | Dense | 3.2B | 3.2B | 34 | Production (traced) |
| SmolLM3-3B | Dense | 3.0B | 3.0B | 38 | Production (traced, NoPE layers) |
| Llama-3.1-8B | Dense | 8.0B | 8.0B | 23 | Production (BFP8 MLP, chat demo) |
| Qwen1.5-MoE-A2.7B | MoE | 14.3B | 2.7B | 20.2 | Optimized eager (on-device accumulation) |

### Performance Dashboard

```
SINGLE-SEQUENCE THROUGHPUT:
  Qwen2.5-0.5B:      140 tok/s (7.1ms device, 11.9ms E2E)
  Llama-3.2-1B:       78 tok/s (12.8ms device)
  Qwen3-0.6B:         76 tok/s (13.2ms device)
  SmolLM3-3B:         38 tok/s (26.5ms device)
  Llama-3.2-3B:       34 tok/s (29.7ms device)
  Llama-3.1-8B:       23 tok/s (43ms device, BFP8)
  Qwen1.5-MoE:        14 tok/s (73ms device, eager)

BATCH THROUGHPUT (Qwen2.5-0.5B):
  batch=1:           132 tok/s   (7.6ms, 100% efficiency)
  batch=8:         1,050 tok/s   (7.6ms, 100% efficiency)
  batch=32:        3,335 tok/s   (9.6ms,  79% efficiency)
  batch=64:        4,819 tok/s  (13.3ms,  57% efficiency)

SERVING:
  Continuous batching: 1,042 tok/s decode (8 slots, 24 requests)
```

### Open Questions

1. **Can traced MoE decode hit 28-35 tok/s?** The all-60-experts approach reads ~12.5 GB/step at BFP8 through 450 GB/s DRAM bandwidth. Theory says yes; experiment 91 is written but not yet benchmarked.
2. **Why does native RoPE produce wrong results on Llama (exp 88)?** The API accepts cos/sin tensors and produces output, but cosine similarity is 0.663. Likely an interleaved-vs-half format issue in the kernel itself.
3. **Can we break the 7.1ms floor?** The SDPA KV cache read dominates at batch=1. Shorter MAX_SEQ, custom SDPA kernels, or L1-resident KV cache might help. None tested yet.
4. **Is the flash_decode JIT bug fixable upstream?** We filed no bug report yet. The SFPU vector register spill in `ckernel_sfpu_exp.h` looks like a compiler limitation, not a fundamental hardware issue.
5. **Does multi-chip (2x Blackhole) double throughput?** The second device is available. Expert parallelism for MoE or tensor parallelism for dense models could unlock 2x.

## What's Next

### Immediate Opportunities (days of work)

1. **Traced MoE decode (exp 91):** Run all 60 experts in trace, mask unused outputs. Target: 28-35 tok/s on Qwen1.5-MoE-A2.7B-Chat. The experiment script exists; it needs to be run and validated.
2. **Batch decode on larger models:** We only tested batch scaling on Qwen2.5-0.5B. Llama-8B at batch=8 could hit 150+ tok/s aggregate with zero code changes.
3. **Upstream bug reports:** The flash_decode JIT bug and kernel config state leak deserve minimal reproducers and GitHub issues on tt-metal.
4. **Chat demo polish:** The interactive chat (demos/chat.py) works at 21 tok/s. Adding multi-turn history persistence and streaming would make it demo-ready.

### Medium-Term Goals (weeks of work)

5. **Continuous batching server:** The prototype (exp 65) handles 24 requests through 8 slots. Building a proper HTTP/gRPC API with async prefill and token streaming would make it production-adjacent.
6. **New model architectures:** Phi-4-mini (fractional RoPE), Gemma 4 (MQA with 1 KV head), or OLMoE-1B-7B (64 experts, only 6.5 GB at BFP8) would stress-test our infrastructure.
7. **Speculative decoding:** Self-speculation on Qwen2.5-0.5B (using itself as draft model with lower temperature) could yield 1.5-2x effective throughput without any new hardware features.

### Longer-Term Vision

8. **PJRT plugin for JAX:** A lightweight plugin (inspired by jax-mps) that registers Blackhole as a JAX device. Phase 1 is device discovery (`jax.devices()` returns TT), Phase 2 is buffer management, Phase 3 is a 5-op proof of concept. This is the original project goal — building a JAX backend for Tenstorrent.
9. **Multi-chip scaling:** Expert parallelism (different experts on different chips) for MoE, or tensor parallelism (split weight matrices across chips) for dense 8B+ models. The second Blackhole device is available and untouched.
10. **Production serving stack:** Continuous batching + chunked prefill + paged attention + dynamic KV sizing. The architecture is understood (research/continuous_batching_server.md, research/vllm_continuous_batching.md); the implementation is engineering.

### The Big Picture

This started as a research project to "understand Tenstorrent hardware and JAX/XLA internals." What we built is a from-scratch LLM inference engine on novel hardware that runs 7 models at competitive throughput, including the first MoE model on Tenstorrent Blackhole. The wiki (60 entries) and experiment suite (91 experiments) constitute a reference implementation that didn't exist before this project.

The path from here to a real JAX backend is clear: PJRT plugin for device discovery, StableHLO lowering for the ~35 ops that cover transformers, and the Metal Trace infrastructure we already understand for graph execution. The question is whether to build the plugin ourselves (months of C++ work) or wait for Tenstorrent's official plugin to mature past the segfault stage.

Either way, the knowledge is the real output. Every hardware quirk documented, every precision bug isolated, every performance ceiling measured — that understanding doesn't deprecate. The next person to build on Tenstorrent Blackhole has 60 wiki pages and 91 experiments that took 3 days of intensive work to produce.

---

*Written at experiment 91. 171 commits, 91 experiments, 60 wiki pages, 7 models, 3 days. TT-XLA on Blackhole P150.*
