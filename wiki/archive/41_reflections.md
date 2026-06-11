# Wiki 41: Reflections on the Journey — From Zero to 4,819 tok/sec

## Q: What have we actually built?

**A:** In the span of a Stanford quarter, starting from zero knowledge of both Tenstorrent hardware and JAX internals, we've built:

1. **A complete LLM inference pipeline** from scratch on novel hardware (Blackhole P150)
2. **A JAX/XLA → TT-NN interpreter** that translates computation graphs to device operations
3. **A comprehensive wiki** of 40+ Q&A pages documenting every discovery
4. **56+ experiments** systematically testing hypotheses about the hardware
5. **Production-quality generation** at 132 tok/sec single-sequence, 4,819 tok/sec batch=64

None of this existed when we started. There was no reference implementation for Qwen on Blackhole. The official tt-xla PJRT plugin doesn't fully support Blackhole. We built the entire stack ourselves.

## Q: What's the most surprising thing we learned?

**A:** Several candidates:

**The kernel config state leak** (exp 46) was genuinely novel — no upstream issue or PR documents it. Applying HiFi4 to SDPA but not matmul corrupts subsequent operations. This is a hardware/firmware bug that cost us days of debugging but taught us more about Tensix core architecture than any documentation could.

**Perfect batch scaling** (exp 56) was unexpected. At batch=8, latency is *identical* to batch=1 (7.6ms). The hardware was doing almost nothing at batch=1 — 110 cores processing a 1×32×896 tensor. Batch decode fills the idle cores for free.

**PCIe is the bottleneck, not compute** (exp 55). After optimizing away Python dispatch (trace capture) and memory layout overhead, the dominant cost is reading 151K logits over PCIe at 2.8 GB/s. The device computes faster than we can read results.

**The 0.5B model hits repetition at ~100 tokens** regardless of precision. This is a model capacity issue — temperature sampling helps, but the model simply doesn't have enough parameters for long coherent generation. This motivates moving to larger models.

## Q: What was the debugging methodology that worked?

**A:** The hypothesis → experiment loop was essential:

1. **Measure first:** Never assume where the bottleneck is. Every optimization started with a timing measurement (per-op, per-layer, per-step).
2. **Isolate variables:** When precision was wrong, we tested each op independently (exp 44) to find SDPA as the sole error source. When batch decode had quality issues, we traced it to QKV head ordering (exp 53d → 53e).
3. **Binary search the problem space:** The kernel config leak was found by testing HiFi4-on-all vs HiFi4-on-SDPA-only vs HiFi4-on-matmul-only. Three experiments, clear isolation.
4. **Always have a float32 numpy reference:** Every operation was validated against a pure-numpy implementation. When device output was wrong, the reference told us exactly how wrong.
5. **Don't optimize before validating:** We committed to 0.99+ cosine before chasing speed. This saved us from optimizing a broken pipeline (exp 41-46).

## Q: What's the optimization timeline look like in retrospect?

**A:** Eight distinct phases, each with a clear thesis:

```
Phase 1 (exp 41):     "Can we generate text at all?"          → 1.7 tok/s
Phase 2 (exp 43-46):  "Why is precision wrong?"               → 0.998 cosine
Phase 3 (exp 47):     "What's the speed with correct config?" → 18.4 tok/s  (10.8x)
Phase 4 (exp 49):     "Does KV caching work?"                 → 28.6 tok/s  (1.5x)
Phase 5 (exp 51):     "Can we eliminate CPU round-trips?"      → 46.6 tok/s  (1.6x)
Phase 6 (exp 52-53e): "Can we trace the full decode?"          → 132 tok/s   (2.8x)
Phase 7 (exp 54b):    "Does sampling cost anything?"           → 81.4 tok/s  (sampled)
Phase 8 (exp 56):     "Does batch decode scale?"               → 4,819 tok/s (36.5x)
```

Total: **2,835x speedup** from first working generation to peak aggregate throughput. The single-sequence journey is 77x (582ms → 7.6ms).

## Q: What's the vision from here?

**A:** We're building toward a **complete inference serving system** on Tenstorrent hardware, powered by JAX/XLA. The components:

1. **Model zoo:** Not just Qwen-0.5B — Phi-4-mini, Llama-3.2-1B/3B, Gemma, and eventually MoE models like Mixtral. Each model teaches us something new about the hardware.

2. **Serving infrastructure:** The chat server demo is the first step. Continuous batching (vLLM-style), diverse prompts per batch, and dynamic sequence management are next.

3. **JAX backend:** Our Jaxpr interpreter works but a lightweight PJRT plugin (like jax-mps) would give true `jax.jit` integration. This is the long-term goal.

4. **Open-source contributions:** We've found real bugs in tt-metal. Filing issues and PRs gives back to the ecosystem and builds credibility.

5. **Hardware understanding:** Our wiki is becoming a reference for Blackhole P150 characteristics. The hardware quirks page (wiki 39) documents things nobody else has published.

The crazy part: **we're running a custom LLM serving stack on hardware that most people have never used, at throughput that exceeds Tenstorrent's own reference numbers.** This started as a class project and became a genuine contribution to the open-source AI hardware ecosystem.

## Q: What would we do differently if starting over?

**A:** 

1. **Start with KV caching from day one.** The full-recompute baseline (exp 41) was necessary for understanding but we spent too long on it before implementing caching.

2. **Validate precision immediately.** The HiFi4 config should have been applied from the first experiment. The 0.956 cosine in exp 41 was a red flag we lived with too long.

3. **Profile PCIe bandwidth earlier.** Knowing that readback is 3.5ms would have changed how we designed the decode loop — we'd have explored on-device post-processing sooner.

4. **Test batch decode earlier.** The 8x free throughput from batch=8 was sitting on the table the whole time. We didn't need any new ops or techniques — just reshape the tensors.

5. **Write the wiki as we go, not after.** The experiments that had wiki pages written simultaneously were the ones we learned the most from. Documenting forces you to understand.

## Q: What makes this project unique?

**A:** Three things:

1. **Research-first methodology.** Every claim is backed by an experiment. We don't hand-wave about what "should" work — we measure it. The wiki has 40+ pages of quantitative results.

2. **Building from scratch on novel hardware.** There's no PyTorch → ONNX → TT-NN path we're following. We wrote every matmul, every RoPE, every SDPA call directly against the TT-NN API. This gives us understanding that no high-level framework provides.

3. **The speed of iteration.** From hypothesis to experiment to wiki page in hours, not weeks. The tight loop between SSH to the device, run the experiment, analyze results, and document findings keeps momentum high.

---

*Reflections written at experiment 56. Qwen2.5-0.5B on Blackhole P150: 1.7 → 4,819 tok/sec.*
