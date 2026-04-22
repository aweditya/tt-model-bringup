# TT-XLA Master Plan & TODO

Living document. Research → Hypotheses → Experiments. Never run out of things to do.

## Current Performance Baseline (2026-04-22)

```
Single sequence:  7.1ms/tok  = 140 tok/sec  (bf8 MLP + native RoPE, traced, paged KV)
Batch=8:          7.5ms/step = 1,073 tok/sec (bf8 MLP, traced)
Batch=32:         9.4ms/step = 3,389 tok/sec
Batch=64:        13.2ms/step = 4,867 tok/sec — PEAK AGGREGATE
Journey:         582ms → 7.1ms = 82x single-sequence speedup
```

---

## Phase 1: Immediate Wins (this week)

### Native RoPE Integration
- [x] Exp 58: Confirm half ↔ interleaved equivalence via permutation (cosine=1.0)
- [x] Exp 59c: `ttnn.experimental.rotary_embedding` = **2.6x faster** (0.053ms vs 0.139ms)
- [x] **Exp 60: Native RoPE in traced decode — 7.1ms/tok = 140 tok/sec** (5% speedup)
  - Theoretical 4.1ms savings → actual 0.3ms (ops pipeline in trace, hidden behind matmuls)
  - Text identical, tokens correct. New single-sequence record!

### Batch + bf8 Scaling Curve
- [x] Exp 59 batch=8+bf8: 1,073 tok/sec
- [x] Exp 59 batch=32+bf8: 3,389 tok/sec
- [x] Exp 59 batch=64+bf8: 4,867 tok/sec

### Per-Layer Quantization Ablation
- [ ] **Exp 61: Quantize ONLY layer i to bf8, keep rest bf16 (24-point sensitivity curve)**
  - H: Layers 0, 21-23 sensitive; layers 1-20 safe
  - Enables targeted asymmetric quantization
- [ ] **Exp 62: Full bf8 (all weights) with traced decode + 100 tokens**
  - H: Exp 57b showed full bf8 works with full recompute; test in traced path

---

## Phase 2: New Models (next 1-2 weeks)

### Architecture Generality
- [ ] **Exp: Llama-3.2-1B port** — same op set as Qwen, different sizes
  - 16 layers, 2048 hidden, 32 Q heads, 8 KV heads (GQA)
  - H: Zero new ops needed, just parameter changes
- [ ] **Exp: Phi-4-mini port** — fractional RoPE (75% of head_dim)
  - H: Single-line RoPE modification
- [ ] **Exp: SmolLM3-3B port** — NoPE variant (skip RoPE every 4th layer)
- [ ] **Exp: Qwen3-0.6B** — latest Qwen with thinking mode

### Larger Models
- [ ] **Exp: Qwen2.5-3B** — 3x larger, test memory/perf scaling
  - H: Fits in DRAM at bf16; ~40-50 tok/sec estimated
- [ ] **Exp: Qwen3-8B with bf8 weights** — aggressive quantization for larger model
  - H: 16GB weights (bf8) fit in 32GB with KV cache room

---

## Phase 3: Serving Infrastructure (weeks 3-5)

### Tier 1: Continuous Batching (days of work)
- [ ] **Per-sequence RoPE** — reshape cos/sin to (1,batch,1,64)
  - H: <1% overhead, enables mixed-position sequences
- [ ] **Sequence masking** — zero embeddings for finished sequences
- [ ] **Host scheduler** — admit/evict at each decode step
- [ ] **Max-batch padding** — capture trace at batch=64, mask empty slots
  - H: Wasted compute < re-trace cost (300ms)

### Tier 2: Production (weeks of work)
- [ ] HTTP/gRPC streaming API
- [ ] Chunked prefill (split long prompts, interleave with decode)
- [ ] Token streaming to client
- [ ] Dynamic KV sizing

---

## Phase 4: Advanced Optimizations

### Speculative Decoding & Medusa
- [ ] **Exp: Self-speculative decoding** — 0.5B speculating on itself
  - H: Alpha ≈ 0.6-0.7, 1.5-2x effective speedup
  - Research: `research/advanced_optimization_techniques.md`
- [ ] **Exp: Medusa head architecture** — 5 heads predicting future tokens
  - H: <1% parameter increase, 2.2-3x speedup
  - Needs: Training data from self-distillation
- [ ] **Exp: Lookahead decoding** — Jacobi iteration, no training needed
  - H: 1.5-2.5x speedup, better at long sequences

### Memory Layout
- [ ] **Exp: HEIGHT_SHARDED activations** — keep tensors in L1 between ops
  - H: 1.5x speedup by eliminating DRAM round-trips (exp 54d showed 2.9x SLOWER for small tensors)
  - Caveat: only viable for larger batch sizes where bandwidth matters
- [ ] **Exp: Fused attention+O projection** — SDPA output stays in L1
  - H: ~1.2x per layer from eliminating one DRAM round-trip

### Quantization Frontier
- [ ] **Exp: SmoothQuant W8A8** — per-channel smoothing for bf8 activations too
  - H: Full data path in bf8, max bandwidth savings
  - Research: `research/mixed_precision_strategies.md`
- [ ] **Exp: GPTQ calibration for bf4** — 4-bit weights with optimal rounding
  - H: Recover from 56.7% naive error to <5% calibrated
- [ ] **Exp: Per-tensor vs per-channel scaling** — find optimal granularity

---

## Phase 5: Compiler & Framework Integration

### C++ Dispatch Path
- [ ] **Exp: Single-layer C++ forward** — measure Python overhead precisely
  - H: 5-10ms overhead reduction (39us/op × 360 ops = 14ms Python overhead)
  - Research: `research/rust_cpp_inference.md`

### PJRT Plugin (JAX native backend)
- [ ] **Phase 1: Device discovery** — `jax.devices()` returns TT device
- [ ] **Phase 2: Buffer management** — allocate/deallocate on device
- [ ] **Phase 3: 5-op proof of concept** — constant, dot_general, add, broadcast, convert
- [ ] **Phase 4: Full transformer** — all ops for Qwen through PJRT
  - Research: `research/stablehlo_lowering.md`, `research/jax_infrastructure_paths.md`

### TT-MLIR Investigation
- [ ] Survey Blackhole support status in tt-forge
- [ ] Test compilation path: tt-forge → StableHLO → TTIR → TTNN

---

## Phase 6: MoE & Multi-Chip

### Mixture of Experts
- [ ] **Exp: Qwen1.5-MoE-A2.7B model load** — 14.3B params, 2.7B active
  - H: Fits in 32GB at INT8; 40-50 tok/sec
  - Research: `research/moe_deep_dive.md`
- [ ] **Exp: Router latency** — linear + softmax + top-k on device vs host
- [ ] **Exp: Host-orchestrated MoE** — dispatch per-expert traces
  - H: 1-2ms dispatch overhead per MoE layer

### Multi-Chip Scaling
- [ ] **Exp: Cross-chip tensor parallel** — split model across 2 Blackhole chips
  - Research: `research/tt_community_contributions.md`
- [ ] **Exp: Expert parallelism** — different experts on different chips

---

## Community Contributions

### Upstream Bug Reports
- [ ] Kernel config state leak (with minimal reproducer)
- [ ] ttnn.argmax 90ms in trace (with timing breakdown)
- [ ] ttnn.split tile padding on Blackhole

### Documentation
- [ ] Tech report: 78x speedup optimization journey
- [ ] Blog post: building a JAX backend for Tenstorrent
- [ ] Contribute test cases to tt-metal CI

### Bounties
- [ ] Auto-optimal matmul config ($2,500 bounty)
- [ ] Model bring-ups ($1,500 each)

---

## Research References

| Area | File | Key Insight |
|------|------|-------------|
| Models | `research/models_landscape_2026.md` | Qwen3, Phi-4, SmolLM3, Llama-3.2 are immediate targets |
| C++/Rust | `research/rust_cpp_inference.md` | 5-10ms Python overhead; C++ dispatch path viable |
| Spec Decoding | `research/advanced_optimization_techniques.md` | Medusa 2.2-3.6x, EAGLE 2.5-3.8x, Lookahead 1.5-2.5x |
| MoE | `research/moe_deep_dive.md` | 120-core mesh suited for expert parallelism |
| vLLM | `research/vllm_continuous_batching.md` | Tier 1-3 serving stack, paged attention |
| StableHLO | `research/stablehlo_lowering.md` | ~35 ops for transformer, PJRT plugin path |
| Precision | `research/mixed_precision_strategies.md` | bf8 MLP safe, attention needs bf16 |
| TT Community | `research/tt_community_contributions.md` | Bounties, bug reports, CI contributions |
| TT Metal | `research/tt_metal_contributions.md` | API patterns, kernel development |
| JAX Paths | `research/jax_infrastructure_paths.md` | Jaxpr interpreter vs PJRT vs TT-MLIR |
