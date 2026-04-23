# Wiki 58: MoE First Light — Qwen1.5-MoE-A2.7B on Blackhole

## The Question

Can we generate text with a Mixture-of-Experts model on Tenstorrent Blackhole?

## What We Did

### Experiment 89: Single Layer Validation (PASSED)
- Downloaded Qwen1.5-MoE-A2.7B (14.3B params, 8 safetensors shards, ~28.6 GB bf16)
- Uploaded 60 experts + 1 shared expert for layer 0 at BFP8
- Single expert forward: **cosine 0.999909** vs numpy reference
- Router (linear + softmax + top-4): correctly selects 4 of 60 experts
- All 60 experts sequential (non-traced): 11.8ms
- Full MoE layer (non-traced): **9.4ms** (MoE block = 97% of time, attention = 3%)
- Flash decode with MHA (16Q/16KV): works without split workaround

### Experiment 90: Full 24-Layer Eager Decode (RUNS, WRONG OUTPUT)
- All 24 layers uploaded at BFP8 (172s upload time)
- Eager decode: device matmuls + CPU routing + top-4 expert dispatch
- **13.6 tok/s** — much faster than expected ~5 tok/s
- BUT: output is garbage ("The capital of France is four儿...")

### Experiment 90b: Numpy Reference (CRITICAL FINDING)
- Pure numpy forward pass produces **the same garbage**
- Top-5 for "The capital of France is": 'for' (13.05), 'four' (12.53), 'the' (11.93)
- "Paris" is nowhere in the top-5
- This means: **the BASE model is weak, not our implementation**

## Bug Found: Shared Expert Gate

The shared expert gate (`shared_expert_gate.weight`) is a **linear projection [1, 2048]**, not a scalar. Our initial code treated the first element as a constant sigmoid value. The correct computation:

```python
# WRONG: seg_val = sigmoid(weight[0])          # constant, input-independent
# RIGHT: seg_val = sigmoid(hidden @ weight.T)  # per-token gating
```

This corrupted the shared expert contribution through all 24 layers. After fixing, the base model produces "Paris" as top-3 and the Chat model produces perfect answers.

## Results: Chat Model (Qwen1.5-MoE-A2.7B-Chat)

**All prompts produce excellent, coherent output on Blackhole:**

| Prompt | Output | Tokens | tok/s |
|--------|--------|--------|-------|
| What is the capital of France? | "Paris" | 2 | 7.8 |
| Write a Python function to check if prime | Complete is_prime() with edge cases + 6k+1 optimization | 101 | 12.9 |
| Explain quantum computing in one sentence | Perfect definition with superposition + entanglement | 43 | 12.8 |

**First working MoE model on Tenstorrent Blackhole!**

## Architecture Notes

```
Qwen1.5-MoE-A2.7B (Qwen2MoeForCausalLM):
  - hidden=2048, 24 layers, MHA (16Q/16KV), head_dim=128
  - 60 routed experts per layer, top-4 routing
  - Routed expert MLP: intermediate=1408 (tiny — 3x smaller than 0.5B Qwen)
  - 1 shared expert: intermediate=5632 (always active)
  - Router: linear [2048, 60] + softmax + topk(4)
  - Shared expert gate: scalar sigmoid
  - Q/K/V have biases, O does NOT (most layers)
  - RoPE: half-format, theta=1e6 (same as Qwen2.5)
  - BFP8 total: ~14.3 GB (fits with 17 GB headroom)
  - norm_topk_prob=false (don't renormalize router probs)
```

## Memory Layout

| Component | Per Layer | 24 Layers |
|-----------|-----------|-----------|
| 60 experts (BFP8) | 520 MB | 12.5 GB |
| Shared expert (BFP8) | 33 MB | 0.8 GB |
| Attention (bf16) | 34 MB | 0.8 GB |
| **Total** | **587 MB** | **14.1 GB** |

Plus embeddings (~0.6 GB) + KV cache. Fits comfortably in 32 GB DRAM.

## Performance Analysis

### Eager decode (exp 90): 78 ms/tok = 12.8 tok/s
- 24 layers × ~3ms/layer ≈ 72ms
- Per-layer breakdown: attention ~0.3ms, MoE routing 4x expert dispatch ~2.7ms
- Host round-trip per layer: ~0.5ms (router readback + expert dispatch)
- Only 10% of time is actual compute; 90% is host-device overhead

### Optimized eager (exp 91): 50 ms/tok = 20.2 tok/s (1.58x speedup)
- On-device expert accumulation (ttnn.multiply + ttnn.add)
- Residual connections stay on device (no CPU round-trip)
- Shared expert gate: CPU sigmoid scalar, device multiply
- Program cache for faster dispatch
- Single sync per layer (router logits only, 60 floats = 240 bytes)
- Remaining bottleneck: ~840 op dispatches per token at ~30μs each

### Predicted traced decode: ~28-35 tok/s
- Run all 60 experts, mask 56 unused → fully traceable
- Bandwidth: 12.5 GB/step at 450 GB/s = ~28ms
- Plus attention (~8ms) = ~36ms total
- No host round-trips = zero dispatch overhead
- Challenge: device-side top-k routing needed (no CPU readback mid-trace)

## What's Next

1. **Fully traced decode** — Device-side top-k routing for ~30 tok/s target
2. **OLMoE-1B-7B** — alternative MoE (64 experts, 7B total, only 6.5GB at BFP8)
3. **Gemma 4** — Google's latest model family, possible MoE variant

## Bugs Found & Fixed

1. **Router shape**: `from_dev(rl, (B * T, n_exp_pad))` crashed because `ttnn.to_torch` unpads tile layout. Fixed: use `n_experts` (60) not `n_exp_pad` (64).

2. **Shared expert gate** (CRITICAL): `shared_expert_gate.weight` is `[1, 2048]` — a linear projection. Our code used `sigmoid(weight[0])` as a constant. The correct computation is `sigmoid(hidden @ weight.T)` — a per-token gate. This was the root cause of garbage output across all 24 layers.
