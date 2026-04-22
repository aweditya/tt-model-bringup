# Wiki 57: MoE Feasibility — Qwen1.5-MoE-A2.7B on Blackhole

## The Question

Can we run a Mixture-of-Experts model on our single 32GB Blackhole chip?

## Model Survey

| Model | Total | Active | BF16 Size | BFP8 Size | Fits 32GB? |
|-------|-------|--------|-----------|-----------|------------|
| **Qwen1.5-MoE-A2.7B** | 14.3B | 2.7B | 28.6 GB | **14.3 GB** | BFP8: yes |
| Qwen3-30B-A3B | 30.5B | 3.3B | 61 GB | 30.5 GB | No |
| Qwen2-57B-A14B | 57B | 14B | 114 GB | 57 GB | No |

**Only Qwen1.5-MoE-A2.7B fits.** At BFP8 it leaves ~17 GB for KV cache + activations.

## Architecture: Qwen1.5-MoE-A2.7B

- hidden=2048, 24 layers, 16 Q heads, 16 KV heads (MHA, not GQA), head_dim=128
- vocab=151936, max_pos=8192, silu activation

### MoE Config
- **60 routed experts** per layer, **top-4 routing**
- Each routed expert: gate/up/down with intermediate=1408 (tiny — 3x smaller than Qwen 0.5B's MLP)
- **1 shared expert** per layer: intermediate=5632 (always active, ~4x a routed expert)
- Router: linear [2048, 60] + softmax + topk(4)
- Shared expert gate: scalar sigmoid

### Parameter Breakdown Per Layer
- 60 routed experts: 60 x 3 x 2048 x 1408 = **519M params**
- 1 shared expert: 3 x 2048 x 5632 = **35M params**
- Attention (MHA): 4 x 2048 x 2048 = **17M params**
- **Per layer total: ~571M** → 24 layers = ~13.7B + embeddings = 14.3B

### Active Per Token
- 4 routed experts + shared expert + attention = ~86M/layer → ~2.7B total

## The Core Challenge: Traced Decode + Dynamic Routing

Our Metal Trace captures a **static op graph**. MoE routing is **dynamic** — different tokens select different experts. Three approaches:

### Approach 1: Run All 60 Experts, Mask Unused (Simplest)

- Execute all 60 expert MLPs in the trace, zero-mask 56 outputs
- Fully traceable — graph is identical every step
- **Bandwidth cost**: reading all 60 expert weight sets per layer = 60 x 3 x 2048 x 1408 x 1 byte (BFP8) = **~520 MB/layer**, 24 layers = **~12.5 GB/step**
- At 450 GB/s = **~28ms** just for expert weights + ~8ms for attention = **~36ms total**
- That's ~28 tok/sec — surprisingly competitive!
- **Compute**: negligible (experts are tiny, 60x1408 intermediate)

### Approach 2: Host-Orchestrated Dispatch (Most Flexible)

- Run attention + router on device, read back top-4 indices to host
- Host dispatches only the 4 needed expert MLPs
- **Bandwidth**: only 4/60 of expert weights = ~0.9 GB/layer = ~21 GB total → ~47ms at 450 GB/s
- But: 24 device-to-host round-trips at ~0.5-1ms each = **12-24ms overhead**
- Total: ~47ms + 12-24ms = **~60-70ms** → slower than Approach 1!

### Approach 3: Pre-group Experts (Advanced)

- Pre-compute which experts are most commonly co-activated
- Group them into "super-experts" that can be traced together
- Requires profiling and is model-specific

### Verdict: Approach 1 Wins

Counter-intuitively, running all 60 experts and masking is likely faster than host-routed dispatch because:
1. Expert weights are tiny (1408 intermediate vs 14336 for Llama 8B MLP)
2. All-experts bandwidth (~12.5 GB at BFP8) is less than Llama 8B's MLP weights (~7 GB)
3. No host round-trips = fully traceable = zero dispatch overhead
4. The waste (56/60 experts unused) is in bandwidth, not compute

**Estimated performance: ~28-35 tok/sec on Qwen1.5-MoE-A2.7B with all-experts traced approach.**

This would be competitive with our Llama-3.2-3B numbers (30 tok/s) while having higher quality (2.7B active from a 14.3B parameter pool).

## Experiment Plan

### Exp 89: Weight Loading + Single Expert Forward
1. Download Qwen1.5-MoE-A2.7B weights (8 safetensors shards, ~28.6 GB BF16)
2. Upload one layer's expert weights at BFP8 — verify memory fits
3. Run a single expert MLP forward pass
4. Test router: linear + softmax + topk(4)
5. Measure per-expert and all-experts-masked timings

### Exp 90: Full MoE Layer
1. Implement one full MoE layer: attention + router + all 60 experts + shared expert
2. Verify correctness against HuggingFace reference
3. Measure layer latency

### Exp 91: Full Model Traced Decode
1. Stack 24 MoE layers with KV cache
2. Prefill + traced decode loop
3. Benchmark against Llama 3B for quality/speed comparison

## Weight Naming Convention (Safetensors)

```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight   # [1408, 2048]
model.layers.{L}.mlp.experts.{E}.up_proj.weight      # [1408, 2048]
model.layers.{L}.mlp.experts.{E}.down_proj.weight    # [2048, 1408]
model.layers.{L}.mlp.shared_expert.gate_proj.weight  # [5632, 2048]
model.layers.{L}.mlp.shared_expert.up_proj.weight    # [5632, 2048]
model.layers.{L}.mlp.shared_expert.down_proj.weight  # [2048, 5632]
model.layers.{L}.mlp.gate.weight                     # [60, 2048] (router)
model.layers.{L}.mlp.shared_expert_gate.weight        # [1] (sigmoid gate)
model.layers.{L}.self_attn.{q,k,v,o}_proj.weight     # [2048, 2048]
```
