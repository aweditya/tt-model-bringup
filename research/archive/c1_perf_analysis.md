# C'1 Performance — ttnn.scatter for on-device KV cache slot writes

**Date**: 2026-05-12
**Host**: qb2, single Blackhole P150
**Change**: replaced KV cache numpy roundtrip in `gated_attn_step_ondevice` with on-device
`ttnn.scatter`. Diff at `experiments/91f_qwen36_27b_full_ondevice.py:273-283`.
**Correctness**: 91r per-layer gate PASS — all DeltaNet ≥ 0.9997, full-attention 0.9989-0.9999,
layer 63 unchanged from Branch III baseline (0.987 pre-existing).

## What changed

Before C'1: each `gated_attn_step` (16 layers × every token) executed:
```python
# 4 to_torch + 2 from_torch — 6 PCIe transfers per call
k_np = ttnn.to_torch(k_tt).float().numpy()
v_np = ttnn.to_torch(v_tt).float().numpy()
cache_k_flat = ttnn.to_torch(kv_cache_k_tt).float().numpy()
cache_v_flat = ttnn.to_torch(kv_cache_v_tt).float().numpy()
# ... numpy slot write ...
kv_cache_k_tt = ttnn.from_torch(...)
kv_cache_v_tt = ttnn.from_torch(...)
```

After C'1: 1 small int32 upload (~4 KB) + 2 on-device scatters, K/V cast to bf16 once.

## Headline numbers

JSON: `~/tt-xla/.cache/perf_baseline_C1_20260512-225706.json`

| Region | C'0 median | C'1 median | Δ ms | Δ % |
|---|---:|---:|---:|---:|
| Full prefill (5 tokens) | 1364.39 | 1126.80 | -237.6 | **-17.4%** |
| prefill + 1 decode | 1631.75 | 1336.86 | -294.9 | -18.1% |
| **Derived decode step** | **267.35** | **210.06** | **-57.3** | **-21.4%** |
| Single deltanet_step | 3.41 | 3.42 | +0.01 | +0.4% |
| Single gated_attn_step | 4.23 | **2.00** | **-2.23** | **-52.9%** |
| Single mlp_step | 1.04 | 1.06 | +0.01 | +1.0% |
| lm_head | 4.21 | 4.23 | +0.02 | +0.4% |
| Compounded estimate | 302.50 | 267.95 | -34.55 | -11.4% |
| **tok/s** | **3.74** | **4.76** | +1.02 | **+27%** |

## Discussion

**C'1 beat its prediction.** The roadmap forecast ~50 ms/tok; we measured -57 ms/tok. Why bigger than expected:

1. **gated_attn dropped 52.9%** in isolation (4.23 → 2.00 ms). 16 layers × 2.23 ms = 35.7 ms saved from this alone.
2. **Additional 21.6 ms** of decode savings beyond the gated_attn delta. Where did it come from? Cross-checking the compounded estimate:
   - C'0: compounded 302.5 ms, measured 267.4 ms — pipelining hides -13.1%
   - C'1: compounded 267.95 ms, measured 210.1 ms — pipelining hides **-21.6%**

**The host roundtrips were synchronization barriers.** Each `to_torch` in the prior path was a synchronous PCIe read that forced the device to drain before the host could read. With 16 of these per token gone (×4 per gated_attn × 16 layers = 64 sync points/token), the ttnn dispatcher can now pipeline deeper across layers — overlapping more of layer N+1's dispatch with layer N's compute.

**Implication for the remaining roadmap:** prior estimates assumed pipelining was "static" at -13%. We now know it's **dynamic** — every host-blocking op we remove unlocks more. This raises the realistic ceiling for C'4 (trace capture). The previous estimate of ~2× decode ceiling underestimated; full Python removal could unlock further pipelining.

**Correctness gate (91r):** all 10 sampled layers match Branch III baseline. DeltaNet ≥ 0.99970 (gate 0.9997), full-attn 0.99989-0.99998, layer 63 at 0.987 unchanged from Branch III's pre-existing deep-layer drift. **Zero regression.**

**Paris demo:** PASS — `experiments/demo_qwen36_27b.py` first token after "The capital of France is" → ` Paris` (token id 11751). Full output: "The capital of France is Paris.\n<think>\n\n\n".

**Note on demo-reported tok/s vs perf_baseline tok/s:** demo shows 3.77 tok/s (265 ms/tok) while perf_baseline measured 4.76 tok/s (210 ms/tok). The gap is because demo's 1.3s ÷ 5 tokens includes the cold first decode after prefill plus per-step `print`/`from_dev` overhead in the demo loop. perf_baseline's `prefill_plus_one_decode − prefill` is the cleaner warm-decode measurement and is the canonical C'1 number.

## Status vs friend's 8 tok/s target

| | ms/tok | tok/s | gap to friend |
|---|---:|---:|---|
| C'0 | 267 | 3.74 | -52% |
| **C'1** | **210** | **4.76** | **-40%** |
| Friend single P150 | ~125 | 8.0 | target |
| Single-chip floor | ~135 | 7.4 | physics |

After C'1 we're at **63% of single-chip floor**. To close to friend's 8 tok/s we need to lop another ~75 ms/tok from C'2+C'3+C'4 combined. Roadmap had budgeted 30-55 ms across those — tight, but the C'4 ceiling just rose.

## Next phase

C'2: bf16 residual stream ablation. Touch points (from `91l_fp32_residual_generate.py:177-211`):
- `upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)` → `bfloat16`
- conv_state init dtype
- ssm_state init dtype
- RoPE table cos/sin dtypes (currently uploaded as `ttnn.float32`)

Validation: `91r` per-layer cosine ≥ 0.999 (slightly looser than current fp32 baseline).
