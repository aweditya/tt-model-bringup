# C'0 Performance Baseline — what the numbers actually say

**Date**: 2026-05-12
**Host**: qb2, single Blackhole P150
**Workload**: Qwen3.6-27B, greedy decode, batch=1, fp32 residual, bf8 projection weights
**Output**: `~/tt-xla/.cache/perf_baseline_C0_20260512-220346.json`

## Headline numbers

| Region | Median | Notes |
|---|---:|---|
| Full prefill (5 tokens) | **1364 ms** | 273 ms/tok prefill |
| `prefill + 1 decode step` | 1632 ms | bundled measurement |
| **Derived decode step** | **267 ms** | = bundle − prefill, our key number |
| Single `deltanet_step` (isolated) | 3.41 ms | × 48 layers |
| Single `gated_attn_step` (isolated) | 4.23 ms | × 16 layers |
| Single `mlp_step` (isolated) | 1.04 ms | × 64 layers |
| `lm_head` (rms_norm + matmul) | 4.21 ms | × 1 per token |

Stdev on every measurement < 1% of median. Numbers are tight.

## The most informative line in the output

```
Compounding estimate (48×deltanet + 16×gated_attn + 64×mlp + lm_head)
  = 302.50 ms  vs  measured decode_step = 267.35 ms
  diff: -35.14 ms (-13.1%)
```

**The diff is NEGATIVE.** Naively summing the isolated per-layer times gives **more** than the measured full-decode time. That means **pipelining inside the full decode is hiding ~35 ms of overhead** that the isolated measurements pay.

**Implication for the roadmap**: trace capture (C'4) was projected as a 5-10× decode speedup. That projection assumed dispatch overhead was BIG and pipelining wasn't yet helping. The numbers say otherwise — dispatch is already largely amortized by the device pipelining ops as we submit them. Trace capture would eliminate the remaining ~35 ms of pipelining-hidden dispatch, but the realistic ceiling is closer to **2× decode speedup**, not 5-10×.

The real attack targets become more concentrated:
- DeltaNet dominates: 48 × 3.41 = **164 ms / 302 ms = 54%** of decode
- Gated attn: 16 × 4.23 = **68 ms = 22%** (this contains the KV roundtrip)
- MLP: 64 × 1.04 = **67 ms = 22%**
- lm_head: 4 ms = 1.5%

## What each planned phase realistically saves

Re-estimated based on the baseline numbers:

| Phase | Target region | Estimated savings | Why |
|---|---|---|---|
| **C'1** paged_update_cache | gated_attn (KV roundtrip) | ~50 ms/tok | 16 layers × ~3 ms/layer numpy roundtrip; eliminates host PCIe traffic |
| **C'2** bf16 residual | residual stream throughout | ~5-10 ms/tok | Halves activation memory bandwidth on the residual; modest |
| **C'3** native RoPE | gated_attn (RoPE region) | ~5-10 ms/tok | 2.6× faster per `feedback_native_rope.md` × 16 layers × small portion |
| **C'4** trace capture | dispatch overhead | ~20-35 ms/tok | Eliminates remaining Python dispatch; less than we thought because pipelining already helps |
| **C'5** chunked prefill | prefill only | ~1000 ms savings on prefill (1364 → ~300) | Batches multi-token forward via `chunk_gated_delta_rule` |

**Sum (all C'1-C'4 stacking)**: 267 ms → ~167 ms/tok (~6 tok/s). Plus prefill becomes near-instant.

To go below 167 ms/tok, we'd need:
- Multi-chip tensor parallelism (C'7) — splits the 27 GB weight read across 4 chips' DRAM banks, lifting the memory-bandwidth ceiling
- DRAM sharding within a chip (C'6) — bandwidth optimization

## Where the floor is

**Theoretical single-chip floor** for batch=1 decode is set by memory bandwidth:
- 27 GB bf8 weights ÷ ~200 GB/s effective Blackhole DRAM ≈ **135 ms/token**

We're at **267 ms/tok**, which is **2.0× the floor**. The C'1-C'4 stack puts us at ~167 ms/tok (1.24× floor). Getting closer to 135 ms/tok requires eliminating ALL non-compute, non-memory-bandwidth work — possible but increasingly diminishing returns.

**Multi-chip TP (C'7)** breaks the single-chip floor — 4 chips reading the weights in parallel = ~33 ms/tok theoretical floor at 4× bandwidth aggregate, modulo cross-chip comms cost.

## Targets (motivated by your friend's numbers)

Friend reports: **8 tok/s single P150, 16 tok/s multiple chips**.

| Phase | Decode ms/tok | tok/s | Status vs friend |
|---|---:|---:|---|
| C'0 (current) | 267 | 3.74 | behind |
| After C'1+C'2+C'3 (mid-stack) | ~190 | 5.3 | catching up |
| After C'4 (full single-chip optimized) | ~167 | 6.0 | close-ish |
| **Friend's single P150** | **~125** | **8.0** | their target |
| Theoretical single-chip floor | 135 | 7.4 | physics limit |
| After C'7 multi-chip TP (4 chips) | ~70-90 (realistic, w/ comms) | 11-14 | within range of friend's 16 tok/s |
| **Friend's multi-chip** | **~62** | **16** | their target |

To hit the friend's single-chip 8 tok/s, we need to be at or near the memory-bandwidth floor. That means C'1+C'2+C'3+C'4 done well, plus possibly some sharded weight optimization. **Tight but achievable.**

To hit 16 tok/s multi-chip, single-chip needs to be optimized first, then C'7 TP. **Achievable if we get the single-chip work right.**

Note: these targets assume the friend's reported numbers were for a **27B-class model**. If it was a 7B model, those are easier targets relative to what we're running. Either way our 27B numbers should be competitive once optimized.

## Pipelining insight — what -13.1% means physically

When we measure `single_deltanet_step` in isolation, the timing includes:
- ~30 µs Python → ttnn dispatch
- ~3.3 ms device compute + DRAM reads
- ~50 µs final sync

When 48 of these run inside a full decode step, the dispatches of layers N+1, N+2, N+3 happen DURING the execution of layer N. So the per-layer dispatch costs overlap with the compute, and we only pay one "final dispatch" worth (~30 µs) at the end.

Across 48 layers: 47 × 30 µs = **1.4 ms of dispatch hidden by pipelining**. Plus similar savings in attention and MLP. Total maybe 3-4 ms savings. But we measured 35 ms savings.

The remaining ~30 ms is probably from:
- L1 cache reuse across consecutive ops in the full forward (data ALREADY warm when next op runs)
- Sync barriers fewer (one at end vs one per region in the isolated case)
- Tile-layout overhead amortized when same tensor flows through multiple ops

The takeaway: **the pipelining is doing real work** and trace capture's value is mostly in capturing those same benefits as a static program (no Python in the loop) — not in adding NEW benefits.

## Next move: C'1 (paged_update_cache)

C'1 attacks the biggest single overhead we can localize: the per-layer KV cache numpy roundtrip. Each gated_attn_step currently does:

```python
# In 91f.gated_attn_step_ondevice:
k_np = ttnn.to_torch(k_tt).float().numpy().reshape(N_KV, HEAD_DIM)
v_np = ttnn.to_torch(v_tt).float().numpy().reshape(N_KV, HEAD_DIM)
# ... read whole cache to host, modify one slot, re-upload
```

This is host I/O for every attention layer of every token. 16 layers × ~3-4 ms estimated cost = **~50 ms/tok savings**.

Two paths to do C'1:
1. **Pad n_kv from 4 to 32 internally** so `paged_update_cache` (which requires tile-aligned shapes) accepts it
2. **Custom MemoryConfig with explicit shard layout** to avoid the tile constraint

Path 1 is simpler and the memory overhead (8× the cache size) is small in absolute terms — KV cache is ~16 MB total even padded.

Expected outcome:
- C'1 baseline JSON shows `single_gated_attn_step` median dropping from 4.23 ms to ~2 ms
- `decode_step_derived` drops from 267 ms to ~217 ms (3.7 tok/s → 4.6 tok/s)
- Correctness gate (91r) shows full_attention layers stay at ≥ 0.9998 cosine
