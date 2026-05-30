# Qwen3.6-27B Performance Roadmap — Branch C'

**Date**: 2026-05-12
**Baseline**: Branch III correctness complete. 3.26 tok/s decode (307 ms/tok), 562 ms/tok prefill on single Blackhole P150.
**Target**: 100-200 ms/tok decode (HW bandwidth-bound theoretical floor ≈ 135 ms/tok for 27 GB bf8 weights at ~200 GB/s DRAM bandwidth).

## Non-negotiables for every perf phase

1. **Correctness gate after every change** — run `experiments/91r_per_layer_diff.py` and confirm DeltaNet ≥ 0.9997, full_attn ≥ 0.9998. Any regression → revert.
2. **One variable at a time** — never bundle two perf hypotheses in one change.
3. **Measure before optimizing** — never optimize on hypothesis alone (B'9's fp32-residual-for-wrong-reason cost us hours).
4. **Demo script (`experiments/demo_qwen36_27b.py`) passes the Paris sanity** at every phase.

## Where the 307 ms/tok currently goes (estimate)

Without instrumentation we can't know precisely, but the breakdown is roughly:
- **Per-op dispatch overhead** (Python → ttnn → device): ~30 µs/op × ~400 ops per token = ~12 ms (small)
- **Weight DRAM reads** (27 GB bf8 at ~200 GB/s effective): ~135 ms (theoretical floor)
- **KV cache numpy roundtrip** (per layer per token, 16 attn layers): ~100 ms (BIG)
- **Other host work** (RoPE table compute, embed lookup, argmax): ~10 ms
- **HiFi4 + fp32 DEST compute**: ~10 ms
- **Synchronization / dispatch latency** (each layer ends with sync): ~40 ms

Total ≈ 307 ms ✓. The KV roundtrip is the biggest fixable chunk.

## Revised phase order (2026-05-12 — daily-driver use case explicit)

**Why the order changed**: user clarified the goal is *daily-driver local LLM replacing cloud* (ChatGPT, etc.). That makes **long-context correctness** the binding constraint, not raw decode tok/s. We also discovered that **Phase A6 v1 chunked-serial DeltaNet already exists** (`experiments/85_deltanet_scan_v1.py` at 800 tok/s prefill — measured), so a usable long-context experience is much closer than the original roadmap implied.

Revised order:

| # | Phase | What | Goal |
|---|---|---|---|
| ✓ | C'1 | ttnn.scatter for KV cache (DONE, commit 6e365a4) | -57 ms/tok (-21%) |
| 1 | **C'0.5** | Scale MAX_POS to 32k+: KV cache resize, RoPE extension, build bf16-vs-fp32-reference long-context comparison test | **prereq** for everything else; unblocks real correctness gate |
| 2 | **C'5a** | Wire A6 v1 chunked-serial DeltaNet (from exp 85) into 91l prefill | 32k prefill: 80 min → ~40 sec |
| 3 | **C'2 (re-test)** | Re-evaluate bf16 residual ablation against the long-context gate | safely accept or reject the bf16 change |
| 4 | **C'5b** | Replace within-chunk serial with parallel `(I-attn)^{-1}` via Neumann series — see `research/c5_chunked_prefill_plan.md` | 32k prefill: 40 sec → ~5-10 sec |
| 5 | **Serve loop** | Chat template, OpenAI/Anthropic-compatible HTTP, tool-call parsing | model becomes daily-driver usable |
| 6 | C'3 | Native `ttnn.experimental.rotary_embedding` | -5-10 ms/tok decode |
| 7 | DN-fusion | DeltaNet 4-linear input-projection fusion (see `research/deltanet_4linear_fusion_plan.md`) | -3-5 ms/tok decode |
| 8 | C'4 | Trace capture for full decode step | -20-50 ms/tok decode (ceiling raised by C'1's pipelining surprise) |
| 9 | C'6 | DRAM-sharded weights | ~10% bandwidth win |
| 10 | C'7 | Multi-chip TP across 4 P150s on qb2 | -60%; unlocks larger context cache too |

**Why this ordering**:
- **C'0.5 first** because every later optimization needs a long-context gate. Without that gate we ship "fast but maybe broken at 32k" code that the user can't actually rely on for daily work.
- **C'5a before C'2** because we can't even MEASURE bf16 correctness at long context until prefill is fast enough to run the test. A6 v1 at 40 sec/32k makes the test loop feasible.
- **C'5b before C'4** because long-context usability is a hard prereq; trace capture's win is decode-only and gains relative weight after long-context is solved.
- **Serve loop before C'3/DN-fusion/C'4** because once the model is usable end-to-end at long context, the user can BENCHMARK it against ChatGPT for actual work, not just synthetic decode metrics. That benchmark informs which further perf phases matter.

## Original phase order (deprecated — kept for diff history)

### C'1 — KV-cache `paged_update_cache` (target: -80 to -100 ms/tok)

**Hypothesis**: eliminate the per-layer numpy roundtrip for KV cache writes. Currently `gated_attn_step_ondevice` does `ttnn.to_torch → numpy modify → ttnn.from_torch` for the K/V slot write because we couldn't tile-align `n_kv=4`.

**Approach**:
1. Pad n_kv from 4 to 32 internally (the sharded `paged_update_cache` requires it)
2. OR use a custom non-default `MemoryConfig` with explicit padded shard shape
3. Replace the numpy roundtrip with `ttnn.experimental.paged_update_cache(cache_tensor, new_kv, update_idxs_tensor)`

**Risks**: memory overhead from padding (cache grows ~8× for the unused KV slots — small absolute size, fine). The op exists but may have shape constraints we haven't hit yet.

**Validation**: 91r per-layer cosine ≥ 0.9997 for full_attn layers (linear_attn unchanged).

### C'2 — bf16 residual ablation (target: small win on residual, halves activation memory)

**Hypothesis**: B'9 promoted the residual stream to fp32 to fix the wrong bug. With Q-scaling fix (#7) in place, bf16 residual likely produces equivalent quality.

**Approach**:
1. Revert 91l's `x_tt = upload(..., dtype=ttnn.float32)` → `bfloat16`
2. Revert RoPE tables and A_log/dt_bias loads to bf16
3. Revert conv_state init to bf16

**Validation**:
- 91r per-layer cosine ≥ 0.999 (slightly looser than fp32 baseline)
- Demo script `Paris` sanity passes
- 60-token greedy output still coherent (eyeball)

**Why this is fast to test**: trivial dtype change in 91l.

### C'3 — Native RoPE (`ttnn.experimental.rotary_embedding`) (target: ~5-10 ms/tok)

**Hypothesis**: prior measurement (`feedback_native_rope.md`) showed 2.6× speedup over our rotation-matrix RoPE.

**Approach**: replace our `apply_partial_rope` function in `91f.gated_attn_step_ondevice` with `ttnn.experimental.rotary_embedding`. The partial-rotary (first 64 of 256 dims) split is handled by slicing the appropriate range and reassembling.

**Validation**: 91r per-layer cosine for full_attn layers unchanged.

### C'4 — Trace capture for decode (target: 5-10× decode speedup)

**Hypothesis**: per-op Python dispatch is currently overhead-bound at batch=1. Trace capture replays the entire decode-token forward as a single device program. Prior experiments (`feedback_trace_capture.md`) showed 4-5× on transformer blocks.

**Approach**:
1. Use `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` to capture one full decode step
2. Use `ttnn.copy_host_to_device_tensor` for the per-step input (embed lookup, cur_pos)
3. `ttnn.execute_trace` to replay
4. Handle the KV-cache state mutation correctly — the trace captures a single state, so the cache update needs to be part of the trace, not external

**Risks**:
- Python scalars baked into trace (per `feedback_trace_capture.md`). Use device tensor buffers for `cur_pos`.
- Trace may not capture host reads. Eliminate any `to_torch` mid-forward.
- Cache memory growth — each trace pinned in L1.

**Validation**: traced output matches eager output bit-for-bit (no host reads in the path).

### C'5 — Chunked prefill (target: 4-8× prefill speedup)

**Hypothesis**: HF's `torch_chunk_gated_delta_rule` batches multi-token prefill. We currently do sequential single-token even during prefill. For 5 prompt tokens: 5 × 562 ms = 2.8 s; chunked could bring this to ~600 ms total.

**Approach**: implement the chunk algorithm (HF source `modeling_qwen3_5.py:234-313`). The math is more complex (parallel scan via matrix inversion of `I - L` for some triangular L) but mathematically equivalent.

**Risks**: complex algorithm; could introduce regressions. Probably the hardest phase.

**Validation**: chunked-prefill ttnn output cosine ≥ 0.9997 vs sequential-prefill ttnn output, on the same prompt.

### C'6 — DRAM-sharded weights (target: ~10% bandwidth win)

**Hypothesis**: `experiments/99_dram_sharded_*.py` showed bandwidth improvements for batch=1. May help at our scale.

### C'7 — Multi-chip TP across 4 P150s (if needed)

The qb2 has 4 P150s with working fabric. Tensor parallelism across them could give 4× compute + 4× memory bandwidth. Likely overkill for single-batch decode but useful for serving multiple users.

## Measurement infrastructure to build

Before C'1, set up the perf baseline harness:

**`experiments/utils/perf_baseline.py`**: runs the production model with timing instrumentation per phase (prefill / decode / per-layer / per-op type). Outputs a structured JSON snapshot. Compare across perf phases.

This is the C'0 deliverable.

## What we are NOT doing

- bf8 lm_head + GPTQ — modest savings, not worth complexity yet
- Custom kernels — stay on stock ttnn until we've exhausted standard approaches
- Multi-batch — single-batch is the daily-driver use case; revisit only if needed
- Op fusion (swiglu fusion etc.) — wait for trace capture to land first; many fusions become unnecessary post-trace

## When are we done?

When the demo runs end-to-end at ≤ 150 ms/tok with correctness sanity passing, Branch III is "production-grade." Anything beyond is bonus.

## Concrete competitive targets

A friend reports **8 tok/s on single P150, 16 tok/s on multiple chips** for an LLM-class workload. Setting these as motivational targets:

| Goal | Decode latency | tok/s | Means |
|---|---:|---:|---|
| C'0 baseline (today) | 267 ms/tok | **3.74** | where we start |
| Match friend single P150 | 125 ms/tok | **8.0** | C'1+C'2+C'3+C'4 stack |
| Single-chip floor | 135 ms/tok | 7.4 | physics (memory bandwidth) |
| Match friend multi-chip | 62 ms/tok | **16.0** | C'7 tensor parallel across 4 P150s |
| Multi-chip floor | ~33 ms/tok | 30 | 4× aggregate DRAM bandwidth |

Single-chip is squeezed against the memory-bandwidth floor (135 ms/tok). Hitting 8 tok/s means executing close to ideal — every non-bandwidth overhead removed. Multi-chip opens new headroom.

---

# Branch D' — Beyond performance: memory-tier and megamodel work

Motivated by the conversation about exo-explore/exo. qb2/qb1's distinctive advantage: **503 GB of DDR system RAM sits in a separate pool from the chip DRAM**. Mac Studios (exo's hardware target) have unified memory — they can't tier in the same way. Three branches worth pursuing once Branch C' is done:

## D'1 — CPU-RAM-resident weights for super-large models

**Target**: run models that don't fit on 4-chip aggregate DRAM (128 GB).

Approach: keep all weights in 503 GB CPU pool, stream layers to chip DRAM on demand via PCIe.

- PCIe Gen4 ≈ 32 GB/s, Gen5 ≈ 64 GB/s (depending on host hardware)
- Slower than on-chip DRAM (~200 GB/s) but workable for batch=1
- Enables Llama-3-405B (bf8 ≈ 200 GB), Qwen3-Max-style models that don't fit on the chips

**Risks**: PCIe bandwidth becomes the new bottleneck. The "effective floor" shifts to ~PCIe-bound (200 GB ÷ 32 GB/s = 6.25 sec/token at Gen4). Probably interactive only at smaller batch sizes or with overlap.

## D'2 — Megacontext via CPU-RAM KV cache

**Target**: 1M+ token context windows.

KV cache memory grows linearly with sequence length. For Qwen3.6-27B at 1M context: cache ≈ 100+ GB, doesn't fit on chip but easily fits in 503 GB system RAM.

PagedAttention-style: cache pages live in CPU RAM, paged into chip per attention layer. Latency increases (PCIe round-trip per layer per query) but the model can SERVICE the long context, which Mac Studios can't.

## D'3 — Heterogeneous compute (CPU + chip together)

**Target**: utilize the 32 CPU cores that sit idle during inference.

Specific opportunities:
- **Speculative decoding**: a small "draft" model runs on CPU, generates speculation candidates, the chip verifies. Standard technique, well-studied.
- **Embed table on CPU**: 248320 × 5120 × 4 bytes = 5 GB embed table. Currently CPU-resident anyway in our impl (we look up rows on host). Keep it there — this is free.
- **Async beam search / sampling state**: keep all the search bookkeeping on CPU while the chip generates.

## D'4 — Multi-host clustering (exo's actual pitch)

**Target**: connect qb1 + qb2 into one tensor-parallel cluster.

Mostly NOT useful for our purposes — they're in the same building but not co-located in a way that would benefit from this. Unless we want to run a model that doesn't fit even on 4 chips (~128 GB), in which case combining qb1 + qb2 gives us 8 chips total.

Probably skip unless a specific workload demands it.

---

## Sequencing for Branch D'

D'1, D'2, D'3 are independent and unlocked once Branch C' lands. Pick based on the workload need:
- D'2 first if anyone wants 1M context for our 27B
- D'1 first if anyone wants to run a 200B+ model
- D'3 (speculative decoding) is high-impact and well-trodden; might actually be a "speed Branch C'" finale rather than a Branch D' branch

D'4 is parked unless a specific need arises.
