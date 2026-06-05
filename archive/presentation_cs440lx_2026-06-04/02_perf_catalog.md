# Performance Optimizations Catalog (Blackhole P150)

## Framing

Target: Tenstorrent **Blackhole P150** (32 GB GDDR6, 8 banks, 110
ttnn-visible Tensix cores, 1.39 MiB L1/core). qb1/qb2 host a 1x4 P150 mesh
with 2 active 25 GbE links per axis.

- Published DRAM peak: 512 GB/s.
- Measured on-device streaming peak: **404 GB/s** (~79% of 512), bf16 TILE
  layout, 1 GiB `ttnn.clone`. This is the working roofline.
- Host<->device PCIe is layout-conversion-bound (~5.6 GB/s write, ~2.7 GB/s
  read). Anything that crosses the bar per token loses real time.

LLM decode at B=1 is weight-bandwidth-bound: the matmul reads all weights
once per token regardless of B. Ceiling ~ `model_bytes / bw / nchips`.

- Gemma 4 12B bf16, 4 chips: 24 GB / 4 / 404 = **14.85 ms/tok = 67 tok/s**
  ideal. Prod traced: **47.5 ms/tok = 21.05 tok/s** = ~31% of ceiling.
- 27B isolated MLP already streams at **400.7 GB/s = 78% of 512 peak**;
  end-to-end inefficiency lives in DeltaNet recurrence, attention, dispatch.

## Catalog table

| Optimization | Model | Before -> After | Mode | Status | Commit |
|---|---|---|---|---|---|
| Paged SDPA + B3 (HiFi2, no fp32_dest_acc) | 27B TP | 7.02 -> 11.43 tok/s (+62%) | trace | shipped | `4741253` |
| Vocab-sharded lm_head + on-device argmax | 27B TP | 11.43 -> 12.02 tok/s (+5.1%) | trace | shipped | `ef3f336` |
| Vocab-sharded lm_head + on-device argmax | Gemma 4 12B | 51.3 -> 47.5 ms/tok (+8.0%) | trace | shipped | `a24f2ea` |
| `num_links=2` for all_reduce / reduce_scatter | 27B TP | 12.72 -> 12.93 tok/s (+1.65%) | trace | shipped | `d739daf` |
| Owned fused decay/gate kernel (10 ops -> 1) | 27B TP | 12.41 -> 12.72 tok/s (+2.5%) | trace | shipped | `08877d5` |
| Chunked DeltaNet (Neumann inverse), seq<=32 | 27B TP | 1.58x prefill, breaks at C>=64 | eager | shipped (capped) | `dc41d6d` |
| A004 explicit `core_grid=10x11` on MoE batched matmuls | 35B-A3B TP | 141.79 -> 110.4 ms/tok (-30 ms/tok) | trace | shipped | `24fe4f8` |
| A008 bf8 MoE expert weights | 35B-A3B TP | 110.4 -> 81.16 ms/tok (1.75x cumulative) | trace | shipped | `de904c6` |
| A002 fused Q/K L2-norm via `ttnn.rms_norm` | 35B-A3B TP | -1.13 ms/tok | trace | shipped | A002 |
| A003 router topk reorder (softmax+topk -> topk+softmax) | 35B-A3B TP | -0.25 ms/tok | trace | shipped | `64d135a` |
| W2 on-device top-k for sampling | 27B CB B=32 | **2.96x step / 6.5x throughput** | trace | shipped opt-in | `bef03ba` |
| P3.5 logits trace for sampling mode | 27B CB | 2.54x | trace | shipped | `6ef94f2` |
| Continuous batching B=1 -> B=32 (greedy) | 27B TP | 12.96 -> **150.5 tok/s agg** (11.6x) | trace | shipped | CB1-CB4 |
| CB + batched conv1d (shiftacc, padding-free state) | 27B CB B=64 | 208 -> **593 tok/s agg** (45.8x over B=1) | trace | shipped opt-in | -- |
| DRAM-sharded MLP matmul | 27B single chip | 0.930 -> 1.975 ms (**2.1x slower**) | eager | NEGATIVE | -- |
| Async `all_reduce_async` for compute/comm overlap | 27B TP | +4% setup, no overlap win | eager | NEGATIVE | `1f1eef2` |
| bf8 KV cache | 27B | Δcos = -2.7e-5, not memory-bound at MAX_POS<=8k | -- | not shipped | -- |
| fp32 KV cache | 27B | SDPA decode hard-rejects fp32 | -- | DEAD END | -- |
| HiFi4 + fp32_dest_acc on SDPA decode | 27B | top1@500 = 27.8% -> 98.4% (B3), +1.8% latency | -- | shipped B3 | P21 |
| bf8 MLP weights | 27B (already prod) | 1.484 -> 1.019 ms isolated (1.46x baked in) | -- | already prod | -- |

## What worked, ranked by impact

**1. Paged SDPA + B3 compute_kernel_config (27B TP: +62%).** Manual `Q@K^T
-> softmax -> attn@V` at HiFi4+fp32_dest_acc replaced by
`paged_scaled_dot_product_attention_decode`: 3 dispatches/layer collapse
into one fused tile-level kernel, **O(1) in MAX_POS** (manual was
O(MAX_POS)), streams K/V from a paged cache. HiFi2 + no fp32_dest_acc was
the bigger lever than expected — also kills the pos-129 prefill cliff
(below). 1.35x TP -> 2.20x TP, past El Reg's 1.78x 4-chip ceiling.

**2. Continuous batching at trace (11.6x at B=32; 45.8x at B=64 with
batched conv1d).** At B=1 the matmul reads all weights from DRAM but uses
one row of each output tile. Raising B is free per-step until compute
crosses memory. Measured `step_ms ~= 73 + 4.3*B`: B=32 costs 2.76x/step,
services 32x = **150.5 tok/s agg**. Crossover at B~18. Eager looked 31x
(dispatch-bound, misleading). With per-slot vector ops fused (batched
conv1d "shiftacc", 3-column padding-free state, 28.76x faster than kdim),
B=64 -> **593 tok/s agg**.

**3. 35B perf session: 141.79 -> 81.16 ms/tok (1.75x).** Four stacked:
- **A004 (core_grid=10x11 on MoE batched matmuls): -30 ms/tok.** ttnn's
  auto-matmul defaulted to **11 of 110 cores** on the dominant MoE gate_up
  op — kernel-time bottleneck. Full grid = 1:1 trace win.
- **A008 (bf8 MoE expert weights): -29 ms/tok.** MoE weights dominate DRAM
  footprint; halving bytes halves streaming time on the dominant kernel.
- **A002 (fused Q/K L2-norm): -1.13 ms/tok trace** (vs ~17 ms isolated).
- **A003 (router topk reorder): -0.25 ms/tok trace** (vs ~7 ms isolated).

A002/A003 illustrate the kernel-vs-dispatch rule: in trace, fusing N small
ops into 1 buys only the kernel-time delta (~5-10% of isolated eager).

**4. Vocab-sharded lm_head + on-device argmax.** 27B prod ran replicated
`[3840, 248320]` lm_head then read `[1, 248320]` fp32 logits to host (~600
KB) for numpy argmax. Fork: `ShardTensorToMesh(mesh, dim=1)`, chip owns
`[3840, 62080]`, **bf16** all_gather (~500 KB on-device, no PCIe), one
`ttnn.argmax` returns 8-byte int. Isolated 37.7 -> 2.58 ms = **14.6x**.
End-to-end **+5.1% (27B)** and **+8.0% (Gemma 4 12B)**. The 14.6x -> 5-8%
gap is because the traced readback was already partially pipelined.

**5. Owned fused decay/gate kernel for 27B DeltaNet (+2.5%).** 10-op chain
(`add -> softplus -> exp -> neg -> mul -> exp -> sigmoid -> reshape x 2`)
per layer collapses into one tile-level compute kernel — saves dispatch
plus 9 L1 round trips. Cos 0.9988 on 500-step ladder.

**6. `num_links=2` (+1.65%).** Blackhole P150x4 exposes 2 active eth links
per axis; prod was using `num_links=1`. One-line flip, isolated:
`reduce_scatter -12.9%`, `all_reduce -11.2%`. Isolated->prod realization
~32% (typical for sub-step CCL).

**7. The HiFi4 SDPA cliff (B3 fix).** Precision unlock, not perf:
HiFi4+fp32_dest_acc on Blackhole SDPA decode produces a pos-129 cliff
(top1@500 = 27.8%). Switching to **HiFi2, no fp32_dest_acc, no
packer_l1_acc** (matching llama3-70b-galaxy) takes top1@500 to **98.4%**
at only 1.8% latency cost. The "lower fidelity" config is what actually
produces faithful long-context behavior.

## What didn't work and why

**DRAM-sharded MLP (27B, 2.1x SLOWER).** Canonical tt-metal recipe
(`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` + L1 width-sharded
activation) is designed for **Galaxy 32-chip Llama-70B**. On single-chip
P150, B=1, HIDDEN=5120, INTERMEDIATE=17408: 0.930 ms (INTERLEAVED) ->
1.975 ms (DRAM-sharded). Reasons: (a) per-core L1 = 1.57 MB; with 4 chained
MLP ops only `in0_block_w_cap <= 2` fits -> more streaming iterations;
(b) shape too small (Galaxy: 56 MB/matmul vs ours 95 MB amortizes fixed
overhead worse); (c) batch=1 doesn't deliver Galaxy's compute density.
INTERLEAVED already hits 60-78% of peak — no headroom for layout tricks.

**Async `all_reduce_async`.** +4% setup vs sync; residual-stream decode
has no parallel compute to overlap (each all_reduce feeds an immediate
residual_add). Probe also showed TT-Metal **already auto-pipelines sync
ops on the same CQ**: `sync(ar+matmul) = 0.585 ms` vs `(sync ar) + (sync
matmul) = 0.675 ms` = 13% of matmul already overlapped. Async only wins
on parallel CCLs we don't have.

**fp32 KV cache (dead end).** `paged_scaled_dot_product_attention_decode`
hard-rejects fp32 Q/K/V at validation. fp32 storage + typecast to bf16 at
read gives Δcos = +0.000000 (rounding happens at typecast). Real fix needs
a custom C++ kernel. We dodged it via B3 instead.

**bf8 KV cache (neutral).** SDPA reads bf16 internally; bf8 storage between
two bf16 boundaries adds rounding below the noise floor (Δcos = -2.7e-5).
Saves 2x cache RAM, but we're not memory-bound at MAX_POS <= 8192.

## The architectural lesson

**Rule 1 (kernel vs dispatch).** Trace amortizes dispatch to ~0. Kernel-time
wins (A004: 60 ms isolated -> 30 ms trace = ~50% realization) translate
1:1. Dispatch-only fusions (A002 QK norm: 17 ms -> 1.13 ms = 7%; A003
router: 7 ms -> 0.25 ms = 4%) buy almost nothing in trace because the
dispatch cost was already 0. Predict trace gain by asking "kernel work or
dispatch?"

**Rule 2 (eager pipelining already hides ~13%).** Sum of isolated per-op
times (302 ms) > measured full-forward (267 ms) at 27B baseline. Trace's
incremental ceiling at B=1 is **~2x**, not 5-10x. Gemma 4 12B traced gives
a 3.56x speedup once Python-in-the-loop is also removed — consistent.

## Continuous batching sidebar

CB is the single biggest perf lever because matmul work is weight-bandwidth-
bound and B is free until compute crosses memory:

| B  | step_ms trace | agg tok/s | vs B=1 |
|----|----:|----:|---:|
| 1  | 77.15  | 12.96  | 1.0x |
| 8  | 106    | 75     | 5.8x |
| 32 | 212.62 | 150.45 | 11.6x |
| 64 | ~350   | 183.5  | 14.2x |

With shiftacc batched conv1d: B=32 -> **376.92 tok/s (29.1x)**, B=64 ->
**593.12 tok/s (45.8x)**. Lesson: matmuls **amortize across the batch**
(weight-bound); cost that scales with B is per-slot **vector ops** (conv1d,
RMSNorm, GQA-repeat). Fuse those, not the matmuls.
