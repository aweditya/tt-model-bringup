# Multi-Chip TP Optimization Menu — qb2, 4 × Blackhole P150

**Built:** 2026-05-14 by Agent O (doc survey) + verification.
**Purpose:** Ranked menu of optimization candidates, indexed by Tracy outcome. Consume when Agent M returns with the per-op breakdown.

## Current state

`experiments/serve/server_tp.py:651-674` runs a full decode trace on (1, 4) mesh. Sustained **7.02 tok/s** vs single-chip **5.19 tok/s** — only **1.35×** scale-out, far below the 4× ideal. Collectives use eager `ttnn.all_reduce` / `ttnn.all_gather` (no persistent buffers). RMSNorm is plain `ttnn.rms_norm` on REPLICATED residual data (`server_tp.py:361-376` + `:632` — verified `x_buf` is replicated via `ReplicateTensorToMesh`). LM head is replicated (not vocab-sharded). No sub-devices / dram-prefetcher infra.

## Verification note (2026-05-14)

Agent O's initial summary claimed `_rms_norm_manual` may be numerically WRONG because it operates on "width-fractured" data. **This is incorrect.** Verified by reading `server_tp.py:632, 637` — `x_buf` and `cur_pos_buf` are uploaded with `ReplicateTensorToMesh`, and `all_reduce(partial)` returns a replicated tensor. So `x_tt` entering `_rms_norm_manual` is REPLICATED across all 4 chips. Plain `ttnn.rms_norm` over the last dim is mathematically correct.

What IS true: 4 chips redundantly computing the same norm is wasted work. The distributed-RMSNorm pattern saves time by parallelizing the reduce — it's a perf win, not a correctness fix.

## Top 14 candidates (ranked by ROI given likely Tracy outcomes)

| # | Name | Bottleneck class | Est. win | Effort | Key reference |
|---|------|------------------|----------|--------|---------------|
| 1 | Distributed `rms_norm_pre_all_gather` / `rms_norm_post_all_gather` | collective + dispatch | 15–25 ms/tok | M / low | `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390` |
| 2 | Persistent-buffer `all_reduce_async` | collective | 5–15 ms/tok | M / med | `llama_ccl.py:694-773` (call), `:712` (kernel) |
| 3 | `reduce_scatter_minimal_async` for row-parallel out_proj / w2 | collective (BW) | 3–10 ms/tok | M / med | `llama_ccl.py:1057`; usage `llama_mlp.py:123-149` |
| 4 | DRAM-sharded matmul program configs (decode w1/w3, w2, out_proj, lm_head) | memory layout / BW | 10–25 ms/tok | H / med | `llama_mlp.py:175-185`; `tech_reports/LLMs/llms.md:601-618` |
| 5 | Sub-devices + `dram_prefetcher` | dispatch + DRAM BW | 10–30 ms/tok | H / high | `llama_model.py:797-805`; `prefetcher_common.py:67-120`; `tech_reports/SubDevices/SubDevices.md` |
| 6 | Vocab-sharded LM head | replicated work + BW | 4–10 ms/tok | L / low | `models/demos/llama3_70b_galaxy/tt/lm_head.py:39-105`; `tech_reports/LLMs/llms.md:868-967` |
| 7 | `fused_rms_minimal` (RMSNorm + AllGather + residual-add fused) | collective + dispatch | 5–10 ms/tok | M / med | `llama_ccl.py:1393-1429`; `llama_decoder.py:148, 155, 157, 175, 179, 181` |
| 8 | Fused `all_gather_minimal_matmul_async` | collective + dispatch | 2–5 ms/tok | M / med | `llama_ccl.py:1201-1265`, kernel at `:1247`; used at `llama_mlp.py:304` |
| 9 | Second CQ for `update_input_buffers` writes | per-step host overhead | 2–5 ms/tok | L / low | `tech_reports/AdvancedPerformanceOptimizationsForModels.md:157-378` |
| 10 | Sharded residual stream in L1 (`DECODE_RESIDUAL_MEMCFG`) | memory layout | 2–5 ms/tok | M / med | `llama_decoder.py:138-141`; `tech_reports/LLMs/llms.md:804-867` |
| 11 | `paged_fused_update_cache` (K+V in one op) | dispatch | 1–2 ms/tok | L / low | `llama_attention.py:509-511` |
| 12 | Drop redundant `synchronize_device` calls in warmup | sync (startup) | startup-only | L / low | `server_tp.py:758, 761` |
| 13 | `paged_scaled_dot_product_attention_decode` on mesh (retry) | dispatch | 1–3 ms/tok | M / med | `llama_attention.py:523-533` |
| 14 | `nlp_create_qkv_heads_decode` to replace manual slice/reshape | dispatch | 1–2 ms/tok | L / low | `llama_attention.py:649`; `tech_reports/LLMs/llms.md:436-441` |

## Per-candidate "Why"

1. **Distributed RMSNorm** — Today `_rms_norm_manual` calls `ttnn.rms_norm` on a replicated tensor (`server_tp.py:376`); 4 chips do redundant full reductions. Production splits into pre-AG (partial stats only, tiny vector) + AG of stats + post-AG. Memory note `feedback_ttnn_fused_ops_gap_analysis.md` projects ~18 ms/tok at 305 calls/tok.
2. **`all_reduce_async`** — Replaces eager `ttnn.all_reduce(partial)` (`server_tp.py:462, 499, 607`). Uses pre-allocated persistent buffer + pre-allocated semaphore (no per-call alloc), and `use_optimal_ccl_for_llama=True` selects a tuned kernel.
3. **`reduce_scatter_minimal_async`** — All-reduce = reduce_scatter + all_gather. If the next op consumes shard-distributed data (which a distributed RMSNorm does), skip the gather. Natural pair with #1.
4. **DRAM-sharded matmul** — Decode is DRAM-bound (weights >> activations). Default `ttnn.linear` doesn't pick optimal `in0_block_w` / `per_core_M/N` for our TP-sharded weight shapes. Memory note `feedback_qb1_mlp_at_78pct_peak.md` shows isolated single-chip MLP is 78% of 512 GB/s — the TP server is likely worse.
5. **dram_prefetcher + sub_devices** — Streams all layer weights into a Global Circular Buffer on a "prefetcher" SubDevice while compute proceeds on a "worker" SubDevice. Hides DRAM weight load behind compute. Biggest multi-chip lever Galaxy uses.
6. **Vocab-sharded lm_head** — Memory note `feedback_p3_lm_head_replicated_pass.md` measured 4.16 ms replicated at 72% of 512 GB/s; sharding pushes to ~1 ms. Lowest-effort big win.
7. **`fused_rms_minimal`** — One op = residual add + RMSNorm + all_gather + layout shuffle. Strictly stronger than #1 once you accept #1.
8. **`all_gather_minimal_matmul_async`** — Pipelines AG with matmul tiles. Free win after #3 ships.
9. **Second CQ** — We issue 4 host→device `copy_host_to_device_tensor` serially before every `execute_trace` (`server_tp.py:614-648`). On CQ1 with event sync to CQ0, overlaps with previous step.
10. **Sharded residual** — Default `ttnn.add` likely round-trips residual through interleaved DRAM between layers. Keep it in L1 width-sharded.
11. **`paged_fused_update_cache`** — Combines two `paged_update_cache` (`server_tp.py:572, 575`) into one dispatch.
12. **Drop warmup syncs** — Startup-only; deprioritize.
13. **paged_sdpa on mesh** — `feedback_p1_sdpa_decode_breaks_on_mesh.md` says SDPA decode failed on (1,4); paged variant may work since it's documented as the canonical mesh path in Galaxy. SDPA is only 2% of attn (`feedback_attn_per_op_findings.md`), so modest.
14. **`nlp_create_qkv_heads_decode`** — Replaces 5-6 manual ops at `server_tp.py:531-540` and `:399-405`.

## Candidates NOT included

- **All-to-all** — no MoE in Qwen3.6-27B.
- **`rotary_embedding_llama`** — abandoned (`feedback_c3_native_rope_abandoned.md`).
- **bf8 KV / bf8 MLP** — already shipping / already neutral.
- **mul_reduce_scalar conv1d** — C++ only.
- **2D weight fracturing** — needs (8,4) mesh; we have (1,4).
- **In-place scatter** — superseded by `update_cache_for_token_`.

## Tracy → recommendation decision matrix

| Tracy result (top time consumer?) | Top candidate | Backup |
|----------------------------------|---------------|--------|
| Device-side all_reduce/all_gather (CCL) kernels | #2 (`all_reduce_async`) | #3 (reduce_scatter) |
| RMSNorm kernels or many small `to_layout`/`add` in norm region | #1 (distributed RMSNorm) | #7 (`fused_rms_minimal`) |
| Matmul kernels at <50% of 512 GB/s | #4 (DRAM-sharded matmul) | #5 (dram_prefetcher) |
| Large host gaps between trace executes (host-bound) | #9 (second CQ) | #12 |
| `to_memory_config` / `interleaved_to_sharded` visible | #10 (sharded residual) | #4 |
| `lm_head` matmul dominates a single token | #6 (vocab-sharded LM head) | #4 |
| Many small ops on attention | #14 (`nlp_create_qkv_heads_decode`) | #13 (paged_sdpa) |
| `paged_update_cache` dispatch gaps | #11 (`paged_fused_update_cache`) | — |
| Gaps inside trace between matmul + next op | #5 (dram_prefetcher) | #4 |
| Single biggest = collective, compute also low BW | Stack #1 + #2 + #3 + #6 | — |
| Single biggest = matmul, compute is BW-bound | Stack #4 + #5 + #6 | — |

## Anchor recommendation (no Tracy required)

**Ship #6 (vocab-sharded lm_head) first.** Lowest effort in the top 6 (~50 LOC), pure win with no risk, well-documented reference in `models/demos/llama3_70b_galaxy/tt/lm_head.py`. Then #1 (distributed RMSNorm). Then let Tracy decide #2 vs #3 vs #4.

Sequence: **#6 → #1 → (Tracy gates rest) → #2/#3 → #4 → #5**.

## Surprising finds from the survey

- **Galaxy fuses residual-add into the *next* layer's RMSNorm** via `tt_sharded_distributed_rmsnorm`'s `residual_input_tensor` arg (`llama_ccl.py:1413-1429`). We do separate adds (`server_tp.py:468, 504, 611`).
- **`ttnn.fused_rms_minimal`** is a single op that fuses RMSNorm + AllGather + residual-add (`llama_ccl.py:1413`) — strictly stronger than candidate 1 alone.
- **Galaxy uses `all_reduce_create_qkv_heads`** (`llama_ccl.py:794`) — fuses the QKV input gather with head splitting in one kernel.

## What was skipped

- DeepSeek V3 reference (Galaxy was sufficient breadth).
- Deep dive into `qwen_model_config.py` (2207 lines, mostly shape constants — low priority).
- TT-Distributed multi-host arch (not relevant for single-host 4-chip).
