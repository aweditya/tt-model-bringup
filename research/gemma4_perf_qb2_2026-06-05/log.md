# Gemma 4 12B perf work on qb2 — 2026-06-05 session log

Subagent operating on qb2; main agent on qb1 doing Nemotron-3 Nano bringup.
Append-only. Each entry is timestamped.

## State at session start

- qb1 canonical perf: 47.5 ms/tok traced (21.05 tok/s) single-stream after P1 vocab-shard (`a24f2ea`).
- qb2 state: tt-xla was stale (May 17, non-git). Active surface deployed via `TT_HOST=qb2 bash scripts/deploy.sh` (99 paths).
- qb2 tt-metal builds: `~/tenstorrent/tt-metal/build/` (non-Tracy) and `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/` (Tracy). NOTE: scripts default `TT_BUILD_DIR=build_Release` — must override to `build` for qb2.
- Gemma 4 12B IT freshly downloaded to qb2 HF cache (23 GB).
- Backing kernels validated on qb2: `qwen36_gdn_*` family + `paged_fused_update_cache` present in `ttnn.experimental`.

## Plan

1. Baseline traced perf on qb2 (fork dev-harness pattern for fast iter).
2. Tracy capture of one signposted forward via `experiments/utils/tracy_profile_one_gemma4_layer.py`.
3. Identify TOP-3 kernel-time ops via tt-perf-report.
4. Land one optimization (P2 distributed RMSNorm, P3 paged SDPA on globals, or paged_fused_update_cache — pick by profile).
5. Re-measure + regression-gate.

## Entries

- 23:24 — **Baseline established** on qb2 via `gm4_v04_trace_validate.py` (IT variant). Eager 474.1 ms/tok, **traced 51.4 ms/tok (19.45 tok/s)**, 100/100 token-for-token match. Note: qb1 headline was 47.5 ms/tok post-P1; the ~4 ms gap may be qb2 hardware/thermals or measurement noise (n=100 not n>3 of n=100). Treating 51.4 as the qb2-local baseline. Vocab-shard `_lm_head_argmax` is in deployed `server_gemma4_unified_ttnn.py`. Log: `.cache/gemma4_perf_qb2/baseline_v04_run1.log` on qb2.
- 23:28 — Tracy probe ran; DRAM marker buffer overflow zeroed device kernel times but op COUNTS were intact. Per signposted forward: 176 PagedUpdateCache (= 40 sliding × 4 + 8 global × 2), 88 SdpaDecode, 329 Matmul (≈7/layer × 48), 337 LayerNorm. CSV: `.cache/perf_logs/tracy_gemma4_layer/reports/2026_06_05_23_29_04/`.
- 23:35 — **First fused-cache attempt FAILED**: `paged_fused_update_cache` TT_FATAL: "input_tensor1 and input_tensor2 must not overlap" (`paged_fused_update_cache_device_operation.cpp:226`). Both K and V landed on core (0,0) because we share `paged_write_mem_cfg_sliding`/`_global` for both. Llama-Galaxy avoids this via `llama_rs_create_heads` which lands K and V on disjoint cores naturally; we need to add disjoint V mem configs.
- 23:40 — **paged_fused_update_cache LANDED**: added `paged_write_mem_cfg_sliding_v` (core (1,0)) + `paged_write_mem_cfg_global_v` (core (1,0) for NKV=1) so K + V shards are disjoint. `gm4_v04_trace_validate.py` VERDICT PASS (100/100 token match). **Eager 474.1 → 236.4 ms/tok (-50.1%)**; **traced 51.4 → 50.9 ms/tok (-0.97%, n=1)**. Traced delta is small as expected — the cache writes are dispatch-bound, and trace already amortizes dispatch. Eager delta is huge because each forward eliminates 88 device dispatches (~3 ms/dispatch on host = ~264 ms saved). Log: `.cache/gemma4_perf_qb2/fused_cache_v04_run2.log`. Needs reproducibility: re-run 2+ times.
- 23:45 — **Reproducibility runs 3 & 4 PASS** (3×100/100 token match). Final aggregate (n=3 traced after the fix):
  - **Eager**: 474.1 → mean(236.4, 214.8, 222.8) = **224.7 ms/tok (-52.6%, 2.11× speedup)**
  - **Traced**: 51.4 → mean(50.9, 51.4, 51.1) = **51.13 ms/tok (-0.5%, within noise floor)**
  - Logs: `.cache/gemma4_perf_qb2/fused_cache_v04_run{2,3,4}.log`
  - Verdict: **paged_fused_update_cache ships ~2× eager speedup, zero traced regression, zero correctness regression**. Validates [[feedback-kernel-vs-dispatch-realization]]: dispatch fusions are nearly invisible at the trace level. Eager wins matter for dev-harness iteration speed (per-probe wall time roughly halved).

- 23:45 — **Op-count census from the failed-tracy CSV** (warnings stripped):
  - 337 LayerNorm (= 4 per-layer × 48 + 1 final + 3 attn × 48 = 192 + 1 + 144 = 337) ← biggest opportunity, but distributed RMSNorm requires sharding hidden state
  - 329 Matmul (~7/layer × 48 = QKV+O+gate+up+down)
  - 88 SdpaDecode (~2/sliding-layer × 40 + 1/global-layer × 8 = 88)
  - 88 (was 176) PagedFusedUpdateCache after fix
  - 1932 BinaryNg, 1504 UntilizeWithUnpadding, 1028 TilizeWithValPadding — layout-shuffle volume hints redundant `to_memory_config` may be present (tt-metal #44958 audit candidate)

