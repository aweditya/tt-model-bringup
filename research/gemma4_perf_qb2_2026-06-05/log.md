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

- 23:48 — Wrote `experiments/utils/tracy_profile_one_gemma4_layer_v2.py` (single-forward, no-warmup) to dodge the marker-buffer overflow. Captured a forward (CSV: `.cache/perf_logs/tracy_gemma4_v2/reports/2026_06_05_23_49_39/`). Partial-data findings (cold JIT contaminates first-pass kernel times; ops that survived):
  - **Matmul: 340 ops × 56.7 μs mean = 19.3 ms / forward** (37% of 51 ms traced budget)
  - **TilizeDevice: 316 valid samples × 61.0 μs mean = 19.3 ms / forward** (37% — huge surprise; matches matmul!)
  - **TilizeWithValPadding: 676 × 12.6 μs = 8.5 ms** (~16%)
  - LayerNorm, SDPA, PagedFusedUpdateCache, AllGather, ReduceScatter, Untilize: marker-dropped, kernel time unknown (likely the remaining ~10 ms)
- Conclusion for the next opt round: **Tilize + TilizeWithValPadding total ~28 ms / forward = >50% of traced budget**. The "1504 UntilizeWithUnpadding + 1728 Slice + 704 ReshapeView" volume hints that `_shard_for_paged_write` (5+ ops per K/V staging × 4 per sliding layer = ~800/forward) is a fat dispatch-tax site. Audit candidate for #44958 (tt-metal) / tt-metal-adoption-plan §"memory-config audit" items.

## Final state

- **Landed**: `paged_fused_update_cache` on Gemma 4 12B sliding+global decode paths. Commit `de0384a`. **Traced delta: -0.5% (within noise)**, eager delta: -52.6% (2.11×). 3×100/100 token match.
- **NOT landed (risk/scope vs time budget)**: distributed RMSNorm (P2, +12-15 ms/tok projected, requires sharding hidden state across mesh — hour-scale work + medium regression risk). Tracy probe v2 confirmed that LayerNorm marker data is missing so we can't sanity-check the projection without more profiling work.
- **Open next**: eliminate redundant TM ops in `_shard_for_paged_write` (5-op chain per K/V staging; ~800 dispatches/forward). Could fold the row-major→pad→tile→to_memory_config into the upstream matmul output spec. Estimated savings: 0.5-2 ms/tok traced.
- **Open further out**: distributed RMSNorm via `ttnn.rms_norm_pre_all_gather` + `all_gather` + `ttnn.rms_norm_post_all_gather` (or the all-in-one `ttnn.fused_rms_minimal`). All three ops are available in qb2's ttnn build. Requires sharding `h` from REPLICATED → 1/4 width on each chip. Major surgery.

---

## Round 2 (subagent, picks up where round 1 left off)

### Plan + lever pick — `_shard_for_paged_write` simplification

- 00:11 — **Baseline re-confirm run 1** with main HEAD on qb2: eager **231.9 ms/tok**, traced **50.9 ms/tok**, 100/100 PASS. Matches round 1's mean (eager 224.7, traced 51.13) within noise. Log: `logs/baseline_round2_run1.log`.
- **Lever picked**: `_shard_for_paged_write` 5-op chain → minimal `to_memory_config`-only reshard.
  - **Why now**: round 1's Tracy v2 measured Tilize + TilizeWithValPadding = 28 ms / forward (>50% of traced budget). `_shard_for_paged_write` fires 176×/forward (88 K + 88 V across 40 sliding + 8 global decoder layers); each invocation calls `to_layout(ROW_MAJOR)` → `reshape` → `pad` → `to_layout(TILE)` → `to_memory_config`. The post-RoPE input is ALREADY tile-layout, and slicing `[1, head_dim]` from a TILE-padded `[NKV, head_dim]` tensor gives byte-identical layout to the post-pad+tile `[1, 1, BLOCK_SIZE=32, head_dim]` (tile-pad rows 1..31 = zeros either way; kernel only writes row 0). So untile→pad→tile is pure overhead.
  - **Mechanism**: replace `_shard_for_paged_write` chain with a single `to_memory_config` reshard from the post-slice tile-layout `[1, head_dim]` to the L1-sharded `[BLOCK_SIZE, head_dim]` memory config — bytes match, only metadata changes.
  - **Predicted delta**: tighter (kernel-time, not just dispatch tax). Eager: maybe -30 to -80 ms / tok (4 dispatches × 176 calls × ~100 μs each). Traced: should land 1-5 ms / tok if the Untilize+Tilize ops are kernel work (round 1's Tracy v2 said 19.3 ms Tilize + 8.5 ms TilizeWithValPadding kernel time / forward).
  - **Risk**: medium. Kernel `paged_fused_update_cache` validates `input.padded_shape[1] == 1` (B), `input.padded_shape[-1] == cache.padded_shape[-1]`, input.is_sharded, shard.shape[1] == padded_shape[-1]. Need to confirm a `[1, head_dim]` tile-layout (padded to `[32, head_dim]`) reshape-viewed as `[1, 1, 32, head_dim]` satisfies these. If reshape's volume check fails, fall back to `ttnn.view` (no volume check). If even THAT fails, fall back to a one-call `_shard_for_paged_write_v2` that does just `to_memory_config` and validate that the padded_shape automatically becomes `[1, 1, 32, head_dim]` after the reshard.
  - **Gate**: 100/100 token-for-token + n=3 traced runs.

### Probe + production land

- 00:17 — **Isolation probe PASS**: `experiments/cb/isolate/gm4_shard_for_paged_write_v2.py` runs single-device, fakes `paged_fused_update_cache` with N_KV=1, HEAD_DIM=256 (sliding shape). Both variants land K, V at the correct cache slot at cos = 0.999999. **Cross-variant max|delta| = 0.0** (K and V) — the simplified variant is bit-identical to the old chain. Confirms the post-RoPE tile-layout `[1, head_dim]` (padded to `[32, head_dim]`) reshape-viewed as `[1, n_kv_heads, 1, head_dim]` carries the same bytes after `to_memory_config`. Log: `logs/probe_shard_v2.log`.
- 00:20 — **Production `_shard_for_paged_write` simplified** (`server_gemma4_unified_ttnn.py:1066-1106`). 5-op chain → 2-op (reshape + to_memory_config). Deployed to qb2. **Validator run 1**: eager 199.8 ms/tok, traced 49.7 ms/tok, 100/100 PASS.
- 00:22 — Validator run 2: eager 211.7, traced 49.5, 100/100 PASS.
- 00:24 — Validator run 3: eager 214.9, traced 49.5, 100/100 PASS.

### Final aggregate (n=3 after the fix)

| Metric | Baseline (round 2) | Simplified | Delta |
| --- | --- | --- | --- |
| Eager mean ms/tok | 221.8 (std 8.8) | 208.8 (std 6.5) | **-5.9% (-13 ms)** |
| Traced mean ms/tok | 51.27 (std 0.32) | 49.57 (std 0.09) | **-3.3% (-1.7 ms)** |
| Traced tok/s | 19.50 | 20.17 | **+3.4%** |
| Token-for-token | 100/100 | 100/100 | clean |

- Baseline logs: `logs/baseline_round2_run{1,2,3}.log`
- Post-fix logs: `logs/shard_v2_v04_run{1,2,3}.log`
- **Both eager AND traced moved** — unlike round 1's paged_fused_update_cache where only eager moved. The 4 dispatches per `_shard_for_paged_write` × 176 calls × ~17 μs / dispatch ≈ ~12 ms / forward of pure dispatch tax is now gone; that's what the eager delta reflects. The traced delta (~1.7 ms) is the kernel-time portion (Untilize + Tilize work that the kernel was actually doing — caught by Tracy v2 at 28 ms / forward but the simplification only removes a piece of that, since the FUSED-UPDATE kernel itself still untilizes a single tile per head inside).

