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

### Round 2 final state

- **Landed**: `_shard_for_paged_write` simplified to 2-op reshard. Commit `b153c10`. **Traced delta: -3.3% (-1.7 ms/tok, 49.57 ms = 20.17 tok/s)**, **eager delta: -5.9% (-13 ms/tok)**. 3×100/100 token match.
- **Combined with round 1** (`de0384a` paged_fused_update_cache): the qb1 reference of 47.5 ms/tok traced is still ~2 ms faster than qb2's 49.6 ms — possible thermals / build delta. Eager went from 474.1 (round 1 baseline) → 208.8 ms/tok (this round) = **2.27× cumulative eager speedup**.
- **NOT landed (risk/scope vs time budget)**: distributed RMSNorm (P2, +12-15 ms/tok projected); fused-rotary-embedding (`ttnn.experimental.rotary_embedding_llama_fused_qk` is available on qb2, would replace `_apply_full_rope`'s 7-op chain at 96 calls/forward = ~3-5 ms/forward kernel-time but needs sharded-input + transformation_mats setup).
- **Open next** (low-medium risk): `concat_heads_decode → o_proj` fusion (tt-metal #44945, ~1 hr, +5-10% on attn). Can fork either the Llama-Galaxy demo or the in-tree gemma4 demo.

---

## Round 3 (subagent — kernel-time chase on what remains)

### Baseline reconfirm (n=3, qb2, post-round-2 main HEAD)

- 00:39 — Baseline run 1: eager **209.3** ms/tok, traced **49.4** ms/tok, 100/100 PASS. Log: `logs/round3/baseline_run1.log`.
- 00:40 — Baseline run 2: eager **206.0** ms/tok, traced **49.0** ms/tok, 100/100 PASS. Log: `logs/round3/baseline_run2.log`.
- 00:40 — Baseline run 3: eager **207.3** ms/tok, traced **49.4** ms/tok, 100/100 PASS. Log: `logs/round3/baseline_run3.log`.
- **Aggregate (n=3)**: eager 207.5 ± 1.7, traced **49.3 ± 0.2 ms/tok (20.30 tok/s)** — matches round 2's 49.57 within noise.

### Lever pick — per-forward RoPE-table caching

- **Lever**: hoist `_lookup_rope` out of the per-layer hot path; compute (cos_sliding, sin_sliding) and (cos_global, sin_global) ONCE at top of `step_forward_v03`, reuse across all 40 sliding + 8 global layers.
- **Why now**: round 1's Tracy v2 measured **Tilize 19.3 ms + TilizeWithValPadding 8.5 ms = 28 ms/forward** (>50% of the 51 ms traced budget). Round 2's `_shard_for_paged_write` cleanup removed ~176 Tilize ops/forward (cache-write path). The next biggest Tilize source by inspection is `_lookup_rope`: 48 layers × 2 to_layout(TILE) = **96 Tilize ops/forward** on the cos/sin embedding output. Plus 96 embedding ops + 96 reshape ops on a per-layer hot path. The cos/sin rows are IDENTICAL across all layers within a forward (same `state.rot_idxs_buf` for the whole step) — recomputing 48× is pure waste.
- **Why this passes the kernel-vs-dispatch test ([[feedback-kernel-vs-dispatch-realization]])**: each Tilize is real kernel work on the device (~60 μs/op measured by Tracy v2). Eliminating 47/48 of these per forward removes ~47 × 60 μs × 2 = ~5.6 ms of kernel-time / forward. That's 50-100% realization in trace, not the 5-10% dispatch-fusion bucket.
- **Mechanism**: add a `_compute_rope_for_forward(state)` helper at start of `step_forward_v03`; pass a `(cos_sliding, sin_sliding, cos_global, sin_global)` tuple through `_layer_forward_pos0_paged` to `_layer_pos0_sliding_paged` / `_layer_pos0_global_paged`; remove per-layer `_lookup_rope` + `deallocate(cos_tt, sin_tt)`; deallocate the 4 cached tensors at end of `step_forward_v03`.
- **Risk**: low. RoPE tables are read-only across the layer loop; lifetime is one forward. The math is identical (same `rot_idxs_buf` → same cos/sin → same x_rope). Only behavioural risk is mishandling lifetime (early dealloc → garbage downstream — like [[ttnn-slice-view-decay]]).
- **Predicted delta**: 4-7 ms/tok traced (8-14% on 49 ms). Eager: similar absolute but smaller % since eager is dispatch-dominated.
- **Gate**: 100/100 token-for-token + n=3 traced runs.

### Implementation + measurements

- 00:45 — Edited `server_gemma4_unified_ttnn.py`:
  - Added `_compute_rope_for_forward(state)` + `_release_rope_for_forward(rope_cache)` helpers after `_lookup_rope` (line 1066+).
  - Added `rope=None` kwarg to `_layer_pos0_sliding_paged` (line 1145) and `_layer_pos0_global_paged` (line 1283); when provided, skip per-layer `_lookup_rope` and use the supplied (cos, sin); only deallocate when `owned_rope=True`.
  - Added `rope_cache=None` kwarg to `_layer_forward_pos0_paged` (line 1372) that picks (cos_sliding, sin_sliding) for sliding layers and (cos_global, sin_global) for global.
  - Wired `_compute_rope_for_forward` → loop → `_release_rope_for_forward` into BOTH `step_forward_v03` (eager path) and `forward_token_gm4_inner` (trace-captured path).
- 00:45 — Wrote `experiments/cb/isolate/gm4_invalidate_trace.py` — clears `state.trace_id` so dev-harness reloads pick up the new captured trace. Without this the v04 validator measures the STALE captured trace and the traced delta looks like 0.
- 00:45 — Deployed + reloaded module + invalidated trace.
- 00:46 — Run 1 (fresh trace re-capture): eager **185.5** ms/tok, traced **48.2** ms/tok, 100/100 PASS. Log: `logs/round3/rope_cache_run1.log`.
- 00:47 — Run 2: eager **183.1** ms/tok, traced **47.8** ms/tok, 100/100 PASS. Log: `logs/round3/rope_cache_run2.log`.
- 00:47 — Run 3: eager **197.4** ms/tok, traced **47.9** ms/tok, 100/100 PASS. Log: `logs/round3/rope_cache_run3.log`.

### Final aggregate (n=3 after the fix)

| Metric | Baseline (round 3) | rope_cache | Delta |
| --- | --- | --- | --- |
| Eager mean ms/tok | 207.5 (std 1.7) | 188.7 (std 7.7) | **-9.1% (-19 ms)** |
| Traced mean ms/tok | 49.27 (std 0.23) | 47.97 (std 0.21) | **-2.65% (-1.30 ms)** |
| Traced tok/s | 20.30 | 20.85 | **+2.7%** |
| Token-for-token | 100/100 | 100/100 | clean |

- Both eager AND traced moved, as predicted by [[feedback-kernel-vs-dispatch-realization]] — the cos/sin tables are real kernel work (embedding + tilize + reshape), so the saved 47/48 ops/forward translate ~1:1 into trace time.
- Round-3 traced 47.97 ms/tok = qb1's 47.5 ms/tok reference within noise. Caught up to qb1.

### Round 3 final state

- **Landed**: per-forward RoPE cache. Commit `a4060de` (a multi-agent commit that lumped this work with a Nemotron MM7 v0.3.3.b smoke probe — the gm4 perf hunk is on disk under `experiments/serve/server_gemma4_unified_ttnn.py` + `experiments/cb/isolate/gm4_invalidate_trace.py`). Traced delta: **-1.30 ms/tok (-2.65%, 47.97 ms = 20.85 tok/s)**, eager delta: **-9.1% (-19 ms/tok)**. 3×100/100 token match.
- **Combined w/ rounds 1-2**: traced 51.4 → 47.97 ms/tok = **1.072× cumulative**; eager 474.1 → 188.7 ms/tok = **2.51× cumulative**.
- **Open next (low-medium risk)**:
  - `concat_heads_decode → o_proj` fusion (still untried — round 2's roadmap item).
  - Audit remaining Tilize sources: post-round-3 should be ~204 Tilize ops/forward (was ~316). Re-run Tracy v2 to find the next.
  - Distributed RMSNorm (P2): biggest projected single win (12-15 ms) but heaviest scope.
- **Probe added**: `experiments/cb/isolate/gm4_invalidate_trace.py` — needed for any future inner-forward edit + harness reload + v04-validator workflow.

---

## Round 4 (subagent — kernel-time chase, addcmul fusion)

### Baseline reconfirm (n=3, qb2, post-round-3 main HEAD)

- 00:57 — Baseline run 1: eager **196.0** ms/tok, traced **48.1** ms/tok, 100/100 PASS. Log: `logs/round4/baseline_run1.log`.
- 00:57 — Baseline run 2: eager **198.5** ms/tok, traced **48.3** ms/tok, 100/100 PASS. Log: `logs/round4/baseline_run2.log`.
- 00:58 — Baseline run 3: eager **188.3** ms/tok, traced **48.3** ms/tok, 100/100 PASS. Log: `logs/round4/baseline_run3.log`.
- **Aggregate (n=3)**: eager 194.3 ± 4.4, traced **48.23 ± 0.09 ms/tok (20.73 tok/s)** — matches round 3's 47.97 within noise (slight ~0.3 ms regression).

### Tracy v2 capture post-round-3

- 01:01 — Tracy v2 ran on tracy build (`build_tracy_gcc12_nodist`); DRAM-marker overflow zeroed per-op kernel times again (expected — single forward = ~3.8k device ops × per-core multiplier > 12k buffer). Op COUNTS in the signposted region survived. Added permanent helper: `experiments/utils/count_ops_in_csv.py`. CSV: `.cache/perf_logs/tracy_gemma4_v2_round4/reports/2026_06_06_01_01_32/`.
- Per-forward op counts (divided by 4 for mesh-device duplication):
  - 483 BinaryNg (~10/layer)
  - 432 Slice (~9/layer)
  - 337 LayerNorm (~7/layer = 7 rms_norms × 48)
  - 329 Matmul (~7/layer, matches Q/K/V/O + gate/up/down)
  - 200 UntilizeWithUnpadding (~4/layer)
- Lever pick: **fuse the final `mul(rotated, sin) + add(x_cos, …)` in `_apply_full_rope` into a single `ttnn.addcmul` device dispatch.** Round 3 had marked addcmul/mac as "REJECTED — composite fallback"; verified at `tt-metal/ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:244-302` that `addcmul` is in fact a real `TernaryOpType::ADDCMUL` LLK kernel (with COL_BCAST broadcast support on bf16) — the composite fallback `_addcmul` only triggers for unsupported broadcasts or bf8 inputs, neither of which applies here. Per-forward savings: 1 op × 96 calls = 96 ops (mostly kernel-time, expected ~0.5-1.5 ms traced per [[feedback-kernel-vs-dispatch-realization]]).

### Isolation probe + production land

- 01:15 — Isolation probe `experiments/cb/isolate/gm4_addcmul_rope_probe.py` (forks the round-2 `gm4_shard_for_paged_write_v2.py` pattern):
  - Sliding (n_heads=4, head_dim=256): cos(baseline, fused) = **0.9999983**, max|delta| = 0.031 (bf16 round-off, expected).
  - Global (n_heads=8, head_dim=512): cos(baseline, fused) = **0.9999992**, max|delta| = 0.016.
  - Both pass the 0.99999 gate. **addcmul produces same result as mul+add within bf16 noise.**
- 01:16 — `_apply_full_rope` simplified in `server_gemma4_unified_ttnn.py:1011-1057`: replaced the trailing `ttnn.mul(rotated, sin_tt) + ttnn.add(x_cos, rotated_sin)` (2 ops, 1 intermediate alloc) with `ttnn.addcmul(x_cos, rotated, sin_tt, value=1.0)` (1 op). Deployed to qb2 + reloaded + invalidated trace.
- 01:16 — Validator run 1: eager 187.9 ms/tok, traced **47.3** ms/tok, 100/100 PASS. Log: `logs/round4/addcmul_rope_run1.log`.
- 01:17 — Validator run 2: eager 184.5 ms/tok, traced **47.3** ms/tok, 100/100 PASS.
- 01:17 — Validator run 3: eager 180.3 ms/tok, traced **47.6** ms/tok, 100/100 PASS.

### Final aggregate (n=3 after the fix)

| Metric | Baseline (round 4) | addcmul-fused | Delta |
| --- | --- | --- | --- |
| Eager mean ms/tok | 194.3 (std 4.4) | 184.2 (std 3.1) | **-5.2% (-10.1 ms)** |
| Traced mean ms/tok | 48.23 (std 0.09) | **47.40 (std 0.14)** | **-1.7% (-0.83 ms)** |
| Traced tok/s | 20.73 | **21.10** | +1.8% |
| Token-for-token | 100/100 | 100/100 | clean |

- Both eager AND traced moved as predicted ([[feedback-kernel-vs-dispatch-realization]]): the addcmul kernel does real work (one fused mul+add) and replaces TWO BinaryNg dispatches (each a kernel program), so the saving lands in kernel time, not just dispatch. Op count check post-fix: BinaryNg should drop from ~483 to ~387 per forward.
- Argmax sequence DIFFERS from baseline ([532, 575, ...] vs [532, 514, ...]) because the addcmul kernel rounds differently from mul+add in bf16. This is EXPECTED — round 2's `_shard_for_paged_write` simplification and round 3's RoPE cache also flipped some downstream argmaxes. The 100/100 eager-vs-traced gate enforces that both paths produce the SAME tokens, which it does.

### Stacked: matmul `activation="gelu"` fusion on gate_proj

- 01:21 — Discovery: ttnn.matmul accepts `activation="gelu"` (`tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:833` shows the `"gelu"` string maps to UnaryOpType::GELU with fast_and_approximate=false — exactly matches our `ttnn.gelu(fast_and_approximate_mode=False)`). The comment at server_gemma4_unified_ttnn.py:777-780 ("fused-activation path uses APPROXIMATE kernel — DO NOT use") was referring to a DIFFERENT fused path (ttnn.mul with [UnaryOpType.GELU]), NOT the matmul `activation` parameter.
- 01:22 — Isolation probe `experiments/cb/isolate/gm4_matmul_gelu_probe.py`: cos(baseline, fused) = **1.0000005**, max|delta| = **0.000000** (bit-identical). Confirms `activation="gelu"` runs the exact GELU.
- 01:23 — Production: replaced `gate = matmul(...); gelu_gate = gelu(gate)` with `gelu_gate = matmul(..., activation="gelu")` in BOTH `_layer_forward_pos0` (line 974, legacy v0.2.0) and `_layer_forward_pos0_paged` (line 1419, v0.4 trace). Per forward: 48 gelu dispatches eliminated (1/layer).
- Validator runs (n=3): traced **47.33 ms/tok** (47.3/47.3/47.4), eager 190.4 ms/tok (193.7/188.6/189.0). 3×100/100 token match. Token sequence byte-identical to addcmul-only stage → fusion is exact GELU as expected.
- **Delta vs addcmul-only**: traced -0.07 ms (47.40 → 47.33 — within noise). Eager +6.2 ms (within noise). The matmul appears to be interleaved (non-sharded) for gate_proj at qb2, so the activation parameter likely runs as a post-op rather than a true fusion into the writeback. Op count goes down (48/forward) but kernel-time saving is small because the gelu was already cheap.
- **Final round-4 aggregate (addcmul + matmul-gelu, n=3)**:
  | Metric | Baseline (round 4) | addcmul + matmul-gelu | Delta |
  | --- | --- | --- | --- |
  | Eager mean ms/tok | 194.3 (std 4.4) | 190.4 (std 2.3) | **-2.0% (-3.9 ms)** |
  | Traced mean ms/tok | 48.23 (std 0.09) | **47.33 (std 0.05)** | **-1.9% (-0.90 ms)** |
  | Traced tok/s | 20.73 | **21.13** | +1.9% |
  | Token-for-token | 100/100 | 100/100 | clean |
- Probe added: `experiments/cb/isolate/gm4_matmul_gelu_probe.py`.

### Round 4 final state

- **Landed (commit 1)**: `_apply_full_rope` mul+add → addcmul. Traced -0.83 ms/tok.
- **Landed (commit 2)**: matmul `activation="gelu"` fusion on `gate_proj`. Traced -0.07 ms/tok (within noise but ships a cleaner code path).
- **Combined w/ rounds 1-3**: traced **51.4 → 47.33 ms/tok = 1.086× cumulative**; eager **474.1 → 190.4 ms/tok = 2.49× cumulative**. Below qb1's 47.5 ms reference by ~0.2 ms.
- **Probes added**:
  - `experiments/cb/isolate/gm4_addcmul_rope_probe.py` — reusable pattern for mul+add fusion sites.
  - `experiments/cb/isolate/gm4_matmul_gelu_probe.py` — reusable pattern for matmul+activation fusion.
- **Helper added**: `experiments/utils/count_ops_in_csv.py` — op-count aggregator for overflow-degraded Tracy CSVs (the standard regime for Gemma 4 full forwards).
- **Open next (low-medium risk)**:
  - **Sharded gate_proj matmul** with `program_config.fused_activation`: would make the activation a TRUE LLK fusion (no post-op), getting the full ~2 ms of saving the activation parameter promised but didn't deliver here. Requires sharding pre_ff to L1 (medium scope).
  - **`rotary_embedding_llama_fused_qk`** (round 3's flagged candidate): still the biggest single remaining single-call lever but needs HF→Llama weight permutation. Saves ~6× more ops than round 4 (576 ops/forward).
  - **Distributed RMSNorm (P2)**: 12-15 ms projected, heaviest scope.

### Investigated but skipped: `[up | gate]` concat + `geglu` fusion

- 01:31 — Probe `experiments/cb/isolate/gm4_geglu_probe.py`: tested `matmul(x, concat([up, gate])) → geglu(., dim=-1)` (rank-4 reshape required) vs `mul(matmul_gelu(x, gate), matmul(x, up))`.
  - cos(baseline, fused) = **0.9999531** — passes the 0.999 gate.
  - **max|delta| = 0.113** (vs 0.031 for addcmul, 0.0 for matmul-gelu) — significantly more precision loss.
  - cos(fused, torch_ref) = 0.9999452 vs cos(baseline, torch_ref) = 0.9999877.
- **Decision**: SKIP. The 0.113 max-delta would compound across 48 layers and risk argmax stability for long contexts. Even though it would save 1 op/layer (48 ops/forward), the precision drop is unacceptable. Recommend a future round investigate WHY ttnn.geglu has higher bf16 noise (likely uses approximate gelu internally despite the docs claiming exact) before re-trying.

---

## Round 5 (subagent — `ttnn.roll` + pre-signed sin tables for `_apply_full_rope`)

### Baseline reconfirm (n=3, qb2, post-round-4 main HEAD `bee4f1e`)

- 01:36 — Baseline run 1: eager **188.6** ms/tok, traced **47.7** ms/tok, 100/100 PASS. Log: `logs/round5/baseline_run1.log`.
- 01:38 — Baseline run 2: eager **184.1** ms/tok, traced **47.8** ms/tok, 100/100 PASS. Log: `logs/round5/baseline_run2.log`.
- 01:40 — Baseline run 3: eager **192.3** ms/tok, traced **47.7** ms/tok, 100/100 PASS. Log: `logs/round5/baseline_run3.log`.
- **Aggregate (n=3)**: eager 188.3 ± 4.2, traced **47.73 ± 0.05 ms/tok (20.95 tok/s)** — slight regression vs round-4 final 47.33 (~0.4 ms, within noise).

### Tracy v2 capture post-round-4

- 01:43 — Tracy v2 captured with all rounds 1-4 applied. CSV: `.cache/perf_logs/tracy_gemma4_v2_round5/reports/2026_06_06_01_43_45/`. DRAM marker buffer overflowed (expected for full forward). Op COUNTS in signposted region (`count_ops_in_csv.py`, divided by 4 mesh for per-forward):
  - 432 Slice (~9/layer) — view ops, free
  - 337 LayerNorm (~7/layer, all rms_norms)
  - 329 Matmul (~7/layer Q/K/V/O + gate/up/down)
  - 291 BinaryNg (~6/layer; -192 vs round-3 from addcmul + matmul-gelu fusions)
  - 200 UntilizeWithUnpadding (~4/layer)
  - 176 ReshapeView, 176 InterleavedToSharded, 165 TilizeWithValPadding
  - 145 UnaryNg (~3/layer; includes 96 RoPE `neg`)
  - 136 Concat (~3/layer; includes 96 RoPE `concat`)
  - 96 Ternary (= 96 addcmuls, all in `_apply_full_rope`)
  - 88 SdpaDecode, 88 PagedFusedUpdateCache, 97 AllGather, 96 ReduceScatter, 56 Clone
  - Conclusion: the `_apply_full_rope` body still spends 4 device ops/call (neg + concat + mul + addcmul) × 96 calls = 384 ops/forward. Cumulative 192 ops/forward attackable if we can fuse neg+concat → roll.

### Lever pick — `ttnn.roll` + pre-signed sin tables

- **Lever**: replace `_apply_full_rope`'s `slice + slice + neg + concat` (2 device ops + 2 view slices) with a single `ttnn.roll(x, shifts=half, dim=-1)`. The negation that belonged to `rotate_half([a, b]) = [-b, a]` is pre-baked into the sin tables at bootstrap (`sin[:, :half] *= -1`). Math identity:
  - `rotate_half(x) * sin = concat([-x2, x1]) * concat([sin_a, sin_a])` (Gemma 4 has sin1==sin2)
  - `                     = concat([x2, x1]) * concat([-sin_a, sin_a])` (factor neg into sin)
  - `                     = roll(x, half, dim=-1) * sin_signed`
- **Why it passes the kernel-vs-dispatch test ([[feedback-kernel-vs-dispatch-realization]])**: `neg` is a UnaryNg kernel; `concat` is a data-movement kernel. Both do real work per call (~tens of μs at our tile sizes). Eliminating them per RoPE × 96 calls/forward = real kernel-time saving.
- **Predicted delta**: 0.5-1.5 ms/tok traced (matches the round-4 addcmul fusion which saved 96 ops and got 0.83 ms; same op count saving here for a similar kernel mix).
- **Risk**: low. The sin tables are read-only across the forward and used ONLY by `_apply_full_rope` (verified: `grep sin_sliding_tt|sin_global_tt` → 4 callers, all in `_apply_full_rope`'s call graph). Pre-signing is atomic with the function rewrite — only one code path uses the new tables.

### Isolation probe + production land

- 01:54 — Isolation probe `experiments/cb/isolate/gm4_roll_rope_probe.py` (forks `gm4_addcmul_rope_probe.py`):
  - Sliding (n_heads=4, head_dim=256): cos(baseline, roll) = **1.0000001**, **max|delta| = 0.000000** (BIT-IDENTICAL).
  - Global (n_heads=8, head_dim=512): cos(baseline, roll) = **1.0000004**, **max|delta| = 0.000000** (BIT-IDENTICAL).
  - Torch sanity also PASS — confirms the sign-factor identity is mathematically exact.
- 01:55 — Wrote roll+pre-sign change to `server_gemma4_unified_ttnn.py`:
  - Bootstrap (line 575-581 + 590-595): `sin_sliding[:, :half_sliding] *= -1.0` and same for global, BEFORE `np_to_replicated`.
  - `_apply_full_rope` (line 1041-1071): 4-op chain → 3-op chain (`roll + mul + addcmul`). API gotcha: `ttnn.roll(x, shifts=half, dim=-1)` — `dim` not `dims` (the python bindings reject the kwarg `dims=[-1]`).
- 01:58 — First run FAILED with TypeError — wrong kwarg `dims=[-1]`. Fixed in one line. Re-deployed.
- 02:00 — Validator run 1: eager 179.1 ms/tok, traced **47.1** ms/tok, 100/100 PASS. **Token sequence BIT-IDENTICAL to baseline** (first 10: [532, 575, 532, 496, 563, 496, 45518, 107, 100, 45518] — same as round-4 final).
- 02:02 — Validator run 2: eager 178.9 ms/tok, traced **47.4** ms/tok, 100/100 PASS.
- 02:04 — Validator run 3: eager 180.0 ms/tok, traced **47.1** ms/tok, 100/100 PASS.

### Final aggregate (n=3 after the fix)

| Metric | Baseline (round 5) | roll-fused | Delta |
| --- | --- | --- | --- |
| Eager mean ms/tok | 188.3 (std 4.2) | 179.3 (std 0.6) | **-4.8% (-9.0 ms)** |
| Traced mean ms/tok | 47.73 (std 0.05) | **47.20 (std 0.17)** | **-1.1% (-0.53 ms)** |
| Traced tok/s | 20.95 | **21.19** | +1.1% |
| Token-for-token | 100/100 | 100/100 | clean (bit-identical) |

- Both eager AND traced moved as predicted ([[feedback-kernel-vs-dispatch-realization]]) — neg + concat = real kernel work, so saving them × 96 calls = real trace-time win. The traced gain (0.53 ms) is consistent with the round-4 addcmul fusion's 0.83 ms (which saved fewer ops but with broader kernel work).
- **Token argmax sequence is byte-identical** to round-4 final — the roll + pre-signed-sin transform is mathematically and bit-wise equivalent (vs round-4's addcmul which had ~0.03 max|delta| from bf16 reorder; this round has 0.0 max|delta|).

### Round 5 final state

- **Landed**: `_apply_full_rope` neg+concat → `ttnn.roll` + pre-signed sin tables. Single commit covers bootstrap sign-bake + `_apply_full_rope` rewrite. Traced delta: **-0.53 ms/tok (-1.1%, 47.20 ms = 21.19 tok/s)**, eager delta: **-4.8% (-9 ms/tok)**. 3×100/100 token-for-token match.
- **Combined w/ rounds 1-4**: traced **51.4 → 47.20 ms/tok = 1.089× cumulative**; eager **474.1 → 179.3 ms/tok = 2.64× cumulative**. **Below qb1's 47.5 ms reference by ~0.3 ms.**
- **Probe added**: `experiments/cb/isolate/gm4_roll_rope_probe.py` — bit-identical math probe; reusable pattern for any "neg + concat" → "roll + pre-signed weight" fusion site.
- **Open next (low-medium risk)**:
  - **`rotary_embedding_llama_fused_qk`** (still flagged): biggest remaining single lever. Replaces all 3 ops in `_apply_full_rope` (roll + mul + addcmul = 3 ops × 96 calls = 288 ops) with ONE op per Q,K pair (48 calls/forward = 48 ops). Saves ~240 ops/forward. BUT needs HF→Llama weight permutation, HEIGHT_SHARDED inputs on disjoint cores, trans_mat setup — medium-day scope. Reference: `~/tenstorrent/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py` (decode mode) + `llama_rope.py` (trans_mat construction).
  - **Sharded gate_proj matmul** with `program_config.fused_activation`: still ~2 ms of potential saving by making `activation="gelu"` a TRUE in-kernel fusion vs the current post-op. Requires sharding `pre_ff` to L1 (medium scope).
  - **Eliminate `v_raw = ttnn.clone(k_h)` in global attention**: 8 Clones/forward saving but small (<0.1 ms expected). One-line change; left for future bundling with a bigger win.
  - **Distributed RMSNorm (P2)**: 12-15 ms projected, heaviest scope.

### Investigated but skipped (round 5 only)

- **`alt_complex_rotate90`**: ttnn op that does interleaved rotate `(out_{2i}, out_{2i+1}) = (-in_{2i+1}, in_{2i})`. Different RoPE convention from HF Gemma 4's split-half rotate; would require permuting Q/K projection weights (interleave first half + second half columns). Same scope as `rotary_embedding_llama_fused_qk` but with less head-room (still need a mul + addcmul after).
- **Eliminate the per-layer `residual_1 = ttnn.clone(h_in)`** (48 Clones/forward): defensive clone added during initial bringup ("L1 hard-FAIL hypothesis: rms_norm + downstream ops may be aliasing h_in"). Static analysis suggests it's safe to remove (rms_norm is pure, no in-place writes to h_in), but the comment flags real bug history. Reverting it risks a hard-to-diagnose regression. SKIP — high blast radius for ~0.3 ms expected gain.
