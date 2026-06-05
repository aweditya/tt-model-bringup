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

