# Repo archive audit — 2026-06-04

**Read-only proposal.** This document lists files in the working tree
that are no longer referenced by the active code/research path, and
proposes target locations under `archive/legacy/` for each. **No files
were moved, deleted, or edited as part of this audit.** The user (or a
follow-up commit) decides whether to act on any proposal.

The motivating event: after Stanford CS440LX demo (2026-06-04) and the
pivot to Nemotron-3 Nano 30B-A3B (MM7), several earlier bringups (27B
single-chip, 35B-A3B drift work pre-fix, Gemma 4 v0.x correctness
ladder) left behind staged probes, validators, and plans that are no
longer on the live path but still complicate `grep` / `Read` /
codebase-wide reasoning.

---

## TL;DR

- **62 candidate files** flagged for archive (of ~300 audited under
  `experiments/{serve,cb,utils,owned_ops,*.py}`, `research/`, and
  `scripts/`). Total ~540 KB.
- **5 thematic buckets**:
  1. **Pre-CB single-process server stack** (10 files) — `server.py`,
     `server_35b.py`, the three `client*.py` flavours, three pre-CB
     shell wrappers, `import_smoke.py`, and the matching protocol
     mock test. Live stack is `server_*_cb.py` + `serve_cb.sh`.
  2. **Top-level `experiments/` one-off probes** (10 files) — fusion
     benches and MoE/QK isolation tests from the 27B/35B perf hunt
     that have been collapsed into `server_tp.py` / `server_35b_ttnn.py`
     as inline comments only.
  3. **35B-A3B drift sub-probes superseded by 2026-06-04 resolution**
     (5 files in `experiments/cb/dev/cb35_drift_*.py` excl. the
     `cliff_search` + `ladder` core) — drift cliff GONE per
     `feedback_35b_drift_resolved_2026-06-04.md`; the fp32-H /
     bf16-manual variants were dead ends.
  4. **CB engine + chunked-prefill scaffolding** (10 files in
     `experiments/cb/validate/engine*.py`, `sampling.py`,
     `cb_alternating_scheduler.py`, `long_context.py`,
     `prefill_transplant.py`, `engine_chunked_prefill.py`, plus
     `experiments/cb/serving_demo.py`) — validators for CB1–CB4
     milestones that landed and were promoted into
     `cb_engine.py`/`cb_scheduler.py`. Plan refs all point to
     `research/production_server_plan.md` and `27b_s2_chunked_prefill_milestones.md`,
     both archive-candidates themselves.
  5. **Superseded `research/` plans + 27B/35B early-stage docs**
     (27 files) — bringup-research write-ups, two clean-up plans
     (`repo-cleanup-plan.md`, `maintainability_pass.md`,
     `moe-cleanup-plan.md`), the early CB plan family, kernel
     scoping docs, and the original Qwen3.6 arch notes that were
     folded into the active 35B/Gemma 4/Nemotron plans.

**Files that LOOK orphan but are LIVE-imported (DO NOT ARCHIVE)**:
`experiments/serve/ondevice_27b.py` and `experiments/serve/generate_27b.py`.
Both are imported by `experiments/serve/server_tp.py` (lines 724-810,
search `_91f`/`_91l` aliases). They are the canonical TP weight-loader
+ embed/lm-head loader and the only reason `server_tp.py` boots. They
are NOT in CLAUDE.md's exclusion list; flagging here for the user.

---

## Methodology

For each file in scope, I ran three checks:

1. **Last-modified date** (`git log -1 --format='%ad' --date=short -- <file>`).
   Files unmodified for ≥7 days are *eligible*; for >3 weeks they
   become *candidates* unless a "reuse-from" reference is found.
2. **Active references** (`grep -rl '<basename>' --include='*.py'
   --include='*.md' --include='*.sh' .`, excluding
   `__pycache__/`, `experiments/.refs/` (vendored tt-metal), `archive/`,
   and `.claude/worktrees/`). A file is an orphan candidate if every
   surviving reference is either inside the file itself, inside an
   already-archived doc (`research/archive/`,
   `experiments/utils/archive/`), or inside another archive candidate.
3. **HANDOFF.md + active-plan citation check.** Files explicitly
   pointed at by `HANDOFF.md` (the post-compaction one-pager) or by
   the active plans the user enumerated were unconditionally kept.

### Hard exclusions actually applied

Per the user's brief, the following are NEVER proposed for archive
even if they look orphan:

- `HANDOFF.md` and everything it references by path
- `research/nemotron3_nano_30b_a3b_bringup_plan.md` and
  `research/nemotron3_nano_architecture_brief.md`
- `research/model_bringup_recipe.md`
- `research/mm7_g1_mamba2_kernel_design.md`
- `research/audit_gdn_kernel_us_vs_tt_metal.md`,
  `research/audit_gemma4_opts_us_vs_tt_metal_44962.md`,
  `research/audit_qwen36_us_vs_qwen9b_p150_branch.md`
- `wiki/` (entire dir — pedagogical reference)
- The active CB stack in `experiments/serve/`:
  `server_tp.py`, `server_tp_cb.py`, `server_35b_ttnn.py`,
  `server_35b_cb.py`, `server_gemma4_unified_ttnn.py`,
  `server_gemma4_unified_cb.py`, `cb_engine.py`, `cb_scheduler.py`,
  `cb_api.py`, `openai_endpoint.py`, `live_slot_store.py`,
  `protocol.py`, `cb_metrics.py`, `experiments/serve/scripts/` (the
  active server-launch dir)
- The G0 numpy oracle + G0a harness for Nemotron Mamba2:
  `experiments/utils/mamba2_numpy_oracle.py`,
  `experiments/utils/test_mamba2_decode_isolated.py`
- All 9 `experiments/owned_ops/` kernels (in production or pending
  production — particularly the new `nemotron3_mamba2_decode_owned/`
  fork). Confirmed each is referenced by at least one of:
  `research/35b_moe_ffn_kernel_build_plan.md`,
  `research/kernel_dataflow_representation.md`,
  `research/nemotron3_nano_30b_a3b_bringup_plan.md`,
  `research/mm7_g1_mamba2_kernel_design.md`
- `archive/presentation_cs440lx_2026-06-04/` (poster + measurements; archived 2026-06-04 after demo)

### "Reuse-from" pattern: kept anything cited by an active plan

The 27B/35B bringup left a deep utility shelf at `experiments/utils/`
that the Gemma 4 and Nemotron plans explicitly cite as fork-from
templates. Each of these was found cited in
`research/gemma4_12b_bringup_plan.md` or
`research/nemotron3_nano_30b_a3b_bringup_plan.md` and is **kept**:

- `experiments/utils/cosine_ladder_35b.py`,
  `cosine_ladder_hf_ref.py`, `cosine_ladder_aggregate.py`
- `experiments/utils/test_fused_swiglu_isolated.py`,
  `test_fused_binary_activations_isolated.py`
- `experiments/utils/test_pattern_a_moe_np.py`,
  `test_pattern_a_moe_tt.py`,
  `test_batched_expert_matmul_isolated.py`
- `experiments/utils/test_async_all_reduce_overlap_isolated.py`
- `experiments/utils/gdn_kernel_oracle.py`,
  `moe_ffn_kernel_oracle.py`
- `experiments/utils/p22_vocab_sharded_lm_head_probe.py`
- `experiments/utils/p150_memory_bandwidth_probe.py` (cited in
  perf memos)
- `experiments/utils/full_layer_tp_probe.py`,
  `tp_attn_traced_probe.py` (Gemma 4 plan §"REUSE MANDATE")
- `experiments/utils/paged_vs_nonpaged_sdpa_latency.py` (Gemma 4
  sliding-SDPA perf reference)
- `experiments/utils/needle_haystack_35b_ttnn.py` (cited by
  `model_bringup_recipe.md`)
- `experiments/utils/needle_haystack_35b_hf.py` (companion of
  the above)
- `experiments/utils/tracy_profile_traced_decode.py`,
  `run_tracy_probe.sh`, `tracy_profile_one_dn.py`,
  `tracy_profile_one_moe.py`, `tracy_analyze_ops.py`,
  `tracy_profile_one_gemma4_layer.py`, `_patch_tracy_assertion.py`,
  `analyze_ops_perf_results.py`, `bench_dn_total.py`,
  `profile_35b_ttnn.py`, `profile_blocks_35b_ttnn.py`,
  `profile_dn_sections.py`, `delete_line_range.py`,
  `gemma4_gelu_variant_probe.py`, `trace_demo_full_step.py`,
  `test_owned_gdn_greedy_generation.py`,
  `p21_fp32_sdpa_cliff_probe.py` (drift fix story),
  `needle_haystack_probe.py`, `needle_haystack_qb2_tp.py`
- `experiments/utils/hf_reference_35b.py`,
  `hf_reference_gemma4_12b.py`, `ttnn_introspect.py`,
  `npz_inspect.py`, `hf_download.py`, `syntax_check.py`,
  `run_with_tracy_build.sh`, `README.md`

All 35B-A3B bringup gates (`experiments/cb/validate/cb35_v0_*.py`,
`cb35_v1_*.py`, `cb35_v2_trace.py`, `cb35_v1_probe.py`) are
**kept** — they are cited as current acceptance gates in
`research/35b_cb_bringup_plan.md` and `code_cleanup_plan_2026-06-04.md`
even though 35B perf/drift is currently de-prioritised.

All Gemma 4 v0.x probes (`gm4_v0*.py`, `gm4_v1*.py`,
`gm4_v2_wireup_smoke.py`, `gm4_sliding_write_read.py`,
`gm4_global_write_read.py`, `gm4_rope_lookup.py`,
`gm4_v033c_needle_haystack.py`, `gm4_per_layer_drift_pos1.py`) are
**kept** — cited by `research/gemma4_12b_bringup_plan.md` and HANDOFF
as the staged bringup evidence ladder.

`experiments/cb/dev/cb35_dev_harness.py`, `cb35_drift_ladder.py`,
`cb35_drift_cliff_search.py`, `gm4_dev_harness.py` are **kept** —
they are the persistent dev harnesses, cited by
`research/35b_drift_next_session_plan.md` and HANDOFF.

`experiments/cb/isolate/paged_sdpa.py`,
`paged_update_cache.py`, `chunked_sdpa.py` are **kept** —
explicitly cited as fork-templates in Gemma 4 + Nemotron plans.

---

## Per-bucket candidates

Format: `<path> | <last_mod> | <one-line reason>`. Sorted oldest first.

### Bucket 1 — Pre-CB single-process server stack (10 files)

The active server stack is `server_*_cb.py` driven by
`scripts/serve_cb.sh`. The files below predate continuous batching
and have no live importer outside themselves and other archive
candidates.

```
experiments/serve/scripts/run_drift_seed_sweep.sh         | 2026-05-14 | drives `client.py generate_long` for the bf16 drift hunt (resolved by B3 SDPA, 2026-05); see `feedback_p21_fp32_sdpa_cliff_probe.md`
experiments/serve/scripts/run_drift_sweep.sh              | 2026-05-14 | as above; companion sweep
experiments/serve/scripts/run_drift_dry_tune.sh           | 2026-05-14 | as above; tunes DRY penalty for drift
experiments/serve/scripts/run_chat_vs_raw.sh              | 2026-05-14 | calls `experiments.serve.client generate_long`; pre-CB chat-template smoke
experiments/serve/scripts/run_chat_quick.sh               | 2026-05-14 | as above; quick variant
experiments/serve/scripts/serve_35b.sh                    | 2026-05-21 | launches `experiments.serve.server_35b` (pre-CB 35B server). Active 35B is `server_35b_cb.py` via `serve_cb.sh`.
experiments/serve/scripts/serve_tp.sh                     | 2026-05-30 | launches `experiments.serve.server_tp` as standalone; CB stack now wraps via `serve_cb.sh`. NOTE: still references `client_tp.py` for shutdown — pair-archive.
experiments/serve/scripts/serve.sh                        | 2026-05-30 | single-chip `server.py` wrapper; superseded by `serve_cb.sh`
experiments/serve/client.py                               | 2026-05-28 | Unix-socket client for pre-CB `server.py`; protocol superseded by OpenAI-HTTP
experiments/serve/client_tp.py                            | 2026-05-28 | TP variant of `client.py`; only used by `serve_tp.sh` (also a candidate)
experiments/serve/client_35b.py                           | 2026-05-28 | 35B variant of `client.py`; only used by `serve_35b.sh` (also a candidate)
experiments/serve/server.py                               | 2026-05-28 | 92KB single-chip socket server; only callers are `demo_qwen36_27b.py` (also a candidate) and the protocol-mock test
experiments/serve/server_35b.py                           | 2026-05-28 | 20KB single-chip 35B server; only caller is `serve_35b.sh` (also a candidate)
experiments/serve/import_smoke.py                         | 2026-05-28 | sanity test that `ondevice_27b` + `generate_27b` import cleanly. Useful but lives in old path. Confirmed by grep no other ref.
experiments/serve/tests/test_protocol_mock.py             | 2026-05-28 | spawns `server.py --mock` over a socket. Pair-archive with server.py. The `protocol.py` it covers is in the keep list, but this test exercises the old wire format.
```

**Total**: 14 files, ~290 KB (dominated by 92KB `server.py` + 44KB
`client_tp.py` + 44KB `ondevice_27b.py`'s pre-CB twin).

**WARNING**: `experiments/serve/ondevice_27b.py` and
`experiments/serve/generate_27b.py` look like peers but ARE LIVE
IMPORTED by `server_tp.py` (`_91f`/`_91l` aliases). Do not archive.

### Bucket 2 — Top-level `experiments/` one-off probes (10 files)

Live `server_tp.py` / `server_35b_ttnn.py` only cite these files in
comments (`# bench (experiments/bench_dn_in_proj_fusion.py)`); no
import. Findings are baked into the production code paths.

```
experiments/test_qk_l2_norm_fusion.py                     | 2026-05-27 | one-off fusion isolation bench; result landed in `server_35b_ttnn.py` as a comment
experiments/test_moe_router_topk_reorder.py               | 2026-05-27 | router top-k reorder bench; finding integrated
experiments/test_moe_gate_up_core_grid.py                 | 2026-05-27 | core-grid sweep for MoE gate-up; finding integrated as comment
experiments/test_moe_bf8_weights_correctness.py           | 2026-05-27 | bf8 weights pcc isolation; result in `server_35b_ttnn.py` comment
experiments/needle_haystack_35b_ttnn_inproc.py            | 2026-05-27 | superseded by `experiments/cb/isolate/cb35_needle_haystack.py` (which forks it)
experiments/bench_dn_in_proj_fusion.py                    | 2026-05-28 | DN in-proj fusion bench; result baked into `server_35b_ttnn.py`
experiments/bench_step_forward_traced.py                  | 2026-05-28 | step-traced bench; superseded by `experiments/cb/bench/trace.py` (active)
experiments/demo_qwen36_27b.py                            | 2026-05-28 | only-caller of the old `server.py` + `client.py`; pair-archive with Bucket 1
experiments/needle_haystack_35b_dry_isolation.py          | 2026-05-28 | DRY-penalty needle isolation; pre-CB
experiments/test_lm_head_core_grid.py                     | 2026-05-28 | lm-head core-grid bench; result integrated
experiments/test_moe_gate_up_h_in_l1.py                   | 2026-05-28 | gate-up "H in L1" bench; result integrated
```

**Total**: 11 files, ~104 KB.

### Bucket 3 — 35B-A3B drift sub-probes superseded by 2026-06-04 resolution (5 files)

The 35B drift cliff is **RESOLVED** in current state per
`feedback_35b_drift_resolved_2026-06-04.md` (memory index entry).
Tasks #163/#170 closed; cause likely a TT firmware/build update.
The wrapper variants below were dead-end hypotheses (fp32 H_t,
manual recurrence) that the resolution invalidated — kept the core
`cb35_drift_cliff_search.py` + `cb35_drift_ladder.py` per
HANDOFF/plan refs, but the thin wrappers can go.

```
experiments/cb/dev/cb35_drift_bf16.py                     | 2026-06-03 | 13-line wrapper around the ladder; bf16 baseline now invariant
experiments/cb/dev/cb35_drift_fp32_h.py                   | 2026-06-03 | fp32-H_t wrapper; user-memory `feedback_35b_manual_recurrence_path_broken.md` invalidates this branch
experiments/cb/dev/cb35_drift_fp32_h_no_dg.py             | 2026-06-03 | fp32-H_t + no-decay-gate wrapper; same invalidation
experiments/cb/dev/cb35_drift_long_bf16.py                | 2026-06-03 | long-context wrapper of the above
experiments/cb/dev/cb35_drift_long_bf16_manual.py         | 2026-06-03 | manual-recurrence variant (broken path per memory)
experiments/cb/dev/cb35_drift_long_fp32_h.py              | 2026-06-03 | long fp32-H_t variant
experiments/cb/dev/cb35_drift_long_fp32_h_no_dg.py        | 2026-06-03 | long fp32-H_t + no-decay-gate variant
```

**Total**: 7 files, ~28 KB.

**AMBIGUOUS — keep**: `cb35_drift_cliff_search.py`,
`cb35_drift_ladder.py`. Both are referenced by name in
`research/35b_drift_next_session_plan.md` and the HANDOFF "old
headline". If 35B drift work is fully retired, those plans should
also move (see Bucket 5).

### Bucket 4 — CB engine + chunked-prefill scaffolding (10 files)

Validators for CB1–CB4 milestones that landed and were promoted into
`cb_engine.py`/`cb_scheduler.py`. Only inbound references are
themselves archive-candidates (`research/production_server_plan.md`,
`27b_s2_chunked_prefill_milestones.md`,
`27b_chunked_prefill_plan.md`) or
`research/code_maintainability_audit.md` which calls them out by
name as cleanup targets.

```
experiments/cb/validate/sampling.py                       | 2026-05-29 | CB sampling validator; landed in `cb_engine._step_sampled*`
experiments/cb/validate/engine_api.py                     | 2026-05-30 | CB1 engine-API validator; landed in `cb_api.py` (active)
experiments/cb/validate/engine_sampling.py                | 2026-05-30 | CB engine sampling gate; landed
experiments/cb/validate/engine.py                         | 2026-05-30 | CB1 base engine validator; landed
experiments/cb/validate/long_context.py                   | 2026-05-30 | 137-tok needle gate for chunked-prefill milestone S1
experiments/cb/serving_demo.py                            | 2026-05-30 | early concurrency prototype; superseded by `cb_engine.py` + `concurrent_chat.py`
experiments/cb/validate/cb_alternating_scheduler.py       | 2026-05-31 | early alternating-slot scheduler; superseded by Orca-style `cb_scheduler.py`
experiments/cb/validate/engine_chunked_prefill.py         | 2026-05-31 | chunked-prefill S2 gate; milestone landed
experiments/cb/validate/prefill_transplant.py             | 2026-05-31 | prefill-transplant S2 gate; milestone landed
experiments/cb/profile/blocks.py                          | 2026-05-28 | per-block attribution profile; referenced only via `server_tp_cb.py` comment
experiments/cb/profile/dn.py                              | 2026-05-28 | DN-only profile harness; refs only in code comments
experiments/cb/profile/dn_matmul.py                       | 2026-05-28 | DN-matmul micro-profile; pre-fix
experiments/cb/profile/floor.py                           | 2026-05-28 | floor-perf profile; pre-trace-A/B methodology
experiments/cb/isolate/conv_reform.py                     | 2026-05-28 | conv1d reform isolation; landed in `qwen36_conv1d_decode_owned`
experiments/cb/isolate/dn_recurrence.py                   | 2026-05-28 | DN recurrence isolation; landed in `qwen36_gdn_decode_owned`
experiments/cb/isolate/owned_gdn.py                       | 2026-05-28 | owned-GDN isolation; landed
```

**Total**: 16 files, ~104 KB.

**AMBIGUOUS — keep**: `experiments/cb/validate/cb35_v0_smoke.py` +
the `cb35_v1_*.py` family + `cb35_v2_trace.py` are still cited as
acceptance gates for the 35B CB stack.

### Bucket 5 — Superseded `research/` plans + early-stage docs (27 files)

Three sub-themes: (a) early bringup research that has been folded
into active plans; (b) clean-up plans that have themselves been
superseded by `code_cleanup_plan_2026-06-04.md`; (c) intermediate
27B CB + chunked-prefill plans whose milestones landed.

```
research/qwen36_arch_notes.md                             | 2026-05-11 | original Qwen3.6 arch notes; folded into 35B/27B plans + audits
research/qwen36_modeling_excerpts.md                      | 2026-05-11 | HF modeling excerpts; reference data superseded by audit_qwen36_*
research/qwen36_30b_a3b_bringup_research.md               | 2026-05-19 | early 30B-A3B research write-up; the live target is now 35B (and Nemotron 30B)
research/all_reduce_dataflow_teaching.md                  | 2026-05-20 | all-reduce teaching writeup; landed as wiki/kernel_research material
research/all_reduce_kernel_audit.md                       | 2026-05-20 | all-reduce kernel audit; finding landed in `server_tp.py` num_links=2 default
research/qwen36_35b_a3b_config_audit_2026_05_21.md        | 2026-05-21 | early 35B config audit; superseded by `audit_qwen36_us_vs_qwen9b_p150_branch.md` (active)
research/qwen36_35b_a3b_incremental_block_plan_2026_05_21.md | 2026-05-21 | incremental-block plan; complete (35B is up and running)
research/35b_a3b_correctness_plan.md                      | 2026-05-24 | correctness plan; resolved per `feedback_35b_drift_resolved_2026-06-04.md`
research/35b_moe_pattern_a_plan.md                        | 2026-05-24 | MoE Pattern A plan; shipped (cited in moe-cleanup-plan as DONE)
research/35b_tt_perf_report_findings.md                   | 2026-05-25 | KEEP — HANDOFF cites as `[5]`. Not a candidate.
research/profiling-cheatsheet.md                          | 2026-05-25 | Tracy cheatsheet; merged into `profiling-quick-reference.md` (also a candidate) and gemma4 perf briefing
research/profiling-quick-reference.md                     | 2026-05-25 | Tracy quick-ref; cited by gemma4 perf briefing only; safe to subordinate (move both with a redirect-stub)
research/35b_moe_ffn_kernel_build_plan.md                 | 2026-05-26 | MoE FFN kernel build plan; kernel landed (`qwen36_moe_ffn_decode_owned`)
research/35b_moe_ffn_kernel_perf_deferrals.md             | 2026-05-26 | KEEP — cited as reuse-from in `kernel_dataflow_representation.md` (live) and Nemotron plan. NOT a candidate.
research/35b_moe_ffn_kernel_scoping.md                    | 2026-05-26 | KEEP — same reason. NOT a candidate.
research/35b_perf_milestones.md                           | 2026-05-26 | KEEP — HANDOFF cites as `[3]`. NOT a candidate.
research/35b_perf_workflow_log.md                         | 2026-05-27 | workflow log; perf sprint complete
research/27b_continuous_batching_plan.md                  | 2026-05-27 | early CB plan; landed (CB1–CB4 done per memory)
research/27b_cb_scope.md                                  | 2026-05-28 | CB1 scope doc; KEEP — explicitly cited by HANDOFF
research/kernel_dataflow_representation.md                | 2026-05-28 | KEEP — referenced by 27b_cb_scope.md and live plans
research/kernel_design_worksheet.md                       | 2026-05-28 | one-off kernel design worksheet; lessons folded into `mm7_g1_mamba2_kernel_design.md`
research/maintainability_pass.md                          | 2026-05-28 | M1–M6 cleanup plan; DONE (memory index: "M1–M6 DONE 2026-05-28")
research/repo-cleanup-plan.md                             | 2026-05-28 | senior-engineer cleanup plan; superseded by `code_cleanup_plan_2026-06-04.md`
research/27b_chunked_prefill_plan.md                      | 2026-05-29 | chunked-prefill design; milestones landed
research/27b_chunked_prefill_prior_art.md                 | 2026-05-30 | prior-art notes for the above; same fate
research/cb_device_sampling_plan.md                       | 2026-05-30 | device-sampling plan; W2 (topk_k) shipped per recent commit `bef03ba`
research/code_maintainability_audit.md                    | 2026-05-30 | the 600-line maintainability audit; superseded by `code_cleanup_plan_2026-06-04.md`
research/moe-cleanup-plan.md                              | 2026-05-30 | MoE cleanup plan; superseded
research/production_server_plan.md                        | 2026-05-30 | production server plan; superseded by CB stack landing
research/public_release_plan.md                           | 2026-05-30 | public-release plan; demo shipped 2026-06-04
research/qb_hosts_cleanup_audit.md                        | 2026-05-30 | qb host cleanup; one-shot done
research/qwen36_35b_a3b_implementation_plan.md            | 2026-05-30 | 35B implementation plan; complete
research/27b_s2_chunked_prefill_milestones.md             | 2026-05-31 | chunked-prefill S2 milestones; landed
```

(Removed from this list and kept: `35b_tt_perf_report_findings.md`,
`35b_perf_milestones.md`, `35b_moe_ffn_kernel_perf_deferrals.md`,
`35b_moe_ffn_kernel_scoping.md`, `27b_cb_scope.md`,
`kernel_dataflow_representation.md`.)

**Candidates total**: 27 files, ~315 KB.

---

## Proposed archive layout

Match the existing `archive/legacy/pjrt_plugin/` convention — flat,
descriptive subfolders, one per thematic bucket, preserving the
in-repo path under each:

```
archive/
  legacy/
    pjrt_plugin/                            # already exists (PJRT bringup)
    tt_jax/                                 # already exists
    qwen05b_bisect/                         # already exists (JAX dump scripts)
    pre_cb_server/                          # NEW (Bucket 1, 14 files)
      experiments/serve/server.py
      experiments/serve/server_35b.py
      experiments/serve/client.py
      experiments/serve/client_tp.py
      experiments/serve/client_35b.py
      experiments/serve/import_smoke.py
      experiments/serve/tests/test_protocol_mock.py
      experiments/serve/scripts/serve.sh
      experiments/serve/scripts/serve_tp.sh
      experiments/serve/scripts/serve_35b.sh
      experiments/serve/scripts/run_chat_quick.sh
      experiments/serve/scripts/run_chat_vs_raw.sh
      experiments/serve/scripts/run_drift_seed_sweep.sh
      experiments/serve/scripts/run_drift_sweep.sh
      experiments/serve/scripts/run_drift_dry_tune.sh
    top_level_oneoff_probes/                # NEW (Bucket 2, 11 files)
      experiments/demo_qwen36_27b.py
      experiments/bench_dn_in_proj_fusion.py
      experiments/bench_step_forward_traced.py
      experiments/test_qk_l2_norm_fusion.py
      experiments/test_lm_head_core_grid.py
      experiments/test_moe_*.py            (4 files)
      experiments/needle_haystack_35b_*.py (2 files)
    cb35_drift_wrappers/                    # NEW (Bucket 3, 7 files)
      experiments/cb/dev/cb35_drift_bf16.py
      experiments/cb/dev/cb35_drift_fp32_h*.py    (2 files)
      experiments/cb/dev/cb35_drift_long_*.py     (4 files)
    cb_engine_scaffolding/                  # NEW (Bucket 4, 16 files)
      experiments/cb/serving_demo.py
      experiments/cb/validate/engine*.py          (4 files)
      experiments/cb/validate/sampling.py
      experiments/cb/validate/cb_alternating_scheduler.py
      experiments/cb/validate/long_context.py
      experiments/cb/validate/prefill_transplant.py
      experiments/cb/profile/blocks.py
      experiments/cb/profile/dn.py
      experiments/cb/profile/dn_matmul.py
      experiments/cb/profile/floor.py
      experiments/cb/isolate/conv_reform.py
      experiments/cb/isolate/dn_recurrence.py
      experiments/cb/isolate/owned_gdn.py
    research_pre_demo_plans/                # NEW (Bucket 5, 27 files)
      research/qwen36_arch_notes.md
      research/qwen36_modeling_excerpts.md
      research/qwen36_30b_a3b_bringup_research.md
      research/qwen36_35b_a3b_config_audit_2026_05_21.md
      research/qwen36_35b_a3b_incremental_block_plan_2026_05_21.md
      research/qwen36_35b_a3b_implementation_plan.md
      research/35b_a3b_correctness_plan.md
      research/35b_moe_pattern_a_plan.md
      research/35b_moe_ffn_kernel_build_plan.md
      research/35b_perf_workflow_log.md
      research/27b_continuous_batching_plan.md
      research/27b_chunked_prefill_plan.md
      research/27b_chunked_prefill_prior_art.md
      research/27b_s2_chunked_prefill_milestones.md
      research/cb_device_sampling_plan.md
      research/all_reduce_dataflow_teaching.md
      research/all_reduce_kernel_audit.md
      research/kernel_design_worksheet.md
      research/maintainability_pass.md
      research/repo-cleanup-plan.md
      research/code_maintainability_audit.md
      research/moe-cleanup-plan.md
      research/production_server_plan.md
      research/public_release_plan.md
      research/qb_hosts_cleanup_audit.md
      research/profiling-cheatsheet.md
      research/profiling-quick-reference.md
```

The existing `research/archive/` directory (122 files, 2026-04 →
2026-05-30) already follows this pattern; we are extending it with
the post-2026-05-30 wave. Recommend either:

- **Option A** (consistent with the existing `archive/legacy/`
  structure): move under `archive/legacy/<bucket>/` as listed
  above. Preserves the "everything we won't touch sits under
  archive/legacy/" invariant from CLAUDE.md.
- **Option B**: collapse Bucket 5 into the existing
  `research/archive/` directory (already established convention)
  and use `archive/legacy/<bucket>/` only for the code buckets.
  Pros: less disruption to in-`research/` cross-citations
  (relative-path links survive). Cons: two archive roots.

Recommend **Option B** — Bucket 5 → `research/archive/` (extend the
existing pattern); Buckets 1–4 → `archive/legacy/<bucket>/`.

---

## Safety net (verify BEFORE archiving)

Run each of these checks just before any move. **None require
device time**; they are pure repo introspection except where noted.

1. **Re-grep the 62 candidates for *new* recent references.**
   Between this audit and a move, a new commit might wire one of
   these up. Strict per-file check:
   ```
   for f in <candidate paths>; do
     hits=$(grep -rl "$(basename "$f" .py)" \
       --include='*.py' --include='*.md' --include='*.sh' \
       . 2>/dev/null \
       | grep -v __pycache__ | grep -v archive/ | grep -v "$f")
     [ -z "$hits" ] || echo "STILL REFERENCED: $f -> $hits"
   done
   ```
2. **Smoke each live server backend before AND after** so any
   import-graph breakage is caught:
   ```
   ssh qb1 'cd ~/tt-xla && TT_BACKEND=27b HF_HUB_OFFLINE=1 \
       bash experiments/serve/scripts/serve_cb.sh start'
   # wait for bootstrap, then 2-3 chat completions
   ssh qb1 'bash experiments/serve/scripts/serve_cb.sh stop'
   # repeat for TT_BACKEND in {gemma4_12b, 35b}
   ```
3. **Run the only pure-unit test path that survives the cleanup**:
   ```
   .venv/bin/pytest experiments/serve/tests/test_cb_api_routing.py \
                   experiments/serve/tests/test_openai_endpoint.py
   ```
   (We are NOT proposing to archive these two.)
4. **Numpy oracle self-test**:
   ```
   .venv/bin/python experiments/utils/mamba2_numpy_oracle.py --self-test
   .venv/bin/python experiments/utils/test_mamba2_decode_isolated.py --self-test
   ```
   (Kernel-author dependency for Nemotron G1; must stay green.)
5. **Verify `import_smoke.py` is not run by CI** before archiving
   it (check `.github/workflows/`, `Makefile`, `pyproject.toml`).
6. **One commit per bucket** so any breakage bisects cleanly. Don't
   batch all 62 files in a single move.
7. **Update cross-citations**: after archiving, run a final
   `grep -rl 'research/<archived-doc>' .` for each Bucket-5 file
   and either update the link to `research/archive/<doc>` or
   acknowledge the broken link in the moving commit. Bucket 1–4
   moves should not need this because no live code imports them.

---

## First-pilot batch (5 files, recommended)

The lowest-risk archive batch — every reference is itself a
candidate, every file has been untouched ≥7 days, every path is
documented as superseded in user-memory:

1. `experiments/serve/scripts/run_drift_seed_sweep.sh` (2026-05-14)
2. `experiments/serve/scripts/run_drift_sweep.sh` (2026-05-14)
3. `experiments/serve/scripts/run_drift_dry_tune.sh` (2026-05-14)
4. `experiments/serve/scripts/run_chat_vs_raw.sh` (2026-05-14)
5. `experiments/serve/scripts/run_chat_quick.sh` (2026-05-14)

These five shell scripts have no `.py` imports against them, only
drive `experiments.serve.client generate_long` (itself a candidate),
and their only inbound reference is `research/repo-cleanup-plan.md`
(itself a candidate). Move to `archive/legacy/pre_cb_server/` and
verify with `git grep -l 'run_drift_'` showing only the archive
dir.

If that goes cleanly, the next pilot batch (10 files) is Bucket 2
(top-level `experiments/*.py` one-off probes) — they are pointed at
only by inline comments inside `server_*_ttnn.py` (comments survive
the move).

---

## Optional: `scripts/move_to_archive.sh` strawman (NOT WRITTEN)

A future helper would:

1. Take a bucket name + a manifest file as args:
   `scripts/move_to_archive.sh pre_cb_server manifests/bucket1.txt`
2. For each path in the manifest:
   a. Compute the target under `archive/legacy/<bucket>/<path>`
   b. `mkdir -p` the target's parent
   c. `git mv` the file to preserve history (NOT `mv` — `git mv`
      keeps blame and rename detection)
3. After all moves, emit a `git status` summary and stop **without
   committing**. The user reviews and commits.
4. Print a one-line "rollback" hint: `git checkout -- :/` or
   `git reset --hard HEAD` if the user wants to undo before
   commit.

The script does NOT:
- Edit any file's contents (so cross-citations may become stale —
  the safety-net step 7 handles that manually).
- Touch files outside the manifest.
- Run any git command that could lose work
  (`git reset --hard`, `git clean -fd`, etc.).

---

## Surprises worth flagging

1. **`experiments/serve/ondevice_27b.py` (44 KB) and
   `generate_27b.py` (16 KB) are NOT orphans.** Despite the
   "_27b" suffix and looking like demo companions of
   `server.py`/`client.py` (which ARE orphans), both are imported
   under `_91f`/`_91l` aliases by the production
   `experiments/serve/server_tp.py`. Naming history: they used to
   be `91f.py`/`91l.py` (the `91*` prefix was an importlib hack);
   the rename made them look like demos. They should be in
   CLAUDE.md's exclusion list and aren't.
2. **`experiments/cb/dev/cb35_drift_ladder.py` was touched
   2026-06-04** and is still in HANDOFF/plan refs even though
   the drift cliff is resolved per
   `feedback_35b_drift_resolved_2026-06-04.md`. Kept out of
   caution — if the user confirms 35B perf/drift work is fully
   retired, this and `cb35_drift_cliff_search.py` should follow
   Bucket 3.
3. **`research/35b_drift_briefing_2026-06-04.md` and
   `research/35b_drift_next_session_plan.md` are recent (2026-06-04
   / 2026-06-03) but describe work HANDOFF now de-prioritises**
   ("De-prioritise 35B work" — demo plan §3). Kept because
   HANDOFF references them as `[4]` in the cold-start reading
   list. If the user wants a tighter HANDOFF skim list, these are
   the next thing to demote.
4. **`research/code_maintainability_audit.md` (2026-05-30) calls
   out Bucket-4 files BY NAME for removal**, but no follow-up
   commit moved them — this audit is essentially a re-discovery
   of that pending work plus the post-2026-05-30 wave (Buckets 3
   and the post-demo Bucket 5 entries).
5. **`experiments/serve/scripts/v4_precision_sweep.py`
   (2026-05-20)** is referenced only by a *comment* inside
   `server_tp.py` ("see scripts/v4_precision_sweep.py"). The
   comment will outlive an archive move. Could be added to
   Bucket 1 if desired; kept out of the pilot for caution.
6. **`experiments/serve/scripts/compare_paged_*.py` and
   `count_coherent.py` (2026-05-28)** — no live caller; pre-CB
   drift hunt tools. Could join Bucket 1 in a follow-up sweep but
   not included in this audit's count (they live under
   `experiments/serve/scripts/` which CLAUDE.md excludes; this
   audit honored that exclusion strictly).
