# Repo cleanup, refactor & docs-freshness audit — 2026-06-10

Enumeration only — nothing deleted/moved. Verify on host before action.

## 1. Top-line summary

Pivoted from JAX/PJRT (`wiki/01–64` fossil) to direct TT-Metal bringup of
**27B** (`server_tp.py`, 3097 LOC), **35B-A3B MoE** (`server_35b_ttnn.py`,
2040), **Gemma 4 12B** (`server_gemma4_unified_ttnn.py`, 3121 — demo path)
and **Nemotron-3 Nano 30B-A3B** (`server_nemotron3_nano_ttnn.py`, 2961, in
flight). Demo today is the **qb1 Gemma 4 OpenAI HTTP server with chunked
prefill** (HANDOFF 7 h ago). **Verdict**: top-level docs healthy (all ≤ 1 wk);
**`research/` (76) and `experiments/cb/isolate/` (122) are 30–40%
archivable**; servers duplicate layer/RoPE/buffer slabs across 4 files.

## 2. DOCS triage

### Top-level — all CURRENT
- `HANDOFF.md` (7 h), `README.md` (6 d), `CONTRIBUTING.md` (6 d),
  `REPRODUCE.md` (6 d), `CLAUDE.md` (5 d), `Makefile` (11 d).
- `research/README.md` (6 d) + `wiki/README.md` (6 d) — accurate indices.

### `research/` — 76 files

**CURRENT (living/active)**: `gemma4_chunked_prefill_plan_2026-06-08.md` (7 h),
`gemma4_step2_fp32_acc_plan_2026-06-09.md` (33 h), `gemma4_layout_op_elimination_plan_2026-06-08.md` (2 d),
`gemma4_drafter_*` set (2 d), `gemma4_mtp_*` set (2-4 d), `gemma4_verify_kp1_*` (3 d),
`nemotron3_nano_30b_a3b_bringup_plan.md` (4 d), `mm7_g1_mamba2_kernel_design.md` (6 d),
`precision_long_context_2026-06-08.md` (2 d), `model_bringup_recipe.md` (7 d),
`27b_cb_scope.md` (13 d, HANDOFF-cited).

**REFERENCE (CURRENT, stable)**: `kernel_research/01..13_*.md` teaching set;
`multi_trace_orchestration_reference.md`, `tokenizer_chat_template_reference.md`,
`deepseek_v3_alias_page_table_reference.md`, `moe_trace_precedents.md`,
`tt_metal_moe_cb_patterns.md`.

**STALE (point-in-time, factually outdated)**: `35b_drift_briefing_2026-06-04.md` +
`_next_session_plan.md` (drift resolved per memory), `35b_determinism_2026-06-04.md`,
`cb_perf_regression_audit_2026-06-04.md`, `gemma4_pc_chat_template_analysis_2026-06-04.md` +
`_asymmetry_2026-06-04.md` (template now in cb_api), `gemma4_perf_briefing_2026-06-04.md`,
`27b_prefill_trace_plan.md` + `27b_prefix_caching_plan.md` (both shipped),
`35b_perf_milestones.md` + `_tt_perf_report_findings.md` + `_moe_ffn_kernel_scoping.md` +
`_moe_ffn_kernel_perf_deferrals.md` (pre-CB, 2 wk), `home_llm_landscape_2026.md`,
`multi_model_serving_plan.md`, `qb2_rebuild_plan_2026-06-08.md`,
`qwen36_topk_owned_design.md` (shipped), `35b_cb_bringup_plan.md` (shipped),
`gemma4_12b_bringup_plan.md` + `_scoping.md` (done), `gemma4_31b_bringup_scope.md` (speculative),
`qwen36_9b_branch_read_2026-06-08.md`, `gemma4_branch_diff_2026-06-08.md`.

**META-DEBT (DUPLICATE — this doc supersedes)**: `code_cleanup_plan_2026-06-04.md`,
`repo_archive_audit_2026-06-04.md`, `doc_audit_plan_2026-06-05.md`,
`doc_polish_plan_2026-06-05.md`.

**DEAD (JAX/PJRT lineage)**: `research/01..05_*.md` (Apr 20). Already kept
in archive lineage; relocate to `research/archive/`.

**Bulk**: `research/gemma4_perf_qb2_2026-06-05/` (100+ logs) + `probe_logs/`
(>2 wk raw) — bake summary, move raw to archive.

### `wiki/` — 77 entries

`wiki/01..64` (Apr 20–22) are the GPT-2 → Qwen 0.5B → 8B JAX/PJRT sprint —
**STALE-LEARNING** (project lineage, not operative). **CURRENT operative**:
`65_mamba_state_space_models`, `66_blackhole_kernel_dataflow_anatomy`,
`bringup_checklist`, `debugging_methodology`, `profiling_guide`,
`seven_bugs_case_studies`, `qa_correctness_and_architecture`.

### Consolidation plan — 5-doc target

1. **`README.md`** — front door (keep).
2. **`HANDOFF.md`** — rolling state (keep).
3. **`docs/MODELS.md`** (new) — merge per-model status from `27b_cb_scope`,
   `35b_cb_bringup_plan`, `gemma4_12b_bringup_plan`, `nemotron3_nano_30b_a3b_bringup_plan`
   into one page per model.
4. **`docs/RECIPES.md`** (new) — merge `model_bringup_recipe.md` +
   `wiki/bringup_checklist.md` + `wiki/debugging_methodology.md` +
   `wiki/profiling_guide.md` + `wiki/seven_bugs_case_studies.md`.
5. **`docs/KERNELS.md`** (new) — index into `research/kernel_research/01..13`
   and `experiments/owned_ops/README.md` + `INTEGRATION.md`.

Everything else → `research/archive/`.

## 3. CODE triage

### `archive/` (117) — already well-organized; don't prune
- `archive/legacy/{pjrt_plugin,tt_jax}/` JAX lineage (kept per CLAUDE.md);
  `{01..91x}_*.py` (~91) + `bringup/` (68) superseded probes; 6 dated buckets.
  Verify `archive/README.md` indexes the buckets.

### `experiments/cb/isolate/` — 122 probes, ~50 archivable
- **Move 25 superseded `gm4_v0*` + `gm4_round*`** (Gemma 4 is past v0.x;
  precision now gated via `gemma4_long_context_argmax_gate.py`).
- **Move 14 `nemotron3_v01*` + `v02*` bootstrap/section smokes** (phases shipped).
  Keep `_v033_nstep_chain_smoke` (HANDOFF demo), `_v041g_needle_longhorizon`,
  `_v05bench_niah`, `_v05p1_eager_perf`.
- **Move 5 `cb35_*`** (35B drift resolved).
- **Keep workflow-locked**: `gemma4_long_context_argmax_gate.py`,
  `gemma4_spec_dec_cache_invariant_probe.py`, `gemma4_chunked_prefill_{L128,ladder,trace_ladder}.py`,
  `gemma4_long_decode_vs_hf_ladder.py` (new today), `paged_sdpa.py`,
  `paged_update_cache.py`, `prefill_trace.py`, `prefix_cache_*.py`,
  `mamba2_*_smoke.py`.

### `experiments/utils/` (60+) — duplicated patterns
| Pattern | Files | Note |
|---|---|---|
| `cosine_ladder_*` | 4 (`_hf_gemma4_it.py` 9 m ago) | Promote `_aggregate.py` to shared core. |
| `hf_oracle_gemma4_assistant{,_v2}.py` | 2 | Mark v1 DEPRECATED per HANDOFF. |
| `hf_reference_{27b,35b,gemma4_12b,nemotron3_nano}.py` | 4 | Each re-implements safe_open + numpy fp32 loader (~200 LOC base). |
| `needle_haystack_*.py` | 4 | One generic + 3 wrappers. |
| `tracy_profile_one_*` | 4 | Shared shell, model configs. |

`experiments/utils/archive/` (2 wk) already siloed.

### `experiments/serve/` — duplicated server skeletons
- 4 `server_*_ttnn.py` repeat bootstrap + `update_input_buffers` + trace
  orchestration. `server_*_cb.py` family repeats CB slot setup/reset.
- **Naming inconsistency**: drop the `_ttnn` suffix (all are ttnn); use
  `server_<model>.py` + `server_<model>_cb.py`. Today: `server_tp.py` is 27B
  but name doesn't say so; `server_gemma4_12b_assistant_ttnn.py` (1357 LOC)
  is parallel to `_unified_ttnn.py`.

## 4. REFACTORING opportunities

### `server_gemma4_unified_ttnn.py` (3121 LOC)
1. **Extract `_apply_full_rope` / `_apply_full_rope_seq`** (lines 1357, 2456)
   to `experiments/serve/_rope.py`. Same fn is re-forked at
   `server_gemma4_12b_assistant_ttnn.py:610` and `server_gemma4_unified_cb.py:278`
   (`_apply_full_rope_b`). Saves ~120 LOC + locks semantics across forks.
2. **Collapse the 6 `_layer_pos0_*paged*` variants** (lines 1502, 1674, 1786,
   1873, 1934, 2095 = ~800 LOC, 4 are K+1 forks) into one
   `_layer_pos0_paged(..., *, attn_kind, B=1)`.
3. **Move `forward_prefill_chunked_tp` + `step_forward_prefill`** (line 2688)
   to `_prefill.py`.

### `server_tp.py` (3097 LOC, 27B)
1. Extract `_chunked_recurrence_tp` (836), `_neumann_inverse_via_mesh_tp`
   (1010), `_chunked_dn_with_chunked_recurrence_tp` (1071) to `_dn.py`.
   35B server also uses these.
2. Move `handle_bench_decode_tp_components` + `handle_profile_decode_tp_ops`
   (lines 2202–2649, ~450 LOC) to `experiments/cb/bench/decode_tp.py`.

### `server_35b_ttnn.py` (2040 LOC)
1. Three near-clone attention paths: `attn_forward_ttnn_sdpa` (772),
   `attn_forward_ttnn` (920), `attn_forward_ttnn_manual` (947). Manual is
   BROKEN per memory — delete after SDPA-only branch verified.
2. `moe_forward_ttnn` (1180) vs `_pattern_a_batched` (1254) — keep batched.

### `cb_scheduler.py` (848 LOC)
1. Pull `_batched_step` / `greedy_ref` (96, 107) into `cb_dev.py` — they're
   harness bypass paths, not production scheduler.

## 5. Beautiful-docs principles applied

**README.md already beautiful-docs shape** (TOC, model table, quickstart,
troubleshooting, related-projects). Improvements:
- Add `## Examples` linking `experiments/cb/validate/forward.py` as canonical
  "hello world".
- Restate the current perf number directly (not just in HANDOFF) so README
  is independently informative.

**Proposed `docs/` tree**:
- `MODELS.md` — per-model status, perf, demo cmd, code path.
- `RECIPES.md` — bringup recipe + checklist + debugging + profiling merged.
- `KERNELS.md` — owned_ops + `kernel_research/01..13` + INTEGRATION.md.
- `ARCHITECTURE.md` — mesh, fabric, paged caches, trace pattern (link wiki 66).
- `CHAT_API.md` — `scripts/CHAT_TUI.md` + `cb_api` endpoints hoisted.

**Quickstart (3 commands)**:
```
ssh qb1
bash ~/tt-xla/experiments/serve/scripts/serve_cb.sh start
curl http://localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

**`examples/` dir is missing**. Add: `01_chat_curl.sh`, `02_chat_tui.md`,
`03_run_a_probe.sh`, `04_add_an_owned_op.md`.

## 6. PRIORITIZED action list

| # | Scope | Action | h | Why |
|---|---|---|---|---|
| 1 | `research/` (4) | Move 4 meta-debt audit docs to `research/archive/` (this doc supersedes). | 0.25 | One source of truth for cleanup. |
| 2 | `cb/isolate/` (~44) | Move 25 `gm4_v0*`/`round*` + 14 `nemotron3_v01*`/`v02*` + 5 `cb35_*` to `archive/cb_isolate_superseded_2026-06-10/`. | 1.0 | 122 → ~78 probes; signal:noise jumps. |
| 3 | `research/` (15) | Move STALE point-in-time docs (35B drift trio, gemma4 pc-template duo, perf-briefing, 35B MoE FFN trio, qb2 rebuild, qwen36 topk owned design, 9B branch read, 12B bringup) to `research/archive/`. | 1.0 | 76 → ~40. |
| 4 | `serve/_rope.py` | Extract `_apply_full_rope` (+seq+b) into one module; update 3 servers. | 1.5 | Locks RoPE across target+drafter+CB. |
| 5 | `docs/MODELS.md` | New: per-model status (status, perf, demo cmd, code path). | 2.0 | Single-source-of-truth per model. |
| 6 | `docs/RECIPES.md` | Merge `model_bringup_recipe` + `bringup_checklist` + `debugging_methodology` + `profiling_guide` + `seven_bugs_case_studies`. | 2.0 | One recipe for new collaborator. |
| 7 | `server_gemma4_unified_ttnn.py` | Collapse 6 `_layer_pos0_*` variants into one parameterized helper. | 4.0 | 3121 → ~2500 LOC; bug surface shrinks. |
| 8 | `utils/_hf_oracle_base.py` | Promote shared safe_open + numpy fp32 oracle; refactor 4 callers. | 3.0 | Kills triplicated load logic. |
| 9 | `examples/` | Create 4-file examples skeleton. | 1.0 | Beautiful-docs examples; onboarding. |
| 10 | `wiki/01..64` | Move JAX/PJRT-era entries to `wiki/archive/`; keep 65, 66, bringup_checklist, debugging_methodology, profiling_guide, seven_bugs, qa_correctness as operative. | 0.5 | 77 → ~10 operative. |

**Total**: ~16 h; reclaims ~40% navigation ambiguity. Every move into
`archive/`, no history lost.

### Caveats
- Verify in `MEMORY.md` + `HANDOFF.md` before moving any `_v0xx_*` probe.
- Do not rename `tt-xla/` / `~/tt-xla` (CLAUDE.md).
- `server_tp.py` is canary-gated (CONTRIBUTING.md).
- `experiments/owned_ops/` (12 dirs, self-documented) — leave.
- `archive/legacy/{pjrt_plugin,tt_jax}/` explicitly kept (CLAUDE.md).
