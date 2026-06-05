# Doc audit plan — 2026-06-05

Audit + cleanup of every `.md` in the active repo tree (excluding `archive/`,
`experiments/.refs/`, `tt_docs_corpus/`, `home/`, `scratch/`, `.claude/`,
`.venv/`, `.git/`, `.cache/`, `.pytest_cache/`, `.ruff_cache/`).

Style target: <https://github.com/matheusfelipeog/beautiful-docs>
(scannable, TOC up top, fenced code blocks, links over inline mentions,
short paragraphs, badges only when meaningful).

Buckets:

- **KEEP** — current + accurate. No edit, or only style polish if requested.
- **REWRITE** — accurate intent but stale wording, qb1/qb2 leakage, or
  style mismatch. Rewrite in place.
- **ARCHIVE** — superseded, references files that moved/were archived,
  or duplicated by a newer doc. `git mv` to `archive/stale_docs_2026-06-05/`.
- **DELETE** — never appropriate per the brief (use ARCHIVE instead).

Total active `.md` files: 161.

---

## Priority rewrites (top-level)

| Path | Verdict | Reasoning |
|---|---|---|
| `README.md` | REWRITE | Project doc — remove qb1/qb2 leakage; generalise to QuietBox; apply beautiful-docs style. |
| `HANDOFF.md` | KEEP-with-flag | Cold-start one-pager, curated. Stale `qb1` references throughout, day-4 status is fresh. Flag for human; do NOT auto-rewrite. |
| `CONTRIBUTING.md` | REWRITE | Generalise `qb1`/`qb2` references; add explicit QuietBox link; tighten table. |
| `REPRODUCE.md` | REWRITE | qb1/qb2 leakage in setup; align with beautiful-docs style + add TOC. |
| `CLAUDE.md` | KEEP | Project instructions for the agent. Stays as-is (host-specific by design). |

---

## experiments/ (12 files)

| Path | Verdict | Reasoning |
|---|---|---|
| `experiments/README.md` | KEEP | Concise, current. |
| `experiments/owned_ops/README.md` | KEEP | Lists prod kernels + install path; current. |
| `experiments/owned_ops/qwen36_gdn_decode_owned/README.md` | KEEP | Per-op doc; matches code. |
| `experiments/owned_ops/qwen36_gdn_decode_owned/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_decay_gate_decode_owned/README.md` | KEEP | Per-op doc. |
| `experiments/owned_ops/qwen36_decay_gate_decode_owned/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_gdn_delta/README.md` | KEEP | Per-op doc (sub-op). |
| `experiments/owned_ops/qwen36_gdn_delta/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_gdn_decay_state/README.md` | KEEP | Per-op doc. |
| `experiments/owned_ops/qwen36_gdn_decay_state/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_gdn_outer_update/README.md` | KEEP | Per-op doc. |
| `experiments/owned_ops/qwen36_gdn_outer_update/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_gdn_output/README.md` | KEEP | Per-op doc. |
| `experiments/owned_ops/qwen36_gdn_output/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_gdn_prediction/README.md` | KEEP | Per-op doc. |
| `experiments/owned_ops/qwen36_gdn_prediction/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_conv1d_decode_owned/README.md` | KEEP | Experimental op. |
| `experiments/owned_ops/qwen36_conv1d_decode_owned/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/nemotron3_mamba2_decode_owned/README.md` | KEEP | Active bringup (MM7). |
| `experiments/owned_ops/nemotron3_mamba2_decode_owned/INTEGRATION.md` | KEEP | Validation log. |
| `experiments/owned_ops/qwen36_moe_ffn_decode_owned/README.md` | KEEP | In-progress op. |
| `experiments/kernel_patches/qwen36_gdn_decode_owned/README.md` | KEEP | JIT patch doc. |
| `experiments/utils/README.md` | KEEP | Utility shelf index. |

---

## models/ (1 file)

| Path | Verdict | Reasoning |
|---|---|---|
| `models/README.md` | KEEP | Concise demo index. |

---

## scripts/ (1 file)

| Path | Verdict | Reasoning |
|---|---|---|
| `scripts/CHAT_TUI.md` | KEEP | Current; matches `scripts/chat.py`. |

---

## research/ — top level (38 files)

Per the brief: lean toward KEEP unless (a) references files that no longer
exist, (b) duplicated by newer doc, or (c) covers a finished/shipped scope.

| Path | Verdict | Reasoning |
|---|---|---|
| `research/README.md` | KEEP | Index doc, current. |
| `research/01_tenstorrent_hardware.md` | KEEP | Foundational reference. |
| `research/02_tenstorrent_software_stack.md` | KEEP | Foundational reference. |
| `research/03_jax_xla_pjrt.md` | KEEP | JAX/PJRT background (pre-pivot). |
| `research/04_tt_isa_documentation.md` | KEEP | ISA notes. |
| `research/05_jax_from_scratch.md` | KEEP | JAX walkthrough. |
| `research/06_hypotheses_to_test.md` | KEEP | Early hypotheses. |
| `research/27b_cb_scope.md` | KEEP | Cited from HANDOFF; CB design + numbers. |
| `research/27b_prefill_trace_plan.md` | KEEP | Cited active. |
| `research/27b_prefix_caching_plan.md` | KEEP | Cited active (PC implementation reference). |
| `research/35b_cb_bringup_plan.md` | KEEP | Active bringup plan. |
| `research/35b_determinism_2026-06-04.md` | KEEP | Recent finding (still relevant). |
| `research/35b_drift_briefing_2026-06-04.md` | KEEP | Recent; 35B drift resolved doc. |
| `research/35b_drift_next_session_plan.md` | KEEP | Parked but kept per its own header (next-session scaffolding); HANDOFF still references it. |
| `research/35b_moe_ffn_kernel_perf_deferrals.md` | KEEP | Cited from HANDOFF perf trail. |
| `research/35b_moe_ffn_kernel_scoping.md` | KEEP | Companion scoping doc. |
| `research/35b_perf_milestones.md` | KEEP | Cited from HANDOFF (active perf trajectory). |
| `research/35b_tt_perf_report_findings.md` | KEEP | Cited from HANDOFF. |
| `research/audit_gdn_kernel_us_vs_tt_metal.md` | KEEP | Methodology audit (recent). |
| `research/audit_gemma4_opts_us_vs_tt_metal_44962.md` | KEEP | Audit (recent). |
| `research/audit_qwen36_us_vs_qwen9b_p150_branch.md` | KEEP | Audit (recent). |
| `research/cb_perf_regression_audit_2026-06-04.md` | KEEP | Recent audit, still informative. |
| `research/code_cleanup_plan_2026-06-04.md` | KEEP | Living cleanup plan. |
| `research/gemma4_12b_bringup_plan.md` | KEEP | Active bringup plan (Gemma 4 IT live). |
| `research/gemma4_12b_scoping.md` | KEEP | Companion scoping doc. |
| `research/gemma4_pc_chat_template_analysis_2026-06-04.md` | KEEP | Recent investigation. |
| `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md` | KEEP | Recent root-cause; HANDOFF cites it. |
| `research/gemma4_perf_briefing_2026-06-04.md` | KEEP | Recent perf briefing. |
| `research/home_llm_landscape_2026.md` | KEEP | Cited from HANDOFF (candidate models). |
| `research/kernel_dataflow_representation.md` | KEEP | Methodology doc. |
| `research/mm7_g1_dataflow_decisions.md` | KEEP | Active MM7 design log. |
| `research/mm7_g1_mamba2_kernel_design.md` | KEEP | Active MM7 design. |
| `research/model_bringup_recipe.md` | KEEP | Cited from HANDOFF; the staged ladder recipe. |
| `research/multi_model_serving_plan.md` | KEEP | Cited from HANDOFF. |
| `research/nemotron3_nano_30b_a3b_bringup_plan.md` | KEEP | Current MM7 plan-of-action. |
| `research/nemotron3_nano_architecture_brief.md` | KEEP | Architecture brief. |
| `research/repo_archive_audit_2026-06-04.md` | KEEP | The earlier audit; reference for what was moved. |
| `research/speculative_decoding_plan_2026-06-04.md` | KEEP | Forward-looking plan. |
| `research/tokenizer_chat_template_reference.md` | KEEP | Reference. |
| `research/tt_metal_adoption_plan_2026-06-04.md` | KEEP | Living adoption plan. |
| `research/tt_metal_contributions_2026-06-05.md` | KEEP | Fresh from this morning. |
| `research/tt_metal_moe_cb_patterns.md` | KEEP | Cited from HANDOFF. |
| `research/vllm_chat_template_handling.md` | KEEP | Cited from HANDOFF. |
| `research/vllm_prefix_caching_audit.md` | KEEP | Cited from HANDOFF. |

---

## research/kernel_research/ (12 files)

The 01–12 kernel teaching material. Foundational and currently cited.

| Path | Verdict |
|---|---|
| `research/kernel_research/01_tensix_architecture_primer.md` | KEEP |
| `research/kernel_research/02_adding_a_custom_ttnn_op.md` | KEEP |
| `research/kernel_research/03_hello_world_kernel_walkthrough.md` | KEEP |
| `research/kernel_research/04_update_cache_reference_op.md` | KEEP |
| `research/kernel_research/05_memory_configs_deep_dive.md` | KEEP |
| `research/kernel_research/06_trace_capture_internals.md` | KEEP |
| `research/kernel_research/07_sdpa_decode_and_paged_variant.md` | KEEP |
| `research/kernel_research/08_tensix_vs_cuda_programming_model.md` | KEEP |
| `research/kernel_research/09_production_kernel_dataflow_survey.md` | KEEP |
| `research/kernel_research/10_github_design_discussions.md` | KEEP |
| `research/kernel_research/11_external_blog_synthesis.md` | KEEP |
| `research/kernel_research/12_implementation_readiness.md` | KEEP |

---

## wiki/ (72 files)

Per the brief: "wiki entries are working docs and shouldn't be over-polished;
only touch if actively misleading or broken". The wiki is a chronological
learning log (01–66) — keep numbering and ordering. No moves; only the
index README gets a light polish if there are broken links.

| Path | Verdict |
|---|---|
| `wiki/README.md` | KEEP (light style polish if needed) |
| `wiki/01_what_is_jax.md` through `wiki/66_blackhole_kernel_dataflow_anatomy.md` | KEEP |
| `wiki/bringup_checklist.md` | KEEP |
| `wiki/debugging_methodology.md` | KEEP |
| `wiki/profiling_guide.md` | KEEP |
| `wiki/qa_correctness_and_architecture.md` | KEEP |
| `wiki/seven_bugs_case_studies.md` | KEEP |

---

## Items flagged for human judgment

1. **`HANDOFF.md`** is a 972-line curated cold-start doc. The user says "if you
   find issues, write them up in this plan and let the parent agent decide".
   Findings:
   - References `qb1` throughout (cold-start ssh commands, dev harness paths).
     These are real host instructions the user runs — they're load-bearing
     for resuming work. Generalising to `QuietBox` would break the
     copy-paste workflow.
   - The document mixes a "post-win quick-start" headline (latest day-4 PASS)
     with the historical demo-day write-up. The pre-MM7 "OLD HEADLINE" block
     is still informationally useful but is ~700 lines of accumulated history.
   - Recommendation: do NOT auto-rewrite. Parent agent should decide whether
     to (a) trim the OLD HEADLINE blocks down, (b) preserve qb1 references
     as personal-dev-notes, (c) leave entirely.
2. **HANDOFF "Read order when resuming work" §** points at
   `archive/superseded_research_2026-06-04/profiling-quick-reference.md`.
   The doc already acknowledges this is an archived path, so the link is
   honest. KEEP.
3. **CLAUDE.md** ostensibly belongs to the user/agent (project instructions).
   Per the brief it's not part of `memory/` so it's in scope — but rewriting
   it would change the harness's behavioural priors. KEEP unchanged.

---

## Execution order (commits)

1. `docs(plan)`: drop this plan doc.
2. `docs(readme)`: rewrite README in beautiful-docs style; remove qb1/qb2;
   link to QuietBox.
3. `docs(reproduce)`: align REPRODUCE with new README; remove qb1/qb2
   specifics; add TOC.
4. `docs(contributing)`: align CONTRIBUTING; remove qb1/qb2.
5. No archive moves expected — every active doc passed audit.
