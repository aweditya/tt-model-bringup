# research/ — design notes & living plans

Raw research, design docs, and multi-step plans. For the **current** project
state (perf number, prod code path, what's next) read [`../HANDOFF.md`](../HANDOFF.md)
first — these docs are point-in-time and many are historical.

## Living / active

- [`nemotron3_nano_30b_a3b_bringup_plan.md`](nemotron3_nano_30b_a3b_bringup_plan.md) +
  [`nemotron3_nano_architecture_brief.md`](nemotron3_nano_architecture_brief.md) +
  [`mm7_g1_mamba2_kernel_design.md`](mm7_g1_mamba2_kernel_design.md) — current MM7
  bringup target (Nemotron-3 Nano 30B-A3B hybrid Mamba2-Transformer MoE).
- [`model_bringup_recipe.md`](model_bringup_recipe.md) — the staged v0.1→v2
  ladder + isolation-probe + dev-harness recipe (read first when starting a
  new model bringup).
- [`27b_cb_scope.md`](27b_cb_scope.md) — the continuous-batching design + numbers
  (CB0–CB4, kept as the HANDOFF-cited reference).
- [`35b_moe_ffn_kernel_perf_deferrals.md`](35b_moe_ffn_kernel_perf_deferrals.md) +
  [`35b_perf_milestones.md`](35b_perf_milestones.md) +
  [`35b_tt_perf_report_findings.md`](35b_tt_perf_report_findings.md) — the 35B MoE
  FFN perf trajectory + tt-perf-report findings.

## Methodology (reusable)

- [`kernel_dataflow_representation.md`](kernel_dataflow_representation.md) — the
  tile-dataflow-to-hardware mapping (TDG).
- [`audit_gdn_kernel_us_vs_tt_metal.md`](audit_gdn_kernel_us_vs_tt_metal.md) +
  [`audit_qwen36_us_vs_qwen9b_p150_branch.md`](audit_qwen36_us_vs_qwen9b_p150_branch.md) +
  [`audit_gemma4_opts_us_vs_tt_metal_44962.md`](audit_gemma4_opts_us_vs_tt_metal_44962.md) —
  point-in-time audits of our paths vs upstream.

## Subdirectories

- `kernel_research/` — the 01–12 kernel-architecture deep-dives (the project's
  kernel *teaching* material).
- `archive/` — completed/superseded phase plans (Branch II/III, C', PJRT-era).
  Also see `../archive/superseded_research_2026-06-04/` for the 27-file wave
  of plans archived after the 2026-06-04 demo (27B CB, 35B MoE FFN kernel build,
  pre-CB production server plan, profiling cheatsheets, maintainability audit, etc.).
- `probe_logs/` — raw `.log`/`.json` profiling artifacts.

Everything else is dated bringup/design notes kept for context — grep by topic
(`rope`, `paged`, `tp`, `deltanet`, …). The learning-by-building Q&A lives in
[`../wiki/`](../wiki/README.md).
