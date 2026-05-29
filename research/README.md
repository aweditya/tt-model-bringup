# research/ — design notes & living plans

Raw research, design docs, and multi-step plans. For the **current** project
state (perf number, prod code path, what's next) read [`../HANDOFF.md`](../HANDOFF.md)
first — these docs are point-in-time and many are historical.

## Living / active

- [`maintainability_pass.md`](maintainability_pass.md) — the repo cleanup plan
  (phases, status, how-to-continue). Current focus.
- [`27b_cb_scope.md`](27b_cb_scope.md) + [`27b_continuous_batching_plan.md`](27b_continuous_batching_plan.md)
  — the continuous-batching design + running log (CB0–CB6, kernel work, floor).
- [`35b_moe_ffn_kernel_build_plan.md`](35b_moe_ffn_kernel_build_plan.md) +
  [`35b_moe_ffn_kernel_perf_deferrals.md`](35b_moe_ffn_kernel_perf_deferrals.md) +
  [`35b_perf_workflow_log.md`](35b_perf_workflow_log.md) — the in-progress 35B MoE
  FFN custom kernel (G0–G4).

## Methodology (reusable)

- [`kernel_design_worksheet.md`](kernel_design_worksheet.md) — fill-before-you-code
  worksheet + the TT hard constraints (kernel-size limit, JIT-no-rebuild,
  view-decay, trace-capture host-transfer hang, M=1 GEMV ceiling).
- [`kernel_dataflow_representation.md`](kernel_dataflow_representation.md) — the
  tile-dataflow-to-hardware mapping (TDG).
- [`repo-cleanup-plan.md`](repo-cleanup-plan.md) — the senior-engineer cleanup
  audit (line-level inventory; mine it for de-bloat specifics).

## Subdirectories

- `kernel_research/` — the 01–12 kernel-architecture deep-dives (the project's
  kernel *teaching* material).
- `archive/` — completed/superseded phase plans (Branch II/III, C', PJRT-era).
- `probe_logs/` — raw `.log`/`.json` profiling artifacts.

Everything else is dated bringup/design notes kept for context — grep by topic
(`rope`, `paged`, `tp`, `deltanet`, …). The learning-by-building Q&A lives in
[`../wiki/`](../wiki/README.md).
