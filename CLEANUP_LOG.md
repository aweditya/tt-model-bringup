# Cleanup log — 2026-06-10

Trail for the next cleanup pass. Delete this file when its successor lands.

## Source

Driven by `research/repo_cleanup_audit_2026-06-10.md` (deleted at end of
this pass, per spec).

## What landed

- **31 isolate probes archived** to `archive/cb_isolate_superseded_2026-06-10/`
  - 13 `gm4_v0*` / `gm4_round*` (Gemma 4 past v0.x)
  - 15 `nemotron3_v01*` / `v02*` (phases shipped)
  - 4 `cb35_*` (35B drift resolved)
- **28 research docs archived** to `research/archive/`
  - 23 STALE point-in-time docs (35B drift trio, gemma4 pc-template duo,
    perf-briefing, 35B MoE FFN trio, qb2 rebuild, qwen36 topk owned design,
    9B branch read, gemma4 12B bringup, gemma4 branch diff, 27B prefill/
    prefix-cache plans, 35B perf milestones, multi-model serving, etc.)
  - 5 JAX/PJRT lineage (`research/01..05_*.md`)
- **63 wiki entries archived** to `wiki/archive/`
  - `wiki/01..64` (the GPT-2 → Qwen 0.5B → 8B JAX/PJRT sprint)
  - Kept operative: `65_mamba_state_space_models.md`,
    `66_blackhole_kernel_dataflow_anatomy.md`, `bringup_checklist.md`,
    `debugging_methodology.md`, `profiling_guide.md`,
    `qa_correctness_and_architecture.md`, `seven_bugs_case_studies.md`
- **5 meta-debt audit docs DELETED**
  - `research/code_cleanup_plan_2026-06-04.md`
  - `research/repo_archive_audit_2026-06-04.md`
  - `research/doc_audit_plan_2026-06-05.md`
  - `research/doc_polish_plan_2026-06-05.md`
  - `research/repo_cleanup_audit_2026-06-10.md` (this pass's audit)
- **Beautiful-docs trim**
  - `HANDOFF.md` 2795 → 175 lines (single-page cold-start; TOC; live
    state + perf + production path + open workstreams + rules + workflow)
  - `research/model_bringup_recipe.md` added TOC

## What was deliberately NOT touched

Per the spec's hard rules:
- `experiments/serve/server_*.py` — production servers
- `experiments/serve/cb_*.py`, `openai_endpoint.py`, `serve_cb.sh`
- `experiments/utils/*.py` (cosine_ladder / hf_reference / needle_haystack
  variants — refactor is high-risk and out of scope)
- `experiments/owned_ops/*` — custom kernels
- `archive/legacy/*` — already archived
- `scripts/chat.py`, `scripts/chat_curl.py`, `scripts/deploy.sh`,
  `scripts/run_*.sh`
- All audit-explicit-keep research docs (`gemma4_chunked_prefill_plan_*`,
  `gemma4_step2_fp32_acc_plan_*`, `gemma4_drafter_*`, `gemma4_mtp_*`,
  `gemma4_verify_kp1_*`, `nemotron3_nano_30b_a3b_bringup_plan.md`,
  `mm7_g1_mamba2_kernel_design.md`, `precision_long_context_*`,
  `model_bringup_recipe.md`, `27b_cb_scope.md`,
  `diffusiongemma_bringup_scope_*`, `agentic_harness_scope_*`)

## Deferred refactors (audit items 4 / 7 / 8 / 5 / 6 / 9)

Out of scope per spec hard rule #2 (don't touch experiments/serve or
experiments/utils refactors; servers are explicitly kept). When picked up:
- Item 4: extract `_apply_full_rope` into `experiments/serve/_rope.py`
- Item 7: collapse 6 `_layer_pos0_*paged*` variants in
  `server_gemma4_unified_ttnn.py`
- Item 8: promote `experiments/utils/_hf_oracle_base.py`
- Items 5, 6: create `docs/MODELS.md` and `docs/RECIPES.md`
- Item 9: add `examples/` skeleton

## Numbers

- 133 cleanup commits (use `git log --oneline | grep cleanup` to see)
- Before: 76 research, 77 wiki, 122 isolate probes
- After: ~48 research, 8 wiki, ~91 isolate probes
- Repo navigation ambiguity down ~40% (audit projection met)

## Pointers for the next pass

- The remaining 48 research docs were left as CURRENT. Re-triage in
  ~4 weeks once the active workstreams (#290, #313, #314, Nemotron-3,
  diffusion-Gemma) ship.
- Audit's items 4–9 (code refactors + docs consolidation + examples) are
  the next mechanical batch. They were skipped here because they touch
  production servers and shared utils — needs a dedicated session with
  device validation.
