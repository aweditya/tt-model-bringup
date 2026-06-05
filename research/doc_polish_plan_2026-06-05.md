# Doc polish pass — 2026-06-05 (round 2)

Second-round documentation polish after the first audit
(`research/doc_audit_plan_2026-06-05.md`, commits `eb11a50..6262b4c`).
Goal: get the repo to a state where someone visiting can immediately
see the value. Focus is on **stale framework references**, **a single
clean perf summary**, **archive consolidation**, and **post-Mamba2
spot-check** of the round-1 polish.

`HANDOFF.md` is operator-curated and explicitly out of scope per the
brief.

Style target: <https://github.com/matheusfelipeog/beautiful-docs>
(scannable, TOC up top, fenced code blocks, links over inline mentions,
short paragraphs, no narrative).

---

## Table of contents

- [Task 1 — Strip stale framework references](#task-1--strip-stale-framework-references)
- [Task 2 — Performance summary table (NEW doc)](#task-2--performance-summary-table-new-doc)
- [Task 3 — Consolidate `scratch/legacy-demos/` into `archive/`](#task-3--consolidate-scratchlegacy-demos-into-archive)
- [Task 4 — Spot-check round-1 polish for post-Mamba2 staleness](#task-4--spot-check-round-1-polish-for-post-mamba2-staleness)
- [Execution order (commits)](#execution-order-commits)
- [Items flagged for human judgment](#items-flagged-for-human-judgment)

---

## Task 1 — Strip stale framework references

### 1a. PJRT / XLA / JAX-as-target-backend prose

The project pivoted from a JAX/XLA PJRT backend to direct TT-Metal
bringup. CLAUDE.md keeps a one-line "originally scoped as a JAX/XLA
backend" note — that stays. Everywhere else we scrub prose that talks
about PJRT/JAX as the **active** path. Historical wiki entries (which
literally chronicle the journey) **stay** — those are journal-style
and explicitly about the pre-pivot era.

| File | Lines | Edit |
|---|---|---|
| `README.md` | 7 ("no PJRT, no JAX") | KEEP — this is anti-context, useful |
| `README.md` | 12–13 ("pivoted from a JAX/XLA PJRT backend") | TRIM — drop the "pivoted from" framing; recast as "TT-Metal direct bringup" |
| `README.md` | 88–90 ("originally a JAX/XLA PJRT backend") | TRIM — short footnote, not a full paragraph |
| `README.md` | 142 (archive description) | TRIM — drop "PJRT plugin (`legacy/`)" emphasis; just say "retired era directories" |
| `README.md` | 171–172 (Related projects: tenstorrent/tt-xla) | KEEP — that's a legitimate upstream project link |
| `README.md` | 187–190 (Origin footnote) | DROP — duplicates the line at 88–90 |
| `CONTRIBUTING.md` | 154 ("Frozen names: ... tt-xla") | KEEP — operator instruction (paths) |
| `research/README.md` | 36 ("PJRT-era") | KEEP — `archive/` description; accurate label |

### 1b. `tt-xla` (the stale repo name) in prose

The local working dir `~/tt-xla` is operator-mandated and stays. Look
only at prose like "the tt-xla project" / "fork tt-xla" / titles. The
repo is `tt-model-bringup`.

| File | Lines | Edit |
|---|---|---|
| `research/06_hypotheses_to_test.md` | throughout (H1/H3/H4 ask "is tt-xla slow?") | ARCHIVE — pre-pivot doc, no longer the active hypothesis ladder |
| `research/kernel_research/10_github_design_discussions.md` | 1 (title), 48, 136, 138 | REWRITE — replace "tt-xla" → "tt-model-bringup" in prose; keep doc as kernel reference |
| `research/kernel_research/11_external_blog_synthesis.md` | 1, 4 | REWRITE — replace "tt-xla" → "tt-model-bringup" in prose |
| `research/kernel_research/08_tensix_vs_cuda_programming_model.md` | 3 | REWRITE — replace "tt-xla" → "tt-model-bringup" in audience line |
| `research/02_tenstorrent_software_stack.md` | 15, 90 | KEEP — describes Tenstorrent's official tt-xla project (legitimate ref) |
| `research/03_jax_xla_pjrt.md` | content | KEEP — historical-reference doc; primer on JAX/XLA/PJRT |
| `research/05_jax_from_scratch.md` | content | KEEP — historical primer (JAX walkthrough) |
| Path mentions (`~/tt-xla`, `tt-xla/.cache/`) across many research docs | n/a | KEEP — those are real operator paths |

### 1c. `cs440lx` (user's class) prose references

Recast as "research project / exposition" — not class-specific. The
archive bucket `archive/presentation_cs440lx_2026-06-04/` is a dated
artifact directory and stays as-is (don't rename archive paths).

| File | Lines | Edit |
|---|---|---|
| `README.md` | 12, 142 | EDIT — drop "Stanford CS440LX" framing |
| `wiki/22_pjrt_plugin_deep_dive.md` | 112 ("For CS440LX") | EDIT — "For this project" |
| `wiki/23_jax_mps_style_approach.md` | 588 ("For CS440LX") | EDIT — "For this project" |
| `wiki/60_journey_reflection.md` | 186 ("Stanford CS440LX class project") | EDIT — "research project" |
| `research/tt_metal_contributions_2026-06-05.md` | 1, 3, 446 | EDIT — drop the "CS440LX" subtitle; keep project name |
| `research/audit_gdn_kernel_us_vs_tt_metal.md` | 3–4 | EDIT — drop "CS440LX" qualifier from author/trigger lines |
| `research/audit_gemma4_opts_us_vs_tt_metal_44962.md` | 9 | EDIT — drop "during the CS440LX poster session" → "during a poster session" |
| `research/audit_qwen36_us_vs_qwen9b_p150_branch.md` | 11 | EDIT — "CS440LX poster-session attendee" → "poster-session attendee" |
| `research/repo_archive_audit_2026-06-04.md` | 9, 111 | KEEP — references the dated demo + archive path (descriptive, not branding) |
| `research/nemotron3_nano_architecture_brief.md` | 682 ("CS440LX research use this is fine") | EDIT — "research use this is fine" |
| `research/nemotron3_nano_architecture_brief.md` | 691, 696, 702 (absolute paths under `/Users/.../cs440lx/...`) | EDIT — strip personal-dir prefix; use repo-relative paths |
| `research/cb_perf_regression_audit_2026-06-04.md` | refs to `archive/presentation_cs440lx_2026-06-04/...` | KEEP — those are archive paths, not branding |

---

## Task 2 — Performance summary table (NEW doc)

Create `research/perf_summary_2026-06-05.md`. Layout:

- One headline table per model (eager / traced ms/tok, single-seq tok/s,
  hardware mesh, source citation).
- One CB scaling table per model that has CB measurements
  (B=1, 4, 8, 16, 32, 64 → step ms, aggregate tok/s).
- Roofline ceilings next to each headline.
- Explicit "(not measured)" / "(approx)" annotations where a number
  is uncertain — no hallucination.

Sources for the numbers:

| Source | Used for |
|---|---|
| `HANDOFF.md` §"Live perf headlines" + §"Steady-state perf snapshot" | 27B B=32, Gemma 4 B=32, 35B SLOTS=1, TP single-seq |
| `archive/presentation_cs440lx_2026-06-04/06_live_measurements.md` | 1/8/16/32 client scaling tables for 27B + Gemma 4 |
| `research/27b_cb_scope.md` §"Throughput-vs-B sweep" + DNK-G4 table | 27B B=1..64 step/tok/s, +DNK-G4 593 tok/s |
| `research/35b_perf_milestones.md` | 35B trace ms/tok evolution + bf16/bf8 BW ceiling |
| `archive/superseded_research_2026-06-04/35b_perf_workflow_log.md` | 35B 6-step workflow attempts A001..A008 ms/tok timeline |
| `research/gemma4_perf_briefing_2026-06-04.md` | Gemma 4 51.3 → 47.5 ms/tok + roofline (14.85 ms/tok, 67 tok/s) |
| `research/nemotron3_nano_30b_a3b_bringup_plan.md` + HANDOFF | Nemotron G1 status (not measured tok/s yet) |
| `models/`-era memory entries (linked, not duplicated) | Legacy single-chip demos (Llama, Qwen2.5, SmolLM) |

Models / rows:

1. **Qwen3.6-27B-A3B (dense, TP across 4 P150s)** — production CB +
   prefix cache; richest data.
2. **Qwen3.6-35B-A3B (hybrid GatedDeltaNet + MoE)** — 6-step perf
   trail, B>1 blocked on task #162.
3. **Gemma 4 12B (base + IT)** — vocab-shard lm_head landed, dev
   harness benchmarked.
4. **Nemotron-3 Nano 30B-A3B (Mamba2 hybrid)** — in progress; G1
   single-core PASS; no end-to-end tok/s yet — mark "(not measured)".
5. **Legacy single-chip demos** — link to `REPRODUCE.md` table; don't
   duplicate. Just one summary row per family.

Anti-duplication: when a number lives in `HANDOFF.md` and the perf
summary, the summary cites HANDOFF as source.

---

## Task 3 — Consolidate `scratch/legacy-demos/` into `archive/`

`scratch/legacy-demos/` exists and contains:

- `demo_gpt2.py` — top-level GPT-2 demo (pre-pivot tt-xla naming)
- `demos/` — 12 demo files (chat/serving/benchmark/qwen-traced)
- `generate_moe_qwen15.py` — Qwen-1.5-MoE generation script
- `PLAN_pre_pivot.md` — the pre-pivot master plan (April 2026)

These are pre-pivot demos that belong with the rest of `archive/`. No
active doc references `scratch/legacy-demos/`. Action:

```bash
git mv scratch/legacy-demos archive/scratch_legacy_2026-06-05
rmdir scratch                           # empty after the move
```

Add a one-line `archive/scratch_legacy_2026-06-05/README.md` describing
the bucket if one doesn't already come along. Update:

- `CONTRIBUTING.md` line 40 (`scratch/   legacy demos kept for reference`) — drop the row from the layout block.

Confirmed `grep -rln 'scratch/legacy-demos'` finds zero active-doc
hits, so no other text needs updating.

---

## Task 4 — Spot-check round-1 polish for post-Mamba2 staleness

Recent commits since the round-1 polish (`d4fa2cc`, `978f23e`,
`b2c4ccc`, `9a015d8` — Mamba kernel G1 work) shipped:

- G1 single-core Mamba2 SSD decode kernel at modes 1–5 PASS
  (`b2c4ccc`, `9a015d8`)
- G1 day-4.5 mode=4 PASS at cos 0.999998 (`978f23e`)

What the round-1 polish should now mention:

| File | Update |
|---|---|
| `README.md` §"Models brought up" | Nemotron-3 row already says "In progress — owned Mamba2 SSD kernel (G0→G4)". UPDATE to "G1 single-core kernel complete; G2 multi-core next." |
| `README.md` §"Repo layout" | `experiments/owned_ops/` description — keep generic; the per-op README under `nemotron3_mamba2_decode_owned/` carries the G1 detail |
| `REPRODUCE.md` | Owned_ops kernel-gate table mentions the two production ops; **add row** for `nemotron3_mamba2_decode_owned` with "G1 single-core PCC > 0.999 (mode=5 production)" or similar |
| `CONTRIBUTING.md` | No change needed — Mamba2 follows the existing "Adding a custom kernel" recipe |
| `wiki/README.md` | Already indexes wiki 65 (Mamba SSM primer) + 66 (Blackhole kernel anatomy). No change needed. |

These are small additions, not rewrites.

---

## Execution order (commits)

1. `docs(plan)` — drop this plan doc.
2. `docs` — Task 1a/1b/1c edits to active docs (README, CONTRIBUTING,
   research/* prose). One commit for top-level docs, one for
   research/wiki prose. ARCHIVE `research/06_hypotheses_to_test.md`
   via `git mv` to `archive/superseded_research_2026-06-04/`
   (it pre-dates the pivot).
3. `docs(perf)` — add `research/perf_summary_2026-06-05.md`.
4. `chore(archive)` — `git mv scratch/legacy-demos
   archive/scratch_legacy_2026-06-05/`; add the bucket README; drop
   `scratch/` row from CONTRIBUTING repo map; `rmdir scratch`.
5. `docs(readme)` — Nemotron G1 status update on README +
   REPRODUCE owned-kernels table.

---

## Items flagged for human judgment

1. **`HANDOFF.md`** — explicitly out of scope per the brief; some `qb1`
   prose references and `~/tt-xla` paths persist there. Operator-curated;
   leave alone.
2. **`CLAUDE.md`** — keeps the "originally scoped as a JAX/XLA backend"
   sentence by design. Project instructions; out of scope.
3. **`scratch/` after the move** — once `legacy-demos/` is moved out,
   `scratch/` is empty. Plan removes the directory and the
   CONTRIBUTING row. If someone wants to keep `scratch/` as a
   gitignored scratchpad, leave the directory but add a `.gitkeep`.
   Current plan: drop it.
4. **Wiki "Foundations: JAX/XLA (Wiki 01–05)"** — the wiki index
   keeps the JAX/XLA chronology because the wiki is the project's
   journal-as-learning-log. NOT in the framework-reference scrub.
5. **`research/06_hypotheses_to_test.md`** — proposed ARCHIVE. The
   doc's H1 ("TT-XLA is slow") is no longer the active hypothesis
   ladder. Parent can override and keep it as historical context.
