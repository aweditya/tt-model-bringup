# Memory bank audit — 2026-05-14

## Scope
Audited the entire memory bank at `/Users/adityasriram/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/memory/` (119 files) for gaps, ambiguities, and missing canonical commands. Triggered by recurring agent failures with wrong tooling (`nvidia-smi` for Tenstorrent, `curl http://` for a Unix-socket server) and the newly-written `reference_how_to_run_stuff.md` covering canonical operational commands.

## Files audited
- `MEMORY.md` (cover-to-cover index)
- `reference_how_to_run_stuff.md` (newly written canonical-commands doc)
- All `reference_*.md` (12 files): `reference_remote_host.md`, `reference_inline_script_helpers.md`, `reference_hf_oracle_pattern.md`, `reference_hf_token.md`, `reference_research_sources.md`, `reference_tracy_build_qb1.md`, `reference_p150_roofline_priority.md`, `reference_kimi_glm_bringup_menu.md`, `reference_multi_chip_*.md`, `reference_tt_metal_fused_kernels.md`
- All non-negotiable feedback notes referenced at bottom of MEMORY.md (15 files)
- All `project_*.md` (4 files)
- Selected long-tail feedback notes referenced from `reference_how_to_run_stuff.md` (mesh_recovery, paged_greedy_drift, python_stdout_buffering, etc.)
- All 5 unindexed feedback files (see "Missing index entries" below)

Approximate audit coverage: ~60 of 119 files read in full, ~30 grep-scanned for stale-tooling keywords. Audit found no surviving `ssh tenstorrent` references in body text (only acknowledged as gone), no `nvidia-smi` / `CUDA_VISIBLE` / `/tmp/` leaks outside the new how-to-run note, no AMD/ROCm contamination, no TPU references that imply we're targeting TPU.

## Gaps found, categorized

### 1. Wrong-tooling assumptions
NONE in body text outside `reference_how_to_run_stuff.md` (intentional). Three files (`feedback_non_negotiables.md`, `MEMORY.md`, `reference_how_to_run_stuff.md`) contain `nvidia-smi` only inside markdown comparison tables to discourage its use — that's correct.

### 2. Stale host claims — "two chips" / "qb1 is fallback" / "ssh tenstorrent gone"
- `feedback_non_negotiables.md` line 11 (pre-fix): "qb1 host has two Blackhole chips" — wrong, qb1 has 4. Also said "ssh qb1" as the only host, missing qb2.
- `project_branchIII_27b_complete.md` line 29 (pre-fix): "Active host: qb2... qb1 is fallback" — stale per current dual-host policy.
- `MEMORY.md` bottom-section line 99 (pre-fix): "Remote Host... `ssh qb1` (two chips, use one for now)" — wrong, and doesn't mention qb2.

### 3. Stale path advice — `pjrt_plugin/scripts/` vs `experiments/utils/`
- `feedback_non_negotiables.md`, `feedback_no_inline_scripts.md`, `feedback_permanent_scripts.md` all reference `pjrt_plugin/scripts/` and `pjrt_plugin/tests/` as the only canonical script location. As of 2026-05-14 the de facto canonical location for one-off probes is `experiments/utils/` (verified via repo `ls` — 60+ probe files there).

### 4. Missing operational cross-links
- `feedback_consult_docs_before_acting.md` — covers the semantic doc-consultation rule but doesn't point to `reference_how_to_run_stuff.md` for the operational layer.
- `reference_inline_script_helpers.md` — covers helper templates but doesn't point to the canonical operational doc.
- `reference_research_sources.md` (23 days old) — doesn't mention `experiments/.refs/tt-metal/` as the primary local reference, and doesn't link to `feedback_consult_docs_before_acting.md`.
- `reference_remote_host.md` — mentioned `pkill -9 -f serve.server` without warning that on qb2 this WEDGES THE FABRIC (already documented in `feedback_mesh_recovery_after_kill.md` but the two notes weren't cross-linked).

### 5. Missing memory index entries — 5 unindexed files
Before audit, the following lived on disk but were NOT in MEMORY.md:
1. `feedback_deallocate_unblocks_multistep_tp.md` (mandatory deallocate-after-last-use rule on mesh)
2. `feedback_p6_step2_hangs.md` (multi-step hang at step 2)
3. `feedback_paged_refactor_constraints.md` (full constraint map for paged refactor of server_tp.py)
4. `feedback_rope_scaling_long_context.md` (refutes "YaRN is the long-context bug" hypothesis)
5. `feedback_update_cache_tensor_api_gap.md` (our ttnn build has no `cur_pos_tensor=` — must use paged variant for trace)

### 6. /tmp / scratch-directory guidance — outdated
- `feedback_no_tmp.md` listed `pjrt_plugin/scratch/` and `experiments/scratch/` but missed `.cache/` (which is the de facto convention used everywhere now — server logs, probe results, tracy traces).

### 7. Conflicts between notes
NONE escalated. Two near-conflicts resolved by note-stamping:
- `feedback_p1_sdpa_decode_breaks_on_mesh.md` says SDPA decode FAILS on mesh; `feedback_mesh_paged_sdpa_works.md` (top of MEMORY.md) supersedes it with the program_config fix. Already cross-linked at top of index — no fix needed.
- `feedback_non_negotiables.md` said "single device for now" while `reference_remote_host.md` documented the dual-host policy. Fixed in non_negotiables.md to clarify "single device PER EXPERIMENT".

## Fixes applied

| File | Lines / Section | Change |
|---|---|---|
| `feedback_non_negotiables.md` | lines 10-14 | Replaced "qb1 host has two Blackhole chips" + single-host policy with current dual-host (qb1=4 chips no fabric, qb2=4 chips with fabric). Added cross-links to `reference_remote_host.md`, `reference_how_to_run_stuff.md`, `reference_inline_script_helpers.md`. Added 2026-05-14 note pointing to canonical commands doc. |
| `feedback_non_negotiables.md` | last line | Updated example from `ssh qb1 'python3 path'` to `ssh qbX '.venv/bin/python path'` and replaced `pjrt_plugin/scratch/` with `.cache/`. |
| `feedback_no_inline_scripts.md` | full body | Added qb2 to "ssh qb1" mentions. Added `experiments/utils/` as canonical script location. Appended 2026-05-14 note linking to `reference_how_to_run_stuff.md` + `reference_inline_script_helpers.md`. |
| `feedback_consult_docs_before_acting.md` | bottom | Appended 2026-05-14 note clarifying the rule covers the SEMANTIC layer; pointing operational-layer failures (nvidia-smi, curl http) to `reference_how_to_run_stuff.md`. |
| `reference_inline_script_helpers.md` | bottom | Appended 2026-05-14 note linking to `reference_how_to_run_stuff.md` with a one-line warning about the documented pattern of agents using wrong tooling. |
| `project_branchIII_27b_complete.md` | line 29 | Stamped "qb1 is fallback" as 2026-05-12 state, then added 2026-05-14 note explaining current dual-host policy with cross-links. |
| `reference_research_sources.md` | bottom | Appended 2026-05-14 note pointing to `experiments/.refs/tt-metal/` as the primary local reference + cross-links to `feedback_consult_docs_before_acting.md` and `reference_how_to_run_stuff.md`. |
| `feedback_permanent_scripts.md` | bottom | Appended 2026-05-14 note clarifying `experiments/utils/` is the canonical probe location. |
| `feedback_no_tmp.md` | last line + new para | Added `.cache/` to the canonical scratch-directory list with a 2026-05-14 note that `.cache/` is the de facto convention. |
| `reference_remote_host.md` | middle | Warned that `pkill -9 -f serve.server` wedges fabric on qb2; prefer clean shutdown via `serve.sh stop`. Added 2026-05-14 cross-link to `reference_how_to_run_stuff.md` + explicit "ssh tenstorrent is gone" line. |
| `MEMORY.md` | line 22 (long-context cluster) | Added entry for `feedback_rope_scaling_long_context.md`. |
| `MEMORY.md` | line 75 (TP cluster) | Added 4 entries: `feedback_p6_step2_hangs.md`, `feedback_deallocate_unblocks_multistep_tp.md`, `feedback_paged_refactor_constraints.md`, `feedback_update_cache_tensor_api_gap.md`. |
| `MEMORY.md` | line 100 | Updated "Project Non-Negotiables" description to mention `qb1`/`qb2` + "one chip per workload by default". |
| `MEMORY.md` | line 102 | Updated "Remote Host" description from "ssh qb1 (two chips, use one for now)" to current dual-host summary. |
| `MEMORY.md` | line 124 | Updated "No Inline Scripts" description from "ssh qb1" to "ssh qb1/qb2". |

## Verification
After fixes, ran `diff` between `grep`-extracted memory references in MEMORY.md and `ls memory/*.md`. **Zero unindexed files remain.**

## Recommended follow-ups
1. Consider creating a `reference_canonical_paths.md` consolidating `.cache/`, `experiments/utils/`, `experiments/serve/`, `pjrt_plugin/`, `research/`, `wiki/` into one short note. Currently this info is scattered. (Low priority — `feedback_no_tmp.md` + `reference_inline_script_helpers.md` now cover the basics with cross-links.)
2. `reference_research_sources.md` is still 23 days old; could be refreshed to remove the "To discover" placeholder, but doing so would erase historical context. Left as-is.
3. Several feedback notes carry "memory is N days old" system reminders — those are auto-stamped by the harness; no action needed unless content is actually stale.
4. None of these fixes require remote execution. All audit deliverables are local-only research, per non-negotiable #1.

## Conflicts not resolved
None requiring user escalation.
