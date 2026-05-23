# Housekeeping + fresh-clone sanity check — 2026-05-22

Living plan doc. Update as we progress. If context compacts, read this first
to resume.

## Goal

Two outcomes, both about lowering friction for a brand-new collaborator (or
a future cold-clone Claude):

1. **Housekeeping**: shrink MEMORY.md back under its soft limit, land the
   long-standing repro PR on `main` so a fresh `git clone` actually gets the
   reproducibility files, prune stale git metadata.
2. **Fresh-clone sanity check on qb2**: stop the prod TP server, do a
   from-scratch `git clone` into a new directory on qb2, follow the README
   verbatim, run every model we've brought up. Surface any breakage in the
   cold-clone story.

Success criterion: someone with `ssh qb2` access can clone the repo, run
`uv sync` + the tt-metal install step, and reproduce all four model paths
(legacy 6 demos, Qwen3.6-27B single P150, Qwen3.6-27B 4-chip TP, 35B-A3B MoE
smoke) without manual hand-holding beyond what's in the README.

## Decisions captured (2026-05-22)

- Merge `origin/repro/uv-readme-demos` (6 commits, adds pyproject.toml /
  uv.lock / README.md / .env.example / .gitignore updates) into `main` and
  push to origin. Without this, "anyone can clone" is a lie.
- Sanity check runs on **qb2**, not qb1 — the user wants the TP path
  exercised. Means stopping the prod TP server (~11 min cold-restart after).
- Scope of "all models": all four — legacy 6 demos, Demo A (single P150),
  Demo B (4× P150 TP), 35B-A3B MoE smoke.

## Status

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Living plan doc (this file) | in_progress | being written now |
| 2 | Compress MEMORY.md (44 KB → < 24 KB) | pending | one-line hooks; detail already in topic files |
| 3 | Merge repro PR into main + push | pending | rebase preferred; merge commit if conflicts |
| 4 | Prune stale `tt-xla/` worktree | pending | cosmetic |
| 5 | Status-check qb2, stop prod TP server | pending | use `serve_tp.sh stop`, never `pkill -9` |
| 6 | Fresh clone repo on qb2 to new dir | pending | e.g. `~/tt-model-bringup-fresh` |
| 7 | Run setup (uv sync + ttnn install) | pending | follow merged README verbatim |
| 8 | Run legacy 6 demos | pending | 60/64/67/73/76b/80 |
| 9 | Run Demo A (single-chip 27B) | pending | server.py path |
| 10 | Run Demo B (4-chip TP 27B) | pending | server_tp.py path |
| 11 | Run 35B-A3B MoE smoke | pending | may need a new entry point — flag the gap if so |
| 12 | Restart qb2 prod in original `~/tt-xla/` | pending | bootstrap ~5–10 min |
| 13 | Document results + commit | pending | update REPRODUCE.md table |

## Known constraints

- qb2's 4 chips can only host one ttnn process at a time (ttnn opens all
  chips even with `device_ids=[0]`). Concurrent ttnn = SIGBUS. So the prod
  server must be down for the whole sanity-check window.
- `pkill -9` corrupts mesh fabric — recover with `tt-smi -r 0,1,2,3`. Always
  use `serve_tp.sh stop` (SIGTERM).
- MoE bringup work currently targets qb1 mesh; the qb2 path for 35B-A3B may
  not have a one-shot entry script yet. If so, the smoke step documents the
  gap rather than fakes a pass.

## Open questions / followups (update as they surface)

- Does the merged README's setup actually work on a host that doesn't already
  have `~/tt-xla/` cached? The 2026-05-21 patch validated this on qb1; qb2
  may have different `$TT_METAL_HOME` or firmware quirks.
- Is `hf auth login` already wired on qb2? (Listed as "unauthenticated" in
  `reference_hf_token_setup.md` from 2026-05-21.)
- 35B-A3B is mid-bringup (B17 trace validated; long-context decode 100 toks
  coherent). What's the canonical fresh-clone command for it? May need to
  add one to README as a followup task.

## Pointers (for context-recovery)

- Repro PR commits: `bc2dbf9` → `d26e022` on `origin/repro/uv-readme-demos`.
- Prior fresh-clone validation: `.cache/repro_validation_2026_05_21/readme_fix.patch`
  (qb1 only; this run extends to qb2).
- REPRODUCE.md `Re-verified on qb1 (2026-05-21)` table is the baseline to
  compare against.
- CLAUDE.md non-negotiables stand; ssh `tenstorrent` is gone; `qb1`/`qb2`
  are the only hosts.
