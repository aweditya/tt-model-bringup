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

## Status — ALL DONE

| # | Step | Result |
|---|------|--------|
| 1 | Living plan doc | done |
| 2 | Compress MEMORY.md 44 KB → 19.8 KB | done; -55%; all 146 links live |
| 3 | Merge repro PR into main + push | done; rebased clean; HEAD `bdb64df` |
| 4 | Prune stale `tt-xla/` worktree | done |
| 5 | Stop qb2 prod TP server | done |
| 6 | Fresh clone repo on qb2 | done at `~/tt-model-bringup-fresh` |
| 7 | Run setup (uv sync + ttnn install) | done; uv 0.11.13, torch 2.12.0, ttnn 0.69.1.dev0 |
| 8 | Legacy 6 demos | all PASS — 60/64/67/73/76b ≤ baseline, 80 within 2 tok/s |
| 9 | Demo A (single-chip 27B) | PASS — coherent "Paris." + `<think>` block, 4.01 tok/s cold |
| 10 | Demo B (4-chip TP 27B) | PASS — 13.01 tok/s warm (≥ prod 12.93), required symlink workaround for socket bug |
| 11 | 35B-A3B MoE smoke on qb1 | PASS — ` Paris` first token + 24-tok coherent decode at 522 ms/tok |
| 12 | Restart qb2 prod in `~/tt-xla/` | done; 12.97 tok/s baseline restored |
| 13 | Document + commit | this doc + REPRODUCE.md + protocol.py fix (commit `b17d33e`) |

### Findings so far (will fold into final doc)

- **HF token not configured on qb2** (matches `reference_hf_token_setup.md`). Legacy demos still pass — meta-llama → unsloth fallback inside each script. Rate-limit warning prints but downloads complete.
- **README says `TT_BUILD_DIR=build_Release`** but qb2's serve_tp.sh defaults to `build_tracy_gcc12_nodist`. Both build dirs exist; the `uv pip install -e $TT_METAL_HOME` picks the ttnn shipped in TT_METAL_HOME (0.69.1.dev0). Imports + device open cleanly.
- **Both serve.sh / serve_tp.sh respect `PROJECT_ROOT`** for log/launch paths.
- **35B-A3B weights not cached on qb2** — MoE smoke moves to qb1 (user-confirmed).
- **REAL BUG — `experiments/serve/protocol.py:8-11` hardcodes `~/tt-xla/.cache/...`** for SOCKET_PATH / PID_PATH / CACHE_DIR. So a fresh-clone server writes its socket + pid to the LEGACY `~/tt-xla/.cache/` directory, NOT to its own `.cache/`. Visible consequences:
  - `serve.sh stop` (which respects PROJECT_ROOT) can't find the pid file → reports "server not running" even when it is. We had to SIGTERM by pgrep.
  - Two servers from different clones would clobber each other's socket + pid silently.
  - The fresh-clone client still works only because *it* also reads the hardcoded path, so client + server agree by accident.
  Fix: derive `CACHE_DIR` in `protocol.py` from `os.environ.get("PROJECT_ROOT", <__file__-based default>)`. ~10 lines, low-risk (default resolves to `~/tt-xla/.cache` for the prod checkout). Land as a follow-up after this sanity check; not blocking today.
- **Demo A → Demo B transition gotcha.** Without an explicit chip-free check between Demo A teardown and Demo B start, server_tp.py boot can race server.py's TLB cleanup and crash with `tt_tlb_alloc failed with error code -12` (ENOMEM). Recovery: `tt-smi -r 0,1,2,3`. The pid-file bug above is the upstream cause — `serve.sh stop` thinks it succeeded, but the server is still running.

## Known constraints

- qb2's 4 chips can only host one ttnn process at a time (ttnn opens all
  chips even with `device_ids=[0]`). Concurrent ttnn = SIGBUS. So the prod
  server must be down for the whole sanity-check window.
- `pkill -9` corrupts mesh fabric — recover with `tt-smi -r 0,1,2,3`. Always
  use `serve_tp.sh stop` (SIGTERM).
- MoE bringup work currently targets qb1 mesh; the qb2 path for 35B-A3B may
  not have a one-shot entry script yet. If so, the smoke step documents the
  gap rather than fakes a pass.

## Followups (post-session)

- **MEMORY.md edit:** add a `feedback_protocol_hardcoded_paths.md` topic
  file referencing commit `b17d33e` if this bites us again. Right now the
  history is captured in REPRODUCE.md and the commit message — that's
  probably enough.
- **README:** mention the `serve.sh stop` pid-file race (the bash wrapper
  trusts the python server to write the pid file after bootstrap; if you
  call `stop` before that's written, it reports "not running" and the
  python is still alive). Either document the workaround (pgrep + SIGTERM)
  or have the bash wrapper also write a `.pid.launch` *and* check it on
  stop.
- **35B-A3B repro:** the smoke `experiments/utils/decode_smoke_35b_ttnn.py`
  is the right entry point but its docstring hardcodes `~/tt-xla` paths
  and `build_Release`. After the next MoE milestone, fold it into a
  README "Demo C" section with a clean fresh-clone recipe.
- **qb1 is rsync-managed, not git-cloned.** The fresh-clone test on qb2
  proves the cold-start story; qb1's `~/tt-xla/` doesn't have `.git/` so
  it can't `git pull` updates. Possible cleanup: switch qb1 to a real git
  clone if we want symmetric maintenance, or accept that qb2 is the
  canonical "deployable" host.
- **HF token still not configured on qb2 / qb1.** Demos work via the
  unsloth fallback paths, but cold downloads are rate-limited. Lower
  priority — would only bite a new host bring-up.

## Pointers (for context-recovery)

- Repro PR commits: `bc2dbf9` → `d26e022` on `origin/repro/uv-readme-demos`.
- Prior fresh-clone validation: `.cache/repro_validation_2026_05_21/readme_fix.patch`
  (qb1 only; this run extends to qb2).
- REPRODUCE.md `Re-verified on qb1 (2026-05-21)` table is the baseline to
  compare against.
- CLAUDE.md non-negotiables stand; ssh `tenstorrent` is gone; `qb1`/`qb2`
  are the only hosts.
