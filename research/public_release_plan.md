# Public-release plan — tt-model-bringup

## Goal

Take what we have ("Qwen3.6-27B dense + 35B-A3B MoE brought up on Blackhole,
production CB chat server, custom owned_* kernels") and make it inviting,
legible, and contributable for a stranger: a TT engineer, a hobbyist with a
P150, or a researcher who never had qb1 access. **Polish, not rewrite.** No
architectural changes; the CB stack (`research/production_server_plan.md`
P5.1, shipped 2026-05-30) is load-bearing — every recommendation here is
additive or strips dead weight. The pre-existing code-maintainability sweep
(`research/code_maintainability_audit.md`) is a *separate* prereq and is
already queued; this plan does not duplicate it.

## Current-state assessment

| Area | What's good now | What's missing for release |
|---|---|---|
| `README.md` | §"Chat server (production)" (lines 261-348) and §"Setup" (lines 51-156) are tight, copy-pasteable, and accurate. Quickstart block is real. | Top 50 lines are written *for the user* — open in someone else's tab and "what is this and why should I care" is not answered in the first screenful. Need a 60-second framing + a "what works" table at top. |
| `CONTRIBUTING.md` | Repo map, dev loop, canary gate, non-negotiables already in (lines 1-114). | No "file an issue" / "open a PR" sections; no perf-claim methodology (CLAUDE.md non-negotiable #1 isn't mirrored). No PR template reference. |
| `HANDOFF.md` | Single-page, current perf numbers, hardware ceiling, prod path. Perfect cold-start for *us*. | Visitor-targeted version (or rename / re-frame) is needed; "HANDOFF" reads as internal. Keep file, but link as "current status / what's next" from README. |
| `REPRODUCE.md` | Tested environment table is solid (lines 11-26); reproduces three servers + 6 legacy demos with measured numbers. | Assumes the reader has a P150 box ready; the "I just want to read what was done" path doesn't exist. |
| `LICENSE` | **Missing entirely.** | Blocker for any external use. C++ owned-op files already carry `SPDX-License-Identifier: Apache-2.0` (e.g. `experiments/owned_ops/qwen36_gdn_decode_owned/*.cpp`). |
| `.github/` | `ci.yml` runs ruff + compileall + deploy-sync check on push/PR. | No issue templates, no PR template, no `CODE_OF_CONDUCT.md`, no `CITATION.cff`. |
| `experiments/owned_ops/README.md` | Excellent index — both kinds of kernel clearly distinguished, install command, per-op `INTEGRATION.md` cross-refs. **Ship as-is.** | none |
| `models/README.md` | Good index of legacy demos. | none |
| `archive/` | 122 archived research docs + legacy pjrt_plugin. Clean separation already in place. | One sentence in README pointing at it ("here's where the history lives") — already there at lines 8-11; good. |
| Naming | Repo is `aweditya/tt-model-bringup` (renamed 2026-05-19). README is honest about the legacy `tt-xla` dir. | Repo *description* (the GitHub one-liner) + topics aren't set yet. Local dir name (`tt-xla/` + `~/tt-xla` on hosts) deliberately frozen per `CONTRIBUTING.md` non-negotiable #7 — keep that frozen, just clarify in README why. |

---

## Punch list

### P0 — must-fix before tagging v1.0

1. **Add `LICENSE` (Apache-2.0).** New file at repo root. Apache-2.0 chosen
   because (a) tt-metal is Apache-2.0, (b) the owned-op C++ already declares
   it (`experiments/owned_ops/qwen36_gdn_decode_owned/*.cpp:1`), (c) provides
   patent grant — owned kernels are novel work. MIT is the only other
   reasonable pick; Apache-2.0 wins on alignment.

2. **README landing-page rewrite (top 50 lines only).** Insert before
   current line 17 (`## Quickstart`):
   - **One-line tagline** (e.g. "Tenstorrent Blackhole bringup of Qwen3.6-27B
     (dense) and Qwen3.6-35B-A3B (MoE) — production CB chat server + custom
     `owned_*` compute kernels.").
   - **"What works today" table** (Model / Path / Status / Perf / Caveat) —
     reuse the table in `HANDOFF.md` lines 21-31; promote it to README.
   - **"Why this exists"** (1 short paragraph): research project, JAX/XLA
     pivot to direct TT-Metal, custom kernels, CB stack.
   - **"Try it"** → link to existing Quickstart block (line 17, unchanged).
   - **"Read the design"** → links to `HANDOFF.md`, `research/README.md`,
     `experiments/owned_ops/README.md`.
   - **CI badge** (see P1 #8).
   The rest of README (§Host matrix, §Setup, §Chat server, §Long context,
   §Troubleshooting) stays verbatim — they're solid.

3. **Fix the qb1/qb2-as-only-path assumption in README.** Lines 158-200
   (Demo A / Demo B) hard-code `ssh qb1` / `ssh qb2`. A community visitor
   with their own P150 can't follow this. Fix: replace the literal `ssh qb1`
   on line 161 with a one-line note ("on a TT host with a P150 — `ssh qb1`
   for us; substitute your own host or run locally") and the same for line
   183. Don't touch our internal scripts (`Makefile` `TT_HOST ?= qb1`,
   `scripts/run_remote.sh` — those are dev-loop ergonomics, frozen per
   non-negotiable #7).

4. **Set GitHub repo description + topics.** Description: same one-line
   tagline as README. Topics: `tenstorrent`, `blackhole`, `llm`, `qwen3`,
   `inference`, `moe`, `continuous-batching`, `tt-metal`, `tt-nn`.
   (Done via the GitHub web UI / `gh repo edit`.)

### P1 — should-fix soon after release

5. **`CONTRIBUTING.md` polish (additive only).** Add three new sections
   after current line 114:
   - **Filing an issue** — point to issue templates (P1 #6); ask for host
     (qb1/qb2/your own P150), `tt-smi -s` output, the failing log.
   - **Proposing a perf change** — mirror `CLAUDE.md` non-negotiable #1
     into the public doc: "back every perf number with roofline / op-count
     math; isolate (single-op probe), then integrate (full forward), then
     measure (`experiments/cb/bench/trace.py`). Don't ship a < 5% claim;
     within-noise."
   - **Adding a custom kernel** — already there (lines 81-88); good. Just
     add one sentence pointing to a worked example (`qwen36_decay_gate_decode_owned/INTEGRATION.md`).

6. **Issue / PR templates.** New files:
   - `.github/ISSUE_TEMPLATE/bug_report.md` — fields: host (qb1/qb2/your own),
     tt-metal SHA (paste `cat tt-metal-sha.txt`), `make check` output,
     reproducer command, error log.
   - `.github/ISSUE_TEMPLATE/perf_regression.md` — fields: baseline path
     (`experiments/cb/bench/trace.py`?), measured tok/s, expected tok/s,
     `tt-smi -s` core count (firmware silent-downgrade catch — see
     `feedback_p150_firmware_core_check`).
   - `.github/ISSUE_TEMPLATE/question.md` — pointer to wiki/.
   - `.github/PULL_REQUEST_TEMPLATE.md` — checklist: ran `make lint`; if
     touched `server_tp.py` / `ondevice_27b.py` / `server_tp_cb.py`, pasted
     the canary output (`experiments/cb/validate/forward.py`); for perf PRs,
     pasted bench numbers.

7. **`CODE_OF_CONDUCT.md`.** New file. Adopt Contributor Covenant 2.1
   verbatim (the standard); contact email = `aweditya@gmail.com` (per
   `userEmail` in current env). 1 line of customization, ~120 lines total.

8. **CI status badge** in README (insert in the rewritten top-50). Source:
   `https://github.com/aweditya/tt-model-bringup/actions/workflows/ci.yml/badge.svg`.

9. **`CITATION.cff`.** New file at repo root. Fields: title, authors
   (Aditya Sriram + advisor / collaborators TBD), date-released, version
   `0.1.0` (matches `pyproject.toml:3`), URL, type=software. Lets researchers
   `cite this repository` from the GitHub sidebar.

10. **Visible attribution / prior art block** in README (under the
    rewritten landing page, or in a new `## Acknowledgements` section at
    the bottom). Three callouts:
    - **tt-metal / tt-nn** (Tenstorrent, Apache-2.0) — the runtime + the
      ops we layered on; `tt-metal-sha.txt` pins the build.
    - **Qwen3.6 model family** (Qwen team, Tongyi) — link model card.
    - **vLLM-style continuous batching** — `research/27b_cb_scope.md`
      already credits the Orca paper; surface it in the README.

### P2 — nice-to-have

11. **`examples/` directory.** Not strictly needed — `scripts/chat.py`
    (the stdlib chat TUI) is already a "hello world" demo. Recommend:
    *don't create `examples/`*; instead add one line in the README right
    under the chat-server `curl` block: "see `scripts/chat.py` for a
    minimal stdlib chat client (no external deps)." That's enough.

12. **`SECURITY.md`.** New file, ~15 lines. "This is a research codebase;
    no formal security guarantee. For sensitive disclosures, email
    `aweditya@gmail.com`." Tiny, expected on public repos, satisfies
    GitHub's repo-health check.

13. **`research/README.md` light edit.** Already exists per `CONTRIBUTING.md`
    line 59-60. Confirm it reads as a real index (one line per doc) so the
    122 files in `research/archive/` don't scare off a visitor. (No change
    if already so.)

14. **Trim README's perf-numbers density.** Current README lines 202-204
    ("**12.93 tok/s** with `num_links=2` all_reduce + `owned_gdn` +
    `owned_decay_gate`") read as internal jargon. Either keep but add a
    sentence ("see `research/multi_chip_optimizations_menu.md` for the
    optimization ladder"), or compress to "**12.93 tok/s** (see HANDOFF for
    breakdown)." Don't strip the numbers — they're the headline.

15. **(Optional) Python source license headers.** Apache-2.0 doesn't
    *require* per-file headers (a `LICENSE` file at root is enough), but
    tt-metal style is to add them. Cost: a one-shot pass to prepend
    `# SPDX-License-Identifier: Apache-2.0` to every `.py` in
    `experiments/serve/`, `experiments/cb/`, `scripts/`, `models/`. Low
    value, ~20 min of work. Skip for v1.0; revisit if a contributor asks.

---

## Recommended release sequence

**Wave 1 — license + landing (1 evening).** P0 #1, #2, #3, #4. Tag `v0.9.0-pre`
to mark the public-readable state. Don't announce yet.

**Wave 2 — contributor surface (one weekend).** P1 #5, #6, #7, #8, #9, #10.
Tag `v1.0.0`. Now safe to announce — `gh repo edit --visibility public`,
post to TT Discord / r/MachineLearning, tweet from project handle if any.

**Wave 3 — polish (rolling).** P2 #11-14. Apply opportunistically; no
release gate.

**Hard prereq (parallel track):** the code-maintainability sweep
(`research/code_maintainability_audit.md`) — the P0 there (`scripts/deploy.sh`
default-arg-list gap) and the P1 dead-debug-block strips in `server_tp.py`
should land before v1.0 so a fresh-clone visitor isn't reading 100 lines of
dead `ccl_debug` instrumentation. That sweep is its own PR series; this
release plan does not duplicate it.

---

## Names + identity decisions

- **Local dir name (`tt-xla/`):** keep frozen per `CONTRIBUTING.md`
  non-negotiable #7. Renaming has real cost (`~/tt-xla` baked into
  `scripts/run_remote.sh:22`, `scripts/deploy.sh`, every server bootstrap
  log, Claude settings). README already explains it (lines 8-11). Add one
  more sentence in the rewritten landing: "you'll see `tt-xla/` in clone
  paths — that's the legacy name from when this targeted JAX/XLA; the
  remote dir on the TT hosts is also `~/tt-xla/`."
- **Repo name:** `aweditya/tt-model-bringup`. Keep.
- **Repo description (GitHub):** "Tenstorrent Blackhole bringup of Qwen3.6
  (27B dense + 35B-A3B MoE) — direct TT-Metal + custom `owned_*` kernels +
  production continuous-batching chat server."
- **Topics:** `tenstorrent`, `blackhole`, `llm`, `qwen3`, `inference`,
  `moe`, `continuous-batching`, `tt-metal`, `tt-nn`, `apache-2`.
- **License identifier in `pyproject.toml`:** add `license = { text =
  "Apache-2.0" }` to the `[project]` table (`pyproject.toml:1-7`). PEP 621.

---

## Pre-release checklist (flat, in order)

| # | Item | Effort | Owner gate |
|---|---|---|---|
| 1 | Land code-maintainability P0 (deploy.sh fix) + P1 (dead-debug strip in server_tp.py) | 2-3 h | code review |
| 2 | Add `LICENSE` (Apache-2.0) | 5 min | — |
| 3 | Add `license = { text = "Apache-2.0" }` to pyproject | 1 min | — |
| 4 | Rewrite README top 50 lines (tagline + status table + why + try-it + design-links + CI badge) | 1 h | self-review |
| 5 | Replace literal `ssh qb1` / `ssh qb2` in Demo A / B headings with "on your TT host" framing | 10 min | — |
| 6 | Set GitHub repo description + topics (web UI or `gh repo edit`) | 5 min | — |
| 7 | Tag `v0.9.0-pre`, sanity-check clone-and-run from a non-qb host | 1-2 h | manual |
| 8 | Add `CONTRIBUTING.md` 3 new sections (issues / perf / kernel example) | 30 min | — |
| 9 | Add 3 issue templates + 1 PR template under `.github/` | 45 min | — |
| 10 | Add `CODE_OF_CONDUCT.md` (Contributor Covenant) | 5 min | — |
| 11 | Add `CITATION.cff` | 15 min | confirm author list |
| 12 | Add `SECURITY.md` | 5 min | — |
| 13 | Add acknowledgements block to README (tt-metal / Qwen / vLLM/Orca) | 10 min | — |
| 14 | Tag `v1.0.0`; announce | — | — |

**Total to v1.0.0: ~5-6 hours of doc/config work + the maintainability sweep
(separate PR series, already planned).**
