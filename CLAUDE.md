# tt-model-bringup: Tenstorrent Blackhole LLM bringup (Qwen3.6-family)

## Read First

**For any post-compaction or fresh session: read [`HANDOFF.md`](HANDOFF.md) first.** It is one page and contains the current perf number, the hardware-ceiling reference, the production code path, and what to do next. Don't grep around to figure out where things are.

## Project Overview
Exploratory research project for Stanford CS440LX. **Originally scoped as a JAX/XLA backend (hence the legacy `tt-xla` name); the work pivoted to direct TT-Metal model bringup + custom compute kernels for Qwen3.6 on Tenstorrent Blackhole.** Currently working on Qwen3.6-35B-A3B MoE bringup on qb1 (1,4) P150 mesh. Per-token perf number lives in `HANDOFF.md`; the **target is the hardware BW ceiling**, not any prior model's number.

GitHub: `aweditya/tt-model-bringup` (renamed 2026-05-19). Local working dir stays `tt-xla/` (renaming breaks Claude settings); remote dirs on qb1/qb2 stay `~/tt-xla/` (renaming would break too many rsync paths).

## Non-Negotiables

1. **ALWAYS think first, act later.** Plan, plan, and more planning before implementing. No hand-wavy arguments — every claim must be grounded in an experiment or concrete evidence. If we have a hypothesis, we test it.
2. **Research-driven workflow.** Most work is research, Q&A, and building a wiki of practice-exam-style questions. Learning by building.
3. **No code bloat.** Spend more time thinking, less time implementing. When we implement, it's correct and concise.
4. **Remote execution only — `ssh qb1` or `ssh qb2`.** All experiments and code run on a remote host. (Previous host `ssh tenstorrent` is no longer available.)
5. **Two hosts now available**: `qb1` (4 P150s, **inter-chip fabric WORKS as of 2026-05-21** — both single-chip and multi-chip TP workloads) and `qb2` (4 P150s with working fabric, hosts production Qwen3.6-27B TP server). Either host can run TP work; prefer qb1 for experimental mesh work so qb2 prod stays up. **Owned kernels** (`qwen36_gdn_decode_owned`, `qwen36_decay_gate_decode_owned`) are present in BOTH hosts' ttnn builds — verified on qb1 2026-05-28 (`ttnn.experimental` exposes all 8 `qwen36_gdn_*` ops). The earlier "qb2-only" note was stale. NOTE: `qwen36_gdn_decode_owned` hard-asserts batch=1 (`state_logical[0] == 1`); batching it for continuous batching needs device-op + program-factory changes.
6. **No inline scripts** unless absolutely necessary. Write permanent files in `scripts/`, `experiments/`, or the relevant package.
7. **No `/tmp` for anything.** Use project directories for outputs, logs, caches, scratch — anything.
8. **Frequent commits.** Commit early and often.
9. **No local execution of device code.** The local machine is for editing, research notes, and wiki content only.

## Key Resources

### Tenstorrent
- https://tenstorrent.com — corporate site, product info
- TT-NN, TT-Metalium, LLK, TT-lang — Tenstorrent open-source repos on GitHub
- https://www.corsix.org/content/tt-wh-part1 — Corsix's 8-part Tenstorrent Wormhole blog series
- https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi — Programming Tenstorrent processors

### JAX / XLA
- https://github.com/jax-ml/jax
- https://github.com/openxla/xla
- Focus: how JAX has backends for different devices (CUDA, SYCL, MLX, etc.)

## Project Structure
```
tt-xla/
  wiki/           # Q&A wiki pages (learning-by-building)
  research/       # Raw research notes, scraped content summaries
  experiments/    # Code that runs on the Tenstorrent host
  CLAUDE.md       # This file
```

## PJRT Plugin Development (Custom JAX Backend) — ARCHIVED

The original goal was a custom PJRT plugin compiling JAX → ttnn. The project
pivoted to direct TT-Metal bringup; those sources now live under
`archive/legacy/pjrt_plugin/` (+ `archive/legacy/tt_jax/`), kept for reference.
Not on the active path. (Architectural reference at the time: applejax.)

## Workflow
1. Research a topic (scrape, read docs, explore code)
2. Formulate questions and hypotheses
3. Design minimal experiments to test hypotheses (run on remote host)
4. Document findings in wiki/ as Q&A entries
5. Commit frequently
