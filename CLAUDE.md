# TT-XLA: Tenstorrent Backend for JAX/XLA

## Project Overview
Exploratory research project for Stanford CS440LX. The goal is to understand Tenstorrent Blackhole hardware and JAX/XLA internals deeply enough to build a JAX backend for Tenstorrent devices.

## Non-Negotiables

1. **ALWAYS think first, act later.** Plan, plan, and more planning before implementing. No hand-wavy arguments — every claim must be grounded in an experiment or concrete evidence. If we have a hypothesis, we test it.
2. **Research-driven workflow.** Most work is research, Q&A, and building a wiki of practice-exam-style questions. Learning by building.
3. **No code bloat.** Spend more time thinking, less time implementing. When we implement, it's correct and concise.
4. **Remote execution only — `ssh qb1`.** All experiments and code run on the remote host. (Previous host `ssh tenstorrent` is no longer available — Blackhole was disconnected from it.)
5. **Single device for now.** The qb1 host has two Blackhole chips — stick to ONE device until we've saturated it. Once saturated, scale up to both.
6. **No inline scripts** unless absolutely necessary. Write permanent files in `pjrt_plugin/scripts/`, `pjrt_plugin/tests/`, or `experiments/`.
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

## PJRT Plugin Development (Custom JAX Backend)

Building a custom PJRT plugin to compile JAX programs to Tenstorrent ttnn. Key principles:

1. **Correctness first, performance next.** Every op must pass rigorous unit tests before integration.
2. **Expanding unit test suite.** Tests grow with every new op/feature. Never skip tests.
3. **Problem decomposition.** Think through the design before writing C++. Document non-trivial decisions.
4. **Reflection log.** Note design decisions, trade-offs, and things we'd do differently in `research/pjrt_reflections.md`.
5. **Reference: applejax.** Use the applejax interpretation-based PJRT plugin as architectural reference.

## Workflow
1. Research a topic (scrape, read docs, explore code)
2. Formulate questions and hypotheses
3. Design minimal experiments to test hypotheses (run on remote host)
4. Document findings in wiki/ as Q&A entries
5. Commit frequently
