# CS440LX presentation — outline + workstream tracking

Talk: "Bringing up modern LLMs on Tenstorrent Blackhole — what we did,
what worked, what didn't."

## Sections (target slide count)

1. **Project context + scope** (2 slides) — TT-XLA → tt-model-bringup
   pivot; Stanford CS440LX; one-laptop research → 1×4 P150 mesh.
2. **The bringup workflow / methodology** (3-4 slides) — staging
   ladder, HF oracle pattern, isolation probes, dev harness,
   REUSE mandate, two-phase trace warmup.
3. **Models brought up** (1 slide) — Qwen3.6-27B (DENSE TP), Qwen3.6-35B-A3B
   (MoE), Gemma 4 12B (Unified hybrid sliding/global). Tiny models
   in passing.
4. **What we reused vs reinvented** (2 slides) — TT-team patterns we
   forked (paged SDPA, all-gather/reduce, kernel templates) vs
   work that was original (custom owned_gdn for 35B DeltaNet,
   model-specific bringup recipe, dev harness, oracle infra, CB
   scheduler).
5. **Correctness** (2-3 slides) — cosine ladder gate, needle-haystack,
   teacher-forced vs free-run, bf16 noise floor.
6. **Performance optimizations catalog** (4-5 slides) —
   what worked (vocab-shard +8%, paged SDPA +62%, owned_gdn,
   two-phase warmup), what didn't (DRAM-sharded MLP -2.1×,
   async all-reduce, bf8 weights at this regime), and **why**.
7. **Throughput scaling** (2 slides) — single-seq vs CB at B=4/B=32,
   tok/s per-model.
8. **Challenges + bugs** (2 slides) — representative debugging stories
   (NCHIPS shadow, view-decay, multi-EOS, multi-snapshot HF cache,
   StopIteration in uvloop, etc.).
9. **Demo** (live or screenshots) — chat TUI, tool calls, Q&A latency,
   concurrency. Possibly Gemma image input as a stretch.
10. **Wrap + future work** (1 slide).

## Workstreams (subagents in flight)

| Agent | Output file | Status |
|---|---|---|
| 1. Workflow/methodology | `01_workflow.md` | dispatched |
| 2. Perf catalog with numbers | `02_perf_catalog.md` | dispatched |
| 3. TT-reuse vs reinvent | `03_reuse_vs_reinvent.md` | dispatched |
| 4. Per-model throughput audit | `04_throughput.md` | dispatched |
| 5. Challenges/bugs narrative | `05_challenges.md` | dispatched |

After agents return, I assemble into a slide deck (markdown / pptx).

## Live stress-test workstream (post 35B correctness)

- Multi-turn Q&A through the chat TUI (multiple rounds, retention).
- Concurrent client load — 4 simultaneous sessions, observe tok/s
  degradation curve (should stay flat under CB).
- Traced prefill + traced decode TTFT measurement at L=64..1024.
- Screenshots of the TUI at each setup.

## Stretch features (lower priority; tasks added to queue)

- Tool calls: web search + CLI navigation
  (`scripts/chat.py` currently has shell/read_file/calc; add web-search + cli-nav).
- Gemma 4 12B IT image input (the model is multimodal-capable).

## Branch policy

Output files live under `presentation/` and land on branch
`presentation/cs440lx-prep` so the main branch stays clean for
repo users. I'll cut the branch after the first agent returns.
