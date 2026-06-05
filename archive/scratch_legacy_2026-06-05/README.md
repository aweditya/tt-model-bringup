# scratch_legacy_2026-06-05 — pre-pivot demo dump

Moved here from the top-level `scratch/legacy-demos/` on 2026-06-05
as part of the round-2 doc polish ([`research/doc_polish_plan_2026-06-05.md`](../../research/doc_polish_plan_2026-06-05.md)).

Contents:

- `demo_gpt2.py` — top-level GPT-2 generation demo (pre-pivot tt-xla
  naming).
- `demos/` — 12 demo files (chat client/server, batch serving,
  benchmark, qwen-traced, llama8b generate).
- `generate_moe_qwen15.py` — Qwen-1.5-MoE generation script.
- `PLAN_pre_pivot.md` — the pre-pivot master plan (April 2026, the
  "Qwen2.5-0.5B traced 140 tok/s" era — predates the 27B / 35B / Gemma 4
  bringups).

These are kept for reference only. The supported demo path is
[`../../REPRODUCE.md`](../../REPRODUCE.md) → `models/*.py`.
