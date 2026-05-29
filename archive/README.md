# archive/ — preserved experiments (not the active path)

These scripts taught us the hardware and drove the bringup; they are kept for
reference but are **not** maintained, run in CI, or part of the production path.
Active code lives in `experiments/serve/` (servers + scheduler), `experiments/
owned_ops/` + `experiments/kernel_patches/` (custom kernels), and the validation/
bench suite under `experiments/`. The current architecture + perf numbers are in
`HANDOFF.md` and `research/`.

Eras (by leading number, roughly chronological):

- **01–25 — JAX/XLA + interpreter era** (pre-pivot): jax internals, jaxpr→ttnn
  interpretation, trace capture, the PJRT-backend exploration.
- **26–35 — GPT-2 bringup**: first end-to-end on-device transformer, KV cache,
  SDPA decode, trace-bucketed scaling.
- **91k–91x — Qwen3.6-27B bringup debug**: per-layer HF-reference diffs,
  substep dumps, lm-head inspection, fp32 probes used to root-cause drift.
- **`legacy/` — the founding JAX/PJRT backend** (pre-pivot): `pjrt_plugin/`
  (custom PJRT plugin compiling JAX→ttnn), `tt_jax/`, and the `jax_qwen05b_*`
  probes. Retained in case the JAX-backend direction is revisited.

Multi-model bringup *results* (Llama 1B/3B/8B, SmolLM3, Qwen ports, MoE) and the
load-bearing weight loaders are **not** here — see the active tree / README.
