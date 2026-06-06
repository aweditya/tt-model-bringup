# qwen36_topk_owned

Stable-sort fork of `ttnn::topk` for Qwen3.6 / Nemotron-3 MoE router bring-up.

## Why

`ttnn.topk` does NOT thread the LLK `STABLE_SORT` template parameter (added
in tt-metal PR #31989) through to its call sites. With unstable LLK ties,
the MoE router's per-row argmax is non-deterministic across calls, which
compounds across 23 MoE layers per forward and 40+ decode steps and turns
a 7/7 decode chain regression into 0/7 (see MEMORY:
`feedback_ttnn_topk_tie_break_drift.md`).

This op is a verbatim source fork of the reduction `topk` op with
`stable_sort=true` hardcoded at every LLK call site:
- `ckernel::topk_local_sort<true>(...)` (single-core compute kernel, common funcs)
- `ckernel::topk_rebuild<true>(...)` (common funcs)
- `ckernel::topk_merge<false, true>(...)` and `topk_merge<true, true>(...)` (common funcs)

Everything else is rename-only.

## Design doc

`research/qwen36_topk_owned_design.md`.

## Layout

- `qwen36_topk_owned.{hpp,cpp}` — public host API
- `qwen36_topk_owned_nanobind.{hpp,cpp}` — Python binding (hand-written)
- `device/qwen36_topk_owned_{device_operation,single_core_program_factory,multi_core_program_factory,utils,constants,device_operation_types}.{hpp,cpp}` — device ops + PFs (forked, namespace-renamed)
- `device/kernels/{compute,dataflow}/qwen36_topk_owned_*.{cpp,hpp}` — kernels (forked, `stable_sort=true` in compute kernels)
- `_fork_from_upstream.py` — scaffolding helper that regenerates the forked files from a cached upstream snapshot (see below)
- `integrate_into_ttmetal.py` — copies files into a TT-Metal checkout + patches cmake / nanobind registration
- `INTEGRATION.md` — qb1 step-by-step

## Regenerating from upstream

The forked sources are derived from
`~/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/` on qb1.
A cached snapshot lives at `.cache/qb1_topk_src/`.

```bash
# refresh the cached snapshot (from project root)
mkdir -p .cache/qb1_topk_src
rsync -a qb1:tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/ \
        .cache/qb1_topk_src/

# regenerate all forked files (idempotent, overwrites)
python3 experiments/owned_ops/qwen36_topk_owned/_fork_from_upstream.py
```

The fork script preserves the hand-written `qwen36_topk_owned_nanobind.{hpp,cpp}`,
the integration script, this README, and INTEGRATION.md.

## Install + build

See `INTEGRATION.md`.
