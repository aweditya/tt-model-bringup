# Phase 0.B — Gemma 4 12B determinism audit

Status: **patches B + D already shipped on Gemma 4. Patch A only matters
for sampling-temperature path (not spec-dec greedy)**. Net new work for
spec-dec: zero.

## Audit findings vs `research/35b_determinism_2026-06-04.md` § 5

| 35B patch | Gemma 4 12B current state | Effect for spec-dec |
|---|---|---|
| **A** — cb_scheduler deterministic tie-break by lowest token-id in `_step_sampled_logits` / `_step_sampled_topk` | not ported; lives in 35B's CB sampling path | **N/A for spec-dec** — accept walks run at temperature=0 (greedy argmax). A is for sampling routes only. |
| **B** — `ttnn.argmax(use_multicore=False)` on the on-device argmax | **✅ shipped** at `server_gemma4_unified_ttnn.py:1086` | Removes cross-core argmax-flip race. Required for stable greedy. |
| **D** — `fp32_dest_acc=True` on lm_head matmul | **✅ shipped via HIFI4** at `server_gemma4_unified_ttnn.py:91-98` (fp32_dest_acc_en=True), used by the lm_head matmul at `:1053` | Halves ULP noise at final logits. Cheapest precision lever. |

## What this means for spec-dec α

The Leviathan correctness contract requires `argmax(target_logits[i]) ==
draft[i]` to deterministically match across spec-dec invocations. With
B+D already in place at Gemma 4 12B:

- Greedy argmax on a fixed prompt should produce identical token IDs
  across runs (modulo any remaining all_reduce non-determinism — source
  3 from the 35B doc, unverified).
- Acceptance rate α is bounded by drafter quality, not by argmax instability.

We did **not** verify across-runs stability empirically. Recommend a
quick 100-token-greedy-same-prompt-3x smoke as a pre-Phase-3 sanity gate
(~3 min wall once qb2 is free).

## What's still possibly outstanding (not blocking spec-dec)

1. **Source 3 from 35B doc**: `ttnn.all_reduce` reduce-order is
   undocumented; could be data-order-dependent. Confirmed only by E3
   probe in the 35B doc (not yet run on Gemma 4). If non-deterministic
   reduce is real, it would manifest as per-layer hidden-state drift
   even at fixed prompt — affects target's verify and the drafter's
   reference output equivalently. Suggestion: run E3 alongside Phase 3
   accept-rate measurement.
2. **bf16 chain drift across the 48-layer Gemma 4 trunk** — fundamental
   precision floor per `[[bf16-chain-drift-at-B-gt-1]]`. Unrelated to
   the A/B/D patches; would only be improved by Option G (LayerCast) or
   H (full fp32) from the 35B doc, both rejected. For spec-dec α this
   is a multiplicative floor; estimate ~5% lower α than what fp32 CPU
   would measure.

## Action: NO CODE CHANGES THIS SESSION

Phase 0.B determinism prep for spec-dec is complete — patches were
shipped during Round 8 perf work (commits visible via `git log -S
"use_multicore=False" experiments/serve/server_gemma4_unified_ttnn.py`).

## Sanity smoke (when qb2 freed)

```bash
# Greedy same-prompt 3× determinism probe
for i in 1 2 3; do
    curl -s -X POST http://localhost:8000/v1/chat/completions \
      -d '{"model":"gemma4_12b","messages":[{"role":"user","content":"Count to 20"}],"temperature":0,"max_tokens":40}' \
      | jq -r '.choices[0].message.content' | head -1
done
```

If all 3 outputs byte-identical → spec-dec α floor is bounded only by
drafter quality. If not, we have a source-3 bug to localize before
landing spec-dec.

## Sources

- `research/35b_determinism_2026-06-04.md` (the recipe)
- `experiments/serve/server_gemma4_unified_ttnn.py` (B+D verified at lines 91-98, 1053, 1086)
