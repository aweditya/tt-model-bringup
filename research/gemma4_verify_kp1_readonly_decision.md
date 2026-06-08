# Phase 2.B.1 Decision: Read-Only Verify (v0)

**Date**: 2026-06-07
**Owner**: Phase 2.B.1 (Gemma 4 12B IT spec-dec)
**Status**: ADOPTED for v0; Phase 3 may revisit.

## Decision

The B=K+1 verify trace shipping in Phase 2.B.1 is **READ-ONLY**:

- All K+1 Q candidates are fed into a single `paged_scaled_dot_product_attention_decode` call.
- **NO `paged_update_cache` is invoked for the K+1 verify rows.**
- Each candidate's hypothetical K/V never lands in the cache.
- The verify forward reads K/V history through position N (current position),
  identical for all K+1 alias rows.
- After accept-walk picks the winner, the TARGET's own next B=1 decode step
  performs the real KV write for the accepted token.

## Background

Phase 2.B.0.5 kernel isolation gate (commit `c3124d2`,
`experiments/cb/isolate/gemma4_kp1_paged_kernels_smoke.py`) proved:

- `paged_update_cache(B=K+1, alias page-table)` returns cleanly, no TT_FATAL.
- `paged_scaled_dot_product_attention_decode(B=K+1, alias page-table)` returns cleanly.

A finding surfaced during gate-1: with K+1 alias rows all writing to the
SAME physical slot (row 0 of the cache), **only the LAST writer's K/V
persists**. There is no kernel-level write-merge across the alias rows.

So writing K+1 hypothetical K/Vs into the same slot is destructive — only
candidate K wins; the cache then no longer reflects the original "history
through pos N" state that all rows need to read.

## Options Considered

### Option A — READ-ONLY VERIFY (chosen for v0)

- One SDPA decode call with B=K+1, scale=Gemma4 1.0, sliding_window respected.
- K/V cache is untouched during verify.
- After accept walk picks winning candidate W (0..K-1) or bonus (K), the
  target's own B=1 decode at position N+1 performs the canonical
  paged_update_cache + SDPA for the accepted token, advancing cur_pos.

**Pros**:
- No cache rewind logic needed.
- Single SDPA call per verify round — minimal device dispatch overhead.
- Cache state is provably equivalent to plain greedy decode at the
  accepted-prefix boundary; bug surface area is small.
- The compute time of K+1 SDPA forwards is dominated by the K+1 Q-matrix
  scan over the same K cache; we pay one cache-scan worth of bandwidth
  for K+1 candidate evaluations.
- Drafter is the wrong path for KV bookkeeping — drafter already shares
  target's KV via Phase 2.A `read_shared_kv_for_drafter` (read-only).

**Cons (semantic loss)**:
- The "correct" Leviathan verify would have candidate i attend to history
  through pos N PLUS candidates 0..i-1's hypothetical KVs. We give every
  candidate the same history-through-N view.
- For greedy accept-walk this is **NOT a correctness loss for accept[0]**:
  candidate 0's verify position N+1 logits are computed against history
  through N — identical to what the target's own B=1 step would see at
  position N+1.
- For accept[i>0], candidate i sees history through N (missing
  hypothetical K/Vs at N+1..N+i). The target's B=1 step at position N+i+1
  would have seen those. So accept[i>0] is computed against a slightly
  stale context.
- In practice, the missing K/V contributes at most one softmax weight per
  attention head per layer; bf16 chain drift dominates this for K<=7.

### Option B — WRITE-THEN-REWIND (DeepSeek-V3 pattern)

- Each candidate writes its hypothetical K/V into a distinct alias slot.
- Each candidate's SDPA reads from its own alias slot's history (correct
  semantics for accept[i>0]).
- After accept walk, the cache must be rewound — the unaccepted candidates'
  K/V slots must be zeroed/invalidated.

**Pros**:
- Semantically correct Leviathan verify; accept[i>0] sees the proper context.

**Cons**:
- Requires K+1 distinct cache rows backing the alias page-table (currently
  the active prompt has only 1 slot; would need to allocate K+1 spare slots).
- Rewind logic: must track which candidate won, and re-write the cache to
  reflect just that prefix. This means partial-cache writes per round,
  which DeepSeek-V3 ships as a separate device kernel.
- Doubles the verify dispatch count (write + SDPA per layer).
- Increases the attack surface for the v0 ship; bug debug time is higher.

## Why Option A for v0

1. **Leviathan's accept walk doesn't care about the i>0 context drift.**
   The walk does `argmax(verify_logits[i]) == draft[i]`. If the drafter
   guessed right at pos N+i, the target argmax at pos N+i (computed
   against ANY reasonable context) is overwhelmingly likely to also be
   the same token — that's exactly what makes spec-dec work. The accept
   rate `α` measures drafter–target agreement; small context perturbations
   only affect borderline ties.
2. **Bf16 chain drift dominates at L=48 layers.** Per the Phase 2.A.0
   layout probe, sliding KV cos vs HF is 0.96 from layer-chain drift
   alone. The semantic loss from option A is well below this floor.
3. **Isolates the v0 surface area.** Buffer setup, two-phase warmup,
   trace capture, and accept-walk plumbing all need validating. Adding
   write-then-rewind multiplies bug paths.
4. **Recoverable**: if measured α at K=5 turns out below the projected
   2x speedup floor, Phase 3 revisits with Option B.

## Acceptance Criteria (Phase 2.B.1 smoke)

`experiments/cb/isolate/gemma4_target_verify_kp1_smoke.py` proves the
read-only verify trace is bit-equivalent (modulo bf16 chain drift) to
K+1 independent B=1 SDPA reads:

- Bootstrap target Gemma 4 12B IT on (1,4) qb1 mesh.
- Prefill prompt 0 ("The capital of France is", 5 tokens).
- Get target's B=1 argmax at pos 5 (the canonical "next" token).
- Synthesize K=5 draft candidates (random in [0, VOCAB)).
- Run K+1 INDEPENDENT B=1 forwards (no actual KV write between them).
  Capture K+1 logits.
- Run B=K+1 verify trace once. Replay with same K+1 candidates.
  Capture K+1 logits.
- Per-row gate: `cos(verify_trace[i], independent_B1[i]) >= 0.99`.

A bf16 drift floor of 0.999 is the soft target; 0.99 is the hard gate
(rationale: per-row cos at L=48 layers averages 0.99+ for the existing
B=1 decode trace per Phase 1.A multi-prompt validation).

## Phase 3 Hook

If `α` at K=5 measures < 0.4 (rough heuristic: ~2x speedup floor
assuming verify cost == 1.5x B=1 step), Phase 3 may revisit:

- Implement Option B (per-candidate KV write into distinct alias slots).
- Add cache-rewind kernel (fork DeepSeek-V3 `tt/cache.py` partial-write).
- Expand verify_offset window to K+1 distinct slots.

Either way, Phase 3's `spec_dec_scheduler.step()` orchestration sees the
same `_target_verify_kp1(draft_tokens)` API and returns the same K+1
logits shape. v0 → v1 is an internal swap.
