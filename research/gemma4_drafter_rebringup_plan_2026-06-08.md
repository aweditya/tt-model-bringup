# Gemma 4 drafter re-bringup at parallel L=K — plan of action

Status: PLAN, written 2026-06-08 after Phase 3 spec-dec α≈0 finding.

## Why a re-bringup is needed

Phase 3 v0.0 SHIPPED end-to-end but observed **α=0 across 5 prompts × 2
variants**. Root cause:

- We built drafter at **L=1 hardcoded** (Phase 1 v0.2 `drafter_forward`
  asserts `B == 1 and L == 1`)
- Phase 3 scheduler builds K candidates via **autoregressive K calls**,
  chained by substituting drafter's own hidden output as the next call's
  "current target hidden"
- Drafter was trained on TARGET hidden inputs (48 layers, 12B); its own
  4-layer 0.4B hidden has different distribution
- → Rounds 1+ of drafter chain produce out-of-distribution predictions
- → α≈0 (drafter and target disagree on most rounds)

Re-reading HF source at
`transformers/models/gemma4_unified_assistant/modeling_gemma4_unified_assistant.py:217`:
> "There is no difference for the edge case of `q_len == 1` as it acts
> as full attention no matter what"

The drafter is DESIGNED for arbitrary `q_len` (L>1). At L=K, it produces
K predictions in one forward via bidirectional self-attention + cross-
attention to target KV. **That's the "parallel" we missed.**

## Lessons learned (why we messed up)

Captured as durable feedback for future bringups:
1. **Treat feasibility claims as STRUCTURAL, not theoretical.** Phase 0.A
   said "parallel"; we should have implemented L=K from day 1, not L=1.
2. **HF oracle must cover the production shape.** Our Phase 1 v0.0 oracle
   only captured L=1. Capture L=K BEFORE building the ttnn forward.
3. **Hardcoded `assert L == 1` is a future blocker, not a v0 freebie.**
   Flag it as a known limitation in the task description, not buried in
   the assertion.
4. **Run α=0 sanity check at K=1 EARLY.** If K=1 drafter+target+verify
   produces α>0 on at least one prompt, the framework works. If not,
   STOP and investigate before scaling to K=5.

## Re-bringup plan — 6 phases, ~14h total

### Phase R-1: Research (~2h)

- Read `modeling_gemma4_unified_assistant.py` end-to-end. Document:
  - Exact inputs_embeds construction at L=K (what hidden states go in
    which window? overlapping? sliding?)
  - Bidirectional self-attention semantics (q_idx vs kv_idx for SWA)
  - Output shape at L=K — K predictions, but for WHICH positions?
- Read Gemma 4 model card on HF Hub for usage examples
- Output: `research/gemma4_drafter_parallel_design.md` (~100 lines)

### Phase R-2: HF oracle at L=K (~2h)

- Modify `experiments/utils/hf_oracle_gemma4_assistant.py`:
  - Build `inputs_embeds = [1, L=K, 2*BACKBONE_HIDDEN]` per the
    construction rule from R-1
  - Run on CPU for the 5 canonical prompts at K∈{3, 5, 7}
  - Save `drafter_inputs_embeds_LK.npy`, `drafter_logits_LK.npy`,
    `drafter_hidden_LK.npy`, `drafter_argmax_LK.npy` per prompt/K
- Output: `.cache/hf_oracle_gemma4_12b_assistant_LK/prompt_{i}/K{k}/`

### Phase R-3: Re-bring-up drafter at L=K (~4h)

- Modify `server_gemma4_12b_assistant_ttnn.py:drafter_forward`:
  - Remove `assert L == 1`; accept L > 1
  - SDPA: Q[1, NQ_PER_CHIP, L, head_dim] instead of [..., 1, ...]
  - Cross-attention: same K/V from target, broader Q
  - Attention masks: HF uses `create_bidirectional_mask` +
    `create_bidirectional_sliding_window_mask` — fork to ttnn
- Gate on `L=1` (existing path) vs `L>1` (new path) for backward compat
- Validate per-prompt + per-K against HF oracle:
  - cos(traced_logits, hf_logits) ≥ 0.999 per row
  - argmax exact match per row
- Iterate until 4/5 prompts × 3 K-values all pass

### Phase R-4: Drafter trace at L=K (~2h)

- Modify `setup_drafter_trace_state` to allocate
  `drafter_inputs_buf [1, K, 2*BACKBONE_HIDDEN]` instead of `[1, 1, ...]`
- Two-phase warmup → trace capture → replay validation
- Measure traced wall:
  - L=1 baseline: 6.4 ms × 5 = 32 ms (autoregressive)
  - L=K=5 target: 8-12 ms (one forward, larger Q tensor)
- Update `drafter_step_traced` to return the K argmaxes

### Phase R-5: Scheduler update (~2h)

- Replace `_drafter_autoregressive_K` with `_drafter_parallel_K`:
  - One drafter call instead of K chained calls
  - Construct inputs_embeds from K consecutive target hidden windows
- Need K consecutive target hidden states. Current stash has prev/cur
  only. Need to extend `state.last_target_hidden_history` as a ring
  buffer of K entries (~30 KB at K=5).
- Update `step_forward_v03` to append to history each step (cheap)

### Phase R-6: Bench + report (~2h)

- Run multi-prompt α probe at K∈{3, 5, 7}
- Expected α ≥ 0.3 (target/drafter agreement on at least 30% of candidates)
- Per-round wall:
  - Drafter L=K traced: 8-12 ms
  - Verify B=K+1 traced: 60 ms
  - Target B=1 advance × emit: 47 × ~3 = 140 ms
  - Total: ~210 ms / ~3 emit = 70 ms/tok @ α=0.3 = **0.67×**
  - Still slower than baseline. To beat baseline, need write-during-
    verify (v1.0 perf path, separate refactor)
- Document final α + tok/s
- Update HANDOFF with re-bringup outcome

## Open risks

1. **HF attention mask wiring at L>1** — bidirectional + SWA flip logic
   is complex; may need extensive correctness gates before we trust it
2. **inputs_embeds construction rule unclear from HF source alone** —
   may need to run HF generate() with assistant to observe actual usage
3. **Target hidden ring buffer for K consecutive states** — current
   server only stashes prev/cur; extending to history may need careful
   coordination with the existing trace path
4. **The "K predictions for K positions" interpretation may be wrong** —
   drafter may produce K predictions for the SAME position via different
   window contexts (committee voting). Need to verify via HF oracle

## Non-negotiables

- All Python remote via `ssh qb1` / `ssh qb2`
- All probes permanent files under `experiments/cb/isolate/`
- HF oracle helper at `experiments/utils/hf_oracle_gemma4_assistant_LK.py`
- No `/tmp`; artifacts under `.cache/`
- Frequent commits (one per phase)
- Reuse mandate: fork existing `drafter_forward` + `_drafter_attn_*`
  patterns, don't rewrite from scratch

## Decision point: do this now or park

This is **~14 hours of focused work**. Two paths:
- **GO**: invest the time, ship spec-dec at α>0.3 with real demo numbers
- **PARK**: spec-dec stays at v0.0 (framework correct, α≈0), pivot to
  Gemma 4 perf adoption or Nemotron-3 next

The α>0.3 target STILL doesn't beat baseline (need write-during-verify
too). For a real speedup demo we'd need re-bringup + v1.0 perf = ~3 days
total.
