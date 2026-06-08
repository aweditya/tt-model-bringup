# Gemma 4 spec-dec perf push — plan of action (2026-06-08)

After F-1 RoPE fix shipped (commits `86528f6` + `d34c6a4`), drafter is
**bit-exact to HF** and multi-prompt α is up to 0.133 mean / 0.267 max
(5/5 prompts with α > 0). Correctness work paused; now optimizing tok/s.

## Current perf at α=0.267, K=5

Per spec-dec round:
- drafter eager × 5: ~30 ms
- verify traced (aliased): 60 ms
- target B=1 cache writes × (accept + 1) ≈ 108 ms (2.3 emits × 47 ms)
- accept walk + host: < 5 ms
- **Total ~200 ms / ~2.3 tokens = ~85 ms/tok**

vs baseline (target B=1 traced) = 47 ms/tok → **spec-dec 1.8× SLOWER**.

The two dominant costs are (a) target B=1 cache writes (108 ms) and
(b) drafter eager (30 ms). Both are removable.

## Perf plan — three levers, ranked

### P-1: drafter trace path with cur_pos RoPE (~2h)

**Problem**: F-1 added RoPE to `drafter_forward` (eager) but
`drafter_forward_inner_traced` was NOT updated — it still calls
`drafter_layer_forward(state, h, li, K_tt, V_tt)` without cos/sin →
identity RoPE → bit-wrong for cur_pos > 0. Scheduler currently routes
through the eager path because the trace path would be correctness-broken.

**Fix** (forks `server_gemma4_unified_ttnn` rotary buffer pattern):

1. Bootstrap allocates:
   - `state.drafter_rot_idxs_buf`: uint32 [1] device buffer; default 0
   - `state.drafter_cos_sliding_tt`, `state.drafter_sin_sliding_tt`:
     replicated [MAX_KV, HEAD_DIM_SLIDING] ROW_MAJOR (full RoPE tables)
   - `state.drafter_cos_full_tt`, `state.drafter_sin_full_tt`:
     replicated [MAX_KV, HEAD_DIM_FULL]
2. New trace-safe helper `_lookup_drafter_rope(state, cos_table_tt, sin_table_tt, head_dim)`:
   - `ttnn.embedding(state.drafter_rot_idxs_buf, cos_table_tt)` → cos
     row, `ttnn.to_layout(TILE_LAYOUT)`, reshape to [1, head_dim]
3. `drafter_forward_inner_traced` reads cos/sin via lookup INSIDE the
   captured region (so trace re-execution reflects current
   `drafter_rot_idxs_buf` value).
4. New host helper `update_drafter_rot_idx(state, cur_pos)` writes the
   buffer with `copy_host_to_device_tensor` — called OUTSIDE the trace,
   per round.
5. Scheduler's traced path calls `update_drafter_rot_idx` before
   `drafter_forward_traced`.

**Validation**:
- Re-run chain probe Variant C through traced drafter → expect 5/5
  (matches eager F-1).
- Multi-prompt smoke through traced drafter → expect mean α ≈ 0.133
  (same as eager F-1) at 5× the speed.

**Saves**: ~150 ms / round (K=5 × (eager 30 ms - traced 6 ms)).

### P-2: non-aliased verify trace + K/V writes during verify (~4h)

**Problem**: Phase 2.B.1 shipped READ-ONLY verify (no
`paged_fused_update_cache` in the K+1 forward). After accept walk,
scheduler runs target B=1 × N emitted tokens to write their K/V to
cache. At α=0.267 with 2.3 emits → 108 ms of dead time per round.

**Fix** (DeepSeek-V3 verify pattern, non-aliased):

1. Build verify page-table with K+1 DISTINCT rows (NOT aliased — each
   verify position writes to its own slot).
2. K+1 forward writes K/V to all K+1 slots via
   `paged_fused_update_cache`.
3. After accept walk:
   - First `accept_count + 1` slots are KEEPERS (the emitted tokens'
     K/V is now committed).
   - Remaining K - accept_count slots are ABANDONED — next round
     overwrites them. Cheap because cache is paged.
4. Cache cursor advances by `accept_count + 1` positions.

**Tricky bits**:
- Need `accept_count`-aware cache-cursor logic (slot bookkeeping).
- Slot reuse means small extra fragmentation; with K=5 and
  MAX_BLOCKS=128 (current), trivial.
- `paged_fused_update_cache` in K+1 forward = same plumbing as Phase
  2.B.1 had originally; the read-only constraint was a ship decision,
  not a tech blocker.

**Validation**:
- Existing Phase 2.B.1 trace test still passes (K/V slot 0 = old
  behavior).
- New test: post-spec-dec cache state matches post-baseline B=1 cache
  state for the same emitted sequence (byte-equiv).
- Multi-prompt smoke through non-aliased verify → expect same α as
  before (correctness unchanged), tok/s drops by 108 × accept_rate ms.

**Saves**: ~108 ms × accept_count / round. At α=0.267 with K=5 → ~108
ms / round saved.

### P-3: keep `out["hidden"]` on-device between rounds (~3h)

**Problem**: each drafter round reads back `post_projection` output as
fp32 numpy and re-uploads as bf16 next round. Adds bf16 round-trip
noise (compounds across rounds) + a small per-round upload cost.

**Fix**:
1. `drafter_forward_traced` returns ALSO a persistent device tensor
   handle for `hidden` (alongside readback for scheduler logging).
2. Scheduler caches the device handle across K rounds; uploads only
   round 0's `target_h_last`.
3. Per-round concat is on-device: `ttnn.concat([embed_tt, hidden_tt],
   dim=-1)` → `pre_projection_tt`.

**Validation**:
- Re-run chain probe Variant A — should improve from 1/5 toward 5/5
  (matching IT target's actual hidden output for the IT drafter).
- Multi-prompt smoke α should match or slightly improve.

**Saves**: ~3 ms / round (modest perf) + potential α uplift (correctness).

## Combined projection at α=0.267 (current measured)

| Stage | Round cost (ms) | Tokens emitted | ms/tok |
|---|---|---|---|
| Today (eager drafter, aliased verify) | ~200 | 2.3 | **~85** |
| + P-1 (drafter traced) | ~70 | 2.3 | ~30 |
| + P-2 (non-aliased verify) | ~92 | 2.3 | ~40 |
| Baseline | 47 | 1 | 47 |

Wait — that's wrong. P-1 alone doesn't help unless P-2 also lands
(the target B=1 cache writes still dominate). With P-1 + P-2:
- drafter traced × 5 = 32 ms
- verify traced (writes K/V) = 60 ms
- no target B=1 cache writes
- accept walk = < 5 ms
- **Total ~97 ms / 2.3 tokens = ~42 ms/tok = 1.12× over baseline at α=0.267**

At α=0.5: 3.5 tokens / round → 97 / 3.5 = ~28 ms/tok = **1.7× over baseline**.
At α=0.7: 4.5 tokens / round → 97 / 4.5 = ~22 ms/tok = **2.2× over baseline**.

So we need P-1 + P-2 BOTH for any net win at current α. P-1 alone
worsens nothing but doesn't help. P-2 alone gives ~50 ms/tok at α=0.267
which is still slightly slower than 47 ms baseline.

**Order**: P-1 first (simpler, ~2h, unlocks fast drafter). Then P-2.
Then evaluate.

## Workflow / non-negotiables

- Remote-only on qb2 (target IT + drafter co-loaded).
- Permanent files only — fork existing trace pattern from
  `server_gemma4_unified_ttnn`.
- No /tmp.
- Commit per logical step.
- Reuse mandate: P-1 forks target's `_lookup_rope` / `rot_idxs_buf`;
  P-2 forks DeepSeek-V3's non-aliased verify page-table from
  `tt-metal/models/demos/deepseek_v3/tt/generator.py`.
- Probe BEFORE integration: P-1 has chain probe in trace mode; P-2
  needs a cache-state byte-equiv probe before live smoke.

## Then: actual usage test

Once tok/s beats baseline OR we accept correctness-only ship:
- HTTP path: `server_spec_dec.py` (TBD) wrapping the scheduler with
  the existing OpenAI-compatible endpoint
- Chat smoke: 3-5 multi-turn conversations through the chat TUI
- α distribution check across realistic prompt mix
- Long-context smoke: 1K-context prompt → verify cache state stays
  correct under spec-dec
