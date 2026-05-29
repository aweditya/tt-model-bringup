# Chunked prefill for 27B (productionization step 1)

**Goal:** realistic time-to-first-token + end-to-end task times. Today the CB
path prefills the prompt **one token per scheduler iteration** (a length-L prompt
= L full batched-forward steps), so prompt processing dominates wall-clock and
TTFT is unrealistic. Process the prompt in C-token chunks instead.

## Recon (done 2026-05-28) — the building blocks already exist

In `server_tp.py` (single-sequence, B=1, validated in isolation):
- **Chunked DN**: `deltanet_chunked_neumann_tp` (1281) → `_chunked_dn_with_chunked_recurrence_tp`
  (1067) → `_chunked_recurrence_tp` (842, Neumann factorization). **Capped C=32** —
  bf16 matmuls accumulate too much error at C≥64 (see the cap comment ~927; memory
  [[feedback-v4-chunked-dn-seq32-shipped]]). "Chunked DN strictly wins at seq≤32."
- **Chunked attention**: multi-token SDPA + `paged_fill_cache` multi-position KV
  write (prefill-attn fn ~1763, `paged_fill_cache` ~1860).
- **MLP**: already shape-agnostic in the leading dim (processes a chunk as-is).

What's MISSING:
- `forward_prefill_tp_inner` (1915) is still the **single-token STUB** (Phase B.1 —
  "loops the decode forward once per prompt token").
- The CB scheduler (`cb_scheduler.py:17`, 184-188) advances `cur_pos += 1` per
  iteration — no chunked prefill.
- (Post-M4 the chunked-DN fns are still present, 5 refs; their probe handler was
  deleted, so verify their live entry points when wiring.)

## Recon detail (2026-05-28) — exact contracts

- `deltanet_chunked_neumann_tp(state, x_seq_tt, dn, cfg, seq_len)` (1281): returns
  `[seq_len, HIDDEN]` residual-added; threads `dn['ssm']` (S read line ~1232,
  written ~1238). Chunked Neumann for seq_len∈{4,8,16,32}; **per-position fallback
  for >32**. **Dormant — no callers yet.**
- `gated_attn_step_prefill_tp(state, x_seq_tt, attn, cos_seq_tt, sin_seq_tt, cfg,
  seq_len)` (1750): `paged_fill_cache` writes all seq_len K/V (1860); SDPA is
  **causal over the chunk's own Q/K/V** (1880, is_causal=True) → correct for the
  WHOLE prompt as one chunk, but does NOT attend to a prior prefix. **Dormant.**
- `mlp_step_tp` is leading-dim-agnostic (CB reuses it batched) → takes `[C, HIDDEN]`.
- `forward_token_tp_inner` (1635): embeds via `ttnn.embedding(tok_buf, embed_tt)` +
  cos/sin row lookup; layer loop dispatches `deltanet_step_tp` (DN) vs
  `gated_attn_step_tp` (attn) + `mlp_step_tp`; final rms_norm + vocab-sharded LM
  head. `update_input_buffers(state, token_id, cur_pos)` (1600) sets tok/cur_pos/
  rot_idxs buffers per position.
- **No chunked-prefill forward exists**; `forward_prefill_tp_inner` (1915) is the
  single-token stub (its own docstring: B.2 = parallel attn, B.3 = chunked DN).

**KEY subtlety:** for true C=32 multi-chunk prefill, chunk N's queries must attend
to positions 0..(N·32−1) — a multi-*query* paged SDPA over the cache. The existing
prefill-attn does causal-within-chunk only. So split S1:

## Staged plan

- **S1a — whole-prompt single-chunk prefill (Phase B.2; DOING FIRST).** New
  ADDITIVE `forward_prefill_chunked_tp` (B=1; prod stub untouched → zero regression
  risk): embed all L prompt tokens + cos/sin rows, then per layer dispatch
  `deltanet_chunked_neumann_tp(...,seq_len=L)` (DN) vs `gated_attn_step_prefill_tp(
  ...,seq_len=L)` (one parallel causal SDPA + paged_fill_cache over the whole
  prompt) + batched `mlp_step_tp` over `[L,HIDDEN]`; final norm + LM head → last-pos
  logits. Attn becomes ONE SDPA instead of L steps (the win); DN is per-position
  for L>32 (chunked only ≤32). **Gate:** last-pos logit cosine vs single-token stub
  (≥0.99) + TTFT speedup, on qb1. Flag into `forward_prefill_tp_inner`.
- **S1b — true C=32 chunked DN (Phase B.3).** Chunk the prompt into 32s so DN uses
  the fast Neumann path (S threads across chunks). Needs a multi-*query* paged SDPA
  (chunk N attends to prefix 0..N·32−1) — ISOLATE + validate that primitive first
  (project methodology). Bigger; after S1a lands.
- **S2 — CB integration.** On admit, prefill the request's prompt in C=32 chunks
  into its slot's KV + DN state (a per-request prefill phase), then it joins the
  decode rotation. Scheduler PREFILL status: "prefill in chunks" not "1 tok/iter".
  **Gate:** `cb/validate/forward.py` + `cb/needle.py` (long-context) + realistic
  TTFT/task-time numbers.
- **S3 — mixed continuous prefill (optional, defer).** vLLM-style interleave
  prefill chunks with decode tokens in one batched forward. More complex; only if
  S2's prefill-then-decode phasing isn't enough.

## Constraints / gotchas
- Chunk size = 32 (the validated cap). Prompt > 32 → multiple chunks.
- bf16 prefill drift: B3 SDPA (HiFi2, no fp32_dest_acc) is the fix
  [[feedback-fp32-sdpa-cliff-probe]]; keep it.
- View-decay on slices ([[feedback-ttnn-slice-view-decay]]); `paged_fill_cache`
  wants a sharded mem-config; no `from_torch` inside trace capture.
- Validate-then-integrate; never regress the production decode path.
