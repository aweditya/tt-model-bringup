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

## Staged plan

- **S1 — single-seq chunked prefill forward (non-CB first, lower risk).**
  Assemble `forward_prefill_chunked` (B=1): loop C=32 chunks; per chunk run chunked
  DN (thread S state across chunks) + multi-token attn (`paged_fill_cache` writes
  the chunk's K/V; SDPA over the chunk) + MLP; carry DN state + KV + cur_pos across
  chunks. Ragged tail (L % 32 ≠ 0): pad to 32 or per-position the remainder.
  **Gate:** logits cosine-ladder vs the single-token prefill (expect ≥0.99 to
  match the chunked-DN-at-≤32 result) + measure TTFT speedup. Wire behind a flag
  in the prod (non-CB) `forward_prefill_tp_inner` first.
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
