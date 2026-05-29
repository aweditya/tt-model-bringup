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
  **Executable recipe** (additive `forward_prefill_chunked_tp(state, prompt_ids,
  capture_logits=False)`, place after the stub ~1963):
  1. Multi-pos embed (mirror `update_input_buffers`/`forward_token_tp_inner`):
     `tok=[prompt_ids]` `[1,L]` uint32 → `ttnn.embedding(tok, embed_tt, TILE,
     DRAM)` → reshape `[L,HIDDEN]`; `pos=[0..L-1]` `[1,L]` → embed `cos_table_tt`/
     `sin_table_tt` → reshape `[L, rotary_dim]`.
  2. Layer loop: `linear_attention` → `deltanet_chunked_neumann_tp(state,x,dn,
     cfg,L)`; else `gated_attn_step_prefill_tp(state,x,attn,cos_seq,sin_seq,cfg,L)`;
     then `mlp_step_tp(state,x,mlp)`. Then `_rms_norm_manual(x, final_norm_tt, 1e-6,
     HIDDEN)`.
  3. capture_logits: LM head (linear→all_gather→slice→untilize) over ALL L rows →
     `to_torch(ConcatMeshToTensor dim=0)[:L]` → `[L,vocab]`. Production: last-pos
     logits.
  **HAZARDS to validate (isolate first, don't guess):** (a) last-row slice
  `x[L-1:L]` → `[1,HIDDEN]` is a sub-tile/view-decay hazard
  [[feedback-ttnn-slice-view-decay]] — prefer LM-head-over-all-rows-then-take-last,
  or verify the slice; (b) `ttnn.embedding([1,L])` output shape on-device;
  (c) `gated_attn_step_prefill_tp` writes cache at pos [0,L) (assumes fresh seq at
  pos 0) — fine for B=1 prefill; (d) keep B3 SDPA (bf16 prefill drift). Gate:
  per-position logit cosine vs `forward_prefill_tp_inner` stub + TTFT, on qb1.
- **S1b — block-chunked DeltaNet (Neumann/32-block) — DONE (qb1, 2026-05-28).**
  Recon corrected the earlier framing: NO multi-query paged SDPA needed — the
  whole-prompt attention already handles long L (S1a). S1b just runs the DN in
  32-token Neumann blocks for L>32 (`_prefill_dn_chunked_blocks`) instead of the
  per-position fallback. Correct because `deltanet_chunked_neumann_tp` threads both
  `dn['ssm']` and `dn['conv_st']` (updated in-place per position, server_tp.py:1176)
  across calls. **Validated (long_context.py, L=137): needle retrieved + TTFT stub
  31997 ms → chunked 12573 ms = 2.54×** (S1a was attn-only 1.35× at short L).
- **S2 — CB integration (NEXT; substantial — recon'd 2026-05-29).** On admit,
  prefill the request's prompt in 32-blocks into its slot's state, then it joins
  the decode rotation (prefill-then-admit; avoids vLLM mixed-batching). **Real
  challenges (why it's not a 1-liner):**
  - CB per-slot state is `state.cb_dn[li] = {'ssm':[B,NV,K,V], 'conv_cols': 3×[B,C]}`
    (server_tp_cb.py:180) — **shift-accumulate conv**, whereas S1's chunked DN
    (`deltanet_chunked_neumann_tp`) uses the **kdim conv (`dn['conv_st']`)**.
    Representation mismatch → can't just point the S1 chunked prefill at a CB slot.
  - Per-slot targeting: write `cb_dn[li]['ssm'][s]` + `conv_cols[s]` + slot-s paged
    KV (slot page table, batch_idx). CB batched ops are `[B,1-pos]`; chunked prefill
    is `[1-slot, C-pos]`.
  - **S2a:** a CB-native single-slot block-chunked prefill (32-block Neumann over the
    CB shiftacc conv + per-slot ssm; write slot-s paged KV). Gate: chunked-prefill
    slot s → decode == CB 1-tok/iter prefill → decode (functional/needle, not
    weak-ref cosine — [[validate-against-ground-truth-not-a-weaker-tt-path]]).
  - **S2b:** scheduler — on admit call the CB chunked prefill, then DECODE rotation.
    Gate: `cb_scheduler` functional + `cb/needle.py` + realistic multi-req TTFT.
  Multi-session effort. Simpler alt productionization if deprioritized: sampling
  (temp/top-p), OpenAI-compatible endpoint.
- **S3 — mixed continuous prefill (optional, defer).** vLLM-style interleave
  prefill chunks with decode tokens in one batched forward. More complex; only if
  S2's prefill-then-decode phasing isn't enough.

## S1a first qb1 result (2026-05-28) — runs + faster, FAILS correctness gate

`forward_prefill_chunked_tp` works end-to-end (after fixing `from_torch device=mesh`
on the embed indices — the validator caught it). vs the single-token stub on a
29-token prompt: **26/29 argmax match**, most pos cos > 0.97, but **worst cos 0.631
@ pos 21** + last-pos (28) argmax flip (198→561) → gate (≥0.99) FAILS. **TTFT
6965→5142 ms = 1.35×** (modest: L=29 ∉ {4,8,16,32} so DN took the per-position
fallback — only attn parallelized; need L=32 for the Neumann win + a bigger speedup).

**Next (isolate the divergence, don't guess):** the only chunked-vs-stub deltas at
L=29 are (a) attn = one causal SDPA over the chunk vs N per-position decode SDPAs,
and (b) the DN per-position *fallback* (`_deltanet_step_tp_from_inproj` loop, maybe
a sliced-in_proj view-decay) vs `deltanet_step_tp`. Bisect: chunked-attn-only (DN
via stub) vs chunked-DN-only (attn per-position), or a per-layer hidden ladder at
pos 21, to localize. Also re-run at L=32 to exercise the Neumann chunked-DN path.
Then fix, re-gate, wire behind a flag in `forward_prefill_tp_inner`.

## S1a drift isolation (2026-05-28) — the decode-stub is the wrong reference

Sweep L∈{8,16,29,32} (one bootstrap): worst_cos 0.986(L8) → 0.960(L16) → 0.631(L29)
→ 0.516(L32); worst always at a LATE position; **all lengths fail, both DN paths
(Neumann + fallback)**. So it's accumulation-with-L, not a fallback-only bug.

Read of `gated_attn_step_prefill_tp` (1750): structurally correct — pre-norm, QKV,
q/k-norm, rotate-half RoPE with per-pos cos/sin, paged_fill_cache, causal SDPA
(scale 1/√head_dim, B3 kernel cfg), sigmoid output-gate, out_proj+all_reduce+
residual. Mirrors the decode path; no obvious logic bug.

**Reframe:** the chunked DN (`deltanet_chunked_neumann_tp`) is an independently-
validated, *different + more-accurate* formulation ("chunked DN strictly wins at
seq≤32", [[feedback-v4-chunked-dn-seq32-shipped]]); the single-token stub is itself
bf16-noisy (documented single-token-prefill drift, [[feedback-bf16-prefill-drift-cliff]]).
So chunked-vs-stub per-position cosine conflates "bug" with "different-by-design,"
and that difference compounds in the recurrent DN state → late-position drift.

**Corrected gate (next):** compare chunked prefill to **HF (ground truth)** per
position, OR functionally — chunked-prefill → decode N tokens, check coherent +
matches production generation. If chunked tracks HF as well as / better than the
stub, S1a is correct (the stub gate was just too strict). If chunked diverges from
HF where the stub tracks it, there's a real bug → per-layer hidden ladder at the
worst position to localize DN-layer vs attn-layer. (HF 27B oracle: see the cosine-
ladder / needle harnesses; build a per-position HF logit ref if none exists.)

## S1a FUNCTIONAL GATE — PASS (2026-05-28)

`cb/validate/prefill_generate.py` (prefill 2 ways → decode 40, same decode path):
- stub-prefill→decode: coherent but **loops** back to repeat the prompt.
- chunked-prefill→decode: `"…The city of Paris is located in the northern part of
  France, in the Paris Basin, on the banks of the river Seine. The city is divided
  into…"` — **coherent, factually correct, no loop** (arguably better than the stub).

Confirms the diagnosis: chunked prefill produces a good (better) KV/DN state; the
cosine-vs-stub failure was the stub being a poor (bf16-noisy) reference. **S1a
forward validated** (coherent generation + 1.35× TTFT @ L=29; more @ L=32 Neumann).

**Remaining for S1a:** wire `forward_prefill_chunked_tp` behind a default-OFF flag
in `forward_prefill_tp_inner` (additive, zero prod risk) — but first validate the
non-capture last-row return (`ttnn.slice` row-major) since the functional gate used
the capture path. Then S1b (full Neumann win at L=32) + S2 (CB integration).

## S1a COMPLETE (2026-05-28) — correct, chat-usable, wired in

- Functional gate PASS (coherent generation, better than the looping stub).
- Long-context gate PASS (`cb/validate/long_context.py`): L=137 needle prompt →
  chunked prefill **retrieves the code `7X9Q2` verbatim**; non-capture return path
  validated (first-token capture==non-capture). Stub retrieves it too.
- Wired in: `forward_prefill_tp_inner` delegates to `forward_prefill_chunked_tp`
  when `state.prefill_chunked` is set (default OFF → prod unchanged).
- TTFT: **1.35× @ L=29**; at long L the win is attn-only (DN is per-position
  fallback >32), so the big long-context speedup is **S1b** (Neumann/chunked DN).

**Next:** S1b (chunk the prompt into 32s so DN uses the fast Neumann path — needs a
multi-query paged SDPA over the prefix) for the long-context TTFT win; then S2 (CB
integration — the original "realistic task times" goal). Merge S1a (validated,
additive, default-off) first.

## Constraints / gotchas
- Chunk size = 32 (the validated cap). Prompt > 32 → multiple chunks.
- bf16 prefill drift: B3 SDPA (HiFi2, no fp32_dest_acc) is the fix
  [[feedback-fp32-sdpa-cliff-probe]]; keep it.
- View-decay on slices ([[feedback-ttnn-slice-view-decay]]); `paged_fill_cache`
  wants a sharded mem-config; no `from_torch` inside trace capture.
- Validate-then-integrate; never regress the production decode path.
