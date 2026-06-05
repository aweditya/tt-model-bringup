# Continuous batching for Qwen3.6-27B on (1,4) P150 — plan of action

Goal: serve multiple concurrent decode requests by filling the TILE=32
matmul rows with real tokens instead of 31/32 padding. At batch=1 we read
the full weight set per token and produce 1 token; batched decode amortizes
that weight read across B tokens for ~the same step latency → up to ~B×
aggregate throughput. This is the single largest throughput lever available
(bigger than any per-op kernel win).

**Adopt vLLM design decisions; do not reinvent.** Specifically:
- **PagedAttention** (Kwon et al., SOSP 2023) — KV in fixed-size blocks,
  per-sequence block table, no fragmentation. We ALREADY have paged SDPA;
  generalize the single block-table to N sequences.
- **Iteration-level continuous batching** (Orca, Yu et al., OSDI 2022) —
  re-compose the batch every decode step; free a slot on EOS, admit a
  pending request into it. Keeps the batch full without waiting for the
  slowest sequence.
- **Block manager** — free-list of KV blocks, on-demand allocation, and
  (later) preemption under memory pressure.
- **Prefill/decode separation** — prefill is compute-heavy (M = prompt
  len), decode is memory-heavy (M = 1 per seq). Schedule them distinctly.

**No OpenAI-compatible HTTP layer for now** (user deferred). Internal
batching throughput first; API compat is a later add-on.

## CB0 — Scoping (do BEFORE any code)

Confirm the facts that size everything else. No guessing.

1. **27B architecture confirm.** Qwen3.6-27B is dense-FFN (single MLP, NOT
   MoE experts) but HYBRID sequence mixing: GatedDeltaNet (linear attention,
   recurrent state) layers + full-attention layers. Implications:
   - Dense MLP batches cleanly (no expert routing) — the easy part.
   - **Full-attention layers**: paged SDPA already supports batching via a
     block table — generalize from 1 seq to N.
   - **DeltaNet layers**: each sequence has its OWN recurrent state H_t
     [NV, head_k, head_v] + conv state. Batching means a batch dimension on
     the recurrent state and the recurrence math. THIS is 27B's batching
     wrinkle (the analog of MoE routing for the 35B case). Confirm exact
     layer counts + which are DN vs attention from server_tp.py.
   - Read server_tp.py: confirm num_layers, num_kv_heads, head_dim,
     hidden, DN vs attn layer split, current KV cache shape + paged config.

2. **Memory budget → max batch B_max.** Per chip = 31.8 GB.
   - Weights (bf8, sharded /4): measure actual (~7 GB/chip expected).
   - KV per token per layer = num_kv_heads × head_dim × 2(K+V) × 2(bf16),
     × num_attn_layers, /4 chips. Compute the exact number.
   - DN recurrent state per seq = num_dn_layers × NV × head_k × head_v ×
     dtype, /4. This is per-SEQUENCE (not per-token) — sizes with B, not L.
   - Free budget = 31.8 − weights − activations. Divide by (KV/token ×
     max_context + DN_state/seq) to get max concurrent sequences.
   - Expectation: KV memory is abundant; the binding constraint is the
     matmul batch WIDTH we can run efficiently (a few tiles → B = 32 or 64),
     not memory. Confirm with the arithmetic.

3. **Trace strategy decision.** TT traces bake shapes. Options:
   - (a) Capture ONE decode trace at fixed B_max; always run full width,
     mask/pad empty slots. Simplest; wastes compute when batch < B_max.
     This is the CUDA-graph-equivalent vLLM uses.
   - (b) Capture a few traces at B ∈ {1, 8, 16, 32}; pick the smallest that
     fits the live batch. More traces to manage, less waste.
   - Recommend (a) for CB1-CB4, revisit (b) in CB6 if padding waste is large.

4. **Output**: `research/27b_cb_scope.md` with the numbers + B_max + trace
   decision. Gate: do not start CB1 until B_max and the DN-state-batching
   shape are written down.

## CB1 — Batch dimension in the forward (static batch, no scheduler)

Make `forward_token_tp_inner` (server_tp.py) accept B > 1. Static batch:
all B sequences same length, decode together. No admission/eviction yet.

- Activation: [1, HIDDEN] → [B, HIDDEN]. Dense MLP, attention QKV/out
  projections, lm_head: all M=1 → M=B matmuls. Trivial shape change.
- DN recurrent state: add batch dim → [B, NV, head_k, head_v]. The
  recurrence (state*g, state*k, sum, state+k_delta, state*q) batches over
  the leading B. Validate the manual + owned_gdn paths handle B>1 (owned
  kernel may need a batch-aware launch — check its grid).
- Attention: paged SDPA decode with B query rows. cur_pos becomes a
  per-slot vector [B] not a scalar.
- **Validation gate**: feed B identical sequences; every slot's output must
  be bit-identical to the B=1 result for that sequence. Then feed B
  DIFFERENT sequences (same length) and check each against a B=1 reference
  run. Cos > 0.999 per slot.

Isolation first (per workflow): a standalone harness that batches just the
attention block, just the DN block, just the MLP — each validated B=1 vs
B=8 — before touching the full forward.

## CB2 — Per-slot state (ragged lengths)

Sequences in a real batch have different lengths and positions. Add:
- Per-slot `cur_pos[B]` (already vectorized in CB1).
- Per-slot KV block allocation: generalize the paged block table from 1
  table to B tables. paged_update_cache + paged SDPA decode index per slot.
- Per-slot DN recurrent + conv state: indexed by slot, persists across the
  slot's lifetime, zeroed when a slot is freed/reassigned.
- **Validation gate**: 2 sequences of DIFFERENT lengths (e.g. prompt lens
  5 and 50) decode correctly in the same batch; each matches its own B=1
  reference. This is the real correctness test for ragged batching.

## CB3 — The scheduler (continuous batching)

Python-level iteration scheduler (Orca-style):
- **Slot table**: B_max slots, each {seq_id, state, cur_pos, block_table,
  sampling_params, generated_ids} or EMPTY.
- **Pending queue**: incoming requests waiting for a free slot.
- **Per-step loop**:
  1. Decode step over all active slots (batched, CB1/CB2 forward).
  2. Sample next token per slot (per-slot sampling params).
  3. Check stop conditions (EOS, max_tokens) → free finished slots.
  4. Admit pending requests into free slots: run their PREFILL (process
     prompt), then they join the decode batch next step.
- **Prefill handling** (the ragged-shape problem): prefill M = prompt_len
  ≠ decode M = 1. Options, in order of preference:
  - Prefill admitted requests in a SEPARATE batched prefill pass (one or
    few prompts at a time), then merge into the decode loop. Simplest.
  - Chunked prefill (vLLM): split long prompts into decode-sized chunks
    mixed into the decode batch. Defer to CB6 if needed.
- **Validation gate**: stream 64 requests of varying prompt/gen lengths;
  every request's output matches a serial B=1 reference; aggregate
  throughput measured.

## CB4 — Trace integration

Capture the decode step as a trace at fixed B_max (CB0 decision (a)).
- Inputs (tok_buf[B], cur_pos[B], block tables) written OUTSIDE the trace
  via update-buffers, same pattern as the current B=1 trace.
- Empty slots padded + masked (their output discarded).
- **Measure**: aggregate tok/s at B = 8, 16, 32 vs the 12.93 tok/s B=1
  baseline. Expect near-linear scaling until BW saturates or the batch
  width exceeds efficient matmul tiling.

## CB5 — Block manager + preemption (vLLM memory management)

- Free-list of KV blocks; allocate on demand as sequences grow.
- Preemption under pressure: evict a sequence's blocks (recompute-on-resume
  is simplest; swap-to-DRAM is the vLLM alternative). Only needed when
  concurrent demand exceeds KV budget — likely not binding for 27B per the
  CB0 budget, so this can be deferred / minimal.

## CB6 — Throughput tuning + profile the batched regime

NOW tt-perf-report matters (the bottleneck has moved off batch=1 weight
reads). Profile the batched decode:
- Is it still weight-BW-bound (then bigger B helps) or now compute-bound
  (then B is saturated)?
- KV cache read cost at large B.
- Per-slot sampling overhead.
- Revisit trace strategy (b) if fixed-B padding waste is significant.

## Sequencing note vs other workstreams

- **A010 (DN H_t drift)** runs as a PARALLEL track. Continuous batching
  serves short-to-medium contexts (chatbot turns) where current correctness
  is fine; long-context coherence (A010) is orthogonal. The DN-state
  batching in CB1/CB2 SHOULD coordinate with A010's fp32-state work since
  both touch the recurrent state — whoever lands first informs the other.
- **35B MoE batching** is the follow-on: same scheduler + block manager,
  but adds per-expert token grouping (grouped GEMM / sort-scatter) on top.
  Build it after 27B dense validates the infra.
- **DeepSeek-V4-Flash**: SHELVED. 158 GB > 127 GB (4×P150) — does not fit.
  Needs 8 chips or offload. Plus 3 novel mechanisms (compressed-sparse
  attention, mHC, FP4 matmul). Revisit only with more hardware.

## Workflow discipline (per feedback_perf_workflow)

Every CB phase: isolate the shape change → validate correctness (bit-exact
or cos>0.999 vs B=1 reference) → integrate behind a flag → measure. The
B=1 path stays the default until CB4 proves batched throughput + correctness.
No throughput claim without a measured aggregate tok/s, no correctness
claim without a per-slot reference comparison.
