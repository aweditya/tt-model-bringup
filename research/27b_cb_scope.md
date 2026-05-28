# CB0 scope — continuous batching for Qwen3.6-27B (2026-05-27)

Gate doc for the continuous-batching build. Numbers from the actual
HF config + server_tp.py, not estimates.

## Home: qb1

- 27B HF weights cached on qb1 (52 GB) AND qb2 (52 GB).
- `experiments/serve/server_tp.py` on qb1 is **byte-identical to the
  committed repo** (md5 ee4b44d…); qb2's copy is STALE (older, divergent,
  401 KB vs 339 KB).
- **No live 27B server on either host** (qb2's .sock files are stale from
  May 23, no process behind them). Nothing to disrupt.
- Decision: **build CB on qb1.** Canonical server, weights resident,
  this session's work lives here, satisfies "prefer qb1 for experimental".

## 27B architecture (from config.json)

| Field | Value |
|---|---|
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| full_attention_interval | 4 → layer `i%4==3` is full attn, else DeltaNet |
| → layer split | **48 DeltaNet + 16 full-attention** |
| attn: num_attention_heads | 24 (Q) |
| attn: num_key_value_heads | 4 (GQA) |
| attn: head_dim | 256 |
| DN: linear_num_key_heads | 16 |
| DN: linear_num_value_heads | 48 |
| DN: linear_key/value_head_dim | 128 / 128 |
| DN: linear_conv_kernel_dim | 4 |
| vocab_size | 248320 |
| NCHIPS (TP) | 4 |
| MAX_POS / BLOCK_SIZE / NUM_BLOCKS | 8192 / 32 / 256 (current single-seq) |

## The batching wrinkle: 48/64 layers are DeltaNet

27B is **dense-FFN** (clean batch, no MoE routing) but **hybrid sequence
mixing**: 48 GatedDeltaNet layers + 16 full-attention layers.

- **Full-attention layers (16)**: paged SDPA already supports batching via
  a per-sequence block table. This is standard **vLLM PagedAttention** —
  generalize the current single block-table to N tables. Adopt as-is.
- **DeltaNet layers (48)**: each sequence carries a recurrent state H_t +
  conv state, persisting across its whole lifetime. Batching = add a batch
  dim to the recurrence. **vLLM's PagedAttention does NOT cover this** —
  the right reference is vLLM's **SSM / Mamba state management** (mamba_cache),
  where each sequence has a fixed-size recurrent state slot. So: PagedAttention
  for the 16 attn layers, Mamba-style per-slot state for the 48 DN layers.
  Both are established vLLM patterns; we're not inventing, just combining.

## Memory budget per chip (31.8 GB)

**Weights** (bf8 MLP + bf16 norms/attn, sharded /4): ~7.5 GB/chip
(27B params ≈ 30 GB mixed-precision / 4). **Free ≈ 24 GB/chip.**

**KV cache** (16 attn layers, bf16, sharded by KV head → 1 head/chip):
  per token/chip = 1 head × 256 head_dim × 2 (K+V) × 2 B × 16 layers
                 = 16 KB/token/chip
  B=32 × 8192 ctx = 4.0 GB/chip (worst case, all at max ctx)
  B=32 × 2048 ctx = 1.0 GB/chip (realistic)

**DN recurrent state H_t** (48 DN layers, per SEQUENCE not per token):
  NV_PER_CHIP = 48/4 = 12 value heads/chip; k_dim=v_dim=128
  per seq/chip = 12 × 128 × 128 × 2 B × 48 layers = 18.4 MB/seq/chip
  B=32 = 590 MB/chip (bf16) or 1.18 GB/chip (fp32, for A010 coordination)

**Conv state** (48 DN layers): ~31 MB/chip at B=32. Negligible.

**lm_head output** [B, vocab] bf16 = 32 × 248320 × 2 = 15.8 MB. Fine
(or vocab-shard it, already done for 27B per feedback_vocab_sharded_lm_head).

### Verdict: B=32 fits with huge headroom

Total at B=32, 2048 ctx, bf16 DN state:
  weights 7.5 + KV 1.0 + DN 0.59 + conv 0.03 + acts ~0.1 = **~9.2 GB/chip**
vs 31.8 GB available. **Memory is NOT the constraint.** Could push B=64
or B=128 memory-wise. The binding constraint is matmul batch-width
efficiency + scheduler complexity, so **B=32 (one TILE width) is the
clean first target** as requested. Revisit larger B in CB6 if BW isn't
saturated.

## Trace strategy: fixed B=32, mask empty slots (vLLM CUDA-graph pattern)

Capture ONE decode trace at B=32. Live batches < 32 pad empty slots and
discard their output. This is exactly how vLLM uses CUDA graphs (capture
per batch size, pad up). Simplest; the padding waste at low occupancy is
acceptable for v1. Multi-trace (B ∈ {8,16,32}) is a CB6 optimization if
padding waste proves significant.

## Gate cleared → proceed to CB1

All CB0 unknowns resolved:
- Home: qb1
- B_max: 32 (memory allows far more; 32 is the clean tile width)
- Trace: fixed B=32 + mask
- Batching wrinkle: 48 DN layers need Mamba-style per-slot recurrent state;
  16 attn layers need vLLM PagedAttention block tables (already have paged SDPA)

**CB1 next**: add batch dim to the forward, isolate+validate each block
(attn, DN recurrence, MLP) B=1 vs B=8 before the full forward.

## CB1 isolation results (2026-05-27) — all three blocks de-risked

1. **DN recurrence** (the novel/risky part): `cb_dn_recurrence_batch_isolation.py`
   B=8, NV=4, K=V=128. Per-slot cos vs numpy ~0.99998; batched-vs-8×B=1
   diff = **0.0 (bit-exact slot independence, no cross-slot leak)**. The
   ttnn broadcast/reduce ops handle the leading B dim correctly:
   `mul([B,NV,K,V],[B,NV,1,1])`, `sum(dim=-2)`, outer-product update. CB1
   DN path = pure shape change (add leading B).

2. **Dense MLP**: pure matmul, M=1→B. Standard, low-risk (this is what
   A004's batched MoE matmul already proved at the matmul level).

3. **Paged SDPA attention**: ttnn
   `paged_scaled_dot_product_attention_decode` is **natively batched** —
   docstring confirms `input_tensor_q [1, b, nh, dh]`, `cur_pos_tensor [b]`,
   "parallelizes over b", and critically: **"If a position is given as (-1),
   compute for the corresponding index in the batch is skipped."** That -1
   is the built-in empty-slot mask for a fixed-B=32 trace. This IS vLLM
   PagedAttention; the 27B server just calls it at b=1 today.

## CB1 integration plan (precise — for review before the forward surgery)

The forward (`forward_token_tp_inner` + `gated_attn_step_tp` +
`dn_step_tp` + MLP) currently hardcodes b=1. The B-threading:

- **Input buffers**: `x_buf` [1,HIDDEN] → [B,HIDDEN]; `cur_pos_buf` [1] →
  [B] (with -1 for empty slots); `tok_buf` [1,1] → [B,1]; cos/sin lookup
  by per-slot pos → [B, ROTARY_DIM].
- **DN layers (×48)**: recurrent state [NV,K,V] → [B,NV,K,V]; conv state
  [CONV_DIM,KERNEL] → [B,CONV_DIM,KERNEL]; in_proj matmul M=1→B; recurrence
  ops gain leading B (validated). Per-slot state lives in the slot table.
- **Attn layers (×16)**: q/k/v projections M=1→B; paged SDPA decode with
  q [1,B,nh,dh], cur_pos_tensor [B], per-slot page_table [B, blocks/seq];
  paged_update_cache per slot (update_idxs_tensor [B]).
- **MLP**: matmuls M=1→B.
- **lm_head + sample**: logits [B,VOCAB]; per-slot argmax/sampler.
- **Trace**: capture at fixed B=32; empty slots → cur_pos=-1 (SDPA skips
  them), DN state for empty slots is don't-care (masked at output).

Risk notes: the paged_update_cache + sharded-write mem configs
(server_tp.py:1556-1579) are tuned for b=1 HEIGHT_SHARDED L1 writes —
the per-slot batched write is the fiddliest part; isolate it (CB2) before
wiring. The page_table generalizes 1 table → [B, blocks/seq] (block
manager, CB5).

**Status**: CB1 uncertainties resolved. The forward surgery is mechanical
but large + touches the production decode path → pause for review of this
plan before executing.

## All CB primitives validated (2026-05-27) — integration is now mechanical

| Primitive | Isolation | Result |
|---|---|---|
| DN recurrence batched | `cb_dn_recurrence_batch_isolation.py` | bit-exact slot independence, cos~0.99998 |
| Dense MLP batched | (shape-agnostic) | no change — rms_norm/matmul/add all leading-dim agnostic |
| Paged SDPA batched decode | `cb_paged_sdpa_batch_isolation.py` | B=4 ragged [7,20,3,-1], per-slot cos>0.9997 |
| Per-slot cur_pos + page tables | same | works; each slot attends own history |
| Empty-slot skip (cur_pos=-1) | same | skipped, don't-care output, no cross-slot corruption |
| Memory budget B=32 | CB0 arithmetic | ~9 GB/chip of 31.8 |

**Empty-slot note**: cur_pos=-1 does NOT zero the output — it leaves
don't-care data in that slot's output row. The scheduler must IGNORE
inactive slots' outputs (we know which slots are active). No masking of
other slots needed — they're unaffected.

Every primitive the batched forward depends on is now validated on our
ttnn build. The remaining work — threading B through forward_token_tp_inner
+ gated_attn_step_tp + deltanet_step_tp + I/O buffers + per-slot KV write
— is mechanical integration with no remaining algorithmic uncertainty.
Recommended approach: parallel batched path (new module / new functions),
B=1 validated bit-identical to production before B>1, production B=1
server untouched (zero regression risk). Then CB3 Orca scheduler on top.

## CB1/CB2 build status (2026-05-27)

`experiments/serve/server_tp_cb.py` — imports production server_tp.py
(untouched), redefines the batched forward. DONE so far:

- **setup_cb_state(state, B)**: per-slot page tables into a shared block
  pool (B × blocks_per_seq), per-attn-layer batched KV caches, per-DN-layer
  batched ssm [B,NV,K,V] + conv [B,CONV,K-1] state, batched input buffers.
- **update_input_buffers_batched** + **cb_reset_states**.
- **deltanet_step_batched**: full batched DN, manual recurrence, VERIFIED
  line-by-line vs production (caught + fixed a per-head-norm bug). Reads/
  commits per-slot conv+ssm state.

REMAINING for the first runnable B=1 milestone:

1. **gated_attn_step_batched** — two fiddly bits needing device iteration:
   - **RoPE broadcast**: cos/sin are now [B, ROTARY_DIM]; must broadcast
     over the head axis → reshape to [B, 1, ROTARY_DIM] vs production's
     [1, ROTARY_DIM] over [n_heads, HEAD_DIM]. Get the rank/broadcast right.
   - **paged_update_cache WRITE at B>1**: paged SDPA READ at B>1 is
     validated (cb_paged_sdpa_batch_isolation.py), but the WRITE uses
     HEIGHT_SHARDED L1 mem configs built at bootstrap for B=1 specific core
     grids (server_tp.py:1556-1579). The B-slot sharded write is the one
     genuinely-entangled piece — isolate it in the server context (CB2)
     OR validate it via the B=1 forward gate first (B=1 reuses the proven
     config) then generalize to B>1.
2. **forward_batch_tp_inner**: embed [B], cos/sin lookup [B], layer loop
   (deltanet_step_batched / gated_attn_step_batched + base.mlp_step_tp,
   which is shape-agnostic), final norm, lm_head, per-slot argmax.
3. **cb_validate_27b.py**: gate ladder — (a) B=1 batched == production B=1
   bit-identical, (b) B=8 identical-slots == B=1, (c) B=8 different slots
   each match its own B=1 reference.

The B=1-identical gate (3a) is the key checkpoint: it validates the entire
batched plumbing against the proven production path at B=1 before any B>1
shape risk. Run it before trusting any B>1 number.

## CB1 COMPLETE — batched forward bit-identical to production (2026-05-27)

`cb_validate_27b.py` gate (B=1 vs production, manual DN math) **PASSES all three**:
- **3a** CB B=1 logits == production B=1: **logit_cos = 1.000000 at every position**
  (raw + mean-centered + top-10 = 10/10), argmax matches token-for-token.
- **3b** B=4 identical slots: all four slots == B=1 (no shape-induced drift).
- **3c** B=4 DISTINCT equal-length prompts: each slot == its own B=1 reference
  → per-slot KV + DN recurrent state are fully isolated, no cross-slot leak.

### Root-cause that blocked CB1 (view-decay in the batched attn step)

`gated_attn_step_batched` (+ `_attn_finish`, `rope_b`) was calling
`ttnn.deallocate` on tensors whose `ttnn.slice`/`ttnn.reshape` **views were
still live** — `all_tt`, `qg`, `k_flat`, `v_flat`, `q_r`→`q_sdpa`,
`attn_out`→`attn_ph`, `gated`→`flat`, and `gate_tt` (a view of `all_tt`).
Freeing a source frees the buffer its views read, so q/k/v/gate read
stale/reallocated memory. At pos 0 only V matters (attn_out == V), which is
why the per-layer ladder showed `all_tt` bit-identical (cos 1.0) but the V
slice cos 0.008 — the smoking gun. Production `gated_attn_step_tp` never
deallocates these for exactly this reason. Fix: drop the unsafe early deallocs;
only free independent (materialized) tensors. Localization method: fresh-prod
per-layer hidden ladder → first divergence at the first ATTENTION layer (DN
layers L0-L2 were bit-identical) → attn sub-component probe (`all_tt`/`v`/
`attn_out`/`reduced`) pinned it to the V slice.

**Production server_tp.py is byte-for-byte pristine** (debug hooks were added,
used, then reverted via `git checkout`). All CB code lives in `server_tp_cb.py`.

### Next: CB2 (ragged per-slot lengths) → CB3 (Orca scheduler) → CB4 (trace @ B=32)
The forward is correct at static equal-length B; CB2 adds per-slot positions of
different lengths (the `cur_pos=-1` empty-slot skip + per-slot page tables are
already validated in isolation). Then the Orca iteration-level scheduler, then
capture one decode trace at fixed B=32 and measure tok/s vs the 12.93 baseline.

## Eager throughput scaling confirms the CB thesis (2026-05-28)

`cb_bench_throughput.py` — eager batched decode, 30 timed steps, sync-bounded:

| B  | ms/step (eager) | aggregate tok/s | vs B=1 |
|----|-----------------|-----------------|--------|
| 1  | 252.72          | 3.96            | 1.00×  |
| 8  | 254.01          | 31.50           | 8.0×   |
| 32 | 258.97          | 123.56          | 31.2×  |

Decode at B=1 is **fully memory-bound**: 32× the batch costs only **+2.5%** per
step (252.7→259.0 ms). The per-token matmul `[B,K]x[K,N]` streams the same
weight bytes regardless of B, so they amortize across the batch — aggregate
throughput scales ~linearly in B (31.2× at B=32). This is the "decode isn't
wasting 31/32 of the tile" win, quantified.

These are EAGER numbers (per-op Python dispatch ≈ 252 ms/step dominates;
production traced B=1 = 77 ms/step = 12.93 tok/s, so trace strips ~3.3× of
dispatch). Since compute barely grows with B, traced B=32 PROJECTS to
~12.93 × ~31 ≈ ~400 tok/s aggregate — but that is a projection. CB4 must
capture an actual B=32 trace and measure (never cite projection as measurement,
per feedback_real_vs_projected).

## CB4 — TRACED B=32 throughput MEASURED (2026-05-28)

`cb_bench_trace.py` captures one decode trace of the batched forward at fixed B
(vLLM CUDA-graph pattern) and times `execute_trace`. **traced-correctness PASS**:
traced B=1 teacher-forced argmax == production reference (execute_trace threads
DN ssm/conv + paged KV state in-place).

| B  | execute ms/step | aggregate tok/s | vs B=1 |
|----|-----------------|-----------------|--------|
| 1  | 77.15           | 12.96           | 1.0×   |
| 8  | 106.44          | 75.16           | 5.8×   |
| 32 | 212.62          | 150.50          | 11.6×  |

**Traced B=1 = 12.96 tok/s matches production 12.93** → the trace + manual-DN
path is sound. **B=32 continuous batching = 150.5 tok/s aggregate = 11.6×.**

The eager projection (~400 tok/s) was WRONG and this is why we measure
(feedback_real_vs_projected): eager is dispatch-bound so batching looked free
(31×); under trace dispatch is amortized and *compute* is exposed. Per-step
compute grows 2.76× from B=1→32 because the decode matmuls cross from
memory-bound (weight streaming amortized across the batch) toward compute-bound
(FLOPs ∝ B). Rough crossover: memory≈77 ms flat, compute≈4.2 ms/token, so
compute==memory near B≈18; beyond that compute dominates and throughput scales
sub-linearly. B=32 still delivers 11.6× at 2.76× latency (4.7 tok/s/seq).

Notes:
- DN here is MANUAL recurrence (owned_gdn is B=1-only). A batched owned-GDN
  kernel would cut the B=1 (and B>1) compute further — open lever.
- Higher B (64/128) untested; ~110 cores + memory allow it. The throughput peak
  is likely past B=32 but with diminishing returns as compute-bound. Worth a sweep.
- This is static equal-length B=32. Real serving needs CB2 (ragged per-slot
  lengths) + CB3 (Orca scheduler) on top — orchestration, not new device risk.
