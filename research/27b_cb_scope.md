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

## Throughput-vs-B sweep + roofline model (2026-05-28)

`cb_bench_trace.py --batches 1,16,32,48,64` (traced execute_trace, manual DN):

| B  | execute ms/step | aggregate tok/s | vs B=1 |
|----|-----------------|-----------------|--------|
| 1  | 77.15           | 12.96           | 1.0×   |
| 16 | 140.57          | 113.82          | 8.8×   |
| 32 | 212.69          | 150.45          | 11.6×  |
| 48 | 279.99          | 171.44          | 13.2×  |
| 64 | 348.79          | 183.49          | 14.2×  |

**Linear cost model (fits to <2%):** `step_ms ≈ 73 + 4.3·B`.
- 73 ms = batch-independent floor = weight streaming (memory-bound; the same
  ~7.5 GB/chip of weights is loaded once per step regardless of B).
- 4.3 ms/seq = per-token compute (matmul FLOPs ∝ B, per-slot DN recurrence,
  per-slot SDPA). Crossover (compute == memory) at B ≈ 73/4.3 ≈ 17.
- **Aggregate throughput asymptote = 1000/4.3 ≈ 232 tok/s** (fully
  compute-bound). B=64 reaches 183 = 79% of the asymptote.

**Implications:**
- Sweet spots: B=32 balances throughput (11.6×) and per-seq latency
  (4.7 tok/s/seq); B=64 maxes throughput (14.2×) at 2.9 tok/s/seq. Pick by SLA.
- To raise the ~232 tok/s ceiling, cut the 4.3 ms/token: **batched owned-GDN
  recurrence kernel** (manual DN is the current B>1 path) and/or faster batched
  matmuls are the levers. The kernel-dataflow representation work
  (research/kernel_dataflow_representation.md) targets exactly this.

## CB2 — ragged positions + mid-batch admission VALIDATED (2026-05-28)

`cb_validate_ragged.py` PASS. New primitive `cb_reset_slots(state, [slot])`:
masked-multiply clears ONLY the admitted slot's DN recurrent + conv state
(Mamba-style state-slot reuse). KV needs no reset — per-slot cur_pos bounds the
SDPA read, so a sequence restarting at pos 0 overwrites its own blocks.

Test: slot0 runs A at pos 0..5 continuously; slot1 runs C at pos 0..2 then is
ADMITTED B at step 3 (cb_reset_slots([1])) and runs B at pos 0..2 — so during
steps 3..5 the two slots sit at DIFFERENT positions (slot0=3,4,5; slot1=0,1,2)
in the same forward. Result: slot0 argmax == ref_A (unaffected by slot1's churn
+ reset), slot1 argmax == ref_B (fresh state, own KV). Proves: per-slot ragged
positions, admission DN-reset, KV self-overwrite, slot isolation across a reset.

Device foundation for continuous batching is now COMPLETE: batched forward
(CB1), throughput (CB4), ragged + admission (CB2). All that remains is CB3 —
the Orca iteration-level scheduler (Python control loop, no new device risk).

## CB3 — Orca scheduler WORKS (2026-05-28)

`experiments/serve/cb_scheduler.py` — Orca iteration-level scheduler over the
validated primitives. 5 requests through 2 slots, 44 iterations: **every
request's continuous-batched greedy output is bit-identical to its standalone
B=1 greedy reference.** Admission (cb_reset_slots on a free slot), eviction
(EOS/max_new), queueing (5 reqs > 2 slots → 3 queued + admitted as slots free),
prefill→decode threading, and per-slot isolation all correct.

Design = vLLM/Orca adopted as-is: fixed B slots, iteration-level admit/advance/
evict between steps, Mamba-style per-slot DN state reset on admission, prefill
one-token/step through the decode path, FREE slots parked at cur_pos=0 (isolated,
output ignored). Greedy (argmax) decode.

**CB device foundation + scheduler COMPLETE**: CB1 (correct batched forward),
CB2 (ragged + admission), CB3 (scheduler), CB4 (traced throughput 11.6×@B=32).

### Remaining (perf + productionization, not correctness)
- **Wire the CB4 trace into the scheduler** so step() runs execute_trace
  (150 tok/s @ B=32) instead of the eager forward (the scheduler is currently
  eager). cb_reset_slots + update_input_buffers run eager between execute_trace
  calls — straightforward.
- Chunked prefill (currently one-token/step prefill is simple but slow).
- Sampling (DRY/rep-penalty) instead of greedy; OpenAI endpoint (user deferred).
- Batched owned-GDN kernel to lift the ~232 tok/s compute ceiling.

## CB3 traced — scheduler runs at trace speed (2026-05-28)

`cb_scheduler.py --trace`: step() runs `execute_trace` instead of the eager
forward (admission cb_reset_slots + update_input_buffers run eager between
replays — they mutate persistent buffers in-place, which the next replay reads).
5 reqs / 2 slots: **all bit-identical to references**, 11.8 iters/s (~85 ms/iter
= the 73+4.3·B model at B=2; eager was ~252 ms/iter). Confirms admission +
execute_trace compose. The CB serving system is correct AND production-speed.

**CB1–CB4 all DONE.** A working vLLM-style continuous-batching system for 27B on
Blackhole: bit-identical correctness, 11.6×@B=32 / 14.2×@B=64 throughput, Orca
scheduler with admission/eviction/queueing, traced execution.

## CB5 — per-block compute attribution: DeltaNet owns 97% of the slope (2026-05-28)

`cb_profile_blocks.py` (trace-timing ablation: full − skip_X = block X cost):

| Block | B=1 ms | B=32 ms | scaling ms/seq |
|-------|--------|---------|----------------|
| DeltaNet (48 layers, manual) | 38.93 | **170.70** | **4.25** |
| MLP (64 layers)              | 29.29 | 29.41   | 0.00 (flat) |
| Attention (16 layers)        |  6.56 |  8.33   | 0.06 |
| rest (embed/norm/lm_head/CCL)|  2.36 |  4.15   | 0.06 |
| **full**                     | 77.14 | 212.60  | 4.37 |

**The DeltaNet manual recurrence is 97% of the per-token compute slope**
(4.25 of 4.37 ms/seq) and 80% of the B=32 step. MLP is perfectly memory-bound
(0.00 ms/seq — weight bytes amortize ideally across the batch); attention and
the rest barely scale.

**Conclusion (profile-driven):** the single highest-value CB throughput lever is
a **batched DeltaNet recurrence kernel**. The existing `owned_gdn` kernel is
B=1-only; the CB path uses MANUAL recurrence (per-slot matmuls/muls/outer-
products on [B,NV,K,V] — FLOPs ∝ B, many small ops). Cutting the 4.25 ms/seq by
N× lifts the ~232 tok/s asymptote by ~N× (e.g. 2.5× → ~500 tok/s). This is the
clear next kernel target, and a prime candidate for the TDG dataflow methodology
(research/kernel_dataflow_representation.md).

## CB6 — batched DeltaNet kernel: scope + feasibility (2026-05-28)

CB5 pins the target: DN recurrence = 97% of the per-token compute slope. The
manual CB recurrence (server_tp_cb deltanet_step_batched, mirroring server_tp.py
:810-819) does decay·H + gated·outer(k,v) on the [B,NV,K,V] H-state with ~6 ttnn
ops — so the H-state DRAM traffic (read+write H every step) scales ∝ B. That
traffic, not FLOPs, is almost certainly the 4.25 ms/seq (TEST THIS with a
within-DN profile before building — non-negotiable).

**The fused kernel that fixes this already exists for B=1:**
`ttnn.experimental.qwen36_gdn_decode_owned(H, q, k, v, decay, beta4, ...)`
(production default `deltanet_recurrence_mode="owned_gdn"`) fuses the recurrence
into one op, minimizing H round-trips. A **batched** version (B>1 leading dim)
is the lever to lift the ~232 tok/s asymptote ~N×.

**Feasibility constraints (must resolve before building):**
1. **Host**: the owned kernels are built into **qb2's** tt-metal, NOT qb1's
   (CLAUDE.md §5). CB lives on qb1. Options: (a) build the kernel into qb1's
   tt-metal; (b) move CB to qb2 (has the kernel + the prod TP server — risk to
   prod); (c) keep manual DN on qb1, prototype the batched kernel on qb2.
2. **B=1 in the program factory**: confirm whether qwen36_gdn_decode_owned
   hardcodes batch=1 (typical for custom Metal program factories) — batching
   may need program-factory changes + recompile, not just a bigger input.
3. **Correctness gate (A010)**: H-state precision is the long-context drift
   lever — bf16 H is the minimum; bf8 H is OUT. So we CANNOT shrink H traffic by
   quantizing H; the win must come from fusion / keeping H resident, not
   precision. The batched kernel must preserve bf16 H math (cosine-ladder gate).

**Recommended staging (G0→G4, isolation-first per project pattern):**
- G0: within-DN profile (confirm H-traffic is the cost) + read qwen36_gdn_decode_owned
  source on qb2; numpy oracle for the batched recurrence.
- G1: batched kernel on ONE core, B>1, bit-exact vs numpy + vs manual.
- G2-G3: multi-core / sharding for B=32 (TDG dataflow methodology applies here —
  research/kernel_dataflow_representation.md).
- G4: integrate into server_tp_cb, cosine-ladder long-context gate, re-profile.

This is a multi-session kernel R&D effort + a host/build decision → flagged for
the user. The CB system itself is complete and correct on manual DN today.

## CB6/G0 — owned_gdn is hard-asserted B=1 (2026-05-28)

`cb_owned_gdn_batch_isolation.py`: built the owned-GDN inputs at B=1/2/4 and
called `ttnn.experimental.qwen36_gdn_decode_owned` directly.
- B=1: cos(H_new)=0.999984, cos(out)=0.999986 vs numpy GatedDeltaNet ref —
  confirms the ref + I/O layout (H[B,NV,K,V], q/k[B,NV,1,K], v[B,NV,1,V],
  decay/beta[B,NV,1,1]).
- B=2 and B=4: **`TT_FATAL @ qwen36_gdn_decode_owned_device_operation.cpp:118:
  state_logical[0] == 1`** — the device op hard-asserts batch=1.

GOOD NEWS that supersedes the earlier feasibility note: **the owned-GDN kernels
ARE present in qb1's ttnn build** (`ttnn.experimental.qwen36_gdn_decode_owned`
and 7 siblings) — the CLAUDE.md "qb2-only" note is STALE (like the old "qb1 has
no fabric" note). So no host move is needed; the work is local to qb1.

**The batched-DN-kernel task is now precisely defined:**
1. Relax the `state_logical[0] == 1` assert in the device op AND batch the
   program factory to parallelize the recurrence over B (the compute likely
   hardcodes B=1 layout — must verify in the kernel source, not just the assert).
2. Rebuild ttnn on qb1 (`cmake --install` does NOT update the venv .so —
   cp build_Release/ttnn/_ttnn*.so into .venv). Must NOT regress the B=1 path
   (production prod server uses owned_gdn at B=1).
3. Correctness: bf16 H math preserved (A010 long-context gate); cosine ladder.
4. Then swap CB manual recurrence → batched owned_gdn; re-run cb_profile_blocks
   to confirm the DN slope drops, re-measure the throughput asymptote.

This modifies a custom Metal kernel + rebuilds ttnn (risk to the B=1 prod path,
long build cycle) → flagged for user direction. A no-rebuild alternative worth a
quick timing check first: loop owned_gdn per-slot (B× B=1 calls) inside the CB
DN step and compare traced cost vs the manual batched recurrence — if the fused
per-slot op beats the 6-op manual path even serialized, it's a win with zero
kernel changes.

## CB6/G0b — FOLD trick works for recurrence; kernel has high-S output bug (2026-05-28)

Key insight: owned_gdn parallelizes over `slots = state.shape[1]` and each slot's
recurrence is INDEPENDENT (README SPMD unit = (slot, value_tile)). So fold the
batch into slots: reshape [B,NV,K,V] → [1, B·NV, K,V] and call the UNMODIFIED
kernel — no program-factory batching, no rebuild needed for the work split.

`cb_owned_gdn_batch_isolation.py` (fold mode), per-slot out cos:
| B | S=B·NV | cos(H_new) | cos(out) | pattern |
|---|--------|------------|----------|---------|
| 1 | 12 | 0.99998 | 0.99999 | OK |
| 2 | 24 | 0.99999 | 0.99998 | OK |
| 3 | 36 | 0.99998 | 0.844 | slots 0-16 WRONG (~0.5), 17-35 OK (1.0) |
| 4 | 48 | 0.99999 | 0.605 | slots 0-40 WRONG, 41-47 OK |
| 8 | 96 | 0.99999 | 0.564 | mostly wrong, tail better |

**The recurrence (H_new) is bit-correct at ALL S** — the fold + work-split are
fine. Only `out = q @ state_next` is wrong, and only for EARLY slots once
S > ~24-32 (prod kernel was only tested at slots=12). H doesn't use q/output;
out does → the bug is isolated to the **output path** (state_next→out matmul),
almost certainly a circular-buffer depth / read-after-overwrite when many blocks
pipeline (early slots' state_out tiles get clobbered before the output matmul
consumes them). The "last slots survive" signature fits a too-shallow output CB.

**Revised plan: this is a localized OUTPUT-path kernel fix, not a full batching
rewrite.** Copy the op → `qwen36_gdn_decode_owned_batched` (keep B=1 prod
untouched, per user), fix the output CB pipelining for high slot counts, rebuild
ttnn, re-run this isolation at B=32 (S=384), then fold-integrate into CB.

## CB6/G1 — batched DeltaNet kernel WORKS (debug_mode=10, no rebuild) (2026-05-28)

Root cause of the high-S output bug: the production path (mode 0) computes
`out = q @ state_next` by reading `cb_state_out`, which the WRITER also pops —
a dual-consumer race. Once a core handles >1 block (slots > ~24), the writer
pops a block's state_out tiles before the output matmul reads them → wrong
OUTPUT for early slots (recurrence unaffected; it never reads q/output). Kernel
was only tested at slots=12 (B=1) so it never surfaced.

Fix (patched compute kernel, `experiments/kernel_patches/qwen36_gdn_decode_owned/`):
a `safe_out` path (selected by `debug_mode=10`) routes the output matmul through
`cb_state_next_internal` (compute-owned), leaving `cb_state_out` to the writer
alone. Done as an IN-LOOP CONDITIONAL inside mode 0's branch (a duplicate branch
overflowed the 70656 B TENSIX kernel-config limit at 77024 B; the conditional
fits — add_state_to_two is already in the binary). **Device kernels JIT-compile
from the .cpp → NO ttnn rebuild.** `debug_mode=0` is byte-identical to the
original, so the B=1 prod path (server_tp owned_gdn) is UNTOUCHED.

`cb_owned_gdn_batch_isolation.py` (fold [B,NV,K,V]→[1,B·NV,K,V], debug_mode=10):
  B ∈ {1,2,4,8,16,32}: cos(H_new) & cos(out) ≈ 0.99998 — ALL PASS.
  debug_mode=0: B=1,2 pass, B≥3 out-fails (the original race) — confirms mode 0
  unchanged.

**The batched DeltaNet recurrence kernel is validated.** Next: integrate into
`server_tp_cb.deltanet_step_batched` (produce owned_gdn inputs per fold + call
with debug_mode=10), validate cb_validate_27b bit-identical + cosine ladder,
then re-run cb_profile_blocks/cb_bench_trace to measure the DN-slope drop.

## DNK-G2 — batched owned_gdn integrated + measured (2026-05-28)

`server_tp_cb.deltanet_step_batched` gains a `cb_dn_recurrence_mode="owned_gdn"`
path: fold q/k/v/decay/beta + slot['ssm'] into [1, B·NV, …], call
`qwen36_gdn_decode_owned(..., native_io=True, debug_mode=10)` (the patched
batched-safe kernel). The op updates state in place via the folded view (=the
commit); out [1, B·NV·V] → [B, VAL_DIM_CHIP].

**Correctness** (`cb_validate_27b.py --owned-gdn`, prod ref also owned_gdn):
3a logit_cos=1.0 (bit-identical B=1), 3b/3c PASS. Trace correctness PASS.

**Throughput** (`cb_bench_trace.py --owned-gdn`, traced):

| B  | manual ms | owned_gdn ms | manual tok/s | owned_gdn tok/s | gain |
|----|-----------|--------------|--------------|-----------------|------|
| 1  | 77.15     | 74.80        | 12.96        | 13.37           | +3%  |
| 32 | 212.69    | 190.41       | 150.45       | 168.06          | +11.7% |
| 64 | 348.79    | 307.24       | 183.49       | 208.30          | +13.5% |

Per-block profile (`cb_profile_blocks.py --owned-gdn`): DN slope 4.25 → 3.61
ms/seq (−15%); B=32 DN block 170.7 → 148.5 ms. **Asymptote ~232 → ~277 tok/s.**

**Combined: B=64 owned_gdn = 208 tok/s = 16.1× the B=1 production 12.96 tok/s**,
bit-identical. The remaining DN cost is the surrounding ops (in_proj/out_proj
matmuls, conv1d, q/k + output norms, decay/gate) — all scale with B; owned_gdn
only fuses the recurrence. Those matmuls are the next lever to push past ~277.

## DNK-G3 — DN matmuls are FLAT; the slope is per-slot vector ops (2026-05-28)

`cb_dn_matmul_microbench.py` (traced, real DN weights w_in[5120,4120],
w_out[1536,5120]):

| B  | in_proj ms | out_proj+AR ms |
|----|-----------|----------------|
| 1  | 0.114     | 0.088          |
| 32 | 0.114     | 0.088          |
| 64 | 0.116     | 0.098          |

**The DN matmuls are flat with B (0.000 ms/seq slope)** — weight-streaming /
memory-bound, batch fully amortized (like the MLP). So matmul core_grid tuning
(the A004 lever) is NOT the next win here — contrary to the initial hypothesis.

The 3.61 ms/seq DN slope (owned_gdn) is the **per-slot VECTOR ops** whose
ACTIVATION volume ∝ B: conv1d, q/k L2-norm (×2), output RMSNorm, gqa-repeat,
and the recurrence. owned_gdn already fused the recurrence's vector ops (4.25→
3.61). To push past ~277 tok/s the remaining vector ops must be fused too — a
custom-kernel effort like owned_gdn. **conv1d is the prime candidate** (existing
diagnosis `feedback_conv1d_diagnosis`: 65% sum-reduce, 21% state mgmt, 13% mul;
`feedback_conv1d_circular_buffer` says it needs a tt-metal custom op). The
owned_decay_gate kernel exists but decay/gate is tiny ([B,NV] scalars), so low
value. Next profile: within-DN vector-op attribution (skip conv / qk-norm /
out-gate) to rank the targets before the next fused kernel.

## DNK-G3b — within-DN ranking: conv1d is 71.8% of DN (2026-05-28)

`cb_profile_dn.py` (DN-only forward, skip one sub-op, traced, B=32, owned_gdn):

| DN sub-op | cost (48 layers) | % of DN-only (152.7 ms) |
|-----------|------------------|--------------------------|
| **conv1d**    | **109.70 ms** | **71.8%** |
| recurrence (owned_gdn) | 13.16 ms | 8.6% |
| q/k L2-norm   | 0.51 ms | 0.3% |
| output RMSNorm gate | 0.26 ms | 0.2% |

**conv1d dominates the DN cost by far** (~2.3 ms/layer at B=32). Almost
certainly the K=4→32 TILE-padding tax: the conv builds `conv_input [B, CONV_DIM_
CHIP, K=4]` and in TILE layout K=4 pads to 32, so the concat/mul/sum/silu/slice
all process 8× the logical data. CONV_DIM_CHIP≈2560 → [32,2560,32-padded] ≈ 5 MB
of mostly-padding traffic per layer.

**Next lever = kill the conv K-padding.** Two paths:
1. ttnn reformulation (no kernel): shift-and-accumulate —
   `out = Σ_k w[:,k] · window_k` as 4 muls + 3 adds on [B, CONV_DIM_CHIP] tiles
   (no K dim → no padding), state = 3 separate [B,C] columns. (Old
   `feedback_conv1d_circular_buffer` said slice-shift failed at B=1 single-chip;
   re-attempt given the quantified 72% cost + batched context.)
2. custom conv1d kernel (depthwise 3-tap), like owned_gdn.

Potential: if conv 109→~20 ms, DN-only 152→63 ms, full step ~190→~100 ms →
B=32 ~168→~320 tok/s (≈2×). Highest-value remaining lever by far.
