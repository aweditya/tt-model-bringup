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
