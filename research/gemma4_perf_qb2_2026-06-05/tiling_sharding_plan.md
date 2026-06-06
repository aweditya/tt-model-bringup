# Gemma 4 12B IT decode (B=1) — tiling/sharding landscape + Rounds 10-20 queue

Living document. Companion to `log.md` (per-round implementation history) and
`reports/round8_matmul_bw_breakdown.txt` (durable PM-BW breakdown that anchors
prioritisation).

**Baseline at write time (post-Round-9, qb2)**:
- Traced **47.0 ms/tok** (21.28 tok/s), 100/100 token-for-token gate.
- With `TT_GM4_MLP_DTYPE=bfp8 TT_GM4_LM_HEAD_DTYPE=bfp8` env vars: **46.0 ms/tok**
  (Round-8 lever, reverted in default; safe per Round-9 ablation).
- Hardware: qb2 (1,4) P150 mesh, Blackhole — `dram_grid_size = (8,1)`, compute
  grid `(11,10)` confirmed by `gm4_dram_sharded_mlp_probe.py:15:51:03`.

**Durable diagnosis (Round 7 + Round 8)**:
- PM-BANDWIDTH / PM-COMPUTE = **24.6×** across the full per-forward signposted
  region. Decode at B=1 is **DRAM-bandwidth-bound, NOT math-bound**.
- **Matmul = 99.5% of all bandwidth-bound time** (43.487 ms PM-BW / forward).
- Per-shape (`reports/round8_matmul_bw_breakdown.txt`):
  - MLP `[32, 3840] × [3840, 3840]` triplet × 144/forward = **30.90 ms = 71% PM-BW**
  - lm_head `[32, 3840] × [3840, 65536]` × 1 = 3.66 ms = 8.4%
  - K/V/Q/O projections (per-chip [32, 3840] × [3840, {512, 1024}]) aggregate ~17%
- **Implications**: any future lever class that does NOT reduce matmul DRAM
  traffic — fidelity tuning (HiFi swap), compute-side activation fusion, single-op
  TM elimination after eager-only wins — is bounded by the 4% COMPUTE budget. The
  remaining win pool lives in (a) bytes-per-weight-read reduction (Round 8 bfp8),
  (b) access-pattern reshape (Round 10 DRAM-sharded matmul), and (c) hiding
  DRAM read latency behind prior compute (prefetcher / dual-resident weights).

---

## §1. Landscape map — every applicable tiling/sharding lever

Each row: **What it does** / **Shape regime** / **Gemma 4 fitness** / **Estimated
gain vs 47 ms baseline** / **Implementation cost + risk** / **Precedent
(file:line in tt-metal demos)**.

### A. Matmul memory-config + program-config families

#### A1. WIDTH-sharded DRAM weight + dedicated DRAM-sharded program config

- **What**: weight tensor laid out as WIDTH_SHARDED across all P150 DRAM banks
  (8); matmul kernel `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
  parallelises weight reads across banks via NoC. Compared to default
  INTERLEAVED-DRAM weight (single-bank cyclic read), reduces effective DRAM
  serialisation latency per matmul.
- **Shape regime dominates**: BW-bound matmul with K dimension reused, M small
  (≤32), N wide — exactly the per-chip decode shape.
- **Gemma 4 fitness**: **YES** — Round 10 Phase 3 probe PASSED on all 3 MLP
  shapes (cos=0.9999937, slightly MORE accurate vs fp32 ref). Per-chip shape
  `[32, 3840] × [3840, 3840]` is in the same class as the canonical test
  (`test_matmul_dram_sharded.py:280-283`).
- **Estimated gain**: **-2 to -4 ms/tok traced (-4 to -8%)** if the MLP triplet
  PM-BW (15.5 ms post-bfp8, 30.9 ms current bf16) drops 2-3× per the test's
  documented speedup ceiling, projected through the 24.6× BW/COMP ratio.
- **Cost + risk**: medium — 3 file edits (`np_stacked_to_sharded` mem_config
  kwarg, per-callsite activation reshard, program_config wiring). Risk: at
  M=32 the per-call work is small enough that dispatch noise can mask the
  win; the `in0_block_w = K/num_cores/TILE/4 = 3.75 → 3` rounding hasn't been
  field-tested. Fallback ladder `[3, 5, 15]` covered in probe.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:58-72`
  (W1W3_RING_MEMCFG / W2_RING_MEMCFG); `tt-metal/tests/ttnn/nightly/unit_tests/
  operations/matmul/test_matmul_dram_sharded.py:50-185` (canonical test +
  block-size math); `tt-metal/models/demos/llama3_70b_galaxy/tt/model_config.py:
  2312-2320` (`create_dram_sharded_mem_config` helper).
- **Our probe**: `experiments/cb/isolate/gm4_dram_sharded_mlp_probe.py:82-156`
  (helpers `_dram_weight_mem_cfg`, `_activation_l1_width_sharded`,
  `_dram_sharded_program_config`).

#### A2. WIDTH-sharded DRAM weight for lm_head

- **What**: same as A1 but on the lm_head matmul `[32, 3840] × [3840, 16384]`
  (per-chip after vocab-shard). lm_head needs a SEPARATE helper because vocab
  size 262144 has been padded to a multiple of 24 cores in Llama 70B Galaxy
  (`create_dram_sharded_mem_config_lm_head`); for Gemma 4 we shard 8-way per
  P150 (= the 8 DRAM banks) on `padded_vocab/4` = 65536 cols.
- **Shape regime**: largest single-shot matmul of the forward; widest N (16384
  per chip).
- **Gemma 4 fitness**: **MAYBE** — lm_head is 8.4% of matmul PM-BW; ceiling is
  ~0.7 ms even at perfect DRAM-shard speedup. Worth landing once A1 is proven.
  Bigger issue: lm_head is the only matmul writing into a B-wide N (after
  vocab-shard); the WIDTH_SHARDED L1 output mem_config may interact poorly with
  the existing `ttnn.all_gather(dim=-1)` collective in `_lm_head_argmax`.
- **Estimated gain**: -0.3 to -0.7 ms/tok traced (-0.6 to -1.5%).
- **Cost + risk**: low (one matmul callsite + one helper) but bounded ceiling.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/lm_head.py:60-71`
  (`create_dram_sharded_mem_config_lm_head` selection); `tt-metal/models/demos/
  llama3_70b_galaxy/tt/model_config.py:2321-2336` (the lm_head-specific helper).

#### A3. WIDTH-sharded DRAM weight for Q/K/V/O projections

- **What**: same lever as A1 applied to the 4 attention projections.
- **Shape regime**: per-chip [32, 3840] × [3840, {512, 1024}] (K/V/O sliding,
  Q sliding, Q/K/V/O global). N is much smaller than MLP's 3840, so the wider
  shard width (= N / num_cores) means each core sees a thinner stripe.
- **Gemma 4 fitness**: **MAYBE** — per-shape PM-BW totals 2.29+2.29+2.52+2.29
  = ~9.4 ms (~22% of matmul PM-BW pre-bfp8). Same access-pattern lever applies;
  the open question is whether the N=512 case has enough N-per-core
  (= 512 / 8 = 64 = 2 tiles) to amortise the program-config setup cost.
- **Estimated gain**: -0.5 to -1.5 ms/tok traced (-1 to -3%). Smaller than A1
  because individual per-shape PM-BW is smaller.
- **Cost + risk**: low after A1 lands — the helpers + integration recipe are
  already in place; just 4 more callsite edits + 4 more weight uploads.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py:
  142,192` (`wqkv_mem_config`, `wo_mem_config` via `create_dram_sharded_mem_config`).

#### A4. HEIGHT-sharded matmul (activation-sharded, weight-replicated)

- **What**: opposite of A1 — activation laid out HEIGHT_SHARDED across cores,
  weight replicated per core. Each core does a slice of M rows × full K × full
  N. Dominates when M is large and reduction K is small.
- **Shape regime**: prefill (M >> 1), CB at B>>1 (M = B*TILE), batched matmul.
- **Gemma 4 fitness**: **NO** at B=1 single-stream decode (M=32 = one tile).
  Becomes attractive at CB B=4+ (Gemma 4 CB ships at B=4 per task #167) when
  M = B * TILE = 128 = 4 tiles and the per-core M-rows is non-trivial.
- **Estimated gain**: not applicable to B=1 single-stream. At B=4 CB:
  unmeasured; reserved as Round-18-class lever after CB perf pass.
- **Cost + risk**: medium — different program config (`MatmulMultiCoreReuse`,
  no DRAM-shard tag), different shard math.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py`
  prefill path (`forward_prefill`); `models/tt_transformers/tt/model_config.py`
  prefill matmul configs.

#### A5. BLOCK-sharded matmul (output tile-grid sharded in 2D)

- **What**: output tensor BLOCK_SHARDED across a 2D core grid; each core
  produces a per_core_M × per_core_N tile-block. Best at large M, N, K with
  approximately square output.
- **Shape regime**: prefill at sequence > tile, batched training, attention QKᵀ
  in non-decode mode.
- **Gemma 4 fitness**: **NO** for B=1 decode (M=32, N=3840 — strongly rectangular).
  Could apply to attention's intermediate matmul if we batch across heads, but
  the per-head head_dim_global=512 is already small.
- **Estimated gain**: not applicable to decode B=1.
- **Precedent**: `tt-metal/tests/ttnn/nightly/unit_tests/operations/matmul/
  test_matmul_2d.py`; SDPA chunked-prefill path in the prefill compute kernel.

### B. RMSNorm variants

#### B1. Standard `ttnn.rms_norm` on REPLICATED activation (current Gemma 4)

- **What**: classic LayerNorm op; reads full hidden, writes full hidden,
  REPLICATED across mesh.
- **Shape regime**: any.
- **Gemma 4 fitness**: **CURRENT**. 337 LayerNorm ops/forward (~7/layer).
- **Cost + risk**: zero (baseline).

#### B2. Distributed RMSNorm — `rms_norm_pre_all_gather` + `all_gather` + `rms_norm_post_all_gather`

- **What**: split RMSNorm into a per-chip "compute statistics" kernel that
  produces a small (1, 4) stats tile, all-gather the stats across the cluster
  axis, then a per-chip "apply normalisation" kernel that reads its 1/N share
  of the hidden state. The hidden activation must be SHARDED across mesh
  (dim=-1, 1/N per chip) for this to actually save bytes — otherwise the
  all-gather of stats is added cost with no reduction in the rms_norm read.
- **Shape regime**: TP-sharded hidden state in Megatron-style. Only meaningful
  with sharded `h`.
- **Gemma 4 fitness**: **MAYBE** — Gemma 4 currently uses `all_reduce` after
  o_proj/down_proj which produces REPLICATED `h`. To benefit from distributed
  RMSNorm we'd need to switch the row-parallel matmul output to
  `reduce_scatter` (Llama Galaxy pattern), which itself reduces DRAM write
  traffic (1/N tiles vs N tiles all-reduce). This is the Megatron-TP rewrite —
  192 sites across 48 layers. **Not a single-lever change.**
- **Estimated gain**: combined with reduce_scatter rewrite: -8 to -12 ms/tok
  (projected by Round 8 finding; needs validation). The norm itself is
  COMP-bound not BW-bound; the win comes from the reduce_scatter side.
- **Cost + risk**: HIGH (multi-day, architectural). Risk: needs `tt_ccl`
  semaphore infra for `tt_distributed_rmsnorm` (`llama_ccl.py`).
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/distributed_norm.py:
  1-90` (the full DistributedNorm class); `tt-metal/models/demos/gemma4/tt/
  rms_norm.py:43-80` (Tenstorrent's upstream Gemma 4 demo distributed RMSNorm —
  uses `rms_norm_pre_all_gather` + `all_gather(dim=3, cluster_axis=1)` +
  `rms_norm_post_all_gather`); `tt-metal/models/demos/llama3_70b_galaxy/tt/
  llama_ccl.py` (`tt_distributed_rmsnorm`, `tt_sharded_distributed_rmsnorm`).

#### B3. `LayerNormShardedMultiCoreProgramConfig` (single-device sharded norm)

- **What**: same `rms_norm` op but the input is L1-sharded across a core grid,
  the program_config specifies subblock/block tiling. Saves vs default by
  keeping the activation resident in L1 across norm + downstream matmul.
- **Shape regime**: any layered topology where the input is already sharded.
- **Gemma 4 fitness**: **MAYBE** — feasible without the Megatron rewrite. If
  we shard `h` per chip (e.g. WIDTH_SHARDED L1 across 8 cores) before each
  norm + immediately following matmul, we save the `to_memory_config` round
  trip. Adds reshard ops but eliminates DRAM read between norm and matmul.
- **Estimated gain**: -0.5 to -1.5 ms/tok traced (per Tracy v2: LayerNorm
  marker-dropped, real cost unmeasured but estimated 3-5 ms/forward).
- **Cost + risk**: medium — needs sharded input contract on every norm
  callsite; risk of breaking the rms_norm shape-drift bug ([[ttnn-rms-norm-shape-drift-at-B-gt-1]]).
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/distributed_norm.py:
  21-45` (the program_config + L1 sharded input/output mem_cfgs);
  `tt-metal/models/demos/gemma4/tt/rms_norm.py:48-54` (same construction).

### C. Weight precision (BW reduction without access-pattern change)

#### C1. `bfloat8_b` weights on MLP + lm_head — **Round 8 win, reverted in Round 9**

- **What**: block-floating-point 8-bit shared-exponent-per-tile. Halves weight
  DRAM read bytes.
- **Gemma 4 fitness**: **YES** — Round 9 ablation proved bfp8 is bit-stable vs
  bf16 at decode B=1 across L=128/512/1024 (token-identical output at same
  seed). The Round 8 long-context regression was a false attribution (IT model
  prompt-echo artefact, not precision drift).
- **Estimated gain**: **-0.87 ms/tok (-1.86%) measured**, n=9 100/100 PASS.
- **Cost + risk**: zero — already implemented behind env gate
  `TT_GM4_MLP_DTYPE=bfp8` / `TT_GM4_LM_HEAD_DTYPE=bfp8`. Default reverted in
  Round 9 (conservative).
- **Precedent**: `experiments/serve/server_35b_ttnn.py:320-336` (35B MoE expert
  bfp8); `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:90-96`
  (Llama Galaxy MLP).

#### C2. `bfloat4_b` weights on MLP

- **What**: 4-bit block-fp shared-exponent. Quarters DRAM read.
- **Gemma 4 fitness**: **MAYBE** — Llama Galaxy ships bfp4 on w1/w3 with a
  documented "normally ok here but sub .99 pcc for llama 3.1 weights" caveat.
  Needs full long-context retrieval check. Smaller per-weight precision than
  bf16 → risk surface bigger than bfp8.
- **Estimated gain**: -1.5 to -3 ms/tok if PCC holds (doubles bfp8's win).
- **Cost + risk**: low to implement (one dtype flag in `upload_mlp_layer`);
  high correctness risk.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:90,
  94` (`ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b`).

#### C3. `bfloat8_b` weights on Q/K/V/O projections

- **What**: extend C1 to attention. Each attention projection PM-BW is
  smaller; combined ~17% of matmul PM-BW.
- **Gemma 4 fitness**: **YES** — Round 8 probe already covered Q sliding + O
  sliding (cos=0.9999678/0.9999712). Identical scaffold to C1.
- **Estimated gain**: -0.2 to -0.5 ms/tok traced.
- **Cost + risk**: low — same `_resolve_dtype("TT_GM4_ATTN_DTYPE")` env gate
  as C1.
- **Precedent**: same as C1.

### D. CCL collective changes (DRAM-traffic-aware)

#### D1. `ttnn.reduce_scatter` replacing `all_reduce` after row-parallel matmul

- **What**: row-parallel matmul (o_proj, down_proj) produces partial outputs
  per chip; reduce_scatter sums them AND scatters the result so each chip
  holds 1/N. Saves vs all_reduce by avoiding the all-gather phase.
- **Shape regime**: any Megatron-style TP layer with row-parallel projection.
- **Gemma 4 fitness**: **MAYBE** — direct prerequisite for B2 distributed
  RMSNorm. Standalone the win is small (we'd need to reshard back to
  REPLICATED before the next norm, defeating the saving).
- **Estimated gain**: combined with B2: -3 to -5 ms/tok of the projected
  distributed-norm bundle.
- **Cost + risk**: HIGH (architectural; combined with B2).
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py:
  ~600+` (wo path uses reduce_scatter); `experiments/serve/
  server_nemotron3_nano_ttnn.py` (Nemotron-3 MoE uses `ttnn.reduce_scatter`
  for the replicate→shard pattern at v0.4.1.e commit `a65af53`).

#### D2. `all_gather_matmul` (fused gather + matmul)

- **What**: col-parallel matmul wants REPLICATED input; if upstream gives
  SHARDED input, fuse the all_gather into the matmul kernel.
- **Shape regime**: prefill / large M.
- **Gemma 4 fitness**: **MAYBE** — only useful if we've already split the
  forward into sharded-h regions (i.e. distributed-norm-shaped pipeline).
  At single-stream B=1 with REPLICATED `h` between layers, doesn't apply.
- **Cost + risk**: HIGH — couples with D1 + B2 in the Megatron rewrite.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py`
  (`double_matmul_line_reduce_scatter`).

### E. Cross-layer DRAM prefetch (asynchronous weight load)

#### E1. Llama 70B Galaxy `prefetcher_common.py` — global circular buffer

- **What**: dedicated "prefetcher" cores that asynchronously load layer L+1's
  weights into L1 while layer L's matmul is running. Hides DRAM read latency
  behind compute. Requires sub-device semaphores, persistent global circular
  buffers, and sender/receiver core mappings.
- **Shape regime**: any BW-bound model where matmul takes > 1 DRAM-read worth
  of time per layer.
- **Gemma 4 fitness**: **MAYBE** — fits the BW-bound diagnosis perfectly.
  Multi-week scope. Best return on the BW-bound diagnosis if all easier levers
  are exhausted.
- **Estimated gain**: -5 to -15 ms/tok if it lands (best case: full hide of
  DRAM matmul read time, bounded by COMP budget).
- **Cost + risk**: VERY HIGH (multi-week). Risk: requires the `tt_ccl` infra
  and persistent global circular buffer plumbing; brittle to changes in the
  forward op order.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/prefetcher_common.py`
  (the full infra); `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:
  108-117` (`prefetch` method + per-tensor insert); `:184-188`
  (`global_cb=...` in matmul call).

### F. Operator/dispatch elimination (Rounds 1-6 family)

#### F1. Cross-layer fusion of `paged_fused_update_cache` (K + V same call)

- **What**: Round 1 fused per-layer K and V cache writes into a single op
  (88/forward → 88/forward, but each op now does 2 cache writes). The
  cross-layer version would batch ALL layer K+V writes into a single op.
- **Gemma 4 fitness**: **NO** — Round 8 PM-BW measurement showed this op
  class is COMP-bound (PM-BW=0). Round 1's hypothesis ("Round 8 candidate
  for paged_fused_update_cache cross-layer batching") was deprioritised by
  Round 7+8 findings. **Do not re-attempt.**
- **Precedent**: N/A.

#### F2. `rotary_embedding_llama_fused_qk` with HF→Llama weight permutation

- **What**: replace `_apply_full_rope`'s 3 ops (`roll + mul + addcmul`, post
  Round 5) with one fused op operating on Q+K together. Saves 4-5 ops × 96
  calls → 1 op × 48 calls = ~240 ops/forward.
- **Shape regime**: any RoPE-using model.
- **Gemma 4 fitness**: **NO at current scope** — the fused kernel's
  `trans_mat` is 32×32 tile-granularity and only implements interleaved-half
  rotate ((out_{2i}, out_{2i+1}) = (-in_{2i+1}, in_{2i})). HF Gemma 4 uses
  split-half rotate (rotate_half([a, b]) = [-b, a]) which is head_dim-wide and
  cannot be represented by the trans_mat. Workaround: permute Q/K projection
  weights + cos/sin tables to interleaved layout offline. Multi-hour scope.
- **Estimated gain**: -1.5 to -3 ms/tok traced (in the BW-bound regime the
  RoPE ops are COMP-bound, but ~96 ops eliminated × ~30 μs each = 3 ms).
- **Cost + risk**: HIGH (offline weight permutation across Q + K + cos/sin
  + K-cache; needs a separate experimental branch with weight-permutation
  utilities + reduced-scope probe).
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py:
  489-492` (the fused kernel call); `tt-metal/models/demos/llama3_70b_galaxy/
  tt/llama_rope.py` (trans_mat construction).

### G. Speculative / multi-token decode

#### G1. Speculative decode (draft model + verify)

- **What**: a small draft model proposes K tokens; the main model verifies
  in parallel. At decode B=1 this is essentially batch-1-with-K-positions
  per forward.
- **Gemma 4 fitness**: **MAYBE** — would need an external draft model. Not in
  scope without a second checkpoint. Architecturally compatible (12B IT trace
  already handles batched-B forward via CB v1).
- **Cost + risk**: VERY HIGH (multi-month, infra + drafter).
- **Precedent**: not in tt-metal core demos; vLLM-style. Out of scope for
  Rounds 10-20.

#### G2. SDPA fusion / paged_fused_update_cache + SDPA merge

- **What**: combine the cache write and SDPA read into one op.
- **Gemma 4 fitness**: **NO** — both ops are COMP/dispatch bound, not BW
  bound; merging them doesn't save DRAM traffic. Round 1 already fused the
  K+V cache write side; SDPA stays separate.
- **Precedent**: none.

### H. Activation re-sharding inside a layer (L1 residency)

#### H1. WIDTH_SHARDED L1 activation between RMSNorm and matmul

- **What**: after `rms_norm`, instead of writing the result to interleaved
  L1, write to WIDTH_SHARDED L1 matched to the next matmul's `in0` contract.
  Eliminates a `to_memory_config` reshard. Pairs naturally with A1 (the
  DRAM-sharded matmul wants WIDTH_SHARDED L1 input anyway).
- **Shape regime**: any pipeline where norm → matmul → norm → matmul.
- **Gemma 4 fitness**: **YES** as a follow-on to A1. The Round 10 DRAM-sharded
  MLP probe already constructs a WIDTH_SHARDED L1 activation; landing A1 forces
  this lever to exist at the MLP callsite.
- **Estimated gain**: -0.5 to -1 ms/tok traced (eliminates the per-layer
  `to_memory_config` dispatch that A1 alone would add).
- **Cost + risk**: low (already required by A1).
- **Precedent**: `gm4_dram_sharded_mlp_probe.py:114-136` (`_activation_l1_width_sharded`).

#### H2. L1-resident weights (small weights only)

- **What**: weights small enough fit in L1 stay resident there per forward.
  No DRAM read per call.
- **Shape regime**: tiny weights — small embeddings, scalar buffers. Decoder
  layer weights are 3840×3840×bf16 = 30 MB — way too big.
- **Gemma 4 fitness**: **NO** for MLP/QKV (too big). Could apply to
  `layer_scalar`, q_norm, k_norm, v_norm weights (each ≤ 1 KB).
- **Estimated gain**: negligible (these weights are already cached effectively).
- **Cost + risk**: zero benefit at the layer-weight scale.

### I. Tensor parallelism across mesh dim (4-chip vs single-chip)

#### I1. Mesh-TP across all 4 chips for B=1 decode (Gemma 4 currently does this)

- **What**: Gemma 4 12B is ALREADY TP=4 across the (1,4) mesh — each chip
  holds 1/4 of the hidden dim shard. The lever question is whether to
  DOUBLE-TP (combine mesh-TP with intra-chip tensor-shard).
- **Gemma 4 fitness**: **NO further win** — we're already at TP=4. The
  alternative — single-chip with weights fitting on one Blackhole — would
  require model size << 12B (12B at bf16 = 24 GB, doesn't fit one P150).
  The current TP=4 is forced by memory, not chosen for perf.
- **Estimated gain**: N/A.

#### I2. Pipeline parallelism (split LAYERS across chips, not hidden dim)

- **What**: each chip holds N/4 layers; activation flows chip-to-chip.
- **Shape regime**: B>=4 pipeline-balanced inference.
- **Gemma 4 fitness**: **NO at B=1** — pipeline-fill at B=1 wastes 3/4 chips
  while one chip is computing. Only beats TP=4 at B large enough to fill the
  pipeline.
- **Estimated gain**: N/A at single-stream.

### J. Other tracked candidates from research

#### J1. `concat_heads_decode -> o_proj` fusion (tt-metal #44945)

- **What**: fuse the head-concat reshape with the o_proj matmul.
- **Gemma 4 fitness**: **NO measurable** — Round 7 brief noted "concat and
  reshape are TM ops (metadata-only); the matmul is the real work. Saving
  the TM ops gets <0.3 ms expected." BW-bound diagnosis confirms this.
- **Estimated gain**: <0.3 ms.
- **Precedent**: tt-metal #44945.

#### J2. Sharded `gate_proj` matmul with `program_config.fused_activation`

- **What**: Round 4 already landed `activation="gelu"` on the gate_proj
  matmul (interleaved). The fully-sharded variant would put gelu inside the
  LLK writeback rather than as a post-op.
- **Gemma 4 fitness**: **NO** — Round 7 BW-bound diagnosis: the activation is
  COMP-side (4% of forward); fully fusing it lands ~0.2 ms gain at best.
- **Estimated gain**: ≤0.5 ms.
- **Precedent**: `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:154-160`
  (ttnn.mul with SILU activation inside the matmul mem_config).

#### J3. `paged_fused_update_cache` audit for SDPA contract

- **What**: ensure the disjoint K/V mem_configs used at Round 1 are still
  optimal post-Round-10 sharding changes.
- **Gemma 4 fitness**: monitor, not a lever.

---

## §2. Round 10-20 prioritised queue (Roadmap)

**Ordering principle**: (a) ROI per hour, (b) compounding (DRAM-sharded matmul
infrastructure unlocks sharded-RMSNorm + sharded-attn), (c) risk floor.

| Round | Lever | Expected % win | Prereq | Risk | Time (hrs) | Validation gate |
|-------|-------|----------------|--------|------|------------|-----------------|
| **10** | A1: DRAM-sharded MLP weights + L1 activation reshard + dedicated program_config | -4 to -8% (-2 to -4 ms) | none (probe PASSED Phase 3) | medium (dispatch noise at M=32) | 1 | 100/100 token-for-token × 3 runs + traced ms/tok delta |
| **11** | Re-enable C1 (bfp8 MLP + lm_head) by default after A1 lands | -1.86% (-0.87 ms, **stacks with A1**) | A1 + needle-haystack re-baseline at L=128/512/1024 to confirm Round 9 verdict survives stacking | low | 0.5 | 100/100 + L=128/512/1024 needle Y≥2/3 each |
| **12** | A2: DRAM-sharded lm_head + lm_head-specific helper | -0.6 to -1.5% (-0.3 to -0.7 ms) | A1, A2 helper port | low (small, bounded) | 1 | 100/100 token-for-token × 3 runs |
| **13** | A3: DRAM-sharded Q/K/V/O attention projections | -1 to -3% (-0.5 to -1.5 ms) | A1 (shared helpers) | low (same scaffold as A1) | 1.5 | 100/100 + per-shape probe (forks `gm4_dram_sharded_mlp_probe.py`) |
| **14** | H1: WIDTH_SHARDED L1 activation chain through RMSNorm → matmul (eliminates reshard) | -1 to -2% (-0.5 to -1 ms) | A1+A3 | medium (rms_norm shape-drift bug risk) | 2 | 100/100 + per-callsite probe |
| **15** | C3: bfp8 on attention Q/K/V/O projections | -0.4 to -1% (-0.2 to -0.5 ms) | C1 baselined, needle gate | low | 0.5 | 100/100 + L=128 needle Y≥2/3 |
| **16** | B3: `LayerNormShardedMultiCoreProgramConfig` for all 7 rms_norms/layer | -1 to -3% (-0.5 to -1.5 ms) | H1 (sharded activation pipeline) | medium | 3 | 100/100 + per-callsite probe + cos vs fp32 ref |
| **17** | F2 (spike): `rotary_embedding_llama_fused_qk` with offline HF→Llama weight permutation | -3 to -6% (-1.5 to -3 ms) | dedicated branch + weight-permutation utility | HIGH (multi-day; precision concession on head_dim>128) | 6-8 | sliding-only first; 100/100 token-for-token; cos≥0.999 |
| **18** | C2 (gated): bfp4 on MLP w1/w3 only (keep w2 bfp8) | -2 to -4% (-1 to -2 ms) if PCC holds | C1+C3 stable; long-context regression suite passing on bfp8 | HIGH (precision) | 1 | 100/100 + L=128/512/1024 needle Y≥2/3 each + L=2048 |
| **19** | B2+D1 (architectural): distributed RMSNorm + reduce_scatter — **multi-session experimental branch** | -8 to -12% (-4 to -6 ms) projected, **biggest remaining single lever** | A1-A3+B3+H1+C1+C3 stack landed | VERY HIGH (multi-day; tt_ccl infra; 192 rewrite sites) | 12-20 | separate branch; staged 4-layer microbench → 12-layer → full 48 |
| **20** | E1 (architectural): cross-layer DRAM prefetcher — **multi-week experimental branch** | -10 to -30% (-5 to -15 ms) projected | full Round 19 stack + tt_ccl infra | VERY HIGH (multi-week; brittle) | 40+ | separate branch; only after Round 19 + remaining levers harvested |

**Open queue stop-gate**: rounds 18-20 should each be planned as their own
session-pair (spike → land or revert with documented findings, Round 7
template). Rounds 10-17 are continuous incremental progression.

**Cumulative projection** (rounds 10-17 stacked, midpoint estimates):
- 10: 47.0 → 44.0 ms/tok
- 11: → 43.1 (re-stack bfp8)
- 12: → 42.6
- 13: → 41.6
- 14: → 40.9
- 15: → 40.6
- 16: → 39.6
- 17: → 37.4 (if F2 spike lands)
- **Targeted state after Round 17: ~37 ms/tok (27 tok/s), ~21% cumulative improvement above today's 47 ms/tok.**

After Round 17 the remaining levers (18-20) are higher risk and one-or-more-day
each. The 37 ms/tok target is a credible single-shot milestone for the project.

---

## §3. Reference table — file:line citations

For quick lookup when implementing each lever.

| Symbol | Reference file:line |
|--------|---------------------|
| DRAM-sharded weight mem_config helper | `tt-metal/models/demos/llama3_70b_galaxy/tt/model_config.py:2312-2320` |
| DRAM-sharded lm_head helper | same file `:2321-2336` |
| DRAM-sharded matmul test (block-size math) | `tt-metal/tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py:50-185` |
| DRAM-sharded matmul production callsite | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:58-72,154-188` |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | `gm4_dram_sharded_mlp_probe.py:139-156` |
| Distributed RMSNorm (Llama Galaxy) | `tt-metal/models/demos/llama3_70b_galaxy/tt/distributed_norm.py:1-90` |
| Distributed RMSNorm (Gemma 4 upstream) | `tt-metal/models/demos/gemma4/tt/rms_norm.py:43-80` |
| `tt_distributed_rmsnorm` + sharded variant | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py` |
| Prefetcher infra | `tt-metal/models/demos/llama3_70b_galaxy/tt/prefetcher_common.py` |
| Prefetcher tensor insertion + matmul global_cb call | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:108-117,154-188` |
| `rotary_embedding_llama_fused_qk` callsite | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py:489-492` |
| Trans_mat construction (interleaved RoPE) | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_rope.py` |
| bfp4_b / bfp8_b dtype selection | `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:90-96` |
| MLP `RING_MEMCFG` (DRAM-sharded) construction | `tt-metal/models/demos/llama3_70b_galaxy/tt/model_config.py:1439-1450` |
| Round 8 PM-BW breakdown (our anchor) | `research/gemma4_perf_qb2_2026-06-05/reports/round8_matmul_bw_breakdown.txt` |
| Our Round 10 Phase 3 probe (PASSED) | `experiments/cb/isolate/gm4_dram_sharded_mlp_probe.py` |
| Our MLP production callsite (target) | `experiments/serve/server_gemma4_unified_ttnn.py:289-318` (upload) + `:1531-1537` (forward) |
| Our `np_stacked_to_sharded` (extend with memory_config kwarg) | `experiments/serve/server_gemma4_unified_ttnn.py:163-176` |

---

## §4. Round 10 Phase 4 integration recipe (executable spec for THIS session)

This is the Phase 4 spec the previous Round 10 RETRY (Phase 1-3) handed off.

### Files to touch (only these)

1. `experiments/serve/server_gemma4_unified_ttnn.py`
2. `experiments/cb/isolate/gm4_dram_sharded_mlp_probe.py` — **already PASSED**, no
   edits unless we add the production callsite probe.

### Edits

**Edit 1 — `np_stacked_to_sharded`**: add `memory_config=None` kwarg; pass through
to `ttnn.from_torch`. (Line 163-176.)

**Edit 2 — `upload_mlp_layer`**: behind env gate `TT_GM4_DRAM_PREFETCH=1`,
build the WIDTH_SHARDED DRAM memory config from the probe's `_dram_weight_mem_cfg`
helper for each of gate/up/down per-chip shape `[3840, 3840]`. Pass to
`np_stacked_to_sharded(..., memory_config=...)`. (Lines 309-317.)

**Edit 3 — paged forward `_layer_forward_pos0_paged`**: behind same env gate,
before the gate_proj matmul, reshard `pre_ff` from interleaved L1 to
WIDTH_SHARDED L1 via `ttnn.to_memory_config(pre_ff, _activation_l1_width_sharded(mesh, M=32, K=3840, num_cores=8))`.
Pass `program_config=_dram_sharded_program_config(M=32, K=3840, N=3840, num_cores=8, num_banks=8)`
to all 3 MLP matmuls (gate, up, down). For down_proj, the input is `mid`
(post-mul) — reshard `mid` similarly. Output of each matmul stays WIDTH_SHARDED
L1; reshard back to interleaved before `all_reduce_tt`. (Lines 1531-1537.)

**Edit 4 — legacy `_layer_forward_pos0`**: mirror Edit 3 if it's exercised (the
v04 trace path uses `_layer_forward_pos0_paged`; legacy path is not on the hot
path for our benchmark). Optional.

### Helpers to fork from probe into server

Copy `_dram_weight_mem_cfg`, `_activation_l1_width_sharded`,
`_dram_sharded_program_config` from `gm4_dram_sharded_mlp_probe.py:82-156` into
`server_gemma4_unified_ttnn.py` (top of file, near `np_stacked_to_sharded`).
Gate the imports under the env-var check so non-DRAM-prefetch paths don't pay
any cost.

### Validation gate

- 100/100 token-for-token (`gm4_v04_trace_validate.py`) × 3 runs.
- max|delta| = 0 across runs (token-stable).
- Traced ms/tok delta vs baseline 47.0 ms/tok measured at n=3.
- If positive (regression), revert per Round 7 template (HiFi2 negative finding)
  and document.

### Env gate convention

`TT_GM4_DRAM_PREFETCH=1` enables A1 (DRAM-sharded MLP). Future levers:
`TT_GM4_DRAM_PREFETCH_ATTN=1` (A3), `TT_GM4_DRAM_PREFETCH_LMHEAD=1` (A2).
Names reserved; not yet implemented.
