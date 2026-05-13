# Phase C'7 — Multi-Chip Tensor Parallel Design Doc

**Target:** Qwen3.6-27B on 4 × P150 in qb2 (working NoC fabric, validated A7 collectives).
**Baseline to beat:** 215 ms/tok eager paged decode, single chip (`feedback_paged_eager_landed.md`).
**Performance goal:** 60–80 ms/tok → 12–17 tok/s (3.0–3.5× scaling).
**Correctness gate:** `run_91r --paged` cosine ≥ 0.9998 on every full-attn layer; greedy "Paris is the capital of" sanity ≥ 60 tokens.

---

## 0. Architectural inputs we are sharding (recap)

From `experiments/91f_qwen36_27b_full_ondevice.py` + `research/phase_b_prime_qwen36_27b_plan.md`:

| Symbol | Value | Comes from |
|---|---:|---|
| `hidden` | 5120 | text_config.hidden_size |
| `num_layers` | 64 (48 DeltaNet + 16 gated full-attn) | layer_types `[L L L F] × 16` |
| `intermediate_size` (MLP) | 17408 | text_config |
| `N_Q` (full-attn Q heads) | 24 | num_attention_heads |
| `N_KV` (full-attn KV heads) | **4** | num_key_value_heads — clean 4-way TP |
| `head_dim` | 256 | partial rotary 64 / passthrough 192 |
| `N_V_HEADS` (DeltaNet value heads) | **48** | linear_num_value_heads — clean 4-way (12/chip) |
| `N_K_HEADS` (DeltaNet key heads) | **16** | linear_num_key_heads — clean 4-way (4/chip) |
| `K_DIM, V_DIM` (per DeltaNet head) | 128, 128 | linear_{key,value}_head_dim |
| `CONV_DIM` (DeltaNet conv) | 2·16·128 + 48·128 = 10 240 | 2·KEY_DIM + VAL_DIM |
| `conv_kernel` | 4 | linear_conv_kernel_dim |
| `vocab` | 248 320 | tokenizer |

All three of "shard along value head", "shard along KV head", "shard along intermediate" land on multiples of 4 with **zero padding**. This is the design's biggest gift — we never have to pad to a TP-friendly number.

---

## 1. Model decomposition strategy

### 1.1 DeltaNet: shard along `N_V_HEADS` (12 value heads / chip) — **Option A**

**Why A, not B (replicate state):** the SSM state tensor `H ∈ [N_V_HEADS, K_DIM, V_DIM] = [48, 128, 128]` is 3.0 MB per layer fp32, **6.6 GB total across 48 layers**. Replicating it on each chip is 80% of our weight budget on a 32 GB device and gives no compute reduction. With Option A:
- `H` shards to `[12, 128, 128] = 768 KB` per chip per layer → **192 MB total**, 30× saving.
- The recurrence `H_new = H·decay + k_col·delta` is **per-head independent** (see `91f:218-227`) — no cross-head communication during the scan. Heads stay local.
- conv1d weight and `conv_state ∈ [CONV_DIM, K-1]` also shard along the head dimension (per-head 256-wide stripes; CONV_DIM is per-head laid out as `[2·KEY_DIM_q | 2·KEY_DIM_k | VAL_DIM]`, see §1.1.1).

The DeltaNet input projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, fused to `in_proj_all`) shard their **output** dim head-wise. The output projection `out_proj : [VAL_DIM, hidden] = [6144, 5120]` shards its **input** dim — Megatron-style column→row pattern. This means:

1. RMSNorm on `x` runs identically on every chip (input is replicated).
2. Fused `in_proj_all` shards out-dim, each chip computes its 12 heads worth.
3. conv1d, q/k L2 norm, q-scale, softplus/decay, recurrence — **fully local**.
4. RMSNormGated + silu(z) gate — per-head, local.
5. `out_proj` is row-parallel: each chip emits a partial sum of shape `[1, hidden]`.
6. **One all-reduce** sums the four partials → replicated full-hidden residual.

#### 1.1.1 conv1d sharding gotcha (open question)

`mixed_qkv` layout from `in_proj_qkv` is `[q_flat | k_flat | v_flat]` with `q_flat,k_flat ∈ [N_K_HEADS·K_DIM]`. The GQA-interleave at `91f:191-194` replicates 1 K head into `N_REP = N_V/N_K = 3` Q-head rows.

If we shard out-dim of `in_proj_qkv` head-wise on value heads, each chip gets `12/3=4` k/q heads. Clean. **BUT** the layout in memory is `q_flat | k_flat | v_flat`, not interleaved. We must re-layout the weight so that each chip's slice is contiguous: `[q_local|k_local|v_local]` per chip. Easiest: replace `in_proj_qkv` concat order with per-head-block striping on host before upload.

**Probe (Phase C'7.4 gate):** with random weights, run one DeltaNet step single-chip and 4-chip, compare H_new shard concatenated to single-chip H_new — must be bit-identical (no collective involved; pure local recurrence). If not, the per-head re-striping is wrong.

### 1.2 Gated full attention: shard along `N_KV` heads (1 KV head / chip)

`N_KV = 4` and `mesh = 4`, so each chip gets **exactly one KV head + 6 Q heads** (GQA 24:4 = 6 Q per KV). The paged KV cache shards along `dim=1` (`[max_blocks, N_KV, BLOCK_SIZE, HEAD_DIM]` → `[max_blocks, 1, BLOCK_SIZE, HEAD_DIM]` per chip).

Map of `gated_attn_step_ondevice_paged` (`91f:380-504`) onto the 4-chip mesh:

| step | per chip | collective? |
|---|---|---|
| `rms_norm(x)` (input replicated) | local | — |
| `attn_qkv = linear(h, W_qkv)` column-parallel on out-dim | each chip computes its 6 Q + 1 K + 1 V | — |
| `q_norm`, `k_norm` per-head RMSNorm | local | — |
| partial RoPE (Level 1, `91f:317-344`) | local; cos/sin replicated | — |
| `paged_update_cache` to local KV slice | local | — |
| `paged_scaled_dot_product_attention_decode` | local (chip has 6 Q heads + its 1 KV head + full cache) | — |
| sigmoid gate, residual prep | local | — |
| `o_proj : [N_Q·head_dim=6144, hidden=5120]` row-parallel | each chip emits partial [1, hidden] | **all-reduce on hidden** |

**Crucial property:** because each chip's 6 Q heads all attend to the same 1 KV head, SDPA never needs cross-chip data. No all-gather of Q before SDPA, no all-gather of KV. This is the cleanest tensor-parallel split in the model.

### 1.3 MLP: shard along `intermediate_size` (4352 / chip)

`gate_proj`, `up_proj : [5120, 17408]` → per-chip `[5120, 4352]` (column parallel).
`down_proj : [17408, 5120]` → per-chip `[4352, 5120]` (row parallel).

| step | per chip | collective? |
|---|---|---|
| `post_attention_layernorm(x)` (input replicated) | local | — |
| `gate = silu(linear(h, W_gate))` → `[1, 4352]` | local | — |
| `up   = linear(h, W_up)` → `[1, 4352]` | local | — |
| `mul(gate, up)` → `[1, 4352]` | local | — |
| `down = linear(...,W_down)` → `[1, 5120]` partial | local | — |
| residual sum | — | **all-reduce on hidden** |

This is identical to Megatron MLP TP. Validated pattern, no surprises.

### 1.4 Embedding + lm_head

- `embed_tokens : [vocab=248320, hidden=5120]` ≈ 1.27 GB bf16. **Replicate.** Read-mostly, lookup is local, no need to pay communication.
- `lm_head : [hidden=5120, vocab=248320]` ≈ 1.27 GB bf16. **Shard along vocab** (62 080 / chip). Each chip emits its vocab slice; argmax can be done two ways:
  1. **All-gather the logit slice** (1.27 GB / 4 = 317 MB per chip per step) then argmax local — expensive.
  2. **Two-stage argmax**: local argmax + local max-value → all-gather *only* the 4 (idx, val) pairs → host picks global argmax → broadcast next token. 32 bytes/step instead of MB.

Pick option 2. The 4-pair gather is free and we already gate sampling on host anyway.

### 1.5 Final RMSNorm

Replicated input → identical local op. No collective.

---

## 2. Sharding spec per tensor (per layer)

bf16 unless noted. Per-chip footprint is the slice; "interleaved across chips" means each chip stores a different slice (no overlap).

### 2.1 DeltaNet layer (×48)

| tensor | full shape | shard dim | per-chip shape | per-chip bytes |
|---|---|---|---|---:|
| `input_layernorm` | [5120] | replicate | [5120] | 10 KB |
| `in_proj_all` (fused qkv/z/a/b) | [5120, 12480] | out-dim per-head striped | [5120, 3120] | 32.0 MB |
| `conv1d_weight` | [10240, 4] | per-head striped | [2560, 4] | 20 KB |
| `linear_attn_norm` | [128] (per V head) | replicate | [128] | 256 B |
| `A_log` | [48] | shard | [12] | 24 B |
| `dt_bias` | [48] | shard | [12] | 24 B |
| `out_proj` | [6144, 5120] | in-dim (row) | [1536, 5120] | 16.0 MB |
| `post_attention_layernorm` | [5120] | replicate | [5120] | 10 KB |
| MLP `gate_proj` | [5120, 17408] | col | [5120, 4352] | 44.5 MB |
| MLP `up_proj` | [5120, 17408] | col | [5120, 4352] | 44.5 MB |
| MLP `down_proj` | [17408, 5120] | row | [4352, 5120] | 44.5 MB |
| state `H` | [48, 128, 128] fp32 | shard heads | [12, 128, 128] | 0.75 MB |
| `conv_state` | [10240, 3] | per-head striped | [2560, 3] | 15 KB |

**Per DeltaNet layer per chip: ~182 MB weights + 0.77 MB state.**

### 2.2 Gated full-attn layer (×16)

| tensor | full shape | shard dim | per-chip shape | per-chip bytes |
|---|---|---|---|---:|
| `input_layernorm` | [5120] | replicate | [5120] | 10 KB |
| `attn_qkv` (fused Q+gate, K, V) | [5120, 12288+2048] = [5120, 14336] | col, head-grouped | [5120, 3584] | 36.6 MB |
| `q_norm` | [256] | replicate | [256] | 512 B |
| `k_norm` | [256] | replicate | [256] | 512 B |
| `o_proj` | [6144, 5120] | row | [1536, 5120] | 16.0 MB |
| paged `kv_cache_k` | [num_blocks, 4, block=64, 256] | shard on N_KV=1 | [num_blocks, 1, 64, 256] | (see §4) |
| paged `kv_cache_v` | same | shard on N_KV=1 | same | (see §4) |
| MLP `gate_proj`, `up_proj`, `down_proj` | as above | as above | as above | 133.5 MB |

The attn_qkv layout: per chip we keep `Q_local | gate_local | K_local | V_local` where each "local" is one KV-group worth. The per-chip width is `2 · 6·256 + 1·256 + 1·256 = 3584` ✓.

**Per full-attn layer per chip: ~186 MB weights** (KV cache excluded; tracked separately).

### 2.3 Per-chip total weight footprint

| component | count | MB/chip each | MB/chip total |
|---|---:|---:|---:|
| DeltaNet layers | 48 | 182 | 8 736 |
| Gated full-attn layers | 16 | 186 | 2 976 |
| Embedding (replicated) | 1 | 1 272 | 1 272 |
| lm_head (sharded vocab) | 1 | 318 | 318 |
| Final RMSNorm | 1 | 0.01 | 0.01 |
| **Total weights, bf16** | | | **13 302 MB ≈ 13.0 GB** |

If we apply bf8 to the matmul-heavy weights (`in_proj_all`, `out_proj`, MLP three, `attn_qkv`) as validated in `feedback_bf8_weights.md`, this drops by ~1.8× → **~7.4 GB per chip**, matching the spec in the prompt.

---

## 3. Collective ops needed

Per token, in `forward(x)`:

| location | collective | shape | bytes (bf16) | granularity |
|---|---|---|---:|---|
| DeltaNet `out_proj` partial sum → residual | `ttnn.all_reduce` (sum) on hidden | [1, 5120] | 10 KB | 48× per token |
| Gated-attn `o_proj` partial sum → residual | `ttnn.all_reduce` (sum) on hidden | [1, 5120] | 10 KB | 16× per token |
| MLP `down_proj` partial sum → residual | `ttnn.all_reduce` (sum) on hidden | [1, 5120] | 10 KB | 64× per token |
| lm_head argmax | host-side gather of 4 (idx, val) | 32 B | 32 B | 1× per token |
| **Total per token** | | | **~1.3 MB collective traffic** | **128 all-reduces** |

That is **128 all-reduces per token**. Each all-reduce on 10 KB at 4 P150s over Ethernet has a latency floor dominated by setup (semaphores, link config) not bandwidth — likely **0.05–0.2 ms per collective** (open question: probe in C'7.1). At 0.1 ms × 128 = **12.8 ms/tok** pure collective overhead.

### 3.1 Choice of primitive: all-reduce vs reduce-scatter+all-gather

`models/tt_transformers/tt/ccl.py:tt_all_reduce` shows the two strategies:

- **N300/T3K path (1-row mesh):** `reduce_scatter_minimal_async`. The result is sharded — useful if the downstream op is also sharded. We're not (downstream is the next layer's RMSNorm which expects replicated input), so we'd need an extra all-gather.
- **TG path (2D mesh):** `all_gather` + `fast_reduce_nc`.

For 4×1 (which qb2 effectively is — 4 P150s, treat as 1D row), we should use the **N300/T3K branch**: `ttnn.experimental.reduce_scatter_minimal_async` and then `ttnn.experimental.all_gather_async` to bring the sum back to replicated. That's a **2-hop "composite all-reduce"** — equivalent to `ttnn.all_reduce` but tunable.

For the tiny payloads (10 KB), the simpler `ttnn.all_reduce` is probably better — fewer kernel launches. **Open question (probe in C'7.1):** measure both at our exact payload.

### 3.2 Where we explicitly do NOT communicate

- After RMSNorm — input is already replicated; norm is deterministic, so all chips get the same output without a sync.
- After `apply_partial_rope` — `cos_tt`/`sin_tt` are replicated, math is per-row, no comms.
- After `paged_scaled_dot_product_attention_decode` — each chip works on its own 6 Q heads + 1 KV head + its slice of cache; the output is `[1, 6, 256]` per chip and concatenates *implicitly* through the row-parallel `o_proj` (each chip sees its 6 heads' contribution to the partial sum, then we all-reduce).

This is the Megatron column→row pattern and the reason it scales: the only synchronization point is the residual.

### 3.3 Topology

`ttnn.Topology.Linear` for a 4×1 mesh (vs `Ring`). The 4 P150s in qb2 are connected as a line through QSFP-DD; `Linear` matches the physical wiring. From `ccl.py:118`, `topology=ttnn.Topology.Linear` is the documented default for non-TG meshes.

---

## 4. Memory budget per chip

P150 DRAM: 32 GB. Usable after firmware/scratch: ~30 GB.

### 4.1 Weights

bf8 weights (from §2.3 scaled by 1/1.78): **~7.4 GB**.

### 4.2 KV cache, per user, full-attn only

`kv_cache_k_tt` paged layout: `[max_num_blocks_total, N_KV_local=1, BLOCK_SIZE=64, HEAD_DIM=256]` bf16, ×16 layers, ×2 (k+v).

For 32k context, blocks = ceil(32 768 / 64) = 512 per user.

Per-user-per-chip = 512 × 1 × 64 × 256 × 2 bytes × 2 (k+v) × 16 layers
                  = 512 × 32 768 × 32
                  = **537 MB**.

(Without TP, this would be 512×4×64×256×2×2×16 = 2.15 GB on one chip. The TP shard saves us 4×.)

### 4.3 DeltaNet recurrent state, per user

Per chip per layer: `H : [12, 128, 128] × fp32` = 0.75 MB. `conv_state : [2560, 3] × bf16` = 15 KB. Per user = 48 × (0.75 + 0.015) = **36.7 MB**.

### 4.4 Activation + scratch

Per-token activations are tiny (`[1, 5120]` bf16 = 10 KB per layer). Trace capture buffer is ~10–50 MB (from C'4 v4 history). Budget ~1 GB for activations + L1 staging.

### 4.5 Total at 32k context, 1 user

7.4 GB weights + 0.54 GB KV + 0.04 GB DeltaNet state + 1 GB scratch = **~9.0 GB / 30 GB used**.

### 4.6 Concurrent users (batch)

Each additional user costs 0.54 GB KV + 0.04 GB DeltaNet state = **~0.58 GB/user** at 32k context, **~0.04 GB/user at MAX_POS=2048**.

| context | per-user cost | extra capacity (30−9=21 GB) | users at 32k batched |
|---|---:|---|---:|
| 2 048 | ~70 MB | 21 GB | **~300 users** (batch limit will hit dispatch first) |
| 32 768 | ~580 MB | 21 GB | **~36 users** |
| 262 144 | ~4.6 GB | 21 GB | 4 users (memory-bound) |

We are nowhere near memory-limited for normal usage. Throughput will gate before memory.

---

## 5. Expected performance

### 5.1 Amdahl breakdown

From `project_branchC_perf_state.md`, single-chip 215 ms/tok decomposes as (compounded):

| block | ms (single-chip, eager) | parallelizable | per-chip after TP4 |
|---|---:|---:|---:|
| DeltaNet (48 layers) | 116 (54%) | ✓ 4× | 29 |
| Gated full-attn (16 layers) | 47 (22%) | ✓ 4× | 11.8 |
| MLP (64 layers) | 47 (22%) | ✓ 4× | 11.8 |
| lm_head | 3.2 (1.5%) | partially | 1.0 |
| collectives | 0 | new cost | **~13** (§3) |
| dispatch / sync overhead | residual ~1 | grows | ~3 |
| **Total** | **215** | | **~70** |

Realistic target: **65–80 ms/tok ≈ 12.5–15.4 tok/s**. The 60 ms ambition is plausible only if we shrink collective latency below 0.05 ms each (via `ttnn.experimental.all_gather_async` with persistent semaphores from `tt_ccl.get_and_cycle_*`, which is the production pattern).

### 5.2 Where this estimate could be wrong

- **Compute does not scale linearly.** Matmuls at half the size may not run at half the wall time on Blackhole (small-K matmuls under-utilize the matrix engine). Likely <4× speedup on small-K — call it 3.2–3.5× effective on parallel parts. Updated target: **80–90 ms/tok**.
- **Synchronization stalls.** Each all-reduce is a barrier. The CCL ops in `tt_transformers/tt/ccl.py` use **double-buffered semaphores** (`get_and_cycle_*`) to overlap consecutive collectives; if we don't replicate that pattern, we get serialization → 1.5–2× collective penalty.
- **Trace capture + multi-chip is unproven** — see §8 risks.

### 5.3 Stretch: composite trace + persistent collective buffers

If we land C'4 trace capture on 4 chips and use persistent output buffers (the `persistent_output_buffer` arg in `all_gather_async`), the collective dispatch shrinks to a single per-token kernel launch. Friend's repo (single data point referenced earlier in our memory) hit 15.5 tok/s — we should plausibly land at 14 tok/s clean and 17 tok/s with trace.

---

## 6. Mesh + device management

### 6.1 Opening the mesh

From `ttnn.all_gather` doc and `experiments/87_multichip_primitives.py` (Phase A7 — which validated mesh open and tensor placement, but blocked on fabric on qb1):

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ...)  # qb2 has working fabric
mesh = ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
```

Use `(1, 4)` not `(4, 1)`: `cluster_axis=1` is the populated axis, matching the conventions in `distributed_norm.py:84-104` where the all-gather happens on axis 1.

### 6.2 Mesh tensors

Sharded uploads use `mesh_mapper`:

```python
# Replicated weight (e.g. RMSNorm gamma):
w_tt = ttnn.from_torch(
    arr_torch, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

# Column-parallel weight (out-dim shard, e.g. gate_proj [5120, 17408]):
w_tt = ttnn.from_torch(
    arr_torch, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))   # shards along last dim

# Row-parallel weight (in-dim shard, e.g. down_proj [17408, 5120]):
w_tt = ttnn.from_torch(
    arr_torch, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2))   # shards along second-to-last
```

For the head-grouped `in_proj_all` and `attn_qkv` fused weights, the host-side rebuild is required (group heads contiguously per chip before sharding flat). Document the layout in the loader and add an isolation probe (Phase C'7.2 gate) to confirm equivalence.

### 6.3 Replication-aware state

Single mesh tensor handle ≡ "the tensor as seen from the mesh". Each chip owns its slice. `ttnn.synchronize_device(mesh)` syncs all chips. `cosine` comparisons must use `ttnn.get_device_tensors(mesh_tensor)` to extract per-chip slices, or `ttnn.aggregate_as_tensor(...)` to concat host-side.

### 6.4 CCL semaphore management

Mandatory pattern (from `ccl.py:33-107`): create a `TT_CCL` instance once, pass it through every layer. Internally it owns:
- 3 cluster-axis buckets (axis=0, axis=1, no-axis)
- For each bucket: 2× double-buffered barrier semaphores + 2× double-buffered AG semaphores (each AG = 2 raw semaphores) + 2× double-buffered RS semaphores (each RS = 3 raw semaphores).

We don't need to invent this — copy the `TT_CCL` class verbatim. It's not Qwen-specific; it's the generic CCL semaphore pool pattern.

### 6.5 Per-chip vs global trace capture

**Open question.** `begin_trace_capture(device, cq_id=0)` takes a device. On a mesh, we need `begin_trace_capture(mesh, cq_id=0)` — verified per docs to operate per-chip with broadcast control. The `ttnn.experimental.all_gather_async` API takes `persistent_output_buffer` so the buffer can survive across `execute_trace` calls. Phase C'7.6 probe: capture one decode step, replay 10×, confirm cosine and KV state are stable.

---

## 7. Implementation phases

Every phase ends with a hard correctness gate (cosine + tokens-match) before perf is measured.

| Phase | Goal | Gate (correctness) | Gate (perf) |
|---|---|---|---|
| **C'7.1** | Mesh open + collective smoke test on qb2. Implement `TT_CCL` class. Probe: 4-chip all-reduce on `[1, 5120]` bf16 over 100 iters; measure ms/op + variance. Try both `ttnn.all_reduce` and `composite reduce_scatter+all_gather` (`tt_all_reduce`'s two branches). | Sum across 4 chips of replicated `1.0` returns `4.0` everywhere | Single all-reduce latency report; pick winning primitive |
| **C'7.2** | Shard ONE MLP layer. Upload `gate_proj`, `up_proj` column-parallel, `down_proj` row-parallel. Insert `all_reduce(hidden)` after `down_proj`. Compare vs single-chip MLP at same input. | cosine ≥ 0.9999 on output | Wall time per MLP step vs single-chip; expect ~2.5–3× speedup |
| **C'7.3** | Shard one gated full-attn layer. Shard `attn_qkv` head-grouped, `o_proj` row-parallel, paged KV cache along N_KV. Each chip gets 1 KV head + 6 Q heads + 1/4 of cache blocks. | `run_91r` on this single layer ≥ 0.9998; demo: same 60-token greedy sample matches single-chip | Per-attn-layer ms |
| **C'7.4** | Shard one DeltaNet layer. Hardest. Re-stripe `in_proj_all` so each chip's slice maps to its 12 V heads (and 4 K heads' worth of q/k). Local H + conv_state + recurrence. Row-parallel `out_proj`. | cosine ≥ 0.9999 single layer. Verify H_local across chips concatenates to single-chip H bit-identical (no collective involved). | Per-DN-layer ms |
| **C'7.5** | Full 64-layer forward + embed + lm_head. Argmax via host-side (idx, val) gather. | `run_91r --paged` full sweep ≥ 0.9998. Greedy "Paris" ≥ 60 tok identical to single-chip. | End-to-end ms/tok in eager mode |
| **C'7.6** | Perf tune: switch to `ttnn.experimental.*_async` with persistent output buffers; capture trace if `begin_trace_capture` works on mesh. | Cosine unchanged ≥ 0.9998 across 50-step decode | **Target ≤ 80 ms/tok**; stretch ≤ 65 ms/tok |
| C'7.7 (stretch) | Batch >1 throughput; multi-user paged scheduling | Independent users do not corrupt each other's KV/SSM state | tok/s/user vs single-chip |

Each phase is one commit minimum. C'7.4 is the riskiest; budget **2× the time** of the others.

---

## 8. Risk register

### R1 — Blackhole-specific CCL bugs (HIGH)

Issue class: `#16674`-style hangs with sharded memory configs. Our paged_update_cache already navigated this on single chip (`feedback_paged_eager_landed.md` notes "INTERLEAVED memory config sidesteps Blackhole hang #16674"). The fact that the production CCL ops in `tt_transformers/tt/ccl.py` route N300/T3K through `reduce_scatter_minimal_async` and TG through `all_gather_async + fast_reduce_nc` strongly suggests `ttnn.all_reduce` on Blackhole-style devices has rough edges.

**Probe (Phase C'7.1):** run `ttnn.all_reduce` and the composite alternative for 1000 iters each, check for hangs and timing variance. If `all_reduce` hangs, follow the composite path verbatim.

### R2 — DeltaNet recurrence sharding numerics (HIGH)

The H recurrence is `H_new = H · decay + outer(k, delta)`. With H in fp32 (verified safe and required in single-chip), per-chip slices are independent and numerically identical to single-chip computation at the same heads. **No bf16 drift** is introduced by sharding because no inter-chip reduction touches H.

**But** the `out_proj` all-reduce sums 4 bf16 partials per residual. Order-of-summation matters for bf16. Single-chip computes `sum_i H_i · v_i` in some order inside the matmul; 4-chip computes four partial sums then a tree-reduce. Magnitude is small (per-row ~0.001 in deep layers per `feedback_per_row_diagnostics.md`), so we may eat an extra 1 ULP per all-reduce. Over 48 + 16 + 64 = 128 all-reduces per token, that's bounded but real.

**Probe (Phase C'7.4):** per-row cosine on the residual output of layer 47 (deepest DN). Expect ≥ 0.9998. If lower, switch the all-reduce dtype to fp32 (cast in, cast out) for that op only.

### R3 — KV head sharding with N_KV=4 = chip count (MEDIUM)

Per chip, paged_scaled_dot_product_attention_decode sees `kv_cache_k_tt : [num_blocks, 1, 64, 256]`. The "N_KV=1 case" is the most tested in tt-metal (vanilla MHA pattern), so this is **easier** than N_KV=2 or 3. The GQA broadcast (6 Q : 1 KV) happens inside the SDPA kernel, identical to single-chip with N_KV=1.

**One subtle issue:** `paged_update_cache` requires the input to be HEIGHT_SHARDED with shard shape `[32, head_dim]` per the production single-chip code (`91f:459-475`). On a single chip with N_KV=4 we pad to TILE_HEIGHT=32. On 4 chips with N_KV_local=1 we pad to 32. The pad ratio is *worse* per chip (31/32 wasted) but the absolute cost is unchanged.

**Probe (Phase C'7.3):** measure `paged_update_cache` latency with N_KV_local=1 vs the single-chip N_KV=4 baseline. If significantly slower per chip, we may need to batch multiple users to amortize.

### R4 — Trace capture + multi-chip (MEDIUM)

`begin_trace_capture` on a mesh device is documented but **we have not tested it on qb2**. C'4 v4 succeeded on single chip with cosine=1.0 and 5.5% wall-time win, but the trace fixture is per-device. Multi-chip trace must:
1. Capture per-chip kernel sequences synchronously.
2. Record CCL calls including semaphore acquisition.
3. Replay deterministically across all 4 chips.

`ttnn.experimental.all_gather_async` taking `persistent_output_buffer` is the API signal that this is supposed to work — persistent buffers exist precisely so the trace can refer to the same physical L1/DRAM addresses on replay.

**Probe (Phase C'7.6):** capture one trace, replay 10×, compare output vs eager. If it hangs or diverges, fall back to eager-only multi-chip and accept the perf hit (10–20 ms/tok).

### R5 — In-proj_all head striping correctness (MEDIUM)

The fused weight layout in `91f:108-111` concatenates `[qkv | z | a | b]` along `axis=1`. When we shard out-dim, the chip-local slice must still have all four components for *its* heads. We have to re-stripe the weight on host: instead of one big concat, do `chip_i_slice = [qkv_i | z_i | a_i | b_i]` then upload.

**Probe (Phase C'7.4 setup):** unit test on host — for any random input `x`, single-chip `(x @ in_proj_all)[chip_i_slice]` must equal `x @ chip_i_in_proj_all`.

### R6 — Fabric stability over long runs (LOW, MONITOR)

`tt-smi -ls` heartbeat dropouts can recur. Phase A7 on qb2 validated short collectives; we have no data on sustained 64×128-collective-per-token decode for 1000 tokens. Add a watchdog: every 100 tokens, ttnn.synchronize_device(mesh); if any chip is unresponsive, abort.

---

## 9. What we are deliberately NOT doing in C'7

- **Sequence parallel** (sharding along sequence dim). Decode generates 1 token/step; SP buys nothing at decode. May revisit for prefill in a separate phase.
- **Pipeline parallel** (splitting layers across chips). Adds bubble overhead, no compute reduction per token. Strictly worse than TP for our regime.
- **MoE-style all-to-all** (`ttnn.all_to_all_dispatch`). Qwen3.6-27B has dense MLP, not MoE. Not applicable.
- **2D mesh** (e.g. 2×2 TP+DP). Single host, 1 user-per-stream is fine for ≤ tok/s targets. 2D becomes interesting at >100 concurrent users.
- **Cross-chip KV migration** (sharing cache between users on different chips). KV stays sticky to its chip.

---

## 10. Open questions & probe plan

| # | Question | Probe |
|---|---|---|
| Q1 | `ttnn.all_reduce` vs composite (RS+AG) for 10 KB payloads on qb2? | C'7.1: 1000-iter benchmark of each |
| Q2 | Per-collective latency floor (bandwidth vs setup-bound)? | C'7.1: vary payload 1 KB → 1 MB, fit latency = α + β·bytes |
| Q3 | Does `begin_trace_capture(mesh, ...)` work and capture CCL? | C'7.6: capture, replay, compare |
| Q4 | Does `paged_scaled_dot_product_attention_decode` work cleanly with `N_KV=1` shard? | C'7.3: cosine probe vs single-chip with full N_KV=4 |
| Q5 | Bf16 reduction order matters? | C'7.4: per-row cosine on residual after 48 DN all-reduces |
| Q6 | `set_fabric_config` value for qb2 — `FABRIC_1D` or `FABRIC_1D_RING`? | C'7.1: try Linear topology first, then Ring if hangs |
| Q7 | Persistent output buffer reuse across trace replays? | C'7.6 |

---

## 11. Citations & source pointers

- `ttnn.all_gather`, `ttnn.all_reduce`, `ttnn.reduce_scatter`, `ttnn.mesh_partition` — `tt_docs_corpus/docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/`
- Production TT_CCL semaphore pool pattern — `experiments/.refs/tt-metal/models/tt_transformers/tt/ccl.py:33-107` (treat as the canonical pattern; not Qwen-specific)
- `tt_all_reduce` two-branch composite — `experiments/.refs/tt-metal/models/tt_transformers/tt/ccl.py:110-270`
- `DistributedNorm` with double-buffered AG semaphores and `persistent_output_buffer=None` pattern — `experiments/.refs/tt-metal/models/tt_transformers/tt/distributed_norm.py`
- `ttnn.experimental.all_gather_async` / `reduce_scatter_minimal_async` C++ signatures — `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp` and `.../reduce_scatter/reduce_scatter.hpp`
- Single-chip implementation we are sharding — `experiments/91f_qwen36_27b_full_ondevice.py`
- Single-chip perf baseline + breakdown — memory `project_branchC_perf_state.md`, `feedback_paged_eager_landed.md`
- Paged SDPA validation at 32k — memory `feedback_paged_sdpa_decode_works_at_32k.md`
- Phase A7 fabric status (qb1 broken, qb2 working) — `research/phase_a6_a7_results.md`, `research/fabric_diagnosis.md`
- bf8 weight safety — memory `feedback_bf8_weights.md`

---

## 12. Definition of done for C'7

1. `experiments/serve/server.py` accepts a `mesh=True` mode that boots the model on a 1×4 mesh.
2. `run_91r --paged --mesh` runs the per-layer cosine sweep; every gated full-attn layer ≥ 0.9998, every DeltaNet layer ≥ 0.999 (looser bound from `feedback_isolation_must_match_production.md`).
3. `bench_decode_paged --mesh --max-pos 32768 --n-steps 60 --warmup 5` reports ≤ 80 ms/tok median.
4. Greedy "Paris is the capital of" produces the **same** 60 tokens as single-chip.
5. Single commit per phase, documented in `research/c7_multichip_results.md`.
6. Update `project_branchC_perf_state.md` row C'7 with the achieved tok/s.
