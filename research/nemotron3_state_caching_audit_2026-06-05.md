# Nemotron-3 Nano state-caching audit (decode) — 2026-06-05

Audit of paged decode state (KV cache, page_table, cur_pos buffer, SDPA
configs, lifecycle) across 35B (Qwen3.6-A3B), Gemma 4 12B, and 27B (Qwen3.6
dense) so the v0.3.1.b/c Nemotron-3 fork can lift the pattern verbatim.

All file paths absolute on qb1: `~/tt-xla/experiments/serve/...`

---

## §A. State allocation

### A.1 35B (`experiments/serve/server_35b_ttnn.py`)

State slots declared on the `State` class:

- `self.cur_pos_buf` — `server_35b_ttnn.py:1438` — int32 `[1]` device, ROW_MAJOR, replicated across mesh.
- `self.page_table_tt` — `:1439` — int32 `[1, NUM_BLOCKS]` ROW_MAJOR replicated, identity mapping.
- `self.paged_write_mem_cfg` — `:1440` — HEIGHT_SHARDED L1 mem_cfg for paged_update_cache input tile.
- `self.paged_sdpa_progcfg` — `:1441` — `SDPAProgramConfig(CoreCoord(4,4), q_chunk=0, k_chunk=0, exp_approx=False)`.
- `self.sdpa_compute_kernel_config` — `:1442` — B3 recipe (HiFi2, math_approx=False, fp32_dest_acc=False, packer_l1=False).

Concrete construction at bootstrap (`:1877-1924`):

```
SDPA_BLOCK_SIZE = 32
SDPA_NUM_BLOCKS = MAX_KV // SDPA_BLOCK_SIZE  # 128 at MAX_KV=4096
state.cur_pos_buf = ttnn.from_torch(torch.zeros((1,), int32), ROW_MAJOR, int32, mesh, ReplicateTensorToMesh)  # :1886
state.page_table_tt = ttnn.from_torch(np.arange(SDPA_NUM_BLOCKS).reshape(1, NUM_BLOCKS), ROW_MAJOR, int32, mesh, Replicate)  # :1892
shard_spec = ttnn.ShardSpec(grid_1core, [TILE_HEIGHT=32, HEAD_DIM_ATTN=128], ROW_MAJOR)
state.paged_write_mem_cfg = ttnn.MemoryConfig(HEIGHT_SHARDED, L1, shard_spec)  # :1903
state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(CoreCoord(4,4), 0, 0, exp_approx=False)  # :1909
state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(HiFi2, False, False, False)  # :1919
```

Per-layer KV cache allocation lives inside `State.reset_caches_ttnn()` (`:1488-1505`):

```
cache_shape = (sdpa_num_blocks=128, NCHIPS=4, sdpa_block_size=32, HEAD_DIM_ATTN=128)
kc = ttnn.from_torch(zeros, bfloat16, TILE_LAYOUT, mesh, ShardTensorToMesh(dim=1))
# per-chip rank-4 view: (NUM_BLOCKS=128, 1, BLOCK_SIZE=32, HEAD_DIM=128)
```

K and V are stored separately; the "NCHIPS" axis doubles as the per-chip
KV-head-shard axis (NKV_PER_CHIP=1). bf16 — fp32 KV is rejected by the kernel
([[feedback-fp32-kv-cache]]).

### A.2 Gemma 4 (`experiments/serve/server_gemma4_unified_ttnn.py`)

Same pattern with two head_dim variants (sliding=256, global=512).

- `state.cur_pos_buf` — `:431` — int32 `[1]` ROW_MAJOR replicated.
- `state.rot_idxs_buf` — `:438` — uint32 `[1]` (RoPE table lookup).
- `state.tok_buf` — `:446` — uint32 `[1, 1]` for `ttnn.embedding`.
- `state.page_table_tt` — `:452` — int32 `[1, num_blocks]`, identity.
- `state.paged_write_mem_cfg_sliding` — `:509` — HEIGHT_SHARDED L1, shard `[BLOCK_SIZE=32, HEAD_DIM_SLIDING=256]`, 1 core.
- `state.paged_write_mem_cfg_global` — `:515` — HEIGHT_SHARDED L1, shard `[32, HEAD_DIM_GLOBAL=512]`, NUM_KV_HEADS_GLOBAL cores.
- `state.paged_sdpa_progcfg` — `:522` — sliding: `CoreCoord(4,4)`, chunk=0/0.
- `state.paged_sdpa_progcfg_global` — `:532` — global: `CoreCoord(8,4)`, q_chunk=32, k_chunk=64 (L1 budget at d=512).
- `state.sdpa_compute_kernel_config` — `:536` — same B3 recipe.

Per-layer KV caches (`:470-499`):

- Sliding layer: **TWO** caches per layer (`NKV_PER_CHIP_SLIDING=2` → 2 caches each NKV=1 — Gemma 4 two-call paged decode pattern documented in `[[reference-gemma4-two-call-paged-decode]]`). Shape per cache `(num_blocks, NCHIPS, sdpa_block_size, HEAD_DIM_SLIDING)` sharded dim=1.
- Global layer: **ONE** cache, NKV=1 replicated. Shape `(num_blocks, NUM_KV_HEADS_GLOBAL=1, BLOCK_SIZE, HEAD_DIM_GLOBAL)` replicated across mesh.

### A.3 27B (`experiments/serve/server_tp.py`)

- `state.cur_pos_buf` — `:505` — int32 `[1]` ROW_MAJOR replicated.
- `state.tok_buf` / `state.rot_idxs_buf` — `:510, 514` — uint32 `[1, 1]`.
- `state.page_table_tt` — `:445` — int32 `[1, NUM_BLOCKS]` ROW_MAJOR replicated, identity.
- `state.paged_write_mem_cfg` — `:457` — HEIGHT_SHARDED L1, shard `[TILE_HEIGHT=32, HEAD_DIM]`, 1 core.
- `state.fused_paged_write_mem_cfg_k` / `_v` — `:474, 476` — disjoint K/V mem_cfgs (32 cores each: K on (0,0)-(7,3), V on (0,4)-(7,7)) for the `paged_fused_update_cache` fast path.
- `state.paged_sdpa_progcfg` — `:487` — `CoreCoord(4,4)`, chunk=0/0.
- `state.sdpa_compute_kernel_config` — `:493` — B3 recipe.

Per-layer KV caches at upload (`:335-340`): shape `(NUM_BLOCKS, n_kv_heads=4, BLOCK_SIZE, HEAD_DIM)` sharded along N_KV → per-chip `(NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM)`, bf16, TILE_LAYOUT.

Pre-allocation buffers also include prefill tok/pos buffers at `:521-533` (PREFILL_CHUNK_SIZE=32) for the chunked prefill trace.

---

## §B. Reset / lifecycle

### B.1 35B
- `State.reset_caches_ttnn()` — `:1444-1507` — reallocates DN states and KV caches in place; called e.g. before the smoke test at `:2059`. KV cache slots are recreated for each call (no per-slot masked-zero pattern in single-stream path).
- The CB variant `cb_reset_slots()` lives in `server_35b_cb.py` (not the single-stream server) — uses a per-slot DN masked-multiply zero pattern (see [[feedback-35b-cb-reset-slots-b-gt-1-noop]]).

### B.2 Gemma 4
- KV cache allocated once at bootstrap (`:470-499`) — no separate reset function; new generation reuses the same cache and advances `cur_pos_buf` from 0. Position tracking is via `_set_pos()` (`:1319-1342`) which writes the host int into the pre-allocated buffer via `ttnn.copy_host_to_device_tensor`.
- `_set_pos(state, pos)` writes BOTH `cur_pos_buf` and `rot_idxs_buf` per step.

### B.3 27B
- Caches allocated once per layer at upload (`server_tp.py:335`). No "reset_caches" — fresh generation just starts at cur_pos=0; the old values get overwritten as `paged_update_cache` writes at `cur_pos`. (Caveat: stale future positions remain; the SDPA decode kernel masks via `cur_pos_tensor`.)
- `update_input_buffers(state, token_id, cur_pos)` — `:1592-1620` — the per-step host→device write. Three tiny writes: tok_buf `[1,1]` uint32, cur_pos_buf `[1]` int32, rot_idxs_buf `[1,1]` uint32. All via `ttnn.copy_host_to_device_tensor` into pre-allocated buffers (the canonical pattern — never realloc per step, see [[ttnn-list-rebinding-leaks]]).

---

## §C. Decode shape contract — `paged_scaled_dot_product_attention_decode`

From the device op at
`/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:55-260`:

- **Q**: `[1, B, NH, head_dim]` per chip. If sharded → HEIGHT_SHARDED only; else DRAM interleaved. Layout: TILE (per `input_tensors.at(i).layout() == Layout::TILE` check at line 47, except Q which is exempted).
- **K_cache, V_cache**: `[NUM_BLOCKS, NKV_per_chip, BLOCK_SIZE, head_dim]`, TILE, DRAM INTERLEAVED, bf16/fp32/bfp8/bfp4.
- **cur_pos_tensor**: dtype INT32, layout ROW_MAJOR, padded shape last-dim == B (unsharded) or shard-height per core == B (sharded). DRAM INTERLEAVED if unsharded.
- **page_table_tensor**: INT32 (unsharded) or UINT16 (sharded), ROW_MAJOR, shape `[B, max_blocks_per_seq]`. `max_blocks_per_seq <= NUM_BLOCKS`.
- **Output**: same layout as Q, shape `[1, B, NH, head_dim]`.
- **Causal**: `is_causal=True` (default) → uses cur_pos to mask. Q,K,V same head_dim (unless MLA which is unrelated here).
- All three models concretely use B=1, NH=NQ_PER_CHIP=4, head_dim=128 (or 256/512 for Gemma 4). Per-chip wrap: `q_for_sdpa = ttnn.reshape(q_n, [1, 1, NQ_PER_CHIP, HEAD_DIM])` (`server_35b_ttnn.py:867`, `server_tp.py:1568`).

---

## §D. `paged_update_cache` contract

From `paged_update_cache_device_operation.cpp:23-100`:

- **Input tile**: layout TILE, dim 0 = 1, dim -1 = head_dim, dim 1 = batch (== `page_table[0]`). HEIGHT_SHARDED L1 by convention (the existing tile is a 4D tensor `[1, B (or NKV), BLOCK_SIZE_padded, head_dim]`).
- **Cache**: TILE layout, INTERLEAVED memory_layout (not sharded), bf16/fp32/bfp8/bfp4. Shape `[NUM_BLOCKS, NKV_per_chip, BLOCK_SIZE, head_dim]`.
- **update_idxs_tensor**: INT32 ROW_MAJOR. Either DRAM INTERLEAVED (`[B]`) or HEIGHT_SHARDED L1 `[B, 1]`. Padded dim 0 == B.
- **page_table**: INT32 ROW_MAJOR (or UINT16 if sharded). Shape `[B, max_blocks_per_seq]`. `max_blocks_per_seq <= cache_NUM_BLOCKS`.
- **Mutation**: in-place. The cache tensor handle returned by `from_torch` is the same one mutated by every per-step call — no rebinding needed.

Critical gotcha (memorized in [[feedback-paged-update-cache-nkv-per-chip]]):
`input.padded_shape[1]` (dim 1) must equal `page_table.padded_shape[0]` (==B).
If you want NKV_per_chip > 1 the kernel still treats dim 1 as batch. The
35B-style NKV_PER_CHIP=1 contract is the clean one. Gemma 4 sliding's NKV=2
case is handled by splitting into TWO independent paged_update_cache calls
per layer (one per KV head) — `_layer_pos0_sliding_paged` at
`server_gemma4_unified_ttnn.py:1139-1148`.

35B input prep at `server_35b_ttnn.py:846-851`:
```
def _shard_for_paged_write(t_2d):
    t4d = ttnn.reshape(t_2d, [1, 1, 1, HEAD_DIM_ATTN])      # [B=1, NKV=1, 1, HEAD_DIM]
    t_pad = ttnn.pad(t4d, [[0,0],[0,0],[0,BLOCK_SIZE-1],[0,0]], 0.0)  # pad to TILE_HEIGHT=32 on dim -2
    return ttnn.to_memory_config(t_pad, state.paged_write_mem_cfg)    # HEIGHT_SHARDED L1
```

---

## §E. Prefill vs decode dispatch

### E.1 35B
- Decode-only attention; the SDPA path lives in `attn_forward_ttnn_sdpa` (`:773-913`) dispatched by `attn_forward_ttnn` based on `state.attn_mode`. There is NO prefill path on 35B (the model is decode-only autoregressive — single-token in, single-token out — and the LLM prompt is processed token-by-token with the same decode kernel; long contexts are handled via the chunked DN path, not chunked SDPA).
- Effectively single-function, S=1 always.

### E.2 Gemma 4
- Decode-only via `_layer_pos0_sliding_paged` (`:1062`) and `_layer_pos0_global_paged` (`:1188`). Long prompts: same pattern (no separate prefill).

### E.3 27B
- TWO functions: `gated_attn_step_tp` (decode, `paged_update_cache` + `paged_scaled_dot_product_attention_decode`) at `:1480` and `gated_attn_step_prefill_tp` (prefill, `paged_fill_cache` + `ttnn.transformer.scaled_dot_product_attention(is_causal=True)`) at `:1700`.
- Prefill writes via `paged_fill_cache(cache, [1, NKV, S, HEAD_DIM], page_table, batch_idx=0)` at `:1810-1813`.
- Prefill uses NON-paged SDPA (`scaled_dot_product_attention` with is_causal=True); it does NOT route through paged SDPA for the prefill chunk (SDPA still needs the K/V tensors directly). The cache write is purely so the subsequent decode steps see the prompt context.
- Chunked: PREFILL_CHUNK_SIZE=32 fixed; multi-chunk loop is T3 (deferred).

---

## §F. Nemotron-3 fork plan

### F.1 Shapes (concrete)

```
Nemotron-3 (per attention layer):
  NUM_Q_HEADS  = 32   (NQ_PER_CHIP   = 8   on (1,4) mesh)
  NUM_KV_HEADS = 2    (NKV_PER_CHIP  = ???)
  HEAD_DIM_ATTN = 128
  6 attention layers at L5, L12, L19, L26, L33, L42
  B = 1, MAX_DECODE_LEN = 64 (v0.3.1.b/c first cut)
```

**KV head sharding decision**: NUM_KV_HEADS=2 < NCHIPS=4. The 35B path
shards `dim=1` (the per-chip-KV-head axis) so that per-chip NKV=1 (clean
contract). For Nemotron, replicate the KV across pairs of chips — i.e.
`ReplicateTensorToMesh` for K/V caches with shape `(NUM_BLOCKS,
NUM_KV_HEADS=2, BLOCK_SIZE, HEAD_DIM)` per chip — and let Gemma-4-style
**two-call decode** (one per KV head) keep the clean NKV_per_chip=1
contract. This is the `[[reference-gemma4-two-call-paged-decode]]`
blessed workaround.

Alternative: shard `dim=1` to get NKV_per_chip=0.5 → not valid; or replicate
fully and use NKV=2 paged_update_cache + the documented contract gotcha
(known to bite, see [[feedback-paged-update-cache-nkv-per-chip]]). The
two-call pattern is the production-tested escape hatch.

Per-cache shape (one of TWO caches per attention layer):
```
cache_shape = (NUM_BLOCKS, NCHIPS, BLOCK_SIZE, HEAD_DIM_ATTN)
            = (NUM_BLOCKS, 4,      32,        128)
ShardTensorToMesh(dim=1)  # per-chip view (NUM_BLOCKS, 1, 32, 128)
dtype = bfloat16  # fp32 rejected
layout = TILE
```

With `MAX_DECODE_LEN=64`, BLOCK_SIZE=32 → NUM_BLOCKS=2. For v0.3.1.b/c
that is fine. To grow to 8192 just bump NUM_BLOCKS=256. **Recommend
NUM_BLOCKS=256, BLOCK_SIZE=32 from day 1** to avoid rebuilding caches
when we scale (memory is `256*1*32*128*2B = 2MB/chip per cache × 12
caches × 6 layers = 144 MB/chip` — trivial against 32 GB DRAM).

Q heads to KV head mapping: Q heads (0..15) → KV head 0 (cache_0), Q
heads (16..31) → KV head 1 (cache_1). On (1,4) mesh, NQ_PER_CHIP=8:
chips 0,1 → KV head 0; chips 2,3 → KV head 1. (Or split per-chip into
two SDPA calls — Gemma-style.)

### F.2 Layout decision (paging vs single-block)

Use **proper paging from day 1** (NUM_BLOCKS=256, BLOCK_SIZE=32,
identity page_table). Reasons:
1. Existing 35B / Gemma 4 / 27B code is paged — zero new contract surface.
2. Single-block layout (NUM_BLOCKS=1, BLOCK_SIZE=MAX_DECODE) is not how the
   kernel is shape-validated (block_size must be tile multiple; max=BLOCK_SIZE).
3. CB v1.x will need real paging anyway; pay it now.

### F.3 Setup helpers to lift (verbatim)

From `server_35b_ttnn.py`, lift these bootstrap blocks into Nemotron-3
(no math changes needed):

- `state.cur_pos_buf` init — `:1886-1890` (replicate int32 [1]).
- `state.page_table_tt` init — `:1891-1896` (identity arange replicate).
- `state.paged_write_mem_cfg` — `:1898-1906` (HEIGHT_SHARDED L1, shard `[32, 128]`, 1 core).
- `state.paged_sdpa_progcfg` — `:1909-1914` (CoreCoord(4,4), chunk=0/0).
- `state.sdpa_compute_kernel_config` — `:1919-1924` (B3 recipe).

From `server_tp.py`, lift this lifecycle helper:

- `update_input_buffers(state, token_id, cur_pos)` — `:1592-1620` (the three-tiny-HtoD pattern). Adapt to Nemotron-3 (no rot_idxs_buf if RoPE isn't needed — but Nemotron-3 attention layers DO use RoPE per HF config; keep rot_idxs_buf too).

From `server_gemma4_unified_ttnn.py`, lift this two-call decode pattern:

- `_layer_pos0_sliding_paged(...)` body at `:1079-1180` — fork verbatim for Nemotron-3's NKV=2 attention. Replace HEAD_DIM_SLIDING=256 → HEAD_DIM_ATTN=128; replace NKV_PER_CHIP_SLIDING=2 with the local var; the rest of the structure (per-KV-head slice → shard → paged_update_cache K, paged_update_cache V → paged SDPA decode → concat per-Q-half outputs) maps 1:1.

### F.4 Refactor plan for `attn_block_eager_tt`

Current `attn_block_eager_tt` (`server_nemotron3_nano_ttnn.py:931-967`) is
S-agnostic non-paged SDPA with `is_causal=True`. It is correct for the
v0.1.x prefill validation but allocates fresh K/V every call and writes
NOTHING to a KV cache.

**Recommendation: SPLIT into `attn_prefill_tt` (S>1) and
`attn_decode_step_tt` (S=1)**. Reasons:

1. Prefill (S>1) needs `paged_fill_cache` + non-paged SDPA (`is_causal=True`)
   — 27B's exact pattern at `server_tp.py:1700-1845`. Q/K/V shapes
   `[1, NH, S, HEAD_DIM]`, cache write is bulk.
2. Decode (S=1) needs `paged_update_cache` (single-pos) + paged SDPA
   decode. Q/K/V shapes `[1, 1, NH/NKV, HEAD_DIM]`, cache write is
   tile-aligned single row.
3. Pad/shard machinery differs (HEIGHT_SHARDED [TILE_H, HEAD_DIM] write
   for decode vs unsharded bulk for prefill).
4. Single-function branching means dead branches inside trace — bad for
   v0.4 trace capture (the BIGGEST RISK per task #208).

Concrete sequence:

**Step 1** (lands as v0.3.1.b prep): rename current `attn_block_eager_tt`
to `attn_prefill_tt`. Add `paged_fill_cache(state.kv_K_cache_tt[L], k4, state.page_table_tt, batch_idx=0)` and same for V, gated on `state.kv_K_cache_tt[L] is not None`. Caller passes a flag.

**Step 2** (v0.3.1.b): allocate KV caches in `reset_decode_state` (currently zero-initializes ssm_state only, attention KV is placeholder None — `:707-709`). Replace with:
```
for L, kind in enumerate(state.layer_types):
    if kind == "attention":
        # Two caches per layer (Gemma-4-style NKV=1 per call)
        cs = (NUM_BLOCKS=256, NCHIPS=4, BLOCK_SIZE=32, HEAD_DIM_ATTN=128)
        state.kv_K_cache_tt[L] = [
            ttnn.from_torch(zeros(cs), bfloat16, TILE, mesh, ShardTensorToMesh(dim=1))
            for _ in range(NUM_KV_HEADS=2)
        ]
        # same for V
```
And add bootstrap-time init of `cur_pos_buf` / `page_table_tt` /
`paged_write_mem_cfg` / `paged_sdpa_progcfg` / `sdpa_compute_kernel_config`
(lift from 35B `:1875-1924`).

**Step 3** (v0.3.1.c): implement `attn_decode_step_tt` forking
Gemma 4's `_layer_pos0_sliding_paged` with HEAD_DIM=128.

**Step 4**: add `step_forward_decode` that calls `_set_pos(state, cur_pos)`
(host→device copy into cur_pos_buf) before each step.

### F.5 Per-step dispatcher signature
```
def attn_decode_step_tt(state, h_norm_tt, layer_idx, capture=None):
    """v0.3.1.c — single-token decode through paged KV cache.
    Forks server_gemma4_unified_ttnn.py:_layer_pos0_sliding_paged with
    HEAD_DIM=128, NUM_KV_HEADS=2 (per chip), NQ_PER_CHIP=8.
    """
```

---

## §G. Concerns / gotchas

1. **NUM_KV_HEADS=2 with NCHIPS=4** — pick a sharding scheme NOW. If you
   ReplicateTensorToMesh on K/V cache, dim 1 = 2, and `paged_update_cache`
   asserts `input.dim[1] == page_table.dim[0] == B = 1` →
   FAIL. Must shard along the NKV axis somehow. Cleanest: Gemma-4-style
   two-call decode with per-cache NKV=1. See
   [[feedback-paged-update-cache-nkv-per-chip]].

2. **cur_pos_buf NEVER reallocated** — must update via
   `ttnn.copy_host_to_device_tensor(host_tensor, state.cur_pos_buf)`
   each step (see `server_tp.py:1613, server_gemma4_unified_ttnn.py:1330`).
   Realloc-per-step is the [[ttnn-list-rebinding-leaks]] anti-pattern
   that produces garbage attention past pos 0.

3. **bf16 cache only** — fp32 KV cache is hard-rejected by paged SDPA
   ([[feedback-fp32-kv-cache]]). Use bfloat16 for kc/vc.

4. **HEIGHT_SHARDED L1 mem_cfg for paged_update_cache input** — the input
   tile must be HEIGHT_SHARDED to L1; the cache tensor must be INTERLEAVED
   (NOT sharded). Both 35B and 27B follow this; the existing
   `_shard_for_paged_write` helper at `server_35b_ttnn.py:846-851` is the
   reference. Match TILE_HEIGHT=32 on dim -2 padding.

5. **page_table dtype = INT32** unless sharded (then UINT16). Layout
   always ROW_MAJOR. Replicated across mesh.

6. **cur_pos_buf dtype = INT32**, layout ROW_MAJOR. Padded last-dim must
   equal B (=1 for single-stream); shape `[1]` works.

7. **SDPA program_config CoreCoord** — `(4,4)` is the validated 35B/27B
   default at head_dim=128. Default kernel grabs ~110 cores/head which
   triggers a tree-reduction error on the (1,4) mesh
   ([[feedback-mesh-paged-sdpa-works]]). Use CoreCoord(4,4) for
   head_dim=128.

8. **B3 compute_kernel_config required at large positions** — HiFi4 +
   fp32_dest_acc is buggy on Blackhole P150 for SDPA decode at large
   `cur_pos` ([[feedback-fp32-sdpa-cliff-probe]]). Use the verbatim
   B3 recipe: HiFi2 + math_approx=False + fp32_dest_acc=False +
   packer_l1_acc=False.

9. **Trace-compatibility concerns for v0.4** (task #208):
   - `update_idxs_tensor=state.cur_pos_buf` is trace-safe because the
     buffer is pre-allocated and host writes happen OUTSIDE the captured
     trace.
   - The non-paged `update_cache_for_token_` op took a Python int and is
     NOT trace-compatible (bakes into trace) — DO NOT switch to it
     ([[server_tp.py:329]] comment).
   - `_shard_for_paged_write` uses `ttnn.reshape` + `ttnn.pad` + `ttnn.to_memory_config`
     inside the captured region — all trace-safe per 35B production.
   - DO NOT call `ttnn.deallocate` on a view (slice/reshape) handle
     mid-trace ([[ttnn-slice-view-decay]]). The 35B and Gemma 4 paths
     are already careful here; just preserve the structure when forking.

10. **`paged_fused_update_cache` is optional** — 27B has it
    (`server_tp.py:472-485`) but it requires a >= 8x8 compute grid for
    disjoint K/V cores. For first cut, use TWO separate
    `paged_update_cache` calls (one K, one V) per KV head. Fused fast
    path is a NEXT optimization (task #198 captures the Gemma 4
    equivalent).

11. **Cache size at MAX_KV growth** — at MAX_KV=8192, NUM_BLOCKS=256
    × BLOCK_SIZE=32 × HEAD_DIM=128 × 2B (bf16) × NKV_per_chip=1 × 2 caches
    × 6 attn layers = ~24 MB/chip. Trivial. Allocate generously up
    front to avoid resize.

12. **Bootstrap signature**: `bootstrap(state, log=None)` takes the log
    callback (`server_nemotron3_nano_ttnn.py:516`). Add the paged
    plumbing init at the END of bootstrap, after layer upload, before
    returning. Mirror `server_35b_ttnn.py:1875` placement.

---

## Sources

- 35B: `experiments/serve/server_35b_ttnn.py:773-913, 1430-1507, 1875-1925, 1573, 2059`
- Gemma 4: `experiments/serve/server_gemma4_unified_ttnn.py:425-540, 1062-1170, 1188-1265, 1319-1342`
- 27B: `experiments/serve/server_tp.py:335-505, 1480-1620, 1700-1840`
- Nemotron-3 v0.1.x: `experiments/serve/server_nemotron3_nano_ttnn.py:53-67, 480-712, 780-967`
- tt-metal device-op contracts:
  - `/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:23-260`
  - `/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:14-145`
