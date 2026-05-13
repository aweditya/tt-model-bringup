# 07. SDPA Decode L1 Cliff and the Paged Variant

**Question.** Why does `ttnn.transformer.scaled_dot_product_attention_decode` work at MAX_POS=256 but hard-fail at MAX_POS≥1024 on Blackhole, while `paged_scaled_dot_product_attention_decode` scales to 32k? What changes structurally?

Observed failure (from `research/kernel_profile_qb2_20260513.json`):

```
TT_THROW @ tt_metal/impl/program/program.cpp:1136
Statically allocated circular buffers on core range [(x=0,y=0)-(x=10,y=9)]
grow to 1901120 B which is beyond max L1 size of 1572864 B
```

The 1.9 MB > 1.5 MB L1 budget is the entire story. Below we derive it.

---

## 1. SDPA decode L1 budget math

The op is flash-decode: each compute core holds Q, streams chunks of K and V from DRAM, accumulates softmax + V projection. Per-core L1 footprint is dominated by **double-buffered K and V circular buffers** sized by `Sk_chunk_t`, the K-chunk in tiles.

Sk_chunk_t comes from `k_chunk_size`, which the host derives from cache length `s` in `sdpa_decode.cpp:16-30`:

```cpp
inline uint32_t get_chunk_size(uint32_t s) {
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) break;
    }
    return std::min(512, 1 << i);     // largest pow-2 divisor of s, capped at 512
}
```

Then in `sdpa_decode_program_factory.cpp:133, 388-393`:

```cpp
const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;        // tiles per chunk
const uint32_t DHt  = DH  / TILE_WIDTH;
const uint32_t vDHt = vDH / TILE_WIDTH;
const uint32_t k_tiles  = Sk_chunk_t_cb_size * DHt  * 2;   // double buffer
const uint32_t v_tiles  = Sk_chunk_t_cb_size * vDHt * 2;   // double buffer
const uint32_t qk_tiles = PNHt * Sk_chunk_t_cb_size;
```

CB creation (`sdpa_decode_program_factory.cpp:497-571`) instantiates: `c_0` Q, `c_1` K, `c_2` V, `c_3` mask, `c_5..c_7` scale/m_in/l_in, `c_10` tilized Q, `c_11..c_12` identity scalars, `c_24..c_31` intermediates (qk/out/stats), `c_16..c_20` outputs.

**The dominant per-core formula** (everything in bytes; bf16 tile = 32·32·2 = 2048 B):

```
L1_per_core ≈
    K_cb              = Sk_chunk_t · DHt · 2 · 2048      (double-buffered)
  + V_cb              = Sk_chunk_t · vDHt · 2 · 2048
  + qk_intermediate   = PNHt · Sk_chunk_t · 2048
  + Q_cb              = PNHt · DHt · 2048
  + stats/scratch     ≈ 8..14 small CBs, each ~PNHt or const-tile
  + program/runtime   ~ 200..300 KB fixed overhead
```

### Plug in Qwen3.6-27B per-shard (post 8→4 GQA fold)

`N_KV=4`, `HD=256` so `DHt = vDHt = 8`, `PNHt = 1`.

| MAX_POS | s | pow2-divisor | k_chunk_size | Sk_chunk_t | K_cb (KB) | V_cb (KB) | K+V (KB) |
|---------|------|--------------|--------------|-----------|-----------|-----------|----------|
| 256     | 256  | 256          | min(512,256)=256 | 8     |  256 | 256 | **512** |
| 1024    | 1024 | 1024         | min(512,1024)=**512** | **16** | **512** | **512** | **1024** |
| 8192    | 8192 | 8192         | min(512,8192)=**512** | **16** | **512** | **512** | **1024** |
| 32768   | 32768| 32768        | min(512,…)=**512** | **16** | 512 | 512 | 1024 |

Add ~400 KB of qk/stats/im/out CBs (each `PNHt · 2048` or `qk_tiles · 2048 = 16 · 2048 = 32 KB`, several copies under `c_24..c_31`) plus ~300 KB program overhead and you land at **~1.9 MB** — exactly the failure number. At MAX_POS=256 the same overhead gives ~1.0 MB, which fits the 1.5 MB budget. 

**Cliff location.** Sk_chunk_t doubles at the first `s` where the largest power-of-2 divisor crosses 256→512 (i.e. `s=512`), and saturates there for all larger s. So **every cache size from 512 up has the same per-core L1 footprint** as 8192 — they all overflow.

---

## 2. Paged decode — what's different

Same kernel binary, same flash-decode dataflow. Three things change:

**(a) Default `k_chunk_size = 0`.** From `sdpa_decode.cpp:113-124`:

```cpp
// Use k_chunk_size as override; if k_chunk_size == 0, figure it out in kernels
uint32_t k_chunk_size = 0;
if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
    k_chunk_size = program_config.value().k_chunk_size;
    ...
}
```

This makes `Sk_chunk_t = 0` in the host, and the program factory takes the `max_dynamic_chunk_size = dst_size` branch (`sdpa_decode_program_factory.cpp:356-358`):

```cpp
const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
const uint32_t max_dynamic_chunk_size = dst_size;
const uint32_t Sk_chunk_t_cb_size = Sk_chunk_t == 0 ? max_dynamic_chunk_size : Sk_chunk_t;
```

So the K/V CB is sized for **8 tiles** (`dst_size`), not 16. That alone halves K_cb to 256 KB and brings the total well under L1. The actual chunk count walked at runtime is computed in-kernel from `cur_pos_tensor`, not from `s`.

**(b) K/V layout: `[max_num_blocks, N_KV, block_size, HD]`.** The cache is no longer one contiguous `[B, N_KV, max_seq, HD]` tensor. It's a pool of fixed-size blocks; the per-sequence position-to-block mapping lives in a separate page table. The reader kernel (`reader_decode_all.cpp:259-308`) indexes each chunk through `page_table_ptr_u32[logical_block]` to find the physical block in the pool.

**(c) `S = page_block_size · page_table_columns`** is the logical seq length, not the K-tensor's seq axis (`sdpa_decode_program_factory.cpp:88-97`):

```cpp
if (is_paged_attention) {
    uint32_t block_size = k_shape[2];
    page_block_size_t = block_size / TILE_HEIGHT;
    S = page_table_tensor.value().padded_shape()[-1] * S;
    ...
}
```

So DRAM holds a flat block pool; L1 holds only one chunk at a time; logical seq scales with page-table columns, with **zero growth in per-core L1**.

---

## 3. API signatures side by side

From `sdpa_decode_nanobind.cpp:32-95`:

```python
# Non-paged
ttnn.transformer.scaled_dot_product_attention_decode(
    input_tensor_q,                          # [1, B, NH_pad, HD]
    input_tensor_k,                          # [B, N_KV, max_seq, HD]   (contiguous)
    input_tensor_v,                          # [B, N_KV, max_seq, HD]
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos=[],                              # host list[int] OR
    cur_pos_tensor=None,                     # device [B] uint32/int32
    attention_sink=None,
    scale=None, sliding_window_size=None,
    memory_config=None, program_config=None,
    compute_kernel_config=None, share_cache=None,
)

# Paged
ttnn.transformer.paged_scaled_dot_product_attention_decode(
    input_tensor_q,                          # [1, B, NH_pad, HD]
    input_tensor_k,                          # [max_num_blocks, N_KV, block_size, HD]
    input_tensor_v,                          # [max_num_blocks, N_KV, block_size, HD]
    page_table_tensor,                       # [B, max_num_blocks_per_seq], int32, ROW_MAJOR
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos_tensor=None,                     # device [B] int32 — REQUIRED (no host list)
    attention_sink=None,
    scale=None, sliding_window_size=None,
    memory_config=None, program_config=None,
    compute_kernel_config=None,
)
```

Differences: `page_table_tensor` is positional and required; `cur_pos` (host list) is gone; `share_cache` is gone. K/V cache layout changes from `[B, N_KV, S, HD]` to `[max_num_blocks, N_KV, P, HD]`.

---

## 4. Building the page table — Qwen3.6-27B at 32k context

Parameters: `B=1`, `N_KV=4`, `HD=256`, `max_seq=32768`. Pick `block_size P=64` (matches `feedback_paged_sdpa_decode_works_at_32k.md` defaults — 32k requires `P≥32`; P=64 is the safe choice).

```python
import torch, ttnn

B = 1
N_KV = 4
HD = 256
MAX_SEQ = 32768
P = 64                              # block_size; tile-aligned
max_num_blocks = MAX_SEQ // P       # = 512 blocks per sequence (B=1 ⇒ pool = 512)
max_num_blocks_per_seq = max_num_blocks

# Identity page table: virtual block i -> physical block i
page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(
    B, max_num_blocks_per_seq
)
page_table_tt = ttnn.from_torch(
    page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
)

# K/V cache pool: one [N_KV, P, HD] block per physical slot
k_cache_shape = (max_num_blocks, N_KV, P, HD)
k_cache_tt = ttnn.from_torch(
    torch.zeros(k_cache_shape, dtype=torch.bfloat16),
    device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
)
v_cache_tt = ttnn.from_torch(
    torch.zeros(k_cache_shape, dtype=torch.bfloat16),
    device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
)

# cur_pos must be a device tensor (no host fallback in paged path)
cur_pos_tt = ttnn.from_torch(
    torch.tensor([current_token_idx], dtype=torch.int32),
    device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
)
```

Production page tables can be a random permutation (vLLM-style) to exercise the indirection — see `models/tt_transformers/tests/test_attention.py:128-133`:

```python
permutation = torch.randperm(paged_attention_config.max_num_blocks)
page_table = torch.argsort(permutation).reshape(
    model_args.max_batch_size,
    paged_attention_config.max_num_blocks // model_args.max_batch_size,
)
```

For our identity ordering, `logical_block = floor(cur_pos / P)`, `phys_block = page_table[0, logical_block] = logical_block`.

---

## 5. Trade-offs

| Axis | Unpaged | Paged |
|------|---------|-------|
| Per-core L1 | grows with `Sk_chunk_t = min(512, pow2(s))/32` | constant `Sk_chunk_t = dst_size = 8` |
| Max practical MAX_POS | ~256 on Blackhole | 32k+ verified |
| DRAM bytes | identical (cache same total size) | identical (block-pool same total) |
| Reader memory pattern | sequential DRAM stride | indirect via page-table (1 extra L1 lookup per chunk) |
| Companion ops | `update_cache` | `paged_update_cache` / `paged_fused_update_cache` (writer) |
| Host bookkeeping | none | page-table + per-step `cur_pos_tensor` allocation |
| Perf | flash-decode unfused with indirection-free reads | flash-decode + page-table indirection; vendor folklore ≈ 5–15 % overhead |

**Blackhole gotchas.**
- Writer (`paged_update_cache`) historically hangs on Blackhole under sharded memory configs (Memory: `feedback_paged_sdpa_decode_works_at_32k.md`) — paged variant validated at 32k only when writer uses **sharded mem config** to avoid issue #16674.
- Reader supports `program_config.k_chunk_size > 0` override; setting it on the paged path re-creates the cliff. Leave `program_config=None` or `k_chunk_size=0`.
- `cur_pos_tensor` is a **device** tensor; for trace capture, allocate it once and write in-place each step (Memory: `feedback_trace_capture.md`).

---

## 6. Migration plan for Branch C' Qwen3.6-27B

Current code path (Branch III, 27B): uses contiguous `[1, 4, 256, 256]` K/V with `scaled_dot_product_attention_decode` and host `cur_pos`. This is the MAX_POS=256 cliff.

Steps, all permanent scripts under `experiments/utils/`:

1. **Writer smoke.** Build `paged_fused_update_cache` test at our shape (B=1, N_KV=4, HD=256, P=64). Confirm no Blackhole hang.
2. **Reader smoke.** Allocate K/V pool, identity page-table, write 64 tokens, call `paged_scaled_dot_product_attention_decode(cur_pos=63)`, compare to numpy. Target cosine ≥ 0.999.
3. **MAX_POS sweep.** cur_pos ∈ {255, 256, 257, 511, 1023, 4095, 16383, 32767}; dense ±4 around each (#30362 PCC dips).
4. **Block-size ablation.** P ∈ {32, 64, 128}. P=64 expected sweet spot per Memory: `feedback_paged_sdpa_decode_works_at_32k.md`.
5. **Perf parity.** At cur_pos=255, traced loop, paged vs unpaged. Target within 15 % (vendor folklore).
6. **Wire into 91v decode loop.** Replace `scaled_dot_product_attention_decode` with paged variant; replace `update_cache` with `paged_fused_update_cache`; allocate page-table at startup; allocate `cur_pos_tensor` once for trace.

The C' branch performance target (267 ms/tok at MAX_POS=256) is at the unpaged cliff edge; paged is **required** for daily-driver 32k context — not optional, not a perf optimization. The L1 math leaves no room for tuning `k_chunk_size` to make unpaged scale.

---

## Sources

- `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:16-30, 95-150`
- `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp:29-42`
- `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp:78-95`
- `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:88-97, 133, 356-358, 388-393, 478-571`
- `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:218-308`
- `experiments/.refs/tt-metal/models/tt_transformers/tt/attention.py:705-717` (real-model paged decode call)
- `experiments/.refs/tt-metal/models/tt_transformers/tests/test_attention.py:118-144` (page-table construction)
- `research/kernel_profile_qb2_20260513.json` (observed failure)
- `research/paged_sdpa_decode_usage.md` (companion: usage detail and risk list)
- Memory: `feedback_paged_sdpa_decode_works_at_32k.md`, `feedback_sdpa_decode_max_pos_256_cliff.md`
