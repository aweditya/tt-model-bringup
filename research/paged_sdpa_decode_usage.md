# Paged SDPA Decode — Usage Guide

**Target:** `ttnn.transformer.paged_scaled_dot_product_attention_decode`
**Audience:** us, before we write probes
**Motivation:** Stock `scaled_dot_product_attention_decode` cliffs at MAX_POS=256 for Qwen3.6-27B on Blackhole (CB grows to 1901120 B > L1 1572864 B). For 32k+ daily-driver context we need paged. See `experiments/utils/sdpa_max_pos_ceiling_probe.py`.

All shapes/signatures below are from the authoritative C++ pybind in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp` and from real-model usage in `models/tt_transformers/tt/attention.py` — the published HTML doc page for the paged variant is incomplete (it omits `page_table_tensor` entirely).

---

## 1. Function signature

From the nanobind binding (kw_only marked with `*,`):

```python
ttnn.transformer.paged_scaled_dot_product_attention_decode(
    input_tensor_q,        # ttnn.Tensor, [1, B, NH_pad, HD]
    input_tensor_k,         # ttnn.Tensor, [max_num_blocks, N_KV, block_size, HD]
    input_tensor_v,         # ttnn.Tensor, [max_num_blocks, N_KV, block_size, HD]
    page_table_tensor,      # ttnn.Tensor, [B, max_num_blocks_per_seq], int32
    *,
    is_causal=True,                       # bool
    attn_mask=None,                       # ttnn.Tensor, [B, 1, NH_pad, S], optional
    cur_pos_tensor=None,                  # ttnn.Tensor, [B], uint32, *device*
    attention_sink=None,                  # ttnn.Tensor, optional
    scale=None,                           # float, default 1/sqrt(HD)
    sliding_window_size=None,             # int, optional
    memory_config=None,                   # ttnn.MemoryConfig
    program_config=None,                  # SDPAProgramConfig
    compute_kernel_config=None,           # ttnn.DeviceComputeKernelConfig
) -> ttnn.Tensor                          # [1, B, NH_pad, HD]
```

Key differences vs the non-paged variant: paged **requires** `page_table_tensor` (positional, no default) and **drops** `cur_pos` (the host-list form) and `share_cache`. Position bookkeeping is device-only via `cur_pos_tensor`. This matters for trace capture (see Memory note in `feedback_trace_capture.md`).

Source: tt-metal `sdpa_decode_nanobind.cpp` and `sdpa_decode.hpp`.

---

## 2. The page-table abstraction

Three tensors carry the cache:

| Tensor | Shape | Dtype | Layout |
|---|---|---|---|
| K cache | `[max_num_blocks, N_KV, block_size, HD]` | bf16 / bf8_b | TILE |
| V cache | `[max_num_blocks, N_KV, block_size, HD]` | bf16 / bf8_b | TILE |
| Page table | `[B, max_num_blocks_per_seq]` | int32 (uint32 in some test paths) | ROW_MAJOR, DRAM |

A **page** (block) is `[N_KV, block_size, HD]` worth of K (and another of V). `block_size` must be tile-aligned along its last two axes; in tt-metal practice it is `32` or `64` (the tile dim). The kernel does **not** infer occupancy from the page table — it infers it from `cur_pos_tensor[b]`, the *absolute* logical position of the latest token per batch. The page table is a pure logical→physical map: `phys_block = page_table[b, logical_block]`, where `logical_block = floor(t / block_size)`.

Source for cache shape: `models/tt_transformers/tt/attention.py` (`cache_k = torch.zeros((max_num_blocks, n_local_kv_heads, block_size, head_dim))`).

---

## 3. Comparison to vLLM PagedAttention

vLLM (Kwon et al. 2023) introduced this model; the ttnn mapping is nearly 1:1:

| Concept | vLLM | ttnn |
|---|---|---|
| Page / block size | `block_size` | `block_size` |
| Phys page count | `num_gpu_blocks` | `max_num_blocks` |
| Per-seq map | `block_tables[seq_id]` | `page_table[b, :]` |
| Position counter | `context_lens` | `cur_pos_tensor` |
| Block layout | `[N, n_kv, HD/x, P, x]` (x-vectorized) | `[N, n_kv, P, HD]` (tile-native) |

ttnn does not expose copy-on-write, block refcounting, or cross-seq sharing — fine for batch=1. Our allocator is the trivial free-list `0..max_num_blocks-1`.

---

## 4. Constructing the page table (batch=1, single sequence)

At decode step `cur_pos = N`, block size `P`:

- `num_active_blocks = ceil((N + 1) / P)`
- `max_num_blocks_per_seq = ceil(max_seq_len / P)` (fixed at allocation)
- `page_table[0, i] = i` for `i ∈ [0, num_active_blocks)`; pad rest with `0`

The page table is small (32k @ P=128 → 256 int32 = 1 KB), DRAM, ROW_MAJOR, **allocated once** — only K/V writes and `cur_pos_tensor` change per step. Reference: `create_tt_page_table` in `simple_text_demo.py` uses a random permutation; identity ordering is fine for our first probe.

---

## 5. Memory math at Qwen3.6-27B shapes

Per shard: `N_KV=4` (post 8→4 GQA fold), `HD=256`, bf16. One K page = `N_KV · P · HD · 2 = 2048·P` bytes:

| P  | K+V page bytes | comment |
|----|----------------|---------|
| 32 | 128 KB | tile-minimum |
| 64 | 256 KB | sweet spot |
| 128 | 512 KB | one third of L1 — risky with Q + scratch |
| 256 | 1 MB | recreates the cliff |

The non-paged cliff was `2·4·256·256·2 = 1 MB` of K+V chunk + scratch ≈ 1.9 MB > 1.5 MB L1. **P=64 is the safe default.** DRAM total at 32k: `2 · N_KV · 32768 · HD · 2 = 128 MB` combined — same as contiguous; paging changes access pattern, not storage.

---

## 6. Companion ops for cache management

`paged_scaled_dot_product_attention_decode` is **read-only**. We write separately each step:

```python
# Fused K+V write (one launch instead of two)
ttnn.experimental.paged_fused_update_cache(
    keys,                          # K cache, [max_num_blocks, N_KV, P, HD]
    k_new_1BKD,                    # new K, [1, B, N_KV, HD]
    values,                        # V cache
    v_new_1BKD,                    # new V
    update_idxs_tensor=cur_pos,    # device tensor [B] of abs positions
    page_table=page_table,         # [B, max_blocks_per_seq]
)

# Or separately
ttnn.experimental.paged_update_cache(
    keys, k_new_1BKD,
    update_idxs_tensor=cur_pos, page_table=page_table,
)

# Prefill (writes a whole prompt for one user slot)
ttnn.experimental.paged_fill_cache(
    keys_BKSD, k_user_sliced, fill_page_table, batch_idx=slot_idx,
)
```

These are in `ttnn.experimental.*`, not `ttnn.transformer.*`. The fused variant is preferred at decode (halves dispatch). Known issue: paged_update_cache hangs on Blackhole under certain configs (#16674) — we must validate this works on qb1 before committing to paged.

---

## 7. Paged decode vs chunked prefill

Both can carry a page table; they target different phases.

| Op | Phase | Q seq | K/V seq | Page table |
|---|---|---|---|---|
| `paged_scaled_dot_product_attention_decode` | decode | 1 | up to cur_pos | required, `[B, max_blocks_per_seq]` |
| `chunked_scaled_dot_product_attention` | prefill | chunk_size | full prefix | required, with `chunk_start_idx` for causal alignment |
| `scaled_dot_product_attention` | prefill (short) | full | full | none |

Eventual long-context flow: chunked prefill (writes via paged_fill_cache) → paged decode loop (writes via paged_fused_update_cache, reads via paged SDPA decode).

---

## 8. Risks and unknowns

1. **L1 cliff may still bite.** Same flash-decode kernel underlies both variants; oversized `block_size` or `k_chunk_size` in `SDPAProgramConfig` re-creates the CB blowup. Start P=64, default program_config.
2. **PCC dips at specific positions** (#30362). Sporadic precision failures masked by CI sampling every 71 steps. Sweep positions densely.
3. **paged_update_cache hang on Blackhole** (#16674). Smoke-test the writer first.
4. **dtype.** bf16 is validated; bf8_b cache unverified for decode at our shapes.
5. **Perf penalty.** Vendor folklore 5–15%; verify vs non-paged at MAX_POS=256.
6. **`cur_pos_tensor` is device uint32, not host int** — allocate it up front for trace capture.
7. **Page table dtype.** int32 in C++; some helpers bind it as uint32. Verify after allocation.

---

## 9. Recommended next probes

Permanent scripts under `experiments/utils/`, in order:

1. **API smoke** — minimal call: B=1, N_H=4 pad to 32, N_KV=4, HD=256, P=64, max_blocks=8, cur_pos=0.
2. **Writer smoke** — `paged_fused_update_cache` same shapes; confirms #16674 fixed.
3. **Correctness vs numpy** — write 64 tokens, read at cur_pos=63, cosine ≥ 0.999.
4. **MAX_POS sweep** — cur_pos ∈ {255, 256, 257, 511, 1023, 4095, 16383, 32767}; dense ±4 around each to catch #30362.
5. **block_size ablation** — P ∈ {32, 64, 128}: L1 fit and per-step latency.
6. **Perf parity** — paged vs non-paged at cur_pos=255, traced. Target: within 15%.

Only after all six pass do we wire paged into the decode loop.

---

## Sources

- ttnn nanobind: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp` (authoritative signature)
- ttnn header: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp`
- Real-model usage: `models/tt_transformers/tt/attention.py`, `models/tt_transformers/demo/simple_text_demo.py`
- Tech report: `tech_reports/FlashAttention/FlashDecode.md` (tile/core constraints)
- Doc page (incomplete — missing `page_table_tensor`): https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.transformer.paged_scaled_dot_product_attention_decode.html
- PCC dips bug: https://github.com/tenstorrent/tt-metal/issues/30362
- Blackhole writer hang: https://github.com/tenstorrent/tt-metal/issues/16674
- vLLM PagedAttention paper: Kwon et al. 2023, "Efficient Memory Management for LLM Serving with PagedAttention"
- vLLM design doc: https://docs.vllm.ai/en/latest/design/paged_attention/
- Local discovery probe: `experiments/utils/sdpa_max_pos_ceiling_probe.py`
