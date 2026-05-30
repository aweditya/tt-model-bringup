# tt-metal SDPA Prefill + paged_fill_cache APIs — Research 2026-05-19

**Compiled from background agent investigation. No code written.**

## Available APIs

| API | Mode | Input Q | Input K/V | Multi-pos write | Chunked | Paged | Causal | Status |
|---|---|---|---|---|---|---|---|---|
| `ttnn.transformer.scaled_dot_product_attention` | Prefill | [B,H,S,D] | [B,H,S,D] | via fill_cache | No | No | `is_causal=True` | Production |
| `ttnn.transformer.chunked_scaled_dot_product_attention` | Prefill | [B,H,S_chunk,D] | paged | via page_table | Yes | Yes | Implicit | Production |
| `ttnn.transformer.paged_scaled_dot_product_attention_decode` | Decode | [B,H,1,D] | paged | n/a | No | Yes | Implicit | (1,4) tested |
| `ttnn.fill_cache` | Setup | input: [1,H,S,D]; cache: [B,H,S,D] | dense | Full seq | No | No | n/a | Production |
| `ttnn.experimental.paged_fill_cache` | Setup | input: [1,H,S,D]; cache: paged | paged | Full seq | No | Yes | n/a | Production |
| `ttnn.experimental.paged_update_cache` | Decode | input: [1,H,1,D]; update_idxs | paged | Single | No | Yes | n/a | (1,4) tested |

## Recommended primitives for our build

### Phase 1 — Non-chunked prefill (seq_len ≤ 512)

```python
ttnn.transformer.scaled_dot_product_attention(
    input_tensor_q,  # [1, H_q, seq_len, head_dim]
    input_tensor_k,  # [1, H_kv, seq_len, head_dim]
    input_tensor_v,  # [1, H_kv, seq_len, head_dim]
    is_causal=True,  # implicit lower-triangular mask
    scale=1.0 / math.sqrt(head_dim),
    compute_kernel_config=state.sdpa_compute_kernel_config,  # our existing B3 HiFi2 config
    program_config=...,  # see SDPAProgramConfig below
)
```

### Phase 2 — KV cache populate

```python
ttnn.experimental.paged_fill_cache(
    cache_tensor,       # [num_blocks, n_kv, block_size, head_dim] — our existing layout
    input_tensor,       # [1, n_kv, seq_len, head_dim]
    page_table_tensor,  # [1, max_blocks_per_seq]
    batch_idx=0,
)
```

Writes the full prefill chunk into the cache in ONE call. Replaces N
`paged_update_cache` calls. This is what friend uses (attention.py:1158).

### Phase 3 — Chunked prefill (seq_len > 512)

```python
for chunk_idx in range(num_chunks):
    chunk_start_idx = chunk_idx * chunk_size  # e.g., 256 or 512
    Q_chunk = ttnn.slice(Q, [chunk_start_idx, ...], [chunk_start_idx + chunk_size, ...])
    out_chunk = ttnn.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=Q_chunk,
        input_tensor_k=keys_paged,  # K from cache (already populated)
        input_tensor_v=values_paged,
        page_table_tensor=page_table,
        chunk_start_idx=chunk_start_idx,
        program_config=...,
    )
```

## Friend's exact call (cross-reference)

**Non-chunked prefill** (`tt/attention.py:1200-1212`):
```python
sdpa_seq_len = seq_len // batch_size if batch_size > 1 else seq_len
attn_output = ttnn.transformer.scaled_dot_product_attention(
    q_heads_1QSD_8b, k_heads_1KSD_8b, v_heads_1VSD_8b,
    is_causal=True,
    sliding_window_size=self.sliding_window,
    scale=self.scale,
    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
    program_config=self.args.get_attn_sdpa_program_config(
        Mode.PREFILL, sdpa_seq_len, None, None,
    ),
)
```

**Chunked prefill** (`tt/attention.py:1190-1198`) uses
`chunked_scaled_dot_product_attention` with `chunk_start_idx`.

**Prefill RoPE** (`tt/qwen36.py:1575-1611`):
```python
q_rot = ttnn.experimental.rotary_embedding(
    q_heads_1QSD_pre_rot[:, :, :, :rotary_dim],
    rot_mats[0],  # cos_cached [1, 1, max_pos, rotary_dim]
    rot_mats[1],  # sin_cached [1, 1, max_pos, rotary_dim]
)
# concat with tail dim:
q_heads_1QSD = ttnn.concat([q_rot, q_tail], dim=3)
```
**Key insight:** Full-sequence Q/K input; kernel auto-indexes cos/sin cache
by position. No manual per-position RoPE loop needed.

## SDPAProgramConfig tuning for our shape

For our (1, 4) mesh at seq_len ≤ 512:

```python
ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),  # our existing decode setting
    q_chunk_size=128 or 256,  # inner Q loop granularity
    k_chunk_size=128,         # inner K/V loop
    exp_approx_mode=False,    # we use False for accuracy per B3 config
)
```

For longer prefill (1k-8k), `chunked_scaled_dot_product_attention` with
chunk_size 256-512 amortizes the per-chunk overhead while staying within
kernel limits.

## Mesh compatibility (1, 4)

- Decode paged SDPA: **explicitly tested** on (1, 4) — production-deployed
- Prefill SDPA variants: **not explicitly tested on (1, 4)** but friend's
  similar distributed patterns work. Should work; if wedge, fall back to
  eager Q@K^T softmax V composition (quadratic in seq_len, fine since prefill
  < 8k for our use case).

## Causal masking

- **Implicit (recommended)**: `is_causal=True` flag → kernel applies
  lower-triangular mask internally
- **Explicit (optional)**: construct `[B, H, S, S]` mask with -inf for
  future positions, pass as `attn_mask` with `is_causal=False`

For Qwen3.6 prefill: `is_causal=True` is the correct choice.

## Critical file references

- Friend's non-chunked call: `tt-qwen-36/models/tt_transformers/tt/attention.py:1200-1212`
- Friend's chunked call: `tt-qwen-36/models/tt_transformers/tt/attention.py:1190-1198`
- Friend's prefill RoPE: `tt-qwen-36/models/tt_transformers/tt/qwen36.py:1575-1611`
- tt-metal chunked SDPA tests: `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py`
- tt-metal paged_fill_cache tests: `tt-metal/tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py:560-625`
