# Friend's Prefill Forward Pass — Walkthrough 2026-05-19

**Compiled from background agent investigation. No code written.**

## Call chain

```
Generator.prefill_forward_text()
  ├─ prepare_prefill_inputs_trace()  [host: embedding + RoPE]
  ├─ copy_host_to_device()
  └─ Model.ttnn_prefill_forward()    [DEVICE EAGER — not traced]
       └─ Model.forward(mode=PREFILL)
            └─ for each layer in self.layers:
                 └─ TransformerBlock.forward(mode=PREFILL)
                      ├─ [DeltaNet]   _forward_prefill_as_decode_steps()
                      │                 → seq_len iterations of decode kernel
                      ├─ [Gated Attn] attention.forward_prefill()
                      │                 → parallel SDPA on full seq
                      └─ MLP.forward(mode=PREFILL)
            └─ final norm + lm_head
  └─ process_output_prefill()         [host: sample last token]
```

## 12 key questions answered

| # | Question | Answer |
|---|---|---|
| 1 | Entry point signature | `generator.py:1339`. Args: `tokens [1, seq_len] int32`, `page_table`, `prefill_len`. Returns logits for ALL positions. |
| 2 | Embedding batching | Fully batched: [1, seq_len] → [1, 1, seq_len, hidden_size]. Single ttnn.embedding call. |
| 3 | Per-layer loop | Simple `for i, layer in enumerate(self.layers)` (model.py:787). No special dispatch. |
| 4 | **DeltaNet prefill** | **Sequential token-by-token loop** (`decoder.py:281: for token_idx in range(seq_len)`). Slices one token, calls `forward(mode=DECODE)`, accumulates via `slice_write`. Reuses proven decode kernel. **High latency at long prompts.** |
| 5 | Gated Attn prefill | `ttnn.transformer.scaled_dot_product_attention(is_causal=True)` (line 1203). KV write via `paged_fill_cache` (line 1158). Parallel RoPE on all S positions (line 1093). |
| 6 | MLP prefill | Identical to decode. Reshape for large seq if seq > cutoff (mlp.py:136). |
| 7 | CCL during prefill | reduce_scatter triggered when mode==PREFILL or dim==8192 (mlp.py:176). `all_reduce(sharded=False)` for prefill vs `sharded=True` for decode. |
| 8 | KV cache after prefill | Persists. Cache slots [0, prefill_len) filled. Decode starts at position prefill_len without re-prefill. |
| 9 | Output | Logits for ALL positions [1, 1, seq_len, vocab]. Host post-processing samples only last position. |
| 10 | Trace structure | **Eager prefill, traced decode.** seq_len varies → tracing overhead not justified. Decode is traced (amortized per generated token). |
| 11 | Eager confirmation | `ttnn_prefill_forward()` called eagerly (not via `execute_trace()`). Tracing at generator.py:220-308 is warmup/validation only. |
| 12 | Page table management | Pre-allocated at prefill start: `num_blocks = num_blocks_in_seq(seq_len, block_size)`. Fixed size. No dynamic growth. |

## Prefill vs Decode differences

| Aspect | Prefill | Decode |
|---|---|---|
| Sequence | All S in parallel (gated) or seq_len iterations (DeltaNet) | One token |
| QKV proj | `ttnn.linear` on full seq | Single-token matmul |
| RoPE | Parallel `rotary_embedding` on [1,H,S,D] | Per-token with cur_pos index |
| SDPA | `scaled_dot_product_attention(is_causal=True)` | `paged_scaled_dot_product_attention_decode` |
| KV fill | `paged_fill_cache` (full seq) | `paged_update_cache` (single pos) |
| Trace | Eager | Traced |
| Memory | Interleaved DRAM | Sharded L1 |
| MLP all_reduce | `sharded=False` | `sharded=True` |
| Output | Logits [1, 1, seq_len, vocab] | Logits [1, vocab] |

## Reusable patterns

1. **Mode-aware memory configs**: `get_*_mem_config(mode, ...)` branches on PREFILL vs DECODE
2. **Batch reshape pattern**: `[B, 1, S_per_user, H]` ↔ `[1, 1, B*S, H]` for batched prefill
3. **Sequential fallback for linear attention** (DeltaNet): reuse decode kernel in loop (correctness first)
4. **Per-user page-table slicing**: extract per-user blocks for multi-user prefill
5. **Eager + warmup trace**: compile eagerly, optional trace for repeated lengths
6. **Causal via is_causal=True**: no explicit mask tensor needed

## Things we'd do differently (per build-from-scratch principle)

1. **Parallel DeltaNet prefill via Neumann chunked-scan** — friend ships sequential; we have the primitives for ~50× faster (see `deltanet_parallel_prefill_research.md`)
2. **No multi-user batching** — single user is fine for our daily-driver use case
3. **No traced prefill** — agree with friend, eager is right (variable seq_len)
4. **Simpler page table** — pre-allocate at MAX_POS, no per-prompt resizing

## Friend's effort estimate (port-style)

| Phase | Days |
|---|---|
| Attention prefill (forward_prefill + paged_fill_cache) | 3-4 |
| MLP prefill (mostly identical, reshape) | 1 |
| DeltaNet prefill (sequential decode loop) | 1 |
| Model loop (mode-aware dispatch) | 0.5 |
| Page table setup | 2 |
| CCL integration | 1.5 |
| Memory configs | 2-3 |
| Tracing/warmup | 2 |
| Testing | 3-4 |
| **Total port** | **15-20 days** |

For US: this is faster because (a) we're not porting their abstractions, (b) we can ship sequential DeltaNet first then upgrade to Neumann chunked-parallel later, (c) we already have B3 SDPA compute config + paged cache infra working on (1,4) mesh.

**Realistic estimate: 12-15 days** if we hit no major API wedges.
