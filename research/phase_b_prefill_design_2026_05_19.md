# Phase B — Prefill Design (Synthesis of 4 Research Agents) — 2026-05-19

Synthesizes the four parallel research agents into a unified Phase B design.
**No code written yet.** This doc is the design contract before implementation.

References:
- `deltanet_parallel_prefill_research.md` — Neumann chunked-parallel viability
- `sdpa_prefill_api_research.md` — tt-metal prefill SDPA + paged_fill_cache APIs
- `friend_prefill_walkthrough.md` — friend's prefill structure (the reference, not the copy)
- `our_prefill_codebase_audit.md` — what changes in our server_tp.py

## Why we're doing this

Today `handle_generate_tp` loops the decode trace once per prompt token.
At 83 ms/tok, an 8k prompt = 11 min TTFT (time-to-first-token) before the
model can start responding. Unusable for the daily-driver coding-assistant
use case. Real prefill = compute-bound forward pass over the whole prompt
at once. Targets: ~1 sec TTFT at 500 tokens, ~10 sec at 8k.

## Design contract

**`forward_prefill_tp_inner(state, prompt_ids: list[int]) → Tensor[1, VOCAB]`**

Runs **eager** (not traced — seq_len varies per call). Populates DeltaNet
recurrence state and paged KV cache in-place. Returns logits for the LAST
position only (the prediction for the first generated token).

Caller flow:
```
handle_generate_tp(prompt, max_tokens):
    prompt_ids = tokenizer(prompt)
    _reset_state_buffers(state)
    logits = forward_prefill_tp_inner(state, prompt_ids)   # eager, ~1-10 sec
    next_token = argmax(logits)
    for step in range(max_tokens - 1):
        next_token = traced_decode_step(state, next_token, prompt_len + step)
        yield next_token
```

## Per-layer prefill structure

### Embedding
```
prompt_ids: [seq_len]
  → ttnn.embedding(prompt_ids, state.embed_tt)
  → x: [seq_len, HIDDEN]
```
**Status:** ttnn.embedding supports batched lookup. No code change beyond
the call signature.

### RoPE cos/sin slice
```
positions: [0, 1, ..., seq_len - 1]
  → cos_seq: [seq_len, ROTARY_DIM]
  → sin_seq: [seq_len, ROTARY_DIM]
```
Single slice from our existing `state.cos_table_tt` / `state.sin_table_tt`.
Already at MAX_POS dimension.

### DeltaNet prefill — **two-tier implementation**

**MVP (Phase B.1-B.2)**: sequential decode-step loop. Mirrors friend's
`_forward_prefill_as_decode_steps`. Per token: slice → existing decode
path → accumulate via slice_write. Correct but slow.

**Upgrade (Phase B.3)**: chunked-parallel via Neumann factorization.
- Chunk size: C=64 (per `deltanet_parallel_prefill_research.md`)
- Within chunk: closed-form via `(I - L)^{-1} = (I+L)(I+L²)(I+L⁴)(I+L⁸)(I+L¹⁶)(I+L³²)`
- Between chunks: propagate final H state sequentially
- Cost: ~50-100× faster than sequential at 1k-8k tokens

This 2-stage approach lets us ship Phase B.1+B.2 first (correctness baseline)
then optimize with Phase B.3 (the real win). Build-from-scratch principle:
we implement the Neumann composition ourselves, NOT a port.

### Gated Attention prefill

```
qkv = ttnn.linear([seq_len, HIDDEN], w_qkv)  # naturally batched
q, k, v = slice_q_kv(qkv)
q_rot = ttnn.experimental.rotary_embedding(q[:, :, :, :rotary_dim], cos_seq, sin_seq)
q = ttnn.concat([q_rot, q_tail], dim=-1)  # same for k
ttnn.experimental.paged_fill_cache(cache_k, k, page_table, batch_idx=0)
ttnn.experimental.paged_fill_cache(cache_v, v, page_table, batch_idx=0)
attn_out = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    scale=1.0 / math.sqrt(head_dim),
    compute_kernel_config=state.sdpa_compute_kernel_config,  # B3 HiFi2
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    ),
)
attn_out = ttnn.linear(attn_out, w_o)
return _tp_all_reduce(state, attn_out) + x_residual
```

**Per `sdpa_prefill_api_research.md`**: `scaled_dot_product_attention(is_causal=True)`
is production-ready. Mesh compat on (1, 4) not explicitly tested but the
patterns match our decode path. Fallback: eager Q@K^T softmax V composition
if wedged (quadratic in seq_len but fine for ≤ 8k).

### MLP prefill

```
x: [seq_len, HIDDEN]
gate_out = ttnn.linear(x, w_gate, activation='silu')  # broadcasts on seq_len
up_out = ttnn.linear(x, w_up)
mid = ttnn.mul(gate_out, up_out)
down_out = ttnn.linear(mid, w_down)
return _tp_all_reduce(state, down_out) + x_residual
```

**Per `our_prefill_codebase_audit.md`**: zero code changes needed — matmul
+ rms_norm + all_reduce all broadcast on leading dim.

### LM head

```
x: [seq_len, HIDDEN]
x_last = ttnn.slice(x, [seq_len - 1, 0], [seq_len, HIDDEN])  # [1, HIDDEN]
logits = ttnn.linear(x_last, state.lm_head_tt)  # vocab-sharded → [1, VOCAB_SHARD] per chip
gathered = ttnn.all_gather(logits, dim=-1, cluster_axis=1, num_links=2, topology=Linear)
return gathered  # [1, VOCAB], for host argmax
```

Same as decode last step.

## Implementation phases

| Phase | Goal | Days | Validation |
|---|---|---|---|
| B.1 | Numpy reference + single-chip single-layer eager prefill | 3 | Cosine ≥ 0.999 vs HF Qwen3.6-27B layer-0 prefill output at seq_len=128 |
| B.2 | (1, 4) mesh multi-layer eager prefill (DeltaNet sequential, Attn parallel, MLP parallel) | 3 | Cosine ≥ 0.99 vs HF on 500-token prompt; TTFT ~10-30 sec (slow but correct) |
| B.3 | Chunked-parallel DeltaNet via Neumann | 5 | Cosine ≥ 0.99 vs B.2 baseline; TTFT 1-3 sec at 500 tokens (5-10× improvement) |
| B.4 | Integration into `handle_generate_tp` | 1 | End-to-end generate_tp with 500-token coding prompt + 100-token completion |
| **Total** | | **12 days** | |

## Validation gates per phase

**B.1** (single-chip, single-layer):
- Generate per-position hidden states for layer 0 in numpy ref
- Run our ttnn prefill at seq_len=128 on qb1 (single-chip)
- Cosine ≥ 0.999 against numpy ref at every position
- Output shape: [1, 128, HIDDEN]

**B.2** (mesh, multi-layer):
- 500-token prompt through all 64 layers on qb2 (1,4) mesh
- Cosine ≥ 0.99 vs HF bf16 oracle (single-chip CPU)
- TTFT measured (slow OK — sequential DeltaNet is the bottleneck)
- KV cache populated correctly (cross-validate with decode step that reads it)

**B.3** (Neumann chunked-parallel):
- Replace DeltaNet sequential loop with chunked Neumann
- Cosine ≥ 0.99 vs B.2 baseline (within bf16 noise of same math)
- TTFT improves by ≥5× at 500 tokens
- Test chunk boundary correctness explicitly (positions 63, 64, 65)

**B.4** (integration):
- `generate_tp --prompt "Implement a JSON parser in Rust" --max-tokens 100`
- Coherent generation
- TTFT < 5 sec at 500 tokens (single bar of usability)
- Decode tok/s unchanged (≥12.93 baseline)

## Risks (ordered)

1. **`paged_fill_cache` on (1, 4) mesh** — not explicitly tested. Fallback: loop `paged_update_cache` over seq_len positions (slower but works).

2. **`scaled_dot_product_attention(is_causal=True)` on (1, 4) mesh** — similar story. Fallback: eager Q@K^T softmax V (we have all primitives).

3. **DeltaNet recurrence state shape compatibility** — the state shape produced by sequential prefill must match what the decode path expects. If shapes diverge (likely small issue), need state_reshape adapter.

4. **Numpy reference for DeltaNet prefill** — building a correct reference is its own subtask. The HF Qwen3.6 modeling code is the source of truth; need to extract it cleanly.

5. **`ttnn.experimental.rotary_embedding` at seq_len > 1** — friend uses it. We tried single-position native RoPE today and the G3 cosine ladder FAILED (`93f955c`). Multi-position behavior might be different / cleaner. Need fresh isolation probe before relying.

## Non-goals

- Multi-user batching (single user only for daily-driver use)
- Traced prefill (eager is the right call per friend + our independent analysis)
- Continuous batching (out of scope for Phase B)
- Sliding-window attention (Qwen3.6-27B doesn't use it)

## Files that will change

| File | Change | LoC estimate |
|---|---|---|
| `experiments/serve/server_tp.py` | Add `forward_prefill_tp_inner`, DeltaNet + Attn prefill helpers | ~400 |
| `experiments/serve/server_tp.py` | Refactor `handle_generate_tp` to call prefill | ~10 |
| `experiments/serve/client_tp.py` | Optional `--prefill-mode {sequential, chunked_parallel}` toggle | ~15 |
| `experiments/utils/prefill_numpy_reference.py` | NEW — HF-equivalent numpy prefill for validation | ~150 |
| **Total new** | | **~575 LoC** |

## Order of operations

1. Today/Tomorrow (Phase B.1 start): build numpy reference + single-chip single-layer ttnn prefill on qb1
2. After B.1 passes: B.2 on qb2 (mesh + multi-layer)
3. After B.2 passes: B.3 Neumann chunked-parallel
4. After B.3 passes: B.4 integration + ship

After Phase B ships, **Phase A** (extend MAX_POS to 4k/8k) becomes mechanical:
just change the constant + verify the same gates still pass at larger context.
