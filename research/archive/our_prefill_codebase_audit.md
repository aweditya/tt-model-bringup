# Our Codebase: What Changes for Prefill — Audit 2026-05-19

**Compiled from background agent investigation. No code written.**

## TL;DR

Adding a prefill path is **400-550 LoC** of new code, **mostly composing
existing functions with leading-dim batching**. The hard parts are:
- DeltaNet conv1d (decode-only owned kernel doesn't apply; need scanning variant)
- DeltaNet recurrence (need our Neumann chunked-parallel — see `deltanet_parallel_prefill_research.md`)
- Gated Attention SDPA (need a prefill variant; risk if `paged_scaled_dot_product_attention` non-decode doesn't exist on mesh)

Most other components (MLP, RMS norm, all_reduce, embeddings) broadcast
naturally over a leading `seq_len` dim with zero changes.

## Recommendation: eager prefill, traced decode

Friend's approach is correct here. Each prompt has variable `seq_len` →
trace-per-prompt-length doesn't reuse. Prefill latency amortizes over the
generated decode tokens, so eager is fine. Decode stays in the existing
traced single-token path.

Caller flow:
1. `handle_generate_tp(prompt)` → `forward_prefill_tp_inner(prompt_ids)` (eager, populates KV cache + DeltaNet state)
2. Then loop existing `_traced_forward` for decode tokens

## What changes per function

| Component | Decode shape | Prefill shape | Change |
|---|---|---|---|
| Token embedding | `[1,1]` → `[1,HIDDEN]` | `[seq_len]` → `[seq_len,HIDDEN]` | Batch lookup; ttnn.embedding supports 2D LUT |
| RoPE cos/sin lookup | `[1,1]` → `[1,ROTARY_DIM]` | `[seq_len]` → `[seq_len,ROTARY_DIM]` | Multi-row slice or batched embedding |
| DeltaNet in_proj | `[1,HIDDEN]` linear | `[seq_len,HIDDEN]` linear | **No change** — matmul broadcasts |
| **DeltaNet conv1d** | Single-token state mutation | Causal scan over seq_len | **NEW** — owned decode kernel doesn't apply; need scanning variant or eager loop |
| **DeltaNet recurrence (SSM)** | Single H update | H_0 → H_1 → ... → H_{seq_len-1} | **NEW** — use Neumann chunked-parallel (see other research doc) |
| RoPE rotation | Single rotation `[1,HEAD_DIM]` | Multiple `[seq_len,n_heads,HEAD_DIM]` | Multi-position; reshape + batch rotate |
| Attention KV write | Single via `paged_update_cache(idxs=[cur_pos])` | Vector via `paged_update_cache(idxs=[0..seq_len-1])` | **API already supports** vector idxs — no API change |
| **Attention SDPA** | `paged_scaled_dot_product_attention_decode` (Q=1) | Full cross-attention with causal mask (Q=seq_len) | **Need prefill variant** (see SDPA API research doc) |
| MLP | `[1,HIDDEN]` projections | `[seq_len,HIDDEN]` projections | **No change** — matmul broadcasts |
| LM head | `[1,HIDDEN]` → argmax | `[seq_len,HIDDEN]` → slice last → argmax | Slice to position `seq_len-1` |

## What does NOT change

These functions need ZERO modification:
- `mlp_step_tp` — rms_norm + gate/up linear + mul + down linear + all_reduce broadcast on leading dim
- `_rms_norm_manual` — normalizes along HIDDEN; ignores leading dim
- `_tp_all_reduce` — reduces along cluster_axis=1 (chip dim); leading dims passthrough
- All weight loading and weight tensors
- Embedding table `state.embed_tt`
- RMS norm weights, MLP weights, LM head, final norm
- KV cache memory layout

## What does change (concrete code touch points)

| Function | Action | LoC estimate |
|---|---|---|
| `forward_prefill_tp_inner` (NEW) | Whole new entry mirroring `forward_token_tp_inner` | 250-350 |
| DeltaNet prefill helpers (conv1d scan + Neumann recurrence) | New module | 80-120 |
| Gated Attention prefill branch | If-else inside existing `gated_attn_step_tp` or sibling fn | 40-80 |
| `handle_generate_tp` refactor | Call prefill once before decode loop | ~10 |
| `update_prefill_input_buffers` | New eager input pipe | ~30 |
| **Total** | | **400-550 LoC** |

## Compatibility risks (ordered by severity)

1. **`paged_scaled_dot_product_attention` (prefill, non-decode)** — does it exist on mesh? Per memory `feedback_p1_sdpa_decode_breaks_on_mesh.md`, the decode variant had a tree-reduction wedge that we fixed via explicit CoreCoord(4,4) program_config. The prefill variant may have similar gotchas. **Mitigation:** the SDPA API research agent (still running) is checking this. Fallback: implement eager manual Q@K^T softmax V (quadratic in seq_len but fine since prefill < 8k for our use case).

2. **DeltaNet conv1d decode-only owned kernel** — `qwen36_conv1d_decode_owned` mutates split-column state in place. For prefill, must either (a) loop the decode kernel `seq_len` times (wastes prefill amortization), (b) build a new `qwen36_conv1d_prefill_owned` scanning kernel, or (c) eager prefix-conv1d composition. **Mitigation:** start with (a) for correctness baseline, then optimize.

3. **DeltaNet recurrence parallelization** — addressed by our existing Neumann work per `deltanet_parallel_prefill_research.md`. NOT blocking; high-impact optimization opportunity.

4. **RoPE multi-position lookup** — straightforward batch embedding. Trace will need different shape (only an issue if we trace prefill, which we're not).

5. **Two-trace state management** — `state.trace_id` is single decode trace. If we ever want to trace prefill (variable seq_len makes this hard), we'd need a dict keyed on seq_len. **Not blocking** since eager prefill is the plan.

## Cleanest sketch of the new entry point

```
forward_prefill_tp_inner(state, prompt_ids: list[int]) -> Tensor:
    # Eager. Produces logits for position seq_len-1 only.
    seq_len = len(prompt_ids)
    x = embed_lookup(prompt_ids)  # [seq_len, HIDDEN]
    cos, sin = rope_slice(0, seq_len)  # [seq_len, ROTARY_DIM]
    for layer in state.layers:
        x = (deltanet_prefill_tp if layer['type'] == 'linear_attention'
             else gated_attn_prefill_tp)(state, x, layer, cos, sin, seq_len)
        x = mlp_step_tp(state, x, layer['mlp'])  # broadcasts on leading dim
    x = _rms_norm_manual(x, state.final_norm_tt, EPS, HIDDEN)
    last = ttnn.slice(x, [seq_len - 1, 0], [seq_len, HIDDEN])  # [1, HIDDEN]
    logits = ttnn.linear(last, state.lm_head_tt)  # [1, VOCAB_SHARD]
    return all_gather(logits, dim=-1)  # [1, VOCAB] for argmax
```

## Effort breakdown

- **Phase B.1 — Numpy reference + ttnn-only single-layer prefill** (3 days):
  Validate the math at MAX_POS=512, single layer, single chip. Compare to
  HF Qwen3.6-27B layer-0 prefill output bit-for-bit (cos > 0.999).

- **Phase B.2 — Mesh + multi-layer eager prefill** (3 days):
  Wire through 64 layers on (1, 4) mesh. Sequential DeltaNet conv1d + sequential
  recurrence (slow but correct). Validate vs HF on real prompts.

- **Phase B.3 — Parallel DeltaNet via Neumann chunked-scan** (5 days):
  Replace sequential conv1d/recurrence with the chunked-parallel form. Validate
  correctness at chunk boundaries. Bench TTFT at 500/1k.

- **Phase B.4 — Integration into handle_generate_tp** (1 day):
  Plumb the eager prefill call before the existing decode trace loop. Real
  generate_tp validation with 500-token prompt + 60-token generation.

**Total: ~12 days of focused work.** Aligns with the 2-3 week estimate
from the Neumann research agent.
