# tt-metal MoE + Continuous-Batching patterns — research for Qwen3.6-35B-A3B CB bringup

Background research for porting the working 27B CB stack
(`experiments/serve/{cb_engine,cb_scheduler,cb_metrics,cb_api,openai_endpoint,server_tp_cb}.py`,
plus the model-specific `server_tp.py`) onto the 35B-A3B model
(`experiments/serve/server_35b_ttnn.py`). The 35B server is single-stream
(B=1) today; the goal is iteration-level CB analogous to CB1–CB4 done for 27B.

Hosts (`qb1`, `qb2`) were unreachable at the time of writing, so all references
are GitHub `main` paths; cross-reference against the local `experiments/.refs/`
mirror or `~/tenstorrent/tt-metal/` on either host once they come back up.

---

## 1. tt-metal MoE bringup reference

Three primary upstream candidates were surveyed.

### 1.1 `models/demos/deepseek_v3/` — the active MoE reference

This is the cleanest, most-recent reference and is **the one we should copy
from**. DeepSeek-V3 / R1 is 671B / 37B-active MoE (256 routed experts + 1
shared expert, top-8 routing) — same shape as Qwen3.6-35B-A3B's MoE
sub-layer.

Relevant files (all under `models/demos/deepseek_v3/tt/`):

- `moe.py` — generic MoE forward. Top-k → `ttnn.all_to_all_dispatch` →
  per-expert FFN (`experts.py`) → `ttnn.all_to_all_combine`. Forward is
  classmethod `MoE.forward(cls, x, cfg)`, with explicit `forward_prefill` /
  `forward_decode` entry points (lines 396–461 in upstream `moe.py`).
- `moe_optimized.py` — quad-mesh-only path that replaces the
  dispatch+experts+combine triple with the experimental fused op
  `ttnn.experimental.all_to_all_dispatch_metadata` + `ttnn.experimental.moe_compute`
  (single kernel-level dispatch/compute/combine; lines 598–625). Only valid for
  16-row dispatch + ring fabric — not us.
- `experts.py` — batched per-expert FFN. Inputs come in expert-major
  layout `(1, num_experts_per_device, num_tokens, hidden)`; runs
  `ttnn.linear(x, w1)` + `ttnn.linear(x, w3)` + `ttnn.mul(silu, up)` +
  `ttnn.linear(mid, w2)` once for ALL local experts (no per-expert loop).
- `moe_gate.py` — router. Reshapes logits to `(batch_size_per_iter, 16, 16)` and
  calls `DeepseekMoeGateOp.op(...)`. Routes through `topk_first` semantics
  (matches the trick the 35B server already uses in
  `server_35b_ttnn.py:_moe_router_topk` — see §5). Output shape after
  reshape: `(1, 1, total_batch_size, 8)` for both indices and weights.
- `generator.py` — `class DeepseekGenerator` (lines 495–620). Constructor
  takes `batch_size_per_row` (default `USERS_PER_ROW`), `enable_trace`,
  `enable_mtp`. `batch_size = batch_size_per_row * mesh_device.shape[0]`.
  Per-row paged-attention page tables. Distinct `prefill_forward` and
  `decode_forward` methods.
- `generator_vllm.py` — `class DeepseekV3ForCausalLM(DeepseekGenerator)`
  (lines 60–199). Declares `model_capabilities = {"supports_prefix_caching":
  False}`. `prefill_forward` accepts `empty_slots` (slot map) and loops
  per-user (lines 115–199). `decode_forward` returns either token IDs
  (device-side sampling) or logits (host-side). `allocate_kv_cache` reshapes
  the page table to whatever block size vLLM picks.

Key shape observation: DeepSeek's MoE forward keeps `num_tokens` as a normal
batch-like axis — the same tensor goes through router → dispatch →
expert-batched matmul → combine. It is not running a Python for-loop over
slots. This is the upstream proof point that B>1 MoE is doable without
per-slot dispatch.

### 1.2 `models/demos/llama3_70b_galaxy/` — the dense-CB reference

Dense (no MoE), but it's the closest production-grade CB demo. Files under
`tt/`: `generator.py`, `generator_vllm.py`, `llama_attention.py`,
`llama_decoder.py`, `llama_model.py`, `llama_mlp.py`, `distributed_norm.py`,
`prefetcher_common.py`, `llama_ccl.py`, `model_config.py`.

What's worth copying from this demo:

- **Two-phase trace warmup**: `generator.py` lines 553–621 captures `sp0`
  (no prefix cache) and `sp1` (with cached prefix) traces over a sweep of
  sequence lengths, batch 1 and batch 32. Trace key
  `f"{seq_len}_{batch}_{sp0|sp1}"`. This is the same pattern as
  `feedback_two_phase_warmup.md` — compile ALL paths before capturing any.
- **Per-slot logits buffer**: `self.tt_logits_accumulated = [from_torch(...)
  for _ in range(max_batch_size)]` (line 360-ish). This is what we did with
  per-slot DN state in `cb_dn[li]`.
- **Batched prefill at B=32 with uniform length**: `prefill_forward_text`
  detects batch≥16 + uniform 128-token length and goes down the
  32-user-batched path (lines 628–680). Heterogeneous lengths fall back to
  the per-user single-user path.
- `QwenForCausalLM` (line 244) reuses the LlamaForCausalLM vLLM glue with no
  override — Qwen3 dense models inherit the entire CB stack as-is.

### 1.3 `models/tt_transformers/` — the model-agnostic base

`tt/` contains `load_checkpoints.py`, `model_config.py`, etc. — currently
dense-only. Released for Qwen3 (June 9, 2025) for dense 0.6B–32B. No MoE
support in `tt_transformers`. DeepSeek-V3 lives in its own `demos/`
subdirectory specifically because the MoE plumbing doesn't fit the dense
transformer pattern yet.

**Verdict**: copy DeepSeek-V3's MoE plumbing pattern; copy Llama-70B-Galaxy's
CB / vLLM glue; do NOT depend on `tt_transformers` (it has no MoE).

---

## 2. How Llama-70B-Galaxy / tt-transformers handle batched CB

Concrete shape conventions to mirror in our 35B CB:

- **Batch layout is a leading dim**, not an interleave. Input tokens
  `(batch, seq_len)`. KV cache page table `(batch, num_blocks)`. Page table
  is sliced per-user for the single-user prefill fallback:
  `page_table_user = page_table[user_id:user_id+1, :]`.
- **Per-request state** = page-table-row + cur_pos entry. Empty slots
  signalled by `empty_slots: list[int]` from vLLM. Our 27B CB uses
  `cur_pos = -1` to mean "empty" — Galaxy uses an explicit slot mapping; same
  semantic, different wire format. The 27B convention is fine, keep it.
- **vLLM integration**: `LlamaForCausalLM(Generator)` is just a thin wrapper.
  `prefill_forward / decode_forward` delegate to `super().prefill_forward_text
  / decode_forward`. `allocate_kv_cache` reshapes a paged cache shared with
  vLLM's block manager (the `allocate_vllm_kv_cache` helper at lines 14–40
  builds a `ttnn.as_tensor` paged store directly from the page-aligned host
  tensors that vLLM allocates).
- **`model_capabilities` dict on the class** is how a model declares whether
  it supports `prefix_caching`, `async_decode`, etc. DeepSeek-V3 currently
  declares `supports_prefix_caching: False`; we should declare ours False too
  for v1 — same reasoning as their MoE has no prefix-cache-safe routing path
  yet.

---

## 3. Batched MoE routing — the hard part

The per-slot routing problem is real: every token sees a different set of 8
experts. There are three solutions in production:

1. **Per-slot Python loop**. Correct but kills throughput. No production
   demo uses this. Listed only for v0 fall-back.
2. **Batched matmul over stacked experts** (no dispatch). For B=1 and small
   B, just compute every local expert's output and mask. This is what the
   35B server's `moe_forward_ttnn_pattern_a_batched`
   (`server_35b_ttnn.py:1225`) already does — one
   `ttnn.matmul(h_3d_repeat, experts_gate_up_local)` of shape
   `[E_LOCAL=64, 1, HIDDEN=2048] @ [E_LOCAL=64, HIDDEN, 2*MOE_INTER=1024]`
   covers all 64 local experts in one shot, then a top-k mask zeros the
   unrouted ones. Pattern A masks AFTER compute, so per-slot routing is
   trivial: the router emits `[B, TOP_K]` indices, expand to
   `[B, E_LOCAL, TOP_K]`, equality-test with `local_expert_ids`, sum over
   `TOP_K` to get `[B, E_LOCAL]` routing weight, broadcast over
   `[B, E_LOCAL, 1, HIDDEN]` expert outputs. **This is the v0 → v1 path.**
3. **All-to-all dispatch + expert-major reshuffle**. The upstream
   DeepSeek-V3 path (`moe.py:424–461`):
   `ttnn.all_to_all_dispatch(x_chunk, topk_indices_chunk, expert_mapping_tensors,
   **cfg["all_to_all_dispatch"])` reshuffles tokens to live on the device
   that owns their selected expert; experts run in parallel; then
   `ttnn.all_to_all_combine(...)` puts each token back. Trace-clean. **This
   is the v2 path**; needed once batched matmul becomes wasteful (when
   `B * top_k / E_total` is small — e.g. B=32, top_k=8, E=256 → ~64 of 256
   experts active, so the wasted-compute factor under Pattern A is
   `256 / 64 = 4×`). For our (1,4) mesh with `E_LOCAL=64`, the wasted
   factor is `64 / min(B*top_k, 64) = 64 / min(B*8, 64)`. At B=1 that's
   `64/8 = 8×`; at B=8 it's `64/64 = 1×` (break-even); past B=8 Pattern A
   gets *better* because we'd have hit each expert anyway. So Pattern A is
   the right v1 strategy at our batch sizes (and is what the 35B B=1 code
   already does).

**Reference for v2 (all_to_all path)**: vLLM upstream
`vllm/model_executor/layers/fused_moe/` uses an expert-major
reshuffle (`fused_experts` kernel). Conceptually the same as TT's
`all_to_all_dispatch + moe_compute + all_to_all_combine` but CUDA-fused.
We do NOT need to copy this; we have a working `Pattern A` path.

---

## 4. Qwen3.6-35B-A3B specifics

Architecture (from the HF card and `server_35b_ttnn.py`):

- 40 layers total. Pattern: 10 blocks of `(3 × GatedDeltaNet + 1 × GatedAttention)`.
  Every layer's FFN is MoE (NOT dense).
- GatedDeltaNet: 32 V heads, 16 QK heads, head_dim=128.
- GatedAttention: 16 Q heads, 2 KV heads, head_dim=256, partial_rotary_dim=64.
- MoE: **256 experts** (NOT 64 — the user's prompt said 64, but the HF card and
  `server_35b_ttnn.py:upload_moe_layer` confirm 256 routed + 1 shared, top-8
  routed + 1 shared active). Expert intermediate = 512. `E_LOCAL = 256 / 4 = 64`
  per chip on the (1,4) mesh, hence "64 experts per chip" — that's what was in
  the prompt.
- Hidden = 2048. Vocab = 248320 (padded).
- Context: 262K native, YaRN to ~1M.
- Tokenizer: standard Qwen3 SentencePiece / BPE; chat template is
  `<|im_start|>role\ncontent<|im_end|>` Jinja, with `<think>…</think>` blocks
  by default (no `/think` `/no_think` soft switch — model card explicitly
  says soft-switch is NOT supported, unlike Qwen3-30B-A3B). Same tokenizer
  family as 27B.
- **MTP**: Multi-Token Prediction head exists (one extra MTP layer beyond
  the 40). Out of scope for v1 CB; integrate after v3.

Upstream issues to know about:

- vllm-project/vllm#36493 — Qwen3.5-35B-A3B prefix-cache hit rate ~0.1%
  (regression from Qwen3-30B-A3B's ~20%). Root cause is the hybrid
  GatedDeltaNet state — DeltaNet has no notion of "prefix" so vLLM's block
  manager can't dedupe DN-state. Qwen3.6-35B-A3B has the same architecture.
  **Implication**: we should not bother chasing prefix caching for the
  GatedDeltaNet layers in v1; only the GatedAttention layers (KV cache) are
  prefix-cacheable. Even the upstream vLLM team has given up.
- vllm-project/vllm#38182 — MTP + prefix-cache further harm hit rate. Don't
  combine until v3+.

No open tt-metal issues mention `Qwen3.6-35B-A3B` directly (verified via
GitHub issue search).

---

## 5. Patterns to reuse from the user's own 27B CB work

Files surveyed (all in `experiments/serve/`):

| File | Lines | Model-agnostic? | Notes |
|---|---|---|---|
| `cb_engine.py` | 405 | YES | `RequestHandle`, `CBEngine` — pure async glue. Reuse as-is. |
| `cb_scheduler.py` | 718 | YES | Orca scheduler: `admit`, `_step_prefill_chunked`, `step`, `_step_sampled`. Talks to `state` only via `setup_cb_state`, `cb_reset_slots`, `forward_batch_tp_inner`, `update_input_buffers_batched`. Reuse as-is. |
| `cb_metrics.py` | 136 | YES | Counters only. |
| `cb_api.py` | 250 | YES | Unix-socket RPC. |
| `openai_endpoint.py` | 198 | YES | OpenAI shim. |
| `live_slot_store.py` | 126 | YES | Per-slot rolling text buffer. |
| `protocol.py` | — | YES | Message types. |
| `server_tp_cb.py` | 695 | NO — has 27B shape assumptions | See breakdown below. |

`server_tp_cb.py` breakdown (the **only** file that needs a 35B-specific
twin, to be called `server_35b_cb.py`):

- `setup_cb_state(state, B, blocks_per_seq)` (line 45) — generalizable but
  needs three model-specific changes:
  - Imports `NV_PER_CHIP, K_DIM, V_DIM, CONV_DIM_CHIP` from
    `full_layer_tp_probe`. For 35B these become `NV_PER_CHIP=8` (32 V heads /
    4 chips), `K_DIM=128`, `V_DIM=128`, `CONV_DIM_CHIP=CONV_DIM/4`. Just
    re-import from `server_35b_ttnn` constants.
  - DeltaNet recurrence state shape is `[B, NV_PER_CHIP, K_DIM, V_DIM]` —
    same shape for both models; 35B picks different head counts.
  - Owned-GDN kernel: 27B's CB uses the manual recurrence because
    `qwen36_gdn_decode_owned` hard-asserts B=1. 35B has the SAME constraint
    (MEMORY.md: "hard-asserts batch=1"). v0 must use the manual DN. Batching
    the owned kernel is archive/superseded_research_2026-06-04/maintainability_pass-level work — defer.
- `cb_reset_slots(state, slot_ids)` (line 157) — pure math on DN ssm + conv
  cols. Generalizes immediately.
- `cb_reset_states(state)` (line 146) — same, generalizes.
- `cb_prefill_transplant(state, slot_s, L)` (line 188) — generalizes; needs
  the 35B's GatedAttention layer index list instead of 27B's
  `state.layers[…]['type']=='full_attention'` test (which is already present
  in 35B's `state.layer_types`).
- `update_input_buffers_batched` (line 310) — generalizes.
- `deltanet_step_batched` (line 331) — **bulk of the work**. The 35B's
  current DN forward lives in `dn_forward_ttnn` (`server_35b_ttnn.py:372`).
  Need to add a `B` dim to every slice / matmul / norm. 27B's version
  is the exact template — slice ranges become `[B, …]` not `[…]`, ssm
  reshape becomes `[B, NV_PER_CHIP, K_DIM, V_DIM]`. Conv shift-accumulate
  stays the same.
- `gated_attn_step_batched` (line 519) — **bulk of the work**. 35B's
  GatedAttention is more complex than 27B's: partial RoPE (rotary_dim=64,
  head_dim=256), `attn_output_gate=True` doubles the Q projection (see
  `feedback_qwen36_attn_qgate_chunk_per_head.md`). The per-head chunk
  bug from B16 bringup is already handled in `attn_forward_ttnn_sdpa`
  (`server_35b_ttnn.py:743`). Lift that into `gated_attn_step_batched_35b`
  with a leading B.
- `forward_batch_tp_inner` (line 638) — generalizes; replace
  `base.mlp_step_tp` (27B dense MLP) with a **batched MoE forward**
  (see §3 above).

The two-phase warmup pattern (capture all paths first
without `enable_trace=True`, then capture all back-to-back) is
model-agnostic. It is called from `cb_scheduler.py:_warmup_decode` /
`_capture_decode_trace_only` / `_warmup_prefill` / `_capture_prefill_trace_only`.
The 35B trace capture (`server_35b_ttnn.py:bootstrap`, line 1690 onward)
already follows the same lifecycle. The scheduler should just work once
`state` exposes the right forward function.

**Chunked prefill**: 27B has it via `forward_prefill_chunked_tp`. 35B does
not. **Defer for v1** — single-shot prefill via the existing
`step_forward_inner` loop is fine while context budget is < MAX_POS.

---

## 6. Risks specific to MoE + CB

1. **Memory budget**. 35B has 35B params bf16 = 70 GB total weight bytes,
   plus the per-chip MoE weight slabs already loaded
   (`experts_gate_up_local: [64, 2048, 1024]` bf8_b per chip ≈ 64 MB +
   `experts_down_local` ≈ 64 MB per chip per layer × 40 layers ≈ 5 GB per
   chip). Adding batched activations: per-token MoE intermediate is
   `[B, E_LOCAL=64, MOE_INTER=512] @ bf16 = B * 64 * 512 * 2 = 64 KB · B`.
   Negligible. Routing tensors `[B, TOP_K=8]` × few = tiny. KV cache scales
   linearly with B as in 27B. **Headroom is fine through B=16 with the
   (1,4) mesh.**
2. **Two-phase warmup with MoE**. Pattern A MoE is data-independent at the
   tensor level — the same `ttnn.matmul(h_3d_repeat, experts_gate_up_local)`
   runs every step, and the top-k mask is just data. So trace capture works
   identically to 27B. (This is exactly why we use Pattern A and not
   per-expert dispatch for v1.)
3. **Owned-GDN kernel is B=1-only** (MEMORY.md note: hard-asserts
   `state_logical[0] == 1`). v0 must use the manual DN recurrence in CB.
   v1+: integrate the FOLD-B-into-slots trick from the 27B CB experiment
   (commit `a35fb3c`) — fold `[B, NV, K, V]` into `[1, B*NV, K, V]` and
   call the existing kernel. The 35B build of ttnn has all 8
   `qwen36_gdn_*` ops (verified 2026-05-28); we have the device kernels
   available, this is just integration.
4. **Routing top-k is host-readback in `moe_forward_ttnn` (line 1151) but
   on-device in `moe_forward_ttnn_pattern_a_batched` (line 1225)**. CB
   *must* use the Pattern A path because host readback is trace-incompatible.
   The 35B default already is `state.moe_mode = "pattern_a_batched"` so
   this is a no-op constraint — just don't accidentally take the topk
   reference path.
5. **DN state drift at long context** (MEMORY.md A010). The existing 35B
   forward has a known DN H_t precision issue (cos_final 0.93 at pos 1
   for L32 DN). Going B>1 doesn't worsen this, but it doesn't fix it
   either. v1 CB will inherit the same drift; defer to A010 work.
6. **vLLM prefix-cache hit rate is ~0.1% on this model class** (upstream
   issue #36493). For our v1 we are not running vLLM, we are running our
   own CB. Don't waste effort on prefix-cache invalidation gymnastics
   until v3.

---

## 7. Recommended approach

Five stages, each gated on the previous being correct:

### v0 — single-stream B=1 over the CB scheduler (≈1 day)

- Create `server_35b_cb.py` mirroring `server_tp_cb.py`. Imports
  `server_35b_ttnn` as base, redefines forward only.
- `setup_cb_state(state, B=1, ...)`: reuse 27B's body, swap constants
  (NV/K/V from 35B), allocate `cb_dn[li]` for the 30 GDN layers and
  `cb_kv[li]` for the 10 attention layers.
- `forward_batch_35b_inner`: copy `step_forward_inner` (line 1602) and add
  a single B axis. Loop over layers calling
  `deltanet_step_35b_batched` / `gated_attn_step_35b_batched` /
  `moe_step_35b_batched`. The MoE call is `moe_forward_ttnn_pattern_a_batched`
  with the 3D input reshaped from `[B, HIDDEN]` to `[B, 1, HIDDEN]` and the
  router emitting `[B, TOP_K]`.
- Validate: B=1 forward bit-identical to `step_forward_ttnn` (the existing
  B=1 path). Use the same 3a/3b/3c ladder as CB1 from `cb_validate_27b.py`.
- Effort: ≈ 1 day. Most code is a direct paste from `server_tp_cb.py` with
  shape constants swapped + MoE substituted for dense MLP.

### v1 — batched B>1 forward (≈3-5 days)

- Add B dim to every primitive. Per §3, the existing Pattern A MoE
  forward already handles arbitrary leading dim because `ttnn.matmul`
  broadcasts (line 1282: `h_3d_repeat` has leading `E_LOCAL`, expanding to
  `[B, E_LOCAL, 1, HIDDEN]` is a `repeat` on a fresh axis).
- DN: manual recurrence (owned-GDN blocked on B=1 assert). 27B CB's
  `deltanet_step_batched` (line 331) is the exact template.
- Validate: 3a/3b/3c ladder. Add 3d: B=8 distinct slots vs 8 separate B=1 refs.
- Effort: ≈ 3-5 days. Risk concentration: the partial-RoPE + per-head Q-gate
  split in `gated_attn_step_batched_35b` (35B-specific shape gotcha).

### v2 — trace capture at B=N (≈1-2 days)

- Two-phase warmup: pre-warm B=1 prefill + B=N decode paths
  *without* `enable_trace`, then capture them in one batch. This is the
  pattern from `feedback_two_phase_warmup.md`.
- Reuse `cb_scheduler.py`'s `_warmup_decode` / `_capture_decode_trace_only`;
  point them at `forward_batch_35b_inner`.
- Expect ≈ same trace speedup as 27B (≈ 5–10× over eager). The 35B
  current B=1 trace runs in `bootstrap` already; B=N trace is the same
  pattern with bigger input buffers.
- Effort: ≈ 1-2 days assuming v1 is shape-clean.

### v3 — owned-GDN batched (≈3-5 days)

- Integrate the FOLD-B-into-slots trick from the 27B CB experiment
  (commit `a35fb3c`): fold `[B, NV, K, V]` → `[1, B*NV, K, V]` and call
  `qwen36_gdn_decode_owned` with `debug_mode=10` (race-free output
  variant). Per MEMORY.md, the device kernel is already built into qb1 +
  qb2 ttnn; this is pure Python integration.
- Expected gain: matches the +2.5% on 27B (owned decay/gate shipped).
- Effort: ≈ 3-5 days. Risk: 35B DN has different state shape than 27B —
  verify the fold math separately before plumbing.

### v4 — prefix cache for attention layers only (≈2-3 days, optional)

- KV cache for the 10 GatedAttention layers can be prefix-cached safely
  (block manager identical to vLLM).
- DN layers cannot — the H_t state is autoregressive without a block
  notion. Just skip caching for DN.
- Expected hit rate is low (upstream Qwen3.5 sees <0.1%), so the
  effort-to-payoff ratio is dubious unless you have a use case with very
  high prefix-repetition (system prompt reuse).

### Stop-gate decisions

- If v0 + v1 hit > 60 ms/tok at B=8, v2 trace is mandatory before shipping.
- If v3 (owned-GDN batched) returns < +2% gain, skip it — the integration
  cost outweighs the perf.
- v4 prefix cache should only be attempted after the OpenAI endpoint is
  serving real traffic and you have a measured cache-hit upper bound.

---

## Quick-reference: file map

**Upstream to copy from**:
- DeepSeek-V3 MoE forward: `tt-metal/models/demos/deepseek_v3/tt/moe.py`,
  `experts.py`, `moe_gate.py`, `generator.py`, `generator_vllm.py`.
- Llama-70B-Galaxy CB / trace warmup:
  `tt-metal/models/demos/llama3_70b_galaxy/tt/generator.py` (esp. lines
  315–360 init, 543–680 prefill, 1010–1076 trace capture).

**Local to mirror / extend**:
- 27B CB stack: `experiments/serve/server_tp_cb.py` (template),
  `cb_engine.py`, `cb_scheduler.py`, `cb_metrics.py`, `cb_api.py`,
  `openai_endpoint.py`, `live_slot_store.py` (all reusable as-is).
- 35B model code: `experiments/serve/server_35b_ttnn.py` — `class State`
  (line 1342), `step_forward_inner` (line 1602), `dn_forward_ttnn`
  (line 372), `attn_forward_ttnn_sdpa` (line 743),
  `moe_forward_ttnn_pattern_a_batched` (line 1225 — already CB-friendly!),
  `bootstrap` (line 1690).

**Validation harnesses to clone**:
- 27B's `cb_validate_27b.py`, `cb_validate_ragged.py`, `cb_bench_trace.py`,
  `cb_scheduler.py` (all in `experiments/cb/` per commit history).

---

## Sources

- [tenstorrent/tt-metal — models/demos](https://github.com/tenstorrent/tt-metal/tree/main/models/demos)
- [tenstorrent/tt-metal MODEL_UPDATES.md](https://github.com/tenstorrent/tt-metal/blob/main/models/docs/MODEL_UPDATES.md)
- [DeepSeek-V3 tt directory](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/deepseek_v3/tt)
- [DeepSeek-V3 moe.py](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tt/moe.py)
- [DeepSeek-V3 experts.py](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tt/experts.py)
- [DeepSeek-V3 generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tt/generator_vllm.py)
- [Llama-70B-Galaxy tt directory](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/llama3_70b_galaxy/tt)
- [tt-metal Issue #10581 — Continuous batching in Llama](https://github.com/tenstorrent/tt-metal/issues/10581)
- [Qwen/Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vLLM Issue #36493 — Qwen3.5 35BA3B prefix-cache hit rate](https://github.com/vllm-project/vllm/issues/36493)
- [vLLM Issue #38182 — Qwen3.5-35B-A3B MTP + prefix cache regression](https://github.com/vllm-project/vllm/issues/38182)
