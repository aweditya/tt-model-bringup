# Tenstorrent Qwen3.5-9B branch read (2026-06-08)

**Branch URL**: https://github.com/tenstorrent/tt-metal/tree/1cecd16c43cb73c218310e053fb37eaa1a380033/models/demos/blackhole/qwen3_5_9b
**Pinned commit**: `1cecd16c43cb73c218310e053fb37eaa1a380033` (later than the `14be5b9` HEAD of the 2026-06-04 audit `research/audit_qwen36_us_vs_qwen9b_p150_branch.md`; this read targets adoption rather than full architectural diff).
**Source path cached**: `.cache/qwen35_branch_read/` (read-only mirror; not on path).

The branch was confirmed-existing by Yossi at Tenstorrent (email 2026-06-08). The prior fan-out task (#245, commit 486436e) failed to find it; this read replaces that gap.

---

## 1 — Branch overview

Directory `models/demos/blackhole/qwen3_5_9b/` (note: `qwen3_5_9b` collapses 3.5/3.6 in naming).

```
tt/
  model.py            (89 KB, ~1800 LoC)  — Qwen35Model orchestrator: prefill/decode/trace
  layer.py            — Qwen35DecoderLayer with hybrid (DN / full-attn) dispatch
  model_config.py     — Qwen35ModelArgs (subclass of tt_transformers ModelArgs)
  mlp.py              — DENSE SwiGLU (no MoE in 9B)
  tp_common.py        — DRAM-sharded weight/matmul helpers; ports from qwen35_27b
  weight_mapping.py   — HF state-dict remapper + FP8-checkpoint dequant
  rms_norm.py         — Tiny shim around ttnn.rms_norm with +1 zero-centered weight
  rope.py             — Single-device RoPE setup
  qwen35_vllm.py      — vLLM Generator subclass; ties capture_prefill_trace_chunked + decode trace
  generator_interface.py — prefill_dispatch + decode-trace priming helpers
  common.py           — create_tt_model factory
  attention/
    __init__.py, config.py — single-device Qwen35GatedAttention (paged + concat caches)
    decode.py, prefill.py  — branch dispatch (paged decode / paged prefill / concat prefill)
    tp.py             — TPAttention (TP>1) with chunked-SDPA, paged_update_cache, GQA
    rope_tp.py        — partial-rope (HF split-halves) for TP path
    weights.py        — attention weight loader
  gdn/
    __init__.py       — Qwen35GatedDeltaNet wrapper around experimental ops
    decode.py         — calls gated_deltanet_forward_ttnn (composed, NOT the owned-kernel route)
    tp.py             — TPGatedDeltaNet: per-value-head TP recurrence, no cross-device comms
    weights.py        — GDN weight loader + precomputed conv taps
    state.py          — recurrent/conv state init + split helpers
    config.py         — GDNConfig dataclass
    _experimental_path.py — sys.path shim to models/experimental/gated_attention_gated_deltanet
demo/text_demo.py     — parametrized e2e test (128/2k/4k/8k/16k/32k); references Frankenstein corpus
tests/                — 14 tests including factory, masked-bucket, trace-chunked, weight loading
utils/substate.py     — state-dict prefix-strip helper
```

**Critical absence**: no `moe/`, `experts.py`, `router`, `all_to_all` anywhere in branch. `grep -ri "moe\|expert\|router"` returns zero hits. **The 9B is a dense SwiGLU MLP** — it cannot inform any MoE work.

**Entry points** (in order of how a request flows):
- `Qwen35ForCausalLM` (qwen35_vllm.py) — vLLM Generator subclass.
- `Qwen35Model.from_pretrained` → `Qwen35Model.__init__` (model.py:31).
- Prefill: `prefill_dispatch` → `prefill_traced_chunked` → `_forward_prefill_chunk(_tp)` (model.py:439/479).
- Decode: `Generator.decode_forward` (inherited) → `ttnn_decode_forward` → `_forward_decode` (model.py:421).
- Trace capture: `capture_prefill_trace_chunked` / `WarmupForwardMixin` for decode (qwen35_vllm.py:142).

---

## 2 — Architecture comparison vs our 27B/35B

| Item | 9B (pinned) | Our 27B (`server_tp.py`) | Our 35B (`server_35b_ttnn.py`) |
|---|---|---|---|
| Layer count | 32 | 64 | 64 |
| Layer mix | 24 GDN + 8 full-attn (interleaved per `layer_types` HF config) | Similar DN+attn hybrid (per their config) | Similar |
| Hidden / heads | dim from HF cfg; 16 attn heads, 4 KV heads, head_dim=256 | 4096 / 32 / 4 (GQA8) | larger |
| GDN heads | 16 K-heads, 32 V-heads, head_k=head_v=128, conv_k=4 | matches | matches |
| RMSNorm | Zero-centered (output * (1 + w)); `add_unit_offset=True` | Zero-centered (we do this too) | Zero-centered |
| RoPE | partial (rope_head_dim = head_dim * partial_rotary_factor); HF split-halves | partial | partial |
| **MLP** | **Dense SwiGLU** (mlp.py: w1/w2/w3, bfloat4_b on gate/up) | Has MoE | MoE (Pattern A sharded, 256 experts, 8-top) |
| MoE | **none** | **yes** | **yes** |
| Sampling | **host only** (`_supports_on_device_sampling=False`) | host topk fallback (per `[[feedback-ttnn-topk-tie-break-drift]]`) | topk fallback |
| Prefill bucket | 5-bucket fixed mask: `{128, 256, 512, 1024, 2048}` + chunk-outer trace per 2048 chunk | Single chunk size, partial chunking | block-level only |
| Decode trace | Standard `WarmupForwardMixin` at pos 0; DN state re-zeroed each sequence | Manual capture | Manual capture |
| Mesh shapes | (1,1), (1,2), (1,4), (1,8) supported via `_init_tp_config` | (1,4) only | (1,4) only |

**Key architectural differences worth flagging**:
- The 9B uses an **interleaved DN+attn pattern** read straight from HF `text_config.layer_types`. Layer 0 is DN, layer 3 (or similar cadence) is full-attn. Our 27B/35B set their own layer-type list; the 9B reads it directly.
- The 9B keeps the **DN recurrence per-value-head** with no cross-device comms in the recurrence kernel itself — only an all-reduce after the row-parallel `out_proj`. This is the **same TP topology we use**, validated.
- DN K-head→V-head expansion uses `ttnn.repeat_interleave(rf, dim=...)` where `rf = Nv // Nk = 2`. We use the same approach.

---

## 3 — Patterns worth harvesting

### 3.1 Chunk-outer prefill trace (HIGH-VALUE)

This is the single biggest pattern they have that we don't.

**What they do** (`capture_prefill_trace_chunked` model.py:516, `_forward_prefill_chunk_tp` model.py:479):
- Capture ONE chunk's all-layer forward as a single trace (~2048 tokens worth of dispatches).
- Replay the trace `num_full = actual_len // chunk_size` times per request, advancing `chunk_start_idx_tensor` (a DEVICE tensor) between replays via `copy_host_to_device_tensor`.
- DN recurrent + conv state and paged KV cache update **in-place** across replays — addresses baked into the trace stay valid.
- Tail (< chunk_size) runs through the masked-bucket path (see 3.2).

**Why it matters for us**:
- Today our long-prompt (>chunk_size) TTFT falls back to 1-token-per-iter (per audit_qwen36_us_vs_qwen9b_p150_branch.md §3). That's our biggest open TTFT cap for cold chat.
- A 2048-token chunk trace keeps the captured trace under the tt-metal 4 GiB uint32 ceiling at 128K context, while every GDN call still runs at the validated 16-sub-chunk (2048-token) size.
- The trick that makes this work is the **`chunk_start_idx_tensor` device tensor** in flexible chunked SDPA: SDPA program is position-general (`q/k_chunk=64`, see `attention/tp.py:389-396`) so one captured trace serves every chunk position.

**Adoption notes**:
- DN must `_chunk_inplace_state=True` so the chunk-prefill writes via `ttnn.copy` into persistent state buffers (gdn/decode.py:103-115). We have the analog discipline in `cb_engine` slot resets — needs porting per chunk.
- Persistent input buffers (`_chunk_token_buf`, `_chunk_start_idx_tensor`, `_chunk_full_page_table_buf`, `_chunk_page_table_buf`, `_chunk_cos_buf`, `_chunk_sin_buf`) are uploaded once with `mesh_mapper=ReplicateTensorToMesh` and rewritten via `copy_host_to_device_tensor` per replay (model.py:1108-1142).
- Two-phase warmup is mandatory: compile per-chunk programs first, then compile every masked-bucket program (3.2), THEN `begin_trace_capture`. Their note at model.py:613 calls out this requirement directly (matches our `[[two-phase-warmup]]`).

### 3.2 Masked fixed-bucket short-prompt prefill (HIGH-VALUE)

**What they do** (`_PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)`, model.py:787; `prefill_masked_bucket` model.py:930):
- Round every prompt length up to a bucket (5 entries total).
- Pad to the bucket, run all layers once, then **mask the GDN to the EXACT valid_len** so the recurrent + conv state reflect exactly the real prompt.
- For long tails (after the chunk-outer trace), the tail also flows through this same masked-bucket path with `chunk_start>0`.
- Selecting the last real position uses a **one-hot row matmul** (model.py:974-977) instead of a static slice — because slicing at `actual_len-1` would compile a per-length program. The matmul's program is fixed per bucket.

**Why it matters for us**:
- Bounded program set: only 5 buckets ever compile, all warmed up before any trace parks. This is THE fix for the "compile-clobbers-trace" hang their commentary repeatedly cites.
- The masked-bucket path is **shared between short prompts and the long-prompt tail** — one code path served two cases.

**Adoption notes**:
- Buckets are multiples of 128 (the GDN sub-chunk). Match that.
- They run the bucket compile sweep (`warmup_prefill_masked_buckets` model.py:1008) at `width=1..max_width` so paged_fill_cache compiles for every fill width too.

### 3.3 Flexible chunked SDPA via `chunk_start_idx_tensor` (MEDIUM-VALUE)

**What they do** (`attention/tp.py:418-427`):
```python
attn = ttnn.transformer.chunked_scaled_dot_product_attention(
    input_tensor_q=q8, input_tensor_k=k_paged, input_tensor_v=v_paged,
    page_table_tensor=sdpa_page_table,
    chunk_start_idx_tensor=chunk_start_idx_tensor,  # DEVICE tensor
    ...
    program_config=sdpa_cfg,  # FIXED q/k_chunk=64
)
```

`chunk_start_idx` as a runtime device tensor lets ONE compiled SDPA program serve every chunk position. The host-int variant compiles per start-position (model.py:835 commentary).

**Why it matters for us**:
- This is the trick that makes 3.1 trace-clean. Without it, the captured chunk trace would only be valid for `chunk_start=0`.
- They pad the page table to a multiple-of-32 with zero blocks because the kernel requires it (model.py:404-416). Zero-blocked padding maps to physical block 0 but at K positions beyond the prompt — causality masks them out. We should mirror this trick.

### 3.4 Decode-trace primer (LOW–MEDIUM-VALUE — dormant fallback)

**What they do** (`prime_decode_trace` in generator_interface.py:49-75):
- Standard path: capture decode trace at pos 0 during warmup. Safe because every new sequence re-zeros DN state via `_reset_gdn_state_for_new_sequence` before consuming a token.
- Dormant fallback under `QWEN35_DECODE_PRIME=1`: lazily capture trace on FIRST decode at real post-prefill position, with DN state **snapshot + restore** around the two-pass capture (the stock capture runs forward twice, advancing DN state non-idempotently).

`_save_deltanet_states` (model.py:1743) / `_restore_deltanet_states` (model.py:1764) snapshot via `ttnn.to_torch` and restore via `ttnn.from_torch` + `ttnn.copy` into the original buffer (preserves addresses).

**Why it matters for us**:
- Our `cb_engine` does its own decode-trace capture; if it ever exhibits non-idempotent DN state advance at capture time, this snapshot/restore pattern is the answer. Currently it isn't (we capture at pos 0 too), but document the dormant-flag pattern as the escape hatch.

### 3.5 Composed DN op vs our owned kernel (REFERENCE — do not adopt as-is)

**What they do**: decode goes through `recurrent_gated_delta_rule_decode_ttnn` and `chunk_gated_delta_rule_seq_adapter` from `models/experimental/gated_attention_gated_deltanet`. The recurrence kernel L2-norms + scales internally; we drive `q, k, v, beta, g` and consume `o, new_rec`.

**What we do**: `ttnn.experimental.qwen36_gdn_decode_owned(H, q, k, v, decay, beta, native_io=True, output_memory_config=L1)` — our owned kernel does the whole step (see `server_tp.py:783`).

**Why NOT to adopt their path**:
- We already pay for the owned kernel and it ships as the production default (`State.deltanet_recurrence_mode = "owned_gdn"` per `[[reference-gdn-vs-mamba2-kernel-delta]]`).
- Their composed op is a different implementation with its own L2-norm + scale baked in. Switching back would invalidate the entire MM7 G1/G2 ladder.

**What IS worth measuring**: kernel-time comparison between `qwen36_gdn_decode_owned` and `recurrent_gated_delta_rule_decode_ttnn` on identical shapes. The audit doc (research/audit_qwen36_us_vs_qwen9b_p150_branch.md §5) already flags this as a high-value cross-team experiment.

### 3.6 Tail conv-state capture from chunked prefill (LOW-MEDIUM-VALUE)

**What they do** (gdn/tp.py:299-317): at the end of a chunk that ends a sequence (or each chunk when state-stable), write `conv_states[1..K-1] = last K-1 real conv inputs from this chunk`, `conv_states[0] = zero`. The conv shift register is now decode-ready.

**Why it matters for us**: this is one of the subtle correctness traps when wiring chunk-outer prefill into the existing DN decode. The owned kernel takes the conv state as inputs, but the FIR-conv prefill returns a different layout — match this exact slice-and-copy discipline.

---

## 4 — Patterns we already have that match

- **Zero-centered RMSNorm with +1 weight pre-offset** (rms_norm.py:8) — we do this (per `[[feedback-gemma4-v_norm]]` reasoning extended to all Qwen norms).
- **Per-value-head DN TP, all_reduce only after `out_proj`** (gdn/tp.py:3-13) — matches `deltanet_step_tp` in our `server_tp.py:635`.
- **DRAM-sharded matmul program configs** (tp_common.py:80-118) — our `server_tp.py` builds equivalents.
- **HF split-halves RoPE format with partial rotary** (attention/rope_tp.py) — we do this for Qwen and Gemma 4.
- **Two-phase warmup discipline** (model.py:613, 700, 711) — already in our `[[feedback-two-phase-warmup]]`.
- **DN recurrence in-place state writeback** (`_stable_state` in gdn/tp.py:389; `owned_gdn_inplace` in our server_tp.py:821).
- **One-hot select for varying-length output extraction** (model.py:974) — we use one-hot for masked logit-collection too in some 27B prefill paths.

---

## 5 — Recommended adoptions (priority order)

1. **Chunk-outer traced prefill** (3.1) + masked fixed-bucket fallback (3.2). This is the single biggest TTFT gap vs our current state. Effort: medium-high (one full eng cycle); requires DN state in-place discipline + persistent input buffers + 5-bucket warmup. Already prior-art-mapped in `research/27b_prefill_trace_plan.md`.
2. **Flexible chunked SDPA `chunk_start_idx_tensor`** (3.3) — required to make #1 work; one-line API switch. Pad page table to %32 with zero-blocks (model.py:404).
3. **DN snapshot/restore for decode-trace capture** (3.4) — keep as a dormant safety net under a flag (`TT_DECODE_PRIME=1`). Cheap to add; never needed in steady-state.
4. **Tail conv-state capture pattern** (3.6) — wire it in when we adopt #1.
5. **Skip**: their composed-op DN kernel (3.5). Our `qwen36_gdn_decode_owned` is the production default and validated.

---

## 6 — Open questions

1. **What's the chunk_size sweet spot?** They hardcode 2048 (`_PREFILL_WARMUP_CHUNK`). Their commentary says it keeps the trace under 4 GiB uint32 and matches the validated GDN 16-sub-chunk PCC range. Need a per-Qwen-size calibration if we adopt.
2. **Can we share the chunk trace across model sizes?** Probably not — different layer counts and head dims compile different programs. Worth probing.
3. **Do their owned vs composed-op kernel-time numbers match ours?** Open cross-team experiment per audit §5 / `[[feedback-gdn-vs-mamba2-kernel-delta]]`.
4. **What about MoE chunk-outer trace?** Their branch has nothing on MoE. Our 35B MoE trace question (`[[decode-trace-canonical-pattern]]` MoE trap: 4+ host bridges) is unsolved upstream too. We're on our own there.
5. **Is the masked-bucket valid_len discipline necessary at long context?** They take it as given; we'd want to verify the GDN mask path doesn't drift at multi-chunk concatenation. The 5-bucket fall-back means short prompts never go through the trace, so the trace itself stays narrow.

---

## 7 — Related notes / pointers

- Prior audit `research/audit_qwen36_us_vs_qwen9b_p150_branch.md` (2026-06-04, branch HEAD `14be5b9`).
- `research/27b_prefill_trace_plan.md` — our chunked prefill plan (predates this read).
- `research/moe_trace_precedents.md` — the prior fan-out that missed this branch.
- `[[reference-decode-trace-canonical-pattern]]` — our trace audit; chunk-outer prefill is the missing piece for non-MoE Qwen.
- `[[feedback-two-phase-warmup]]` — required for any multi-trace capture.
- `[[reference-gdn-vs-mamba2-kernel-delta]]` — explains why their composed op and our owned kernel coexist.
- Source mirror: `.cache/qwen35_branch_read/` (this read; gitignored).
