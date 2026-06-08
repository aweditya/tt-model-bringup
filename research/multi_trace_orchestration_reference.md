# Multi-Trace Orchestration — Fork-Ready Reference

Status: research distilled from a 2026-06-07 Explore pass of tt-metal
demos on qb1. Fork-ready snippets for **Phase 2.B (B=K+1 verify trace)**
and **Phase 3 (spec_dec_scheduler accept walk)** of the Gemma 4 spec-dec
build (`research/gemma4_mtp_plan_of_action.md`).

**Sources** (all paths under `~/tenstorrent/tt-metal/` on qb1):
- `models/tt_transformers/tt/generator.py` — shared Generator orchestration (gpt_oss + DeepSeek-V3)
- `models/demos/deepseek_v3/tt/generator.py` — MTP multi-trace + accept walk
- `models/demos/deepseek_v3/tt/mtp.py` — MTP2D module (reference structure)
- `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py` — on-device layout-convert principle

---

## 1. Two-phase warmup + capture (the canonical pattern)

`tt_transformers/tt/generator.py:1214-1274`

### Phase 1: eager compile (fills all lazy-init buffers, JITs kernels)

```python
# Line 1227-1233 — one eager forward BEFORE any begin_trace_capture
self._decode_forward_no_trace_text(
    tokens, current_pos, page_table=page_table,
    kv_cache=kv_cache, sampling_on_device=sampling_on_device,
)
logger.info("Done Compiling Model")
```

### Phase 2: device-input allocation + capture

```python
# Line 1243-1248 — prepare host inputs, copy to PRE-ALLOCATED device buffers
host_inputs = self.model[i].prepare_decode_inputs_host(
    tokens[i], current_pos[i], page_table=user_page_table)
device_inputs_i = copy_host_to_device(
    host_inputs, mesh_device=self.model_args[i].mesh_device)
device_inputs.append(device_inputs_i)

# Line 1257-1268 — capture trace
trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
tt_out_trace.append(
    self.model[i].ttnn_decode_forward(
        *device_inputs[i], kv_cache=user_kv_cache,
        sampling_on_device=sampling_on_device,
        capture_sampling_trace=split_enabled,
    ))
ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
```

**Fork for Phase 2.B**: shape-agnostic — `prepare_decode_inputs_host` +
`copy_host_to_device` just work at B=K+1. Build one capture per shape.

---

## 2. Per-step replay (DMA-only update, no realloc)

`tt_transformers/tt/generator.py:1276-1332`

```python
# Line 1305-1313 — rebuild host inputs per step, DMA write into captured device buffers
if reset_inputs:
    for i in range(self.data_parallel):
        host_inputs_i = self.model[i].prepare_decode_inputs_host(
            tokens[i], current_pos[i], page_table[i])
        copy_host_to_device(
            host_tensors=host_inputs_i,
            device_tensors=self.trace_inputs_decode[sampling_on_device][i],
        )

# Line 1314-1315 — execute the captured trace (non-blocking ok)
for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
    ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0,
                       blocking=False)
outputs = self.trace_output_decode[sampling_on_device]
```

**Invariants**:
- `prepare_decode_inputs_host` is idempotent — call it every step
- Device buffers never realloc; just DMA writes into the same handles
- Output handles are the same captured ttnn.Tensors — no reallocation

This is exactly what our v0.4 drafter trace + Phase 2.B verify trace
will do — one `execute_trace` per round.

---

## 3. Aliased page-table for B=K+1 verify

`deepseek_v3/tt/generator.py:58-99`

```python
def _build_verify_alias_page_table_host(
    base_page_table: torch.Tensor,
    num_prompts: int,
    verify_offset: int,
    prompt_indices: List[int] | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    alias_page_table = base_page_table.clone().to(torch.int32)
    num_rows = int(alias_page_table.shape[0])

    prompt_indices_for_alias = prompt_indices
    if not interleaved and prompt_indices_for_alias is None:
        prompt_indices_for_alias = list(range(num_prompts))

    if interleaved:
        # Multi-prompt MTP: rows 1,3,5... alias to rows 0,2,4...
        if prompt_indices_for_alias is None:
            for row in range(1, num_rows, 2):
                alias_page_table[row] = alias_page_table[row - 1]
        else:
            for i in prompt_indices_for_alias:
                src_row = (2 * i) % num_rows
                dst_row = (src_row + 1) % num_rows
                alias_page_table[dst_row] = alias_page_table[src_row]
    else:
        # Sequential: rows [verify_offset, verify_offset+K) alias to rows [0, K)
        for i in prompt_indices_for_alias:
            src_row = i % num_rows
            dst_row = (verify_offset + i) % num_rows
            alias_page_table[dst_row] = alias_page_table[src_row]

    return alias_page_table
```

**Our usage** (Gemma 4 single-stream, K=5):
- `num_prompts=1`, `verify_offset=1`, `prompt_indices=[0]` (or adapt to alias K+1 rows all to row 0)
- Non-interleaved layout
- See `research/deepseek_v3_alias_page_table_reference.md` for the adaptation we already documented

---

## 4. Multi-trace state in the Generator

`deepseek_v3/tt/generator.py:238-298`

```python
self.enable_mtp = bool(enable_mtp)          # constructor flag
self._trace_id: int | None = None            # main decode trace (B=1)
self._mtp_verify_trace_id: int | None = None # verify trace (B=K+1)
self._mtp_predict_trace_id: int | None = None # drafter predict trace
```

**Our mapping** for `spec_dec_scheduler`:
- `_trace_id` → target B=1 decode (target server already captures this)
- `_drafter_trace_id` → drafter B=1 forward (Phase 1 v0.4)
- `_verify_trace_id` → target B=K+1 verify (Phase 2.B)

Per-trace device input buffers stored alongside:
```python
self.trace_inputs_decode  = device_inputs_b1
self.trace_inputs_verify  = device_inputs_kp1
self.trace_inputs_drafter = device_inputs_drafter
```

---

## 5. Hard constraint: host sampling for spec-dec

`deepseek_v3/tt/generator.py:444-445`

```python
if enable_mtp and sample_on_device:
    raise SystemExit(
        "MTP with sampling on device is not supported. "
        "Disable MTP or sample on host."
    )
```

**Why this is the right constraint for us**:
- Device bf16 topk has SFPSWAP drift (`[[ttnn-topk-tie-break-drift]]`)
- Leviathan correctness requires deterministic argmax tie-break — must match numpy
- Accept walk takes ~µs on host; not a perf concern
- Our `spec_dec_scheduler._argmax_with_tiebreak` already uses host numpy

---

## 6. Accept-walk core

`deepseek_v3/tt/generator.py:1612-1663`

```python
# Line 1612 — single sample call across all K+1 logical rows
pred_all = self._sample_greedy(logits_2b)
pred_next       = pred_all[:num_of_prompts]                          # B=1 target step
pred_after_spec = pred_all[verify_offset : verify_offset + num_of_prompts]  # K+1 verify

# Line 1619-1639 — per-token compare-and-advance
for i in range(num_of_prompts):
    next_value = int(pred_next[i].item())
    accepted = next_value == int(spec_tokens[i].item())
    next_tokens[prompt_uid] = next_value
    positions[prompt_uid]   = positions[prompt_uid] + 1
    generated_counts[i]    += 1
    if accepted:
        total_accepts += 1
    generations[i].append(next_value)
    if next_value in stop_token_ids:
        finished[i] = True

# Line 1652-1663 — bonus token if ALL K accepted
if accepted and skip_accept_decode:
    next_after_spec_value = int(pred_after_spec[i].item())
    next_tokens[prompt_uid] = next_after_spec_value
    positions[prompt_uid]   = positions[prompt_uid] + 1
    generated_counts[i]    += 1
    generations[i].append(next_after_spec_value)
```

**Our `spec_dec_scheduler._accept_walk` (already implemented, sketch matches)**:

```python
def _accept_walk(self, draft_tokens, target_logits_kp1) -> tuple:
    accepted = []
    for i in range(self.K):
        target_tok = self._argmax_with_tiebreak(target_logits_kp1[i])
        if target_tok == draft_tokens[i]:
            accepted.append(draft_tokens[i])
        else:
            accepted.append(target_tok)            # correction
            return accepted, i                      # i drafts accepted
    # All K accepted: emit the K+1-th bonus token
    bonus = self._argmax_with_tiebreak(target_logits_kp1[self.K])
    accepted.append(bonus)
    return accepted, self.K
```

Single-stream is simpler than DeepSeek's multi-prompt CB-style loop —
unroll to per-candidate comparison, no per-user state tracking.

---

## 7. On-device layout conversion (informational; MoE-only relevance)

`gpt_oss/tt/experts_throughput/fused_decode.py:80-90` + `deepseek_v3/tt/moe.py:393-395`

```python
# The principle: NEVER to_torch → host reshape → from_torch in the hot path.
topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_indices_rm = ttnn.reshape(
    topk_experts_indices_rm,
    shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"]),
)
```

Gemma 4 spec-dec has no MoE so this is reference-only. Captured because
if we ever pivot to a MoE drafter or trace MoE on Nemotron-3 (v0.6),
this is the pattern: `to_layout + reshape`, never numpy roundtrip.

---

## 8. MTP2D as reference structure (not a fork)

`deepseek_v3/tt/mtp.py:46-260`

**Constructor fields** (lines 199-240): MTP2D parametrizes by `embedding`
(reused from target), `hidden_norm` + `token_norm`, `eh_proj`
(concatenates target hidden + token embedding), `decoder_block` (MoE),
`head_norm` + `head` (lm_head).

**Our drafter is simpler**:
- No `eh_proj` (we have `pre_projection` which takes concat of 2 target
  hidden states; same idea, different shape contract)
- No MoE in the trunk — vanilla 4-layer transformer
- `lm_head` ties to embed_tokens (per Gemma 4 unified-assistant config)

**Reuse**: weight-loading via `_strip_model_prefix` (generator.py:102-114).
We already do equivalent stripping via `model.` prefix removal in
`server_gemma4_12b_assistant_ttnn.py`.

---

## 9. Trace region size budget

Per `gemma4_mtp_plan_of_action.md` §"Three traces total":
- Default: 50 MB (sufficient for one B=1 trace)
- Phase 2.B requirement: bump to **150 MB** (decode-B=1 + verify-B=K+1)
- Drafter trace (~10-20 MB) fits in the same budget

```python
os.environ["TT_METAL_TRACE_REGION_SIZE"] = str(150 * 1024 * 1024)
# Or pass trace_region_size= to ttnn.open_mesh_device
```

The target server already sets this; just bump the value before
capturing the second + third trace.

---

## 10. What spec_dec_scheduler still needs

Mapped against the 3 `NotImplementedError` seams in
`experiments/serve/spec_dec_scheduler.py`:

| Seam | Maps to | Source pattern |
|---|---|---|
| `_target_step` (line ~104) | Phase 2.A | Modify `attn_decode_step_tt` to expose `state.shared_kv_for_drafter` |
| `_drafter_parallel_forward` (~110) | Phase 1 done | Already shipped at v0.2; this seam just calls `srv.drafter_forward` |
| `_target_verify_kp1` (~127) | Phase 2.B | Fork `_build_verify_alias_page_table_host` + capture B=K+1 trace per §1-§4 |
| `_accept_walk` (~150) | Phase 3 | Already implemented — matches §6 pattern verbatim |

---

## Known gotchas (from the upstream patterns)

1. **Page-table aliasing edge cases**: if `verify_offset >= num_rows`,
   the modulo wraps. Always check `num_rows > verify_offset` before
   capture.
2. **bf16 chain drift suppresses α**: ship `[[gemma4-determinism-audit]]`
   patches A+B+D before measuring α. We already have B+D shipped on
   Gemma 4 12B; A is for the sampling path (not greedy spec-dec).
3. **Prefix-cache + spec-dec interaction**: disable prefix-caching in v0
   spec-dec. Layer back on after the basic accept walk is stable.
4. **Drafter latency variance**: measure drafter forward in isolation
   before capturing trace — variable latency can make trace replay slow.

---

## Phase 2.B → Phase 3 implementation order (fork-driven)

1. Phase 2.A.0: KV layout probe (currently in flight on qb1)
2. Phase 2.A: expose `state.shared_kv_for_drafter`
3. Phase 2.B step 1: fork `_build_verify_alias_page_table_host` into
   `spec_dec_scheduler.py` as a static helper (no device work — pure host)
4. Phase 2.B step 2: add B=K+1 verify capture to target server (fork §1's
   two-phase warmup pattern verbatim)
5. Phase 3 wire-up: `spec_dec_scheduler.step()` chains target_step →
   drafter_forward → verify → accept_walk. Each `execute_trace` is one
   line; the orchestration mirrors §1's structure.

The Phase 3 implementation should be lighter than the Phase 1 + 2 work
because we have the accept-walk in skeleton form already; the new code
is just the trace plumbing per §1-§4.
