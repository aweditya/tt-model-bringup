# MoE + ttnn decode trace — upstream precedents in tt-metal

**Date**: 2026-06-06
**Status**: Survey complete; precedents found. Plan section TBD by team.
**Scope**: Establish whether `ttnn.begin_trace_capture` / `ttnn.execute_trace` has
been successfully used by ANY MoE model in `tenstorrent/tt-metal`, and if so,
under what conditions and via what pattern.

This doc is the prerequisite for v0.6 (MoE decode-trace) on Qwen3.6-35B-A3B.
The 4-bridge MoE trap was first articulated in
`memory/reference_decode_trace_canonical_pattern.md` §"The MoE trap".

## TL;DR

- **Precedent EXISTS for MoE decode trace.** Two shipping MoE demos in tt-metal
  decode-trace the full MoE block end-to-end:
  - **`models/demos/gpt_oss/`** — GPT-OSS 20B / 120B, sparse MoE.
    `enable_decode_trace=True` is the demo default at every parametrization site.
    Two distinct expert paths (standard sparse-matmul + Galaxy
    all_to_all_dispatch), both on-device throughout.
  - **`models/demos/deepseek_v3/`** — DeepSeek-V3 671B-A37B, sparse MoE +
    Multi-Token Prediction (MTP). `enable_trace` is wired through to three
    trace-capture sites (main decode, MTP verify, MTP predict).
- **Both demos use the same shared `Generator` infrastructure for non-MTP
  paths**: `models/tt_transformers/tt/generator.py` →
  `_capture_decode_trace_text` (line 1147) / `_decode_forward_trace_text`
  (line 1200). The model just provides
  `ttnn_decode_forward(*device_inputs, ...)` and `prepare_decode_inputs_host`.
- **The 4-bridge MoE trap IS soluble.** Concrete techniques observed:
  1. **Sparsity tensor instead of dispatch** (gpt_oss standard path,
     `tt/experts/decode.py:54-72`): emit a dense `[B, E]` indicator tensor on
     device via `ttnn.scatter`, feed to `ttnn.sparse_matmul(sparsity=, nnz=K)`.
     Eliminates the dispatch altogether for the loudbox shape regime.
  2. **On-device topk indices + layout conversion** (DeepSeek-V3 `tt/moe.py:413-418`,
     gpt_oss `tt/experts_throughput/fused_decode.py:80-114`):
     `ttnn.to_layout(topk_indices, ROW_MAJOR) + ttnn.reshape` instead of
     `ttnn.to_torch → torch reshape → ttnn.from_torch`.
  3. **Routing-weights mul stays on device** (gpt_oss
     `tt/experts/decode.py:120-126`, fused path lines 175-205): permute scores
     `[M, K] → [K, 1, M, 1]`, broadcast multiply against combine output
     `[K, 1, M, H]`, sum over K. The combine-weights re-upload is replaced by an
     on-device permute + broadcast.
  4. **Fused MoE kernels with metadata-only dispatch**
     (`ttnn.experimental.all_to_all_dispatch_metadata + moe_gpt +
     selective_reduce_combine`) compress the dispatch+compute+combine into 3
     ops, all keeping tensors on device.
- **DeepSeek-V3's default `topk_fallback=True` is a host-bridge path**
  (`tt/moe_gate.py:445-490`, `to_torch` + `torch.topk` + `from_torch` — same
  pattern we use, per `feedback_ttnn_topk_tie_break_drift.md`). The
  `topk_fallback=False` path uses `ttnn.topk` and is the trace-compatible one,
  but the model config gates it behind a flag, suggesting precision/quality
  trade-offs that are still being worked through upstream.
- **Negative results**:
  - `deepseek_v3_b1` (single-batch variant) and `deepseek_v3_d_p` (dispatcher
    refactor) have **no `begin_trace_capture` in their trees**, despite being
    shipping MoE codebases. They appear to be earlier-stage forks.
  - `qwen3_vl` is **not MoE**; no trace sites in the tree.
  - The "throughput experts" Galaxy path in gpt_oss does the full
    `all_to_all_dispatch + all_to_all_combine` ring while still trace-capturable,
    so the host bridges we documented are **avoidable**, not fundamental.

**Implication**: Our v0.6 MoE trace effort is **not pioneering**. The pattern
is upstream-blessed. We should fork the appropriate expert-path (loudbox →
sparse-matmul, multi-chip → fused or all_to_all) and the
`tt_transformers.Generator._capture_decode_trace_text` orchestration shape.

## Demos surveyed

`find ~/tenstorrent/tt-metal/models/demos -type d \( -iname "*moe*" -o -iname
"*deepseek*" -o -iname "*mixtral*" -o -iname "*qwen*moe*" -o -iname "*gpt*oss*"
\)` results (qb1, 2026-06-06):

| Demo                       | MoE? | `begin_trace_capture` present | Decode-trace shipped? |
|----------------------------|------|-------------------------------|------------------------|
| `gpt_oss/`                 | Yes  | Yes (`demo/text_demo.py:814` prefill + via shared `Generator`) | **YES** — `enable_decode_trace=True` default at all 12 parametrizations (`text_demo.py:194-440`) |
| `deepseek_v3/`             | Yes  | Yes (`tt/generator.py:2460,2568,2704`) | **YES** — 3 trace IDs: decode, MTP verify, MTP predict |
| `deepseek_v3_b1/`          | Yes  | No                             | No                     |
| `deepseek_v3_d_p/`         | Yes  | No                             | No                     |
| `qwen3_vl/`                | No   | No                             | n/a                    |
| `wormhole/qwen3_embedding_8b/` | No (encoder)  | n/a                       | n/a                    |

No `mixtral` directory exists in the current tree.

## Demo 1: gpt_oss — sparse MoE, loudbox + Galaxy, decode-trace default

**Trace capture is in shared `models/tt_transformers/tt/generator.py`.**

### How the demo turns on decode trace

`models/demos/gpt_oss/demo/text_demo.py:1056`:
```python
out_tok, _ = generator.decode_forward(
    out_tok, current_pos,
    enable_trace=enable_decode_trace,  # True by default for all configs
    page_table=page_table, kv_cache=tt_kv_cache,
    sampling_params=device_sampling_params,
)
```

Defaults at lines 194-440: every parametrization sets
`enable_decode_trace=True`.

### Canonical trace-capture body

`models/tt_transformers/tt/generator.py:1147-1198` (`_capture_decode_trace_text`):

```python
# Compile run (eager warmup) — fills all lazy-init buffers, JITs all kernels
self._decode_forward_no_trace_text(tokens, current_pos, page_table=, kv_cache=)

# Build per-step device inputs ONCE, store on self
host_inputs = self.model[i].prepare_decode_inputs_host(
    tokens[i], current_pos[i], page_table=user_page_table
)
device_inputs_i = copy_host_to_device(host_inputs, mesh_device=...)

# Capture
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
tt_out_trace.append(
    self.model[i].ttnn_decode_forward(
        *device_inputs[i], kv_cache=user_kv_cache,
        sampling_on_device=sampling_on_device,
        capture_sampling_trace=split_enabled,
    )
)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

Identical pattern to our 27B `server_tp.py:1587` (`update_input_buffers`)
canonical decode-trace. The only model-specific piece is what's inside
`ttnn_decode_forward`.

### The MoE block inside the captured region (loudbox / standard experts)

`models/demos/gpt_oss/tt/mlp.py:102-110` — the entire MoE block is 3 lines:

```python
def __call__(self, hidden_states, is_decode):
    expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
    expert_output = self.experts(
        hidden_states, topk_expert_indices=expert_indices,
        topk_expert_weights=expert_weights, is_decode=is_decode
    )
    return expert_output
```

Both `expert_indices` and `expert_weights` are `ttnn.Tensor` (router output is
on-device, see below). **There is no `to_torch` between router and experts.**

`models/demos/gpt_oss/tt/topk.py:19-49`:

```python
expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
# ... softmax on device ...
if use_throughput_experts:
    return expert_indices, expert_weights
else:
    # Scatter into a dense sparsity tensor entirely on device:
    return expert_indices, ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)
```

The non-throughput (loudbox) path builds a **dense [B, E] sparsity indicator
tensor** on device via `ttnn.scatter` and then feeds it to `ttnn.sparse_matmul`
in `tt/experts/decode.py:73-83`:

```python
gate = ttnn.sparse_matmul(
    hidden_states, weights.gate_proj,
    sparsity=sparsity, nnz=num_experts_per_tok,
    ...
)
```

**No `all_to_all_dispatch`, no host bridge, no topk-indices re-upload.** The
"sparsity tensor" pattern collapses the dispatch into a single MM kernel that
knows which experts are active per token. Routing-weights application
(`tt/experts/decode.py:124-126`) is just `ttnn.mul`. All on device.

### The MoE block inside the captured region (Galaxy / throughput experts)

`models/demos/gpt_oss/tt/experts_throughput/decode.py:325-380`:

```python
dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
    hidden_rm, topk_indices_rm, expert_mapping_tensors, **dispatch_config.as_dict()
)
# ... expert compute ...
combine_output = ttnn.all_to_all_combine(
    expert_output, dispatch_metadata, expert_mapping_tensors, **combine_config.as_dict()
)
```

`topk_indices_rm` is a `ttnn.Tensor`, produced from the router's output via
`ttnn.to_layout + ttnn.reshape` (lines 306-310) — **all on device**.
`dispatch_metadata` flows out of `all_to_all_dispatch` and straight into
`all_to_all_combine` without ever touching the host.

### The fused decode path (newest, explicit "enables trace capture")

`models/demos/gpt_oss/tt/experts_throughput/fused_decode.py:80-83` — comment is
authoritative for our purposes:

> Format conversion: router outputs [M, K] TILE DRAM, dispatch needs
> [M, 1, 1, K] ROW_MAJOR. **All done on-device (no host round-trip),
> following the DeepSeek pattern (moe.py lines 393-395). This enables trace
> capture.**

The fused path uses three custom ops in series:
`ttnn.experimental.all_to_all_dispatch_metadata + moe_gpt +
selective_reduce_combine`. All three take topk indices/scores as on-device
tensors.

## Demo 2: deepseek_v3 — sparse MoE + MTP, three trace IDs

### Trace-capture sites

`models/demos/deepseek_v3/tt/generator.py`:

| Line | Trace ID                | Captured region                                     |
|------|-------------------------|-----------------------------------------------------|
| 2460 | `self._trace_id`        | `RowBatchedModel.forward_decode` — main decode      |
| 2568 | `self._mtp_verify_trace_id` | MTP verification step                            |
| 2704 | `self._mtp_predict_trace_id` | MTP prediction (speculative decode)             |

Lines 2680-2716 (main decode capture):

```python
trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._trace_rot_idxs)
self._trace_output = RowBatchedModel.forward_decode(
    x=self._trace_tokens, position_idxs=self._trace_positions,
    cfg=self.model_run_config_decode, rope_tensors=rope_tensors,
    page_tables=self._trace_page_tables_to_use,
    profile_decode=self.profile_decode,
)
ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
```

Identical input-prep pattern to gpt_oss: `_trace_tokens`, `_trace_positions`,
`_trace_rot_idxs` are pre-allocated device tensors written via host-to-device
copies outside the captured region.

### MoE inside the captured region

`models/demos/deepseek_v3/tt/moe.py:412-420`:

```python
x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]))

topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_indices_rm = ttnn.reshape(
    topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
)
```

This is the "DeepSeek pattern" the gpt_oss fused_decode comment refers to:
the topk indices come out of the router as a TILE-layout `ttnn.Tensor`, get
converted to ROW_MAJOR + reshaped to the dispatch contract — **all on device,
no `to_torch`**. Then `ttnn.all_to_all_dispatch` (one of `cfg` fields,
`AllToAllDispatchConfig` at lines 214 / 265) consumes them directly.

### The `topk_fallback=True` host-bridge path (NOT used in trace mode)

`models/demos/deepseek_v3/tt/moe_gate.py:445-490` is `topk_fallback_op`. It
does `ttnn.to_torch` → `torch.topk` (or `topk_bitonic`) → `ttnn.from_torch`.
This is the path our project takes by default per
`memory/feedback_ttnn_topk_tie_break_drift.md`.

The model config at lines 126-191 has `topk_fallback: bool = False` as the
**class-level default**. When `topk_fallback=False`, the device path
(`ttnn.topk` at lines 343, 366) is used, and that path IS the trace-compatible
one. The fallback exists because of the
`SFPSWAP` tie-break drift bug ([tt-metal#20625]) — i.e. **the same bug we hit
on Nemotron-3 motivates DeepSeek-V3 to keep the host fallback as an option**,
not as the trace-mode default.

Verdict: DeepSeek-V3 trace-mode users have to accept the device-topk precision
trade-off. There is no documented commit message claiming both `enable_trace=True
+ topk_fallback=True` works together — they appear to be mutually exclusive
modes.

## Negative results

### `deepseek_v3_b1` (batch-1 variant)

`models/demos/deepseek_v3_b1/` has 51 `.py` files. Zero hits for
`begin_trace_capture`. Hits for `all_to_all_dispatch` exist in unit tests but
not in a model-forward path. Status: pre-trace-mode codebase, likely earlier
fork.

### `deepseek_v3_d_p` (dispatcher-pattern refactor)

`models/demos/deepseek_v3_d_p/` has a `reference/` and `tests/` tree. Zero
trace sites. Appears to be a dispatcher-pattern refactor that hasn't yet
landed trace.

### `qwen3_vl`

Not MoE. Vision encoder, no relevant patterns.

### Mixtral

No `mixtral` directory exists in the current tree (it has been removed or
never landed in main).

## Patterns observed

### Bridge 1 — Pad-zeros for seq-padding

**Solution**: pre-allocate the padded buffer at bootstrap, copy-in per-step.
In gpt_oss `prepare_decode_inputs_host` (model.py:557) handles padding;
device-side reshape uses `ttnn.pad` (model.py:447). Not a trace blocker as
long as the pad value is constant.

### Bridge 2 — Topk indices readback after router

**Solution**: don't read back. The router output is already a `ttnn.Tensor`.
Either:
- (loudbox) **scatter into dense sparsity** and pass to `ttnn.sparse_matmul`
  (`gpt_oss/tt/topk.py:46` + `tt/experts/decode.py:73`).
- (Galaxy/multi-chip) **layout-convert on device** to the `all_to_all_dispatch`
  input contract (`gpt_oss/tt/experts_throughput/decode.py:306-310` and
  `deepseek_v3/tt/moe.py:413-418`).

### Bridge 3 — Topk indices re-upload for dispatch

**Solution**: never leaves the device, so no re-upload needed (see Bridge 2).
For dense sparsity tensors, dispatch is replaced entirely by sparse_matmul.
For all_to_all_dispatch, the metadata flows from `all_to_all_dispatch` straight
into `all_to_all_combine` (`gpt_oss/tt/experts_throughput/decode.py:325 →
374`).

### Bridge 4 — Topk weights re-upload for combine

**Solution**: permute the (on-device) weights into broadcast-compatible shape,
then `ttnn.mul + ttnn.sum`. Concrete: gpt_oss fused_decode lines 175-205:

```python
# combine_output shape: [K, 1, M, H], scores shape: [M, K]
scores_permuted = ttnn.permute(scores, ...) # -> [K, 1, M, 1]
weighted = ttnn.mul(combine_output, scores_permuted)  # broadcast
output = ttnn.sum(weighted, dim=0)
```

The standard (non-fused) expert path uses a simpler `ttnn.mul(next_states,
routing_weights, output_tensor=next_states)` after a per-token reshape
(`tt/experts/decode.py:124-126`), all on device.

## Implications for our v0.6 MoE-trace effort on Qwen3.6-35B-A3B

### What to fork verbatim

1. **The `Generator` orchestration shape** from
   `models/tt_transformers/tt/generator.py:1147-1245`:
   - `prepare_decode_inputs_host(tokens, current_pos, page_table)` → builds
     small host tensors.
   - `copy_host_to_device` into pre-allocated `device_inputs` buffers.
   - One eager forward to JIT + lazy-init.
   - `begin_trace_capture` → `model.ttnn_decode_forward(*device_inputs)` →
     `end_trace_capture`.
   - Per-step: write new host inputs into the SAME device buffers, then
     `execute_trace`.
   This is identical in spirit to our `server_tp.py:1587` `update_input_buffers`
   and `server_35b_ttnn.py:1527`, so the 35B server is already 80% set up.

2. **The on-device topk-indices layout-convert** from DeepSeek-V3
   `moe.py:413-418`. Replace any `ttnn.to_torch(topk_indices)` with
   `ttnn.to_layout(topk_indices, ROW_MAJOR) + ttnn.reshape`.

3. **The on-device routing-weights mul** from gpt_oss
   `tt/experts/decode.py:120-126`. Replace any
   `ttnn.from_torch(routing_weights)` per step with a `ttnn.permute + ttnn.mul`
   on the (already-on-device) router output.

### What to build (gap-fillers)

1. **Audit 35B MoE block for `to_torch` / `from_torch` in the hot path** —
   grep `server_35b_ttnn.py` and the `qwen36_*` experimental ops. Every hit
   that's not gated behind `if "weight_name" not in w:` (lazy weight upload) is
   a host bridge we have to eliminate.
2. **Verify `ttnn.all_to_all_dispatch` / `ttnn.all_to_all_combine` are
   trace-safe in the qb1 ttnn build** — they're in tt-metal main; check our
   build's `ttnn.experimental` exposure (we have `qwen36_gdn_*` per
   memory/CLAUDE.md, but not necessarily the new MoE ops).
3. **Decide on sparsity-tensor vs all-to-all on (1,4) qb1 mesh**. 35B-A3B has
   ~256 experts per layer; sparsity-tensor would need a 256-wide indicator,
   which may be too sparse for `sparse_matmul` to be efficient. The
   `all_to_all_dispatch` path is more natural for 35B at NCHIPS=4 with EP
   sharding.
4. **Plan for the `topk_fallback` trade-off**. Per
   `feedback_ttnn_topk_tie_break_drift.md`, our project ships host topk because
   of SFPSWAP drift. To trace MoE we have to either:
   - Accept the device-topk precision penalty and re-run accuracy ladder.
   - Wait for the LLK stable-topk flag (PR #31989) to be exposed via
     `ttnn.topk`. (Not exposed as of last memory check.)
   This is the SINGLE largest open question, and it's orthogonal to the
   architectural-bridge work.

### Effort estimate

- **Architecture pattern fork**: 1-2 days. The Generator pattern is mature and
  trivially adaptable.
- **35B MoE-block on-device thread**: 1 week. Each of bridges 2-4 is concrete
  but requires touching `qwen36_moe_*` paths and re-running the per-layer
  ladder.
- **Topk decision (precision-vs-trace)**: out-of-band. Either accept-and-measure,
  or wait for upstream LLK fix. **This is the gating decision before
  scoping v0.6.**

## Open questions (defer to v0.6 planning)

1. Does our qb1 ttnn build expose `ttnn.all_to_all_dispatch` and
   `ttnn.all_to_all_combine` (the basic non-fused variants)? Check via
   `experiments/utils/ttnn_introspect.py`.
2. Does our qb1 build expose `ttnn.experimental.moe_gpt`,
   `all_to_all_dispatch_metadata`, `selective_reduce_combine`? (The
   gpt_oss-specific fused kernels.)
3. Is the `Qwen3.6-A3B` topology — 8 experts per token out of ~256 total at
   hidden=4096 — fundamentally amenable to the dense-sparsity-tensor pattern
   (gpt_oss standard, no dispatch) or does the 256-wide indicator force the
   all-to-all path?
4. What's the `topk_fallback=False` numerical drift on Qwen3.6-35B router?
   Re-run the per-layer ladder with device-topk to find out — this is the
   prerequisite experiment for v0.6.

## Sources

All paths on `ssh qb1` under `~/tenstorrent/tt-metal/`. Verified
2026-06-06.

- `models/demos/gpt_oss/demo/text_demo.py:194-440` — parametrization defaults
- `models/demos/gpt_oss/demo/text_demo.py:814-857` — prefill trace capture
- `models/demos/gpt_oss/demo/text_demo.py:1056` — decode-trace plumbing
- `models/demos/gpt_oss/tt/mlp.py:102-110` — MoE MLP entry
- `models/demos/gpt_oss/tt/topk.py:19-49` — on-device router (scatter to
  sparsity)
- `models/demos/gpt_oss/tt/experts/decode.py:54-72, 120-126` — sparse_matmul
  expert path + on-device routing-weights mul
- `models/demos/gpt_oss/tt/experts_throughput/decode.py:300-380` — Galaxy
  all_to_all_dispatch / all_to_all_combine
- `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py:1-30, 80-205` —
  fused MoE kernels + the "this enables trace capture" comment citing DeepSeek
- `models/demos/deepseek_v3/tt/generator.py:1056, 2460, 2568, 2704` — three
  trace IDs (decode + MTP verify + MTP predict)
- `models/demos/deepseek_v3/tt/generator.py:2680-2716` — main decode-trace
  capture
- `models/demos/deepseek_v3/tt/moe.py:412-420` — on-device topk-indices layout
  conversion (the pattern gpt_oss fused_decode cites)
- `models/demos/deepseek_v3/tt/moe_gate.py:126-191, 343-366, 445-490` — model
  config defaults, on-device `ttnn.topk` path, host `topk_fallback_op`
- `models/tt_transformers/tt/generator.py:1147-1198` — shared
  `_capture_decode_trace_text`
- `models/tt_transformers/tt/generator.py:1200-1245` —
  `_decode_forward_trace_text` (per-step input update + `execute_trace`)

Companion docs in our project:
- `memory/reference_decode_trace_canonical_pattern.md` — the canonical
  invariants and the 4-bridge MoE trap
- `memory/feedback_ttnn_topk_tie_break_drift.md` — SFPSWAP drift, host-fallback
  rationale
- `memory/feedback_two_phase_warmup.md` — JIT-compile-before-capture mandate
