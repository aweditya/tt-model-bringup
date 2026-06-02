# Qwen3.6-35B-A3B → CB chat server — milestone plan (revised 2026-06-02)

Living plan. Replaces the original CB35-0..CB35-7 sketch with a tighter
v0..v4 staging informed by the
[`tt_metal_moe_cb_patterns.md`](tt_metal_moe_cb_patterns.md) research.

**Key insight from research**: the hardest part I had imagined
(CB35-5: MoE inside traced B=N forward) is **basically free**. The 35B
server's existing `moe_forward_ttnn_pattern_a_batched`
(`server_35b_ttnn.py:1225`) already supports arbitrary leading batch dim
because `ttnn.matmul` broadcasts. So CB35-5 collapses into CB35-3.

## Goal (unchanged)

35B-A3B online as a CB chat backend behind `/v1/*` via `TT_BACKEND=35b`
(MM1 shipped). Inherits everything the 27B chat path gives.

## Architecture corrections from research

- **256 routed experts** + 1 shared, top-8 routed (NOT 64; 64 is the
  per-chip slice E_LOCAL = 256/4).
- 40 layers in pattern `10 × (3 GDN + 1 GatedAttention)`. Every layer's
  FFN is MoE.
- GatedDeltaNet: 32 V heads, 16 QK heads, head_dim=128.
- GatedAttention: 16 Q heads, 2 KV heads, head_dim=256, partial RoPE
  (rotary_dim=64), `attn_output_gate=True` (Q proj doubled, per-head
  chunk split — already handled by `attn_forward_ttnn_sdpa`).
- Same tokenizer + Qwen3 chat template as 27B. **`/think` `/no_think`
  soft-switch NOT supported** (model card explicit).
- Default `state.moe_mode = "pattern_a_batched"` — trace-safe routing
  (on-device top-k). **Do NOT take the topk-host-readback path.**

## Reuse map (validated by research)

### Reused as-is — zero changes
- `cb_engine.py`, `cb_scheduler.py`, `cb_metrics.py`, `cb_api.py`,
  `openai_endpoint.py`, `live_slot_store.py`, `protocol.py`
- All test infra (`prefix_cache_store.py`, `prefix_cache_lifecycle.py`,
  `chat_template_invariant.py`, `prefix_cache_smoke.py`)
- Two-phase warmup pattern
- Chat template patches (Qwen3.6 family → same `preserve_thinking + trailing strip`)
- TT_BACKEND env selector (MM1 already shipped, registry has 35b key)

### New — model-specific (~600-1200 LOC)
- `experiments/serve/server_35b_cb.py` — direct paste from
  `server_tp_cb.py` (695 LOC) with shape constants swapped and the dense
  MLP replaced by the existing `moe_forward_ttnn_pattern_a_batched`.

## Milestones (v0..v4)

| ID | What | Gate | Effort |
|---|---|---|---|
| **v0** | Single-slot CB (B=1 through `cb_scheduler.Scheduler`) | B=1 forward bit-identical to standalone `step_forward_ttnn`; 1 request through 1 slot generates correctly | ~1 day |
| **v1** | Batched B>1 forward | B=8 distinct slots: each slot's gen tokens match standalone B=1 ref for its prompt | ~3-5 days |
| **v2** | Two-phase warmup + trace capture at B=N | Traced forward bit-correct vs eager; ~5-10× speedup like 27B | ~1-2 days |
| **v3** | Owned-GDN batched (FOLD-B trick) | +2-3% perf gain vs manual DN | ~3-5 days (optional) |
| **v4** | Prefix cache for attention layers only | Smoke test shows turn-2 cache hit on attn KV; DN layers explicitly skipped | ~2-3 days (LOW PRIORITY) |
| **prod** | TT_BACKEND=35b wire-up + real chat | `chat.py` works; multi-tab smoke | ~quick |

## Per-stage detail

### v0 — Single-slot CB (~1 day)

Smallest possible step: create `server_35b_cb.py`, set `cb_B=1`, route
the existing single-stream forward through the CB scheduler. Validates
all the plumbing without changing the model code.

Tasks:
- Create `server_35b_cb.py` by direct paste of `server_tp_cb.py` (695 LOC).
- Swap constants: import `NV_PER_CHIP=8` (35B), `K_DIM=128`, `V_DIM=128`,
  `CONV_DIM_CHIP=CONV_DIM/4` from `server_35b_ttnn` instead of `full_layer_tp_probe`.
- In `setup_cb_state(state, B=1, ...)`: allocate `cb_dn[li]` for the 30
  GDN layers + `cb_kv[li]` for the 10 GatedAttention layers (per the
  10-block-of-4 pattern).
- `forward_batch_35b_inner`: copy `step_forward_inner`
  (`server_35b_ttnn.py:1602`) with a leading B axis. Layer dispatch:
  - GDN layers → call `dn_forward_ttnn` (existing) wrapped with batch dim.
    For v0 with B=1, this is essentially a no-op wrap.
  - Attention layers → call `attn_forward_ttnn_sdpa` (existing).
  - MoE FFN → call `moe_forward_ttnn_pattern_a_batched` (existing,
    already batch-friendly).

Reuse `cb_engine.CBEngine` + `cb_scheduler.Scheduler` with no changes.

Gate (3a/3b/3c ladder, mirrors CB1's gate for 27B):
- 3a: B=1 forward at slot 0, logits cos vs standalone `step_forward_ttnn` ≥ 0.9999
- 3b: B=4 forward, slot 0 real + slots 1-3 DUMMY_TOK at cur_pos=0;
  slot 0 logits match standalone B=1 ref
- 3c: distinct-slot isolation (B=4, two slots with different prompts;
  each slot's gen matches its own B=1 ref)

Effort: ~1 day. Most code is paste-and-swap.

### v1 — Batched B>1 forward (~3-5 days)

Generalize each primitive to batched.

- **DN batched**: 27B's `deltanet_step_batched` (`server_tp_cb.py:331`)
  is the exact template. State shape `[B, NV_PER_CHIP, K_DIM, V_DIM]`
  identical between 27B and 35B (only head counts differ). Manual
  recurrence (owned-GDN kernel still B=1-only).
- **GatedAttention batched**: 27B's `gated_attn_step_batched`
  (`server_tp_cb.py:519`) handles partial RoPE and Q-gate split.
  Port to 35B with shape constants. The per-head Q-gate chunk split
  bug from B16 bringup is already handled in `attn_forward_ttnn_sdpa`
  — lift that logic into batched form.
- **MoE batched**: **already supports B>1 unchanged.** Pattern A masks
  AFTER compute, so per-slot routing is trivial. `ttnn.matmul` of
  `[B, E_LOCAL, 1, HIDDEN] @ [E_LOCAL, HIDDEN, 2*MOE_INTER]` broadcasts
  over the B leading dim. Just feed `h_3d_repeat` with a leading B dim.
- **Per-slot ragged state**: cur_pos / KV / DN reset mechanisms.
  27B's `cb_reset_states`, `cb_reset_slots` (masked multiply) generalize
  immediately.

Gate (3d adds to v0's ladder):
- 3d: B=8 distinct slots, 8 different prompts; each slot's generation
  bit-identical to its standalone B=1 reference for 50+ decode steps.

Risks:
- Per-head Q-gate split is 35B-specific shape gotcha — copy from
  `attn_forward_ttnn_sdpa` exactly.
- Memory budget headroom is fine through B=16 on (1,4) per research §6.

Effort: ~3-5 days (the bulk of the project).

### v2 — Trace capture at B=N (~1-2 days)

Two-phase warmup pattern (already proven for 27B). Steps:
1. Warmup `forward_batch_35b_inner` at B=N WITHOUT trace (compile only)
2. `ttnn.synchronize_device(mesh)`
3. Capture both prefill (if any) + decode traces back-to-back

The 35B B=1 trace already works (P3 task done). Going B=N is the same
pattern with bigger input buffers. Pattern A MoE is data-INDEPENDENT at
the tensor level (mask is just data, not a control-flow branch), so
trace capture works identically.

Gate: traced batched forward bit-correct vs eager (cos ≥ 0.9999 on
logits); ~5-10× speedup like 27B.

Effort: ~1-2 days assuming v1 is shape-clean.

### v3 — Owned-GDN batched FOLD trick (~3-5 days, optional)

Integrate the FOLD-B-into-slots trick from the 27B CB experiment
(commit `a35fb3c`): fold `[B, NV, K, V]` → `[1, B*NV, K, V]` and call
`qwen36_gdn_decode_owned` with `debug_mode=10` (race-free output
variant).

Device kernel already in qb1+qb2 ttnn (verified 2026-05-28; memory
[[remote-hosts]] section). This is pure Python integration.

Expected gain: ~+2.5% (matches 27B's owned decay/gate ship).

Effort: ~3-5 days. Risk: 35B DN has different state shape than 27B —
verify fold math separately.

**Decision gate**: if v0+v1+v2 hit < 60 ms/tok at B=8, **skip v3** — the
integration cost outweighs the perf.

### v4 — Prefix cache (~2-3 days, LOW PRIORITY)

⚠️ **Likely low ROI**. vllm-project/vllm#36493 reports Qwen3.5-35B-A3B
(same arch class) prefix-cache hit rate ~0.1% — DN layers can't be
safely cached because H_t state is autoregressive without a block
notion. Only the 10 GatedAttention layers' KV cache is dedupable.

Skip unless we have a specific high-prefix-repetition use case (shared
system prompt across users, document Q&A with reused contexts). For
single-user chat through `chat.py`, the slot-level cache hit only on
identical conversation continuations is already covered by 27B's PC; on
35B the DN drift will mean prefix matches don't yield correct gen.

If pursued: cache only attention-layer KV; explicitly mark DN layers as
non-cacheable. Mirror the 27B implementation but with a per-layer
"cacheable" flag.

### prod — Production wire-up (quick)

- MM1 already added `TT_BACKEND=35b` selection.
- Restart `serve_cb.sh` with `TT_BACKEND=35b TT_CB_PREFIX_CACHE=0` (v4
  off by default).
- Real chat smoke through `chat.py`.
- `/v1/models` advertises `Qwen/Qwen3.6-35B-A3B`.
- 4-tab concurrent smoke.

Gate: real chat works; multi-tab no wedge.

## Stop-gate decisions (per research §7)

- If v0+v1 hit > 60 ms/tok at B=8, **v2 trace mandatory before shipping**.
- If v3 owned-GDN gain < +2%, **skip v3**.
- v4 only attempted after measuring real cache-hit potential on the live
  v2/v3 server.

## Reuse summary

| Effort | What | Reused from 27B |
|---|---|---|
| ~1 day | v0 single-slot CB | `server_tp_cb.py` template, all scheduler/engine/api |
| ~3-5 days | v1 batched forward | `cb_reset_slots`, `deltanet_step_batched`, `gated_attn_step_batched`; MoE is FREE because Pattern A already broadcasts |
| ~1-2 days | v2 trace | two-phase warmup pattern from cb_scheduler.py |
| ~3-5 days | v3 owned-GDN | FOLD trick from commit `a35fb3c` (27B experiment) |
| ~2-3 days | v4 prefix cache | live_slot_store + cb_scheduler PC logic (mostly free; just guard DN layers) |

**Total**: ~10-20 days for the full stack (v0..v4). v0+v1+v2+prod alone
is ~6-9 days and gives a working chat server with trace speedup.

## What stays open from previous version

- CB35-0 (audit) now done modulo the chat-template invariant test on
  Qwen3.6-35B-A3B tokenizer (qb1 was offline at plan-write time;
  defer to next qb1 availability).

## Dev iteration harness (2026-06-02)

35B weight upload is ~14 min per bootstrap. To avoid paying that cost
per code change, use `experiments/cb/dev/cb35_dev_harness.py` — a
long-running python process on qb1 that bootstraps once, then watches
`/tmp/cb35_trig/` for trigger files and runs the named test via
`importlib.reload`.

Workflow:
```bash
# one-time: launch harness on qb1 (14 min bootstrap, then idles)
ssh qb1 'cd ~/tt-xla && nohup .venv/bin/python -u experiments/cb/dev/cb35_dev_harness.py > /tmp/cb35_harness.log 2>&1 < /dev/null & disown'

# per iteration (locally):
bash scripts/deploy.sh experiments/serve/server_35b_cb.py experiments/cb/validate/cb35_v0_smoke.py
ssh qb1 'touch /tmp/cb35_trig/v0_smoke'
ssh qb1 'cat /tmp/cb35_trig/last.log'   # ← test output, seconds latency
```

To register a new test, add to the `TESTS` dict at the top of the harness
and make the test module expose `main(state=None)`.

Special triggers:
- `_reload` — importlib.reload `server_35b_cb` only (no test run)
- `_exit` — graceful shutdown

## Known issue (CB35-1 v0 — non-blocking)

**Logits bulk-readback on 35B returns garbage** (2026-06-02). When
`forward_batch_tp_inner` returns logits and the caller does
`ttnn.to_torch(logits_tt, mesh_composer=ConcatMeshToTensor)`, the host
sees garbage that VARIES PER RUN (observed argmaxes: 82530, 1099, 198294
on the same input + state). On-device argmax + topk kernels reading the
**same** tensor find the correct answer (token 8). Suspected ttnn bulk-
readback bug specific to 35B's `[1, 248320]` bf16 tensor shape.

**Workaround**: route 35B through topk-mode (`TT_CB_TOPK_K>0`) in
cb_scheduler. cb_api.py sets `TT_CB_TOPK_K=64` as the default when
`TT_BACKEND=35b` — cb_scheduler's `_step_sampled_topk` reads only [B, K]
indices + values which are small per-slot and proven working in the v0
smoke test (case 3: top-1 = base argmax across all 4 chips).

Investigation deferred. Likely a 35B-shape-specific bulk-DMA timing or
sync issue. Worth checking with smaller `[1, VOCAB]` shapes (e.g., a
hypothetical Qwen3.6 variant with smaller vocab) to localize.

## Reference points

- **DeepSeek-V3 reference (the canonical upstream MoE+CB)**:
  `tt-metal/models/demos/deepseek_v3/tt/{moe.py, experts.py, moe_gate.py, generator.py, generator_vllm.py}`
- **Llama-70B-Galaxy (dense-CB + trace warmup reference)**:
  `tt-metal/models/demos/llama3_70b_galaxy/tt/generator.py` (esp. lines 315-360, 543-680, 1010-1076)
- **Local templates**: `experiments/serve/server_tp_cb.py` (the file to
  paste-and-modify), `server_35b_ttnn.py` (existing B=1 forward to wrap).
- Research: `research/tt_metal_moe_cb_patterns.md`.
