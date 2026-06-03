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

| ID | What | Gate | Status |
|---|---|---|---|
| **v0** | Single-slot CB (B=1 through `cb_scheduler.Scheduler`) | B=1 forward bit-identical to standalone `step_forward_ttnn`; 1 request through 1 slot generates correctly | ✅ BIT-VALIDATED 2026-06-02 (`49778b3`) |
| **v1** | Batched B>1 forward | B=8 distinct slots: each slot's gen tokens match standalone B=1 ref for its prompt | ⏳ NEXT |
| **v2** | Two-phase warmup + trace capture at B=N | Traced forward bit-correct vs eager; ~5-10× speedup like 27B | ⏳ |
| **v3** | Owned-GDN batched (FOLD-B trick) | +2-3% perf gain vs manual DN | optional |
| **v4** | Prefix cache for attention layers only | Smoke test shows turn-2 cache hit on attn KV; DN layers explicitly skipped | LOW PRIORITY |
| **prod** | TT_BACKEND=35b wire-up + real chat | `chat.py` works; multi-tab smoke | ⏳ |

## Per-stage detail

### v0 — Single-slot CB ✅ BIT-VALIDATED 2026-06-02

`server_35b_cb.py` (~150 LOC, commits `112d72a` + `49778b3`). Pure
B=1 wrapper: `forward_batch_tp_inner` delegates to
`base.step_forward_inner` (toggles `state.sampler_topk` for argmax vs
topk routing). `cb_reset_states` aliases `state.dn_caches_tt[li]` and
`state.kv_caches_tt[li]` directly as the per-slot caches.

**Critical fix that unblocked v0**: `base.reset_caches_ttnn()` REBINDS
the list but doesn't deallocate the previous per-layer (cs, rs) /
(kc, vc) tensors. In long-lived processes (the dev harness and the real
cb_engine session) leaked tensors fragment the device allocator until
subsequent matmul outputs land in stale memory → forward returns garbage
that VARIES PER RUN (82530, 198294, 219673, …). `cb_reset_states` now
explicitly deallocates every old per-layer tensor before calling
`reset_caches_ttnn`. Generic lesson: `[[ttnn-list-rebinding-leaks]]`.

`return_logits=True` raises `NotImplementedError` (35B's `[1, VOCAB]`
bulk readback via `ttnn.to_torch` is independently broken — issue #149).
cb_engine routes 35B through topk-mode via `TT_CB_TOPK_K=64` default.

Gate results:
- `cb35_v0_smoke.py`: 3/3 PASS
  - argmax bit-equiv: base=8, cb=8 ✓
  - return_logits raises ✓
  - topk[0]=8 (matches base argmax) ✓
- `cb35_v0_chat.py`: 8-token decode bit-identical `[271]×8` from both
  paths — proves multi-step state evolution is correct.

### v1 — Batched B>1 forward (~3-5 days, NEXT)

Generalize each primitive to batched. v0 proved the wrapper plumbing;
v1 introduces actual per-slot work. This is the bulk of the project.

**Architecture — leading-B dim convention**:
- v0 state shapes (per chip): `cs:[1, CONV_DIM_CHIP, KERNEL]`, `rs:[1, NV, K, V]`,
  KV `[NUM_BLOCKS, 1, BLOCK, HEAD_DIM]`. The "1" everywhere is the slot
  axis. v1 makes that axis `B`.
- `state.cb_dn[li] = {"cs":[B,…], "rs":[B,…]}` allocated once at
  `setup_cb_state(B=N)`.
- Per-iter input buffers: `cb_tok_buf:[B, 1]`, `cb_cur_pos_buf:[B]`,
  `cb_rot_idxs_buf:[B, 1]` (already set up at v0; just need to actually
  use them in the forward instead of the single-stream `state.tok_buf`).

**Sub-stages v1.0 → v1.5** (each ships its own test):

| v1.x | What | Status |
|---|---|---|
| v1.0 | `setup_cb_state(B=8)` allocates B-leading caches | ✅ BIT-VALIDATED 3/3 (`ec01052`) |
| v1.1 | Embed + RoPE batched | ✅ BIT-VALIDATED 4/4 (`cf211a4`) |
| v1.2 | DN layer batched (manual recurrence) | ✅ BIT-VALIDATED cos=1.0 mad=0.0 (`4546d29`) |
| v1.3 | GatedAttention batched (paged SDPA over per-slot KV) | ✅ BIT-VALIDATED cos=1.0 mad=0.0 (`e8f2d82`) |
| v1.4 | MoE batched (per-slot loop, superseded) | ⚠ B=1 bit-id, B>1 13% drift (`2d0f582`) |
| v1.4b | MoE batched (true broadcast Pattern A) | ✅ BIT-VALIDATED cos=1.0 mad=0.0 (`a6ac640`) |
| v1.5 | Full forward at B=2 (multi-step) | ✅ FUNCTIONAL PASS — per-slot indep + det; argmax drift across 40 layers documented (`0a50e97`) |
| **v1 status** | Production-ready batched serving infrastructure | **SHIPPABLE** |

**Reuse map** (audit before writing anything):
- 27B's `server_tp_cb.deltanet_step_batched` is the template. Need to
  diff against 35B's `dn_forward_ttnn` to identify shape gotchas
  (NV_PER_CHIP=8 vs 27B's 16; HEAD_K_DIM=HEAD_V_DIM=128).
- 27B's `gated_attn_step_batched` handles partial RoPE + Q-gate split.
  35B `attn_forward_ttnn_sdpa` has the **per-head Q-gate chunk split**
  gotcha (`attn_output_gate=True`, q_proj doubled). Lift that logic into
  the batched form.
- MoE: `moe_forward_ttnn_pattern_a_batched` (`server_35b_ttnn.py:1225`)
  already broadcasts. v1.4 is "verify it works with `[B, 1, HIDDEN]`
  input" — should be free; just need to make sure topk routes per-slot.

**Lessons learned from v1.0–v1.2** (bake into v1.3+):
- `cb_reset_states` MUST `ttnn.deallocate` old per-layer caches BEFORE
  reset (else fragmentation + garbage matmul outputs — v0 bug, generalized
  in `[[ttnn-list-rebinding-leaks]]`).
- `setup_cb_state` MUST `ttnn.deallocate` on B-change re-setup (B=4→B=1
  leaks B>1 caches; aliases are NOT deallocatable).
- ttnn.rms_norm on bf16 has shape-dependent accumulation drift. Use
  rank-3 `[B, N, D]` shapes, never fold to `[B*N, D]`. Same likely true
  for ttnn.sigmoid/silu reductions but unverified
  (`[[ttnn-rms-norm-shape-drift]]`).
- Intermediate-readback pattern (capture chip-0 view at each stage,
  diff vs B=1 reference) localizes shape-drift bugs in minutes.

**v1.3 specific risks**:
- RoPE: cos/sin become `[B, ROTARY_DIM]` per chip. base's K-broadcast
  workaround (concat K to NQ rows, rope, slice row 0) relies on cos/sin
  being scalar per step — needs revisiting at B>1 since each slot has
  different cos/sin. Likely fix: keep K as `[B, 1, HEAD_DIM]`, broadcast
  cos to `[B, 1, ROTARY_DIM]`, apply rope rank-3.
- Per-slot `cur_pos` in paged SDPA: `paged_update_cache` and
  `paged_scaled_dot_product_attention_decode` both take
  `cur_pos_tensor=cb_cur_pos_buf` (shape `[B]`) per the 27B audit.
  Verified API supports this.
- Per-slot Q-gate split (`attn_output_gate=True`): the per-head chunk
  split (Q-side doubled, split into Q and gate) needs to broadcast over
  B. Lift the rank-3 slicing pattern.

**Harness-driven iteration**: each v1.x ships its own `cb35_v1_*` test
in `experiments/cb/validate/`. The harness's dynamic-discovery means
no harness restart between sub-stages. **Always clear `__pycache__`
before retriggering after a deploy** — stale .pyc masks new imports.

### v2 — Trace capture at B=N ✅ SHIPPED 2026-06-02 (`c547419`)

Two-phase warmup + trace capture pattern (mirrors cb_scheduler.py:197-227).
Inherits all the plumbing via `cb.forward_batch_tp_inner` dispatch.

cb35_v2_trace.py at B=2 results:
  - capture/release_trace: OK
  - mean eager  = 296.7 ms/step
  - mean traced = 149.7 ms/step
  - **speedup = 1.98×**

cb_scheduler already drives the trace path automatically via the
unified `cb.forward_batch_tp_inner` entry — no scheduler changes
needed.

Speedup at higher B will be larger (more dispatch amortization).
1.98× at B=2 already crosses the bar; production trace is shipped.

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

### prod — Production wire-up (task #147 PARTIAL; task #160 in flight)

**Bootstrap + HTTP layer**: ✅ GREEN as of 2026-06-02 late evening.
**First inference step**: ❌ task #160 — cb-engine crashes with
`TypeError: only length-1 arrays can be converted to Python scalars`.

Fixes that landed this session:
1. **`cb_scheduler.py` dispatch hole** (`97abfab`): was hardcoded
   to `import server_tp / server_tp_cb`; now uses BACKENDS dispatch
   mirroring `cb_api.py`.
2. **Deploy gap**: MM1's BACKENDS registry sat in local git for
   weeks without being deployed to qb1. User caught via the chat TUI.
   Memory entry `[[deploy-serve-files-too]]`.
3. **35B bootstrap signature** (`8111d70`): `bootstrap(state, log=None)`
   default-to-print, accepts cb_api's 2-arg call.
4. **27B bootstrap signature regression** (`a7ea0fe`): same fix on
   server_tp.py; caught by the 27B smoke after we proved 35B fixes.
5. **35B `state.tok` alias** (`73fd269`): cb_api reads `st.tok` (27B
   convention); 35B exposed only `state.tokenizer`. One-line alias.

Plus observability (`3e150c0`): `/bootstrap` endpoint + side-file
`~/tt-xla/.cache/server_cb.bootstrap.log`.

**27B smoke**: PASS end-to-end. `/v1/chat/completions` returns
"The capital of France is Paris." in 2.4s, finish_reason=stop.

**35B smoke**: at TT_CB_SLOTS=1, end-to-end COHERENT — "Hello" →
`"Hello! How can I help you today?"` (`finish_reason=stop`).
At TT_CB_SLOTS>1 with ragged admission, prompt-independent garbage
(`两件两特朗...` loop) — the empty slot's cur_pos=-1 poisons batched
SDPA / MoE math. v1.5 (B=8 dev harness) validated only with all slots
active so this case never ran. Workaround: backend-aware default
`TT_CB_SLOTS=1` for 35B in `cb_api.py` (mirrors the existing
`TT_CB_TOPK_K=64` 35B-specific default pattern). 27B unchanged at 4.

Earlier `cb_reset_slots` no-op fix in `1fc039c` (port of 27B's
masked-multiply pattern) was necessary but not sufficient — fixed
warmup-pollution carry-over but not the empty-slot poisoning.

**Task #162** (CB35-B-gt-1) tracks the proper B>1 fix. Likely loci:
SDPA cur_pos=-1 mask handling, MoE expert routing with DUMMY_TOK in
empty slot, or batched DN cross-slot reductions.

**Session lessons saved to memory**:
- `[[cb-backend-dispatch-holes]]`: shared-contract changes in
  cb_scheduler/cb_api must land in BOTH backends.
- `[[deploy-serve-files-too]]`: `deploy.sh experiments/serve/*.py`
  before every `serve_cb.sh start`.
- `[[dev-harness-vs-cb-engine-gap]]`: validators that drive
  `_step_sampled_topk` directly skip the cb_scheduler.advance layer
  where shape/lifecycle bugs hide.

#### Legacy outline (kept for posterity)

The v1 stack is functionally complete. Production wire-up means routing
`cb_engine`/`cb_scheduler` through `forward_batch_tp_inner_batched` at
B>1 instead of the v0 delegate-to-base path.

**What's already done** (no changes needed):
- MM1: `TT_BACKEND=35b` selector in `cb_api.py` (`BACKENDS` registry).
- `TT_CB_TOPK_K=64` default for 35B (works around the [1,VOCAB] bulk
  readback bug, #149).
- `serve_cb.sh` lifecycle (start/stop/status) — model-agnostic.

**What needs verifying / wiring**:
- `cb_scheduler` calls `cb.forward_batch_tp_inner(state, return_topk=K)`
  (logits → host top-K sampling). For 35B at B>1, this must route
  through `forward_batch_tp_inner_batched` (the new batched forward).
  Currently `cb.forward_batch_tp_inner` delegates to base at B=1; for
  B>1 we need to wire it to the batched function (or have the scheduler
  call a different entry).
- The argmax/topk final-op in `forward_batch_tp_inner_batched` needs to
  return the same tensor shape contract the scheduler expects
  (UINT32 [B, 1] for argmax-mode, ([B, 1, K], [B, 1, K]) for topk-mode).

**Sub-stages**:
- prod.0: audit `cb_scheduler.step()` call sites that touch the
  forward — what return shape it expects per B.
- prod.1: add `return_topk` support to `forward_batch_tp_inner_batched`
  (mirror the v0 pattern, the lm_head topk path).
- prod.2: wire `forward_batch_tp_inner` to dispatch on B (B=1 → base
  delegate, B>1 → `forward_batch_tp_inner_batched`).
- prod.3: real chat smoke at TT_CB_SLOTS=2 via `chat.py`.
- prod.4: 4-tab concurrent smoke.

Gate: real chat works at B=2 with two concurrent clients; multi-tab no
wedge.

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

35B weight upload is ~14 min per bootstrap. The harness
(`experiments/cb/dev/cb35_dev_harness.py`) bootstraps once into a
long-lived python process on qb1, then watches
`~/tt-xla/.cache/cb35_runtime/trig/` for trigger files and runs the
named test via `importlib.reload`. Per-iteration cost: ~15 seconds.

**MUST be launched via tmux** — see [[qb1-tmux-for-long-running]].
nohup+disown and setsid+exec both fail to keep the python alive after
SSH disconnect on qb1; the controlling shell's death takes the process
with it.

Workflow:
```bash
# one-time: launch harness on qb1 (14 min bootstrap, then idles)
bash scripts/run_harness_tmux.sh

# per iteration (locally):
bash scripts/deploy.sh experiments/serve/server_35b_cb.py experiments/cb/validate/cb35_v0_smoke.py
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v0_smoke'
ssh qb1 'cat tt-xla/.cache/cb35_runtime/trig/last.log'   # ← test output
```

Test discovery is dynamic: drop a `cb35_<name>.py` file with
`main(state=None)` in `experiments/cb/{validate,isolate,bench,dev}/`,
then `touch .cache/cb35_runtime/trig/<name>` runs it. No harness
restart needed.

Special triggers:
- `_reload` — importlib.reload `server_35b_cb` only (no test run)
- `_exit` — graceful shutdown

## Known issues (carry-overs to v1)

**Logits bulk-readback on 35B returns garbage** (issue #149). When
`ttnn.to_torch(logits_tt, mesh_composer=ConcatMeshToTensor)` is called
on a `[1, 248320]` bf16 logits tensor, the host sees garbage that
varies per run (observed argmaxes: 82530, 1099, 198294). On-device
argmax + topk kernels reading the **same** tensor find the correct
answer.

**Workaround (active)**: v0's `forward_batch_tp_inner` raises
`NotImplementedError` on `return_logits=True`. cb_api.py sets
`TT_CB_TOPK_K=64` as the default when `TT_BACKEND=35b` — cb_scheduler
routes through `_step_sampled_topk` which reads only `[B, K]` indices
+ values (small per-slot, proven working in v0 smoke case 3).

Investigation deferred to task #149.

## Reference points

- **DeepSeek-V3 reference (the canonical upstream MoE+CB)**:
  `tt-metal/models/demos/deepseek_v3/tt/{moe.py, experts.py, moe_gate.py, generator.py, generator_vllm.py}`
- **Llama-70B-Galaxy (dense-CB + trace warmup reference)**:
  `tt-metal/models/demos/llama3_70b_galaxy/tt/generator.py` (esp. lines 315-360, 543-680, 1010-1076)
- **Local templates**: `experiments/serve/server_tp_cb.py` (the file to
  paste-and-modify), `server_35b_ttnn.py` (existing B=1 forward to wrap).
- Research: `research/tt_metal_moe_cb_patterns.md`.

## Task #163 — long-context drift (UPDATE 2026-06-03)

H1 from research report (DN H_t bf16 round-trip per step, Ollama
precedent ollama#15865) — **REJECTED**. fp32 H_t fix (`92b442f`)
verified active via bootstrap log, server boots, tokens generate —
but degenerate-loop output at ~25 tokens is **unchanged** vs the
bf16 baseline. Implementation-correct, hypothesis wrong (or partial).

fp32 residual stream extension (`35ea58f` + `7c3ede6` + `1c650b7`) —
**STUCK in engine.start() warmup**. Bootstrap completes but warmup
hangs >30 min (vs <30 sec for bf16). Presumably typecast dispatch
overhead + fp32 math in the trace capture path.

**Lesson re-learned (non-negotiable #1)**: should have routed drift
experiments through the dev harness above (`run_harness_tmux.sh qb1`)
instead of full server restarts. Model stays resident; iteration
~30 sec vs ~14 min for a fresh server bootstrap.

**Next investigator should**:
1. Spin up the dev harness ONCE (14 min, one-time).
2. Run `experiments/utils/cosine_ladder_35b.py` with
   `--dn-state-dtype fp32 --owned-gdn off` to get per-layer cos@L32
   pos1 numbers vs the HF oracle.
3. Compare to baseline (bf16 default). If fp32 H_t doesn't move
   cos@L32 ≥ 0.99, H1 is provably insufficient — move on to H3
   (decay-only fp32 cast), H4 (RMSNorm precision), or H5 (conv1d).
4. ONLY ship a fix to the production server once a ladder run
   confirms cos@L32 pos 1 ≥ 0.99 AND needle-haystack@100 retrieves.
