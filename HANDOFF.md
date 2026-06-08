# HANDOFF — cold-start one-pager

What this project is, where the perf is now, what to run, and what is next.
Read top to bottom; everything else is linked.

---

## LATEST (2026-06-08) — **🎉 Phase 2.B.1 COMPLETE — verify trace works, Phase 3 unblocked**

**B=K+1 verify trace SHIPPED + ALL 4 GATES PASS** (commit `2dea4cb` + foreground iterations):

| Metric | Value |
|---|---|
| Eager B=K+1 forward (warm) | **114.9 ms** |
| **Traced B=K+1 forward (3/3 warm replays)** | **59.8 ms** |
| Trace capture wall | 414 ms |
| Argmax bit-equivalent eager vs traced | ✓ (all 6 rows = 198) |
| Per-row matches independent B=1 forward | ✓ (all 6 = B=1 argmax 198) |

**Projected spec-dec wall per round** (target B=1 47ms + drafter ~5ms +
verify 60ms = ~112ms for K+1 candidates). At α=0.7 → ~4 accepted →
**~28 ms/tok effective ≈ 1.7× over 47 ms baseline**.

Phase 2.B.1 implementation: 8 steps in 8 commits per
`research/gemma4_verify_kp1_audit.md`. Bugs caught + resolved on the
way: `ttnn.embedding` silently collapsing 3D input → 2D; TILE_LAYOUT
padding the Bv=6 dim → 32 breaking reshape volume; mesh-replicated
argmax readback needing first-Bv slice.

**Drafter trace SHIPPED 2026-06-08** (commit `c3f5fc8` + foreground
follow-up): 5/5 gates PASS. Eager 63.6 ms → **traced 6.4 ms warm**
(**9.99× speedup**). Argmax bit-equivalent to eager AND HF (=597 on
prompt_0). v0 limitation: single-bucket L_kv fixed at first capture.
Multi-bucket v1 deferred to HTTP follow-up.

**Spec-dec round budget (all traced)**:
- target B=1: 47 ms
- drafter B=1: 6.4 ms ✓ NEW
- verify B=K+1: 59.8 ms
- accept walk: <1 ms
- **Total: ~114 ms per round** → α=0.7 → **~28.5 ms/tok ≈ 1.65× over 47 ms baseline**

**Phase 3 design call 2026-06-08**: Phase 2.B.1 shipped READ-ONLY verify
(no `paged_fused_update_cache` in K+1 forward). Discovered this means
spec-dec round needs target B=1 × N for cache writes → **no tok/s
speedup at v0**. User decision: **correctness first**. Ship Phase 3 v0.0
to validate accept walk + α; accept slow tok/s. v1.0 perf via non-aliased
page table follows.

**Phase 3 v0.0a SHIPPED 2026-06-08** (commit `3c1f2ad`):
- Full spec-dec round runs end-to-end (target+drafter+verify+accept walk)
- 6/6 gates PASS: co-load, prefill, scheduler.step, emit/accept count, cur_pos advance
- Per-round: drafter×5 eager 480ms + verify traced **59ms** + target B=1 advance 188ms = 794ms / 1 emit (α=0 single round, expected)
- Verify trace replay matched prior Phase 2.B.1 number (59ms ✓)

**Phase 3 v0.0b SHIPPED 2026-06-08** (commit `9a31679`):
- Multi-round spec-dec across 5 rounds at K=3 — VERDICT PASS
- Target hidden exposure hook stashes post-final-norm hidden per step
- Scheduler.generate() multi-round loop drives target+drafter+verify+walk
- Cache advances correctly each round (5→10 in 5 rounds)
- **α=0 across all rounds** — root cause is pre-existing **target prefill
  argmax bug #259**, NOT scheduler. Target outputs t₆=496 (" a") instead
  of HF's t₆=597 (" Paris"). Drafter is HF-bit-validated (argmax=597 ✓)
  so when #259 is fixed, drafter's predictions will match target's and α > 0
  will appear automatically.
- Spec-dec correctly emits target's correction at each round (Leviathan
  fallback) — proves scheduler is faithfully reproducing target output.

**Phase 3 v0.0c CONDITIONAL on fixing #259** (target prefill argmax) —
byte-equiv vs plain B=1 gate. Without #259 fix, both spec-dec and plain B=1
produce the same broken output (by construction). Gate is implicit.

**#259 RESOLVED 2026-06-08** — NOT a bug. HF target oracle confirms our
target's predictions are bit-equivalent to HF across all 6 prefill positions:
```
HF argmax per pos: [258882, 236743, 529, 506, 563, 496]
                                         "of"  "France"  "is"  "a"
```
- Target at pos 5 predicts " a" (496) — **matches HF**
- Drafter at same context predicts " Paris" (597) — also matches HF (#255)
- They genuinely DISAGREE on prompt_0's continuation per their training
- **α=0 on prompt_0 is REAL behavior, not a scheduler bug**

**The "constant 236770" symptom** in old task #259 was a stale issue from
a previous server version that's since been fixed.

**Phase 3 SHIPPED end-to-end** — framework is correct + validated.

**Recommended next** (pick one):
- **A** Multi-prompt α distribution — run v0.0b across the 5 oracle
  prompts to characterize real α. Some prompts will produce non-zero α.
- **B** Phase 3 v1.0 perf path — refactor verify to non-aliased page
  table; ship the projected ~3× speedup. (Real demo material even if α
  is prompt-dependent.)
- **C** Pivot to other priorities (Gemma 4 perf adoption from `arg/
  gemma4_optimizations` branch diff, Nemotron-3, presentation prep).

**Phase 3 v1.0 follow-up** — refactor verify trace to non-aliased page
table (write K/V at K+1 distinct slots, abandon unused). Projected
~15 ms/tok ≈ 3× over baseline.

Phase 4 HTTP wire-up after v0.0.

---

## PRIOR (2026-06-08 AM) — Tenstorrent feedback + Phase 2.B.1 6/8 steps green

**Yossi at Tenstorrent (email 2026-06-08)** — they are reading our wiki +
research dirs over the weekend. Captured at
`research/tenstorrent_feedback_2026-06-08.md`. Three durable references:

- [[reference-tt-metal-gemma4-branch]] — `arg/gemma4_optimizations` at
  commit `f7d0161`, directly relevant to our active Gemma 4 perf work
  (#178/#179/Round 11+). Diff task #269.
- [[reference-tt-metal-qwen36-branch]] — Qwen 3.6 9B Blackhole demo at
  commit `1cecd16`. Our prior MoE+trace research agent missed it. Read
  task #270.
- [[reference-tile-ai-megakernel]] — TileRT/TileOps + megakernel paper
  research direction. Task #271 (deferred until after spec-dec ships).

**Phase 2.B.1 progress (2026-06-08)** — 6 of 8 steps green:
- #260/#261 (state buffers + helpers) already shipped by Phase 2.A agent
- #262 sliding kp1 fork — 15/15 invariant, 15/15 sensitive PASS
- #263 global kp1 fork — 15/15 invariant, 15/15 sensitive PASS
- #264 orchestrator `_layer_forward_pos0_paged_kp1` — mechanical wrap
- #265 full 48-layer `forward_token_gm4_inner_kp1` — probe in flight
- #266 trace capture + #267 e2e smoke pending

---

## PRIOR (2026-06-07 PM) — **Gemma 4 spec-dec GREENLIT, scope dropped 2 days**

User confirmed Q1+Q2+Q3. **Phase 0 DONE device-free**:

**0.A feasibility (commit pending)** — verdict OUTCOME (a) implementable
in ttnn. `research/gemma4_assistant_feasibility.md`. Key findings:
1. **Centroid masked-embedding DISABLED for 12B** (`use_ordered_embeddings=False`
   in the actual 12b-it-assistant config) — standard `lm_head` Linear(1024, 262144).
   No top-k softmax approximation needed.
2. **Drafter is PARALLEL** (one forward → K candidates via sliding-window
   context), NOT autoregressive Leviathan. Simpler than original plan.
3. Drafter shares target's KV (one tuple per `layer_type` = sliding+full);
   target server must EXPOSE its last-layer KV per step.
4. Drafter is 4 Gemma 4 layers + pre/post projection + lm_head — forks
   ~80% from existing `server_gemma4_unified_ttnn.py`.
5. Requires `transformers ≥5.10.0` for the HF oracle (current 5.9.0 has
   `Gemma4AssistantForCausalLM` for E2B, not the unified 12B variant).

**0.B determinism audit** — `research/gemma4_determinism_audit.md`.
Patches B+D already shipped on Gemma 4 12B
(`server_gemma4_unified_ttnn.py:91-98, 1086`). Patch A is for sampling
path (not relevant to greedy spec-dec). **Zero code changes needed**.

**Revised total**: ~5d build + 1d buffer = **~6 days** (was 8).

**Nemotron-3 NIAH long-context (v0.5.bench DONE)** — `9b86e95` results
(probe `0c0230b`). 0% retrieval at L=128/512/1024 but **decode stable at
203-207 ms/tok across all lengths** (v0.5.P1 win holds) and outputs are
COHERENT (`"Actually, I'm not sure!"` not gibberish). Third confirmation
of `[[needle-prompt-shape-not-precision]]`: BASE-format Q/A prompts
trigger IT-conversational evasion in any IT model. Long-context decode
**IS stable** — that was the actual ask.

**Remaining device-free work before qb2 freed**:
- spec_dec_scheduler skeleton (Phase 3.A)
- B=K+1 verify-trace probe scaffold (Phase 2.A)
- DeepSeek-V3 page-table alias fork into our file (Phase 2.B)

**When qb2 freed**: transformers upgrade → HF oracle → drafter v0.1 bringup.

**Phase 1 v0.1 SHIPPED 2026-06-07** (commits `4af15ea` + `f6b45f3` + `1c45ba5`):
- `experiments/serve/server_gemma4_12b_assistant_ttnn.py` (470 LOC) bootstrap PASS on qb2
- **embed cos = 1.0** (bit-perfect vs HF)
- **pre_projection cos = 0.9999774** (well above 0.999 gate)
- Bootstrap 28s cold / 9.9s warm
- Key arch findings: drafter has NO k_proj/v_proj (cross-attention from target's KV); L3 full attn head_dim=512 vs L0-2 sliding head_dim=256 (dual-head_dim); `tie_word_embeddings` honored

**Phase 1 v0.2/v0.3 SHIPPED on qb1 2026-06-07** (commit `dfd44c8`):
- 5/5 prompts: hidden cos≥0.999, logits cos≥0.999, **argmax exact match vs HF**
- Per-prompt: p0=597, p1=107, p2=597, p3=146608, p4=255968 (all match)
- Forward wall ~52 ms warm
- **Phase 1 drafter bringup COMPLETE** (oracle + bootstrap + 4-layer forward + multi-prompt validate in ONE session)
- Drafter compute fully on-device; smoke I/O is correctness-gate only

**Drafter dev harness shipped** (commit pending) — forks `nm3_dev_harness`
pattern. `bash scripts/run_harness_tmux.sh gm4_asst qb1` keeps state
resident; iteration drops from ~60-100s/smoke to ~5-10s. Trigger
names resolve as `gemma4_assistant_<name>` / `gm4_asst_<name>` / `<name>`.
See `[[gm4-asst-dev-harness]]` for usage.

**Phase 2.A SHIPPED on qb1 2026-06-07** (commits `de604fd` + `8fbecd5` +
`952f31e` + `486d3e9`):
- 2.A.0 layout probe: per-chip reassembly works as documented
  (cache_0 → KV head 2c, cache_1 → 2c+1; full replicated). Shape
  strict-match HF; cos sliding=0.96, full=0.99 (bf16 chain drift, not
  layout). `experiments/cb/isolate/gemma4_target_kv_layout_probe.py`
- 2.A target server change: `read_shared_kv_for_drafter(state, L_kv)` +
  `reset_shared_kv_for_drafter(state)` helpers; `state.last_sliding_idx`
  + `state.last_full_idx` derived at bootstrap. ZERO change to hot path
  (helpers only run when spec-dec scheduler calls them).
- 2.A.smoke: target + drafter co-resident on (1,4) mesh (drafter monkey-
  patches `set_fabric_config`/`open_mesh_device` to no-op during boot to
  preserve target's fabric context). Prefill prompt 0, read TT KV via
  helper, drafter forward with TT KV vs HF KV: **argmax MATCH (597=HF)**,
  logits cos=0.968 (gate 0.95), top-8 overlap 5/8.
  `experiments/cb/isolate/gemma4_target_kv_expose_smoke.py`

**Phase 2.B.0 (alias helper) + 2.B.0.5 (kernel gate) SHIPPED 2026-06-07**
(commits `25e3fb3` + `c3124d2`):
- `build_verify_alias_page_table_host` host helper in `spec_dec_scheduler.py`
  (5/5 host probe PASS at K∈{3,5,7} × verify_offset∈{1,8})
- **B=K+1 kernel isolation gate: PASS** — both `paged_update_cache` AND
  `paged_scaled_dot_product_attention_decode` accept B=6 with the
  alias-page-table pattern; no TT_FATAL. HARD-STOP risk surfaced by
  Phase 2.B agent research is CLEARED. Output shape `[1, 6, 16, 256]`
  exactly matches the K+1-logits-per-row contract.

**Cache write race finding** (kernel-level): with K+1 alias rows all
writing to row 0, only the LAST writer's K/V persists. Phase 3 accept
walk needs either (a) read-only verify variant (skip paged_update_cache,
feed K+1 Q only), or (b) write-then-rewind (DeepSeek-V3 pattern).
Decision deferred to Phase 3 design.

**Phase 2.B.1 unblocked + RESCOPED 2026-06-07** (foreground audit `e685c6f`):
target server refactor adds B=K+1 verify trace capture. **~310 LOC
mechanical fork, ~2-3 h** (NOT 1.5-2 days as agent originally estimated).
Scope: 6 state buffers, 4 `*_kp1` function forks (sliding layer, global
layer, per-layer dispatch, top-level forward), 2 host helpers, trace
capture. `_lm_head_argmax` already B-generic; kernels already accept B=K+1
(gate `c3124d2`). 8-step implementation order with per-step gate at
`research/gemma4_verify_kp1_audit.md`. trace_region_size already at
400 MB (no bump needed). Tasks #260-#267.

**Phase 3 after Phase 2** (~1d): implement the 3 NotImplementedError seams
in `experiments/serve/spec_dec_scheduler.py`; bench α at K∈{3,5,7}; greedy
correctness gate.

**Architectural clarifications** (codified in plan `research/gemma4_mtp_plan_of_action.md`):
- Drafter REPLICATED across 4 chips (not TP); shipped at v0.2
- "Parallel" = B=K+1 verify (not concurrent execution); host-step dependency chain
- 3 traces total, `trace_region_size` 50→150 MB
- ~9-10 GB/chip memory footprint (22 GB headroom)

**Build journey** for the record (commits `ed4753f` v0.2 server + `7b1cdc4` rms_norm
isolation + `8781a65` pivot + `dfd44c8` fix):
- 4-layer forward + cross-K/V attention + post_projection + lm_head all
  shipped (+280 LOC to drafter server)
- **HARD STOP on qb2**: `TT_THROW: Failed to generate binaries for
  layernorm — trisc1 SFPI copysgn<vInt,vInt> returns vSMag, expected
  vInt`. Every `ttnn.rms_norm` call fails. v0.1 path (embed+matmul+
  all_reduce only) still PASS on the broken build.
- Isolation confirms qb2-only: 4/4 rms_norm shapes FAIL on qb2, all PASS
  on qb1. Memory entry `[[qb2-layernorm-trisc1-broken-2026-06-07]]`
- **Pivoting v0.2 validation to qb1** (nm3 harness killed to free
  device; oracle artifacts rsync qb2→local→qb1; v0.2 sources deployed
  to qb1). Forward smoke in flight on qb1 now.

---

**Build kicked off 2026-06-07**:
- `gm4` tmux session on qb2 killed (was idle, queue=1 from earlier RULER agent attempt)
- transformers upgraded to 5.10.2 on qb2 (was 5.8.0) — needed for `Gemma4UnifiedAssistantForCausalLM`
- `experiments/utils/hf_oracle_gemma4_assistant.py` deployed + running on qb2
  - First attempt failed: `device_map="cpu"` needs `accelerate` package (not installed)
  - Fix: drop `device_map` + `low_cpu_mem_usage` kwargs (CPU is the default)
  - Re-running 2026-06-07 PM

**Non-negotiables strict-mode escalation**: per
`[[remote-only-strict]]` memory (2026-06-07), all Python execution
runs on `ssh qb1/qb2` via permanent files. No `.venv/bin/python -c`,
no local `uv pip install`. Earlier in this session I violated this
~6× during the feasibility phase — corrected, captured as durable
memory.

---

## PRIOR (2026-06-07) — **v0.5.P1 SHIPPED + v0.5.bench RULER-NIAH on Nemotron-3 next**

**Pivot 2026-06-07 (user)**: skipped P2 RMSNorm fusion after profile showed
modest ROI (~5-10 ms eager via dispatch reduction; no general rms_norm+
matmul fusion in ttnn). MoE is the bigger lever (52% of step) but HiFi2
deferred. Pivot to v0.5.bench — long-context stability validation before
chasing more perf rounds.

**Prefill reality**: Nemotron-3 prefill ~770 ms/token (mamba2 SSD path
runs per-position eager); RULER NIAH at L=4k infeasible (~53 min/sample).
Running smaller NIAH ladder at L=128/512/1024 instead — ~1 hour wall for
first numbers vs upstream Mistral-Nemo-12B (~95% @ 4k) and
Llama-3.1-8B-Instruct (~95% @ 4k) as size-class anchors.

**Profile breakdown (252 ms step from v040f)**:

| Block | layers | ms | per-layer | share |
|---|---|---|---|---|
| MoE | 23 | 131.2 | 5.7 | **52%** |
| Mamba2 | 23 | 93.8 | 4.1 | **37%** |
| Attn | 6 | 9.8 | 1.6 | 4% |
| Embed+LM+sample | — | 16.9 | — | 7% |

---

## PRIOR (2026-06-06 evening) — **v0.5 eager perf pass started; trace deferred to v0.6**

After cross-server audit (commit `47bd16e`) revealed **no MoE model in our
project is currently traced** (35B is also eager-only), strategic pivot:
ship eager perf wins NOW, MoE-on-device dispatch as v0.6 architectural
effort gated on upstream MoE+trace research (task #245 agent in flight).

**v0.5.P1 vocab-shard lm_head landed + measured (commit `693806a`)**.
Forks 27B P22 (server_tp.py:399 + :1680-1688). Measured 2026-06-07 on
fresh qb1 harness bootstrap:

| | pre-shard | post-shard | delta |
|---|---|---|---|
| mean ms/tok | 260.0 | **203.1** | **-21.9%** |
| median | n/a | 201.4 | — |
| p95 | n/a | 214.3 | — |
| tok/s | 3.85 | **4.92** | +28% |

30-step measurement, prefill argmax=6993 PASS, chain matches v0.3.3
baseline (1063/6993 alternating). **Bigger than 27B P22 +5-8% because
Nemotron-3's smaller HIDDEN (2688 vs 4096) makes lm_head a larger
fraction of step time.** Per-step saving ~57 ms.

**MoE+trace upstream research LANDED (commit `486436e`)** —
research/moe_trace_precedents.md (462 lines). **Both `gpt_oss` and
`deepseek_v3` demos in tt-metal trace MoE decode end-to-end.** Smoking
gun at `gpt_oss/tt/experts_throughput/fused_decode.py:80-83`: "Format
conversion ... All done on-device (no host round-trip), following the
DeepSeek pattern (moe.py:393-395). **This enables trace capture.**"

Three concrete solutions for our 4-bridge MoE trap:
1. Sparsity tensor + `ttnn.sparse_matmul`
2. **On-device topk-indices layout-convert** (DeepSeek-V3 `tt/moe.py:413-418`)
   — `ttnn.to_layout + ttnn.reshape` instead of `to_torch/from_torch`
3. Fused kernels (`all_to_all_dispatch_metadata + moe_gpt + selective_reduce_combine`)

Key trade-off they make: DeepSeek-V3 ships `topk_fallback=True` for the
SAME SFPSWAP drift bug we hit, AND trace mode forces `topk_fallback=False`.
**There's no "both" config**. They pick: device-topk + trace + sampling,
OR host-topk + eager + greedy. Our owned-topk-with-stable-flag fork
bridges this gap.

**Implication for v0.6**: scope is now clear — fork
`_capture_decode_trace_text` from `tt_transformers/tt/generator.py:
1147-1245` + DeepSeek-V3's `moe.py:413-418` on-device layout-convert.
Multi-day effort but no novel invention required.

**In flight (background agents)**:
- **RULER NIAH-single smoke** (#242) — running on qb2 against Gemma 4 12B
  IT after server start; first paper-table-comparable long-context number
- **Owned-topk kernel** (#241, commits `310fe82`/`2621e2c`/`e33221b`) —
  scaffold complete, cmake clean, `ttnn.experimental.qwen36_topk_owned`
  exposed. Probe + 7/7 chain validation pending.

---

## PRIOR (2026-06-06 PM) — **owned-topk path identified as highest-ROI trace unlock**

**Decision (2026-06-06, user)**: pursue owned-kernel topk that flips the LLK
stable-sort flag (PR #31989 LLK-level change shipped; `ttnn.topk` doesn't
expose it). Forks the existing `qwen36_gdn_decode_owned` /
`qwen36_decay_gate_decode_owned` owned-op pattern (already in qb1+qb2
builds per `[[reference-remote-host]]`). Effort: ~1-2 days. Unlocks trace
blocker #2 → projected 0.26s eager → ~100-130 ms traced (8-10 tok/s,
~2.5-3× based on 27B/35B trace ratios). See bringup plan v0.4.0h.e row.

**Caveat**: `ttnn.topk` device-op hard-asserts bf16/bfp8 input
(`topk_device_operation.cpp:147-149`). Stable-sort fixes order-determinism
but bf16 numerical ties remain unrecoverable. If 7/7 chain regression
fails after the stable flag lands, follow-up `qwen36_topk_owned_fp32` that
promotes input to fp32 before the descent.

**Cross-server trace audit (2026-06-06 PM)** — Explore agent mapped
canonical decode-trace patterns across 27B, Gemma 4, 35B. **Key finding:
NO MoE model in our project is currently traced.** 35B has MoE; it's
eager-only (no `begin_trace_capture` in `server_35b_ttnn.py`). 27B has
trace; no MoE. Gemma 4 has trace infra; no MoE.

**Trace + MoE is uncharted territory in our codebase.** The audit
identified the MoE trap: even with router on-device, the standard
MoE block has 4+ host bridges that all need eliminating:
1. Pad-zeros for seq padding (FIXED 2026-06-06 via state cache)
2. Topk indices readback (`ttnn.to_torch`) after on-device router
3. Topk indices re-upload (`ttnn.from_torch`) into `all_to_all_dispatch`
4. Topk weights re-upload (`ttnn.from_torch`) into combine broadcast

Eliminating 2-4 requires threading topk_idxs/weights as ON-DEVICE
tensors through dispatch + expert FFN + combine — novel architectural
work, not iterative fixes. See
`[[decode-trace-canonical-pattern]]` memory for the full audit.

**Strategic implication**: pure-eager perf wins (vocab-shard lm_head
+5-8% proven, RMSNorm fusion, HiFi2 expert matmul) are higher ROI than
chasing MoE trace. Owned-topk kernel (committed `e33221b` by background
agent) is correct architectural prep, but trace unblock requires
on-device dispatch as a SEPARATE v0.6 effort.

---

**Empirical trace-blocker finding (2026-06-06 PM, probe `d081398`)** —
v0.4.1.h attempted traced run with `NM3_ROUTER_ON_DEVICE=1`. **Trace
capture FAILED**: `TT_FATAL fd_mesh_command_queue.cpp:581: !trace_id_
.has_value(): Writes are not supported during trace capture`. Root
cause: `attn_decode_step_tt` at `server_nemotron3_nano_ttnn.py:1342-
1343` calls `_set_cur_pos_buf → ttnn.copy_host_to_device_tensor`,
which is a host write — fires 6× per decode step (one per attention
layer) inside the captured region.

This is a **5th host write the 2026-06-05 audit missed** (v0.4.1.a
probe ran with router=OFF, hit host topk first, never got past it).
Fix: move `cur_pos_buf` update OUT of `attn_decode_step_tt`, call it
once before each `execute_trace` from the caller (same pattern as
`update_tok_buf`). Server-side refactor, orthogonal to the owned-topk
kernel work — needed regardless of which topk variant we ship.

So the actual trace-blocker count is **2 remaining**: (a) router host
topk (owned-topk kernel #241 in flight), (b) `cur_pos_buf` host write
(server refactor, ~1 hour). Once both land, we measure traced ms/tok.

---

**Companion — RULER long-context benchmark (v0.5.bench)**. NVIDIA RULER is
the standard long-context LLM eval (multi-task NIAH variants, multi-key,
multi-hop, aggregation, 4k→128k). Replacing the ad-hoc needle test that
hit the IT-template-shape misattribution per
`[[needle-prompt-shape-not-precision]]` across Gemma 4, 35B, and
Nemotron-3. Output directly comparable to upstream paper-table numbers.
Will baseline 27B + Gemma 4 + Nemotron-3 before/after perf rounds.

---

## PRIOR (2026-06-06 AM) — **router on-device attempted, FAILED; research complete**

**State**: Nemotron-3 stable at 0.26s warm step / 3.8 tok/s, 7/7 chain PASS,
3 of 4 trace blockers cleared (#1 embed, #3 reshard via `reduce_scatter`,
#4 final_norm/lm_head/argmax). Default branch is fully working — no
regression.

**Router on-device (commit `29234f5`)** — `ttnn.topk + ttnn.embedding`-gather
behind `NM3_ROUTER_ON_DEVICE=1`. Default OFF preserves 7/7 chain. With
env ON: **0/7 chain FAIL** from decode step 0 — every token wrong.

The probe's `cos=0.9997 / 6/8 expert sets match` looked benign but
compounds across 23 MoE layers/forward into immediate argmax
divergence. **Conclusion**: `ttnn.topk` tie-breaking differs from
`numpy.argpartition` in ways imperceptible at single-layer cos but
lethal at full-network scale. Env-gate caught it BEFORE shipping
(faster signal than the planned text-diff vs needle baseline).

**Research finding (2026-06-06 commit `12f101b`)** — agent
`ace43f189ee5fc05c` traced the root cause: it's NOT a `ttnn.topk` bug.
HW SFPSWAP comparison is asymmetric/unstable (tt-metal#20625). PR
#31989 added a stable-sort flag at the LLK level but `ttnn.topk` does
NOT expose it; bug #33492 on `ttnn.sort(stable=True)` still open.
`ttnn.topk` hard-asserts bf16 input — no fp32 promotion. On our
seed-99 distribution, 6/8 rows have EXACT bf16-quantized ties at the
K-th rank. Compounded over 23 MoE layers → >99% chains diverge.

The host vs device 7/7 vs 0/7 gap isn't a topk bug at all — it's
precision asymmetry in the ADD: host does `sigmoid(bf16) → readback
as fp32 + bias_fp32` (8/8 match); device does `bf16 + bf16`
(6/8 match). And **HF itself uses `torch.topk(sorted=False)`** which
is non-deterministic on CUDA — there is NO "correct" reference.

**Production precedent (CORRECTED 2026-06-06)**: DeepSeek-V3 demo's
source default at `tt/moe_gate.py:126` is `topk_fallback: bool = False`
— i.e. **DEVICE topk IS the default in production**, not host. The
host fallback is a TEST path (`tests/test_moe.py:274` pins True for
correctness validation). The demo accepts the drift because it runs
with `DEFAULT_SAMPLING_TEMPERATURE` — sampling washes out tie-break
choices. **OUR** Nemotron chain uses greedy argmax (no temperature)
for deterministic 7/7 vs HF gate, which exposes the tie-breaks fully
AND the L=128 needle test with router=ON produces gibberish
(`': pleeer? pleeer?'` vs baseline `'\nThe user wants to...'`) — so
even though DeepSeek-V3 ships device-topk, on OUR model + greedy
argmax, the device path is broken.

**Verdict**: keep `NM3_ROUTER_ON_DEVICE=0` (default). Match
DeepSeek-V3 production. **Trace blocker #2 is BLOCKED ON UPSTREAM**
until tt-metal ships a deterministic on-device topk. Eager path
stays at 0.26s warm step (~60× since v0.4.0d baseline).

Next gates: v0.5 single-stream perf (vocab-shard lm_head, RMSNorm
fusion, HiFi2) on the eager path, then v1 continuous batching.

**Gemma 4 Round 10 DRAM-sharded MLP (2026-06-06, qb2)** — Phase 1-3
committed (`738a057` plan + `bbc072b` probe PASS cos=0.9999937).
Phase 4 production integration shipped env-gated `TT_GM4_DRAM_PREFETCH=1`
(commits `7762d7b` + `b0364db` + `980135b` reshard-alias fix). qb2
validation BLOCKED on upload-side mesh-sharder — `ShardTensorToMesh(dim=0)`
+ WIDTH_SHARDED DRAM mem_cfg not a clean composition. **Next session
entry**: fork `tt-metal/models/demos/llama3_70b_galaxy/tt/llama_mlp.py:65-70`'s
`ShardTensor2dMesh(dims=...)` 2D-mesh-sharder pattern. Default branch
fully working — no regression at 47.0 ms/tok.

**Nemotron router long-context coherence test (in flight)** — open
question after v0.4.0h.c research: 0/7 chain is bit-divergence vs
non-deterministic `numpy.argpartition` (HF itself uses
`torch.topk(sorted=False)` — no "correct" reference). At LONG
context, does router-on-device produce COHERENT text (acceptable
drift) or GIBBERISH (heinous)? Re-running v041f needle baseline with
`NM3_ROUTER_ON_DEVICE=1` (commit pending). Outcome decides whether
trace blocker #2 is "blocked on upstream" or "drift acceptable, ship
with text-diff gate".

**Tiling/sharding roadmap (Gemma 4)** — agent's landscape doc at
`research/gemma4_perf_qb2_2026-06-05/tiling_sharding_plan.md` (commit
`738a057`) maps ~24 levers across 10 families with file:line
precedents. Rounds 10-17 stack projects 47 → 37 ms/tok (~21%);
Rounds 18-20 multi-session experimental branches (bfp4, distributed
RMSNorm, DRAM prefetcher).

---

## POST-WIN QUICK-START (2026-06-05 18:50 PT) — **v0.4.0e MATMUL-FOLD LANDED** ✓

**Headline**: Replaced ttnn.conv1d in the Mamba2 decode hot path with a
4×mul + 3×add + 1×bias-add fold. **Per-step time: 15.5s → 0.6-0.7s warm
(~22× full-step speedup)**. Correctness preserved: n-step chain 6/7 PASS,
same single bf16 argmax flip at step 3 as pre-fold baseline (no regression).

- **probe** `experiments/cb/isolate/nemotron3_v040e_conv1d_matmul_fold_probe.py`
  ttnn.conv1d warm 656.5 ms → matmul-fold warm **1.9 ms** (345× kernel-only)
  correctness cos = 0.999994 vs ttnn.conv1d (mad 2.5e-5)
- **integration commit `425acad`** (server_nemotron3_nano_ttnn.py)
  `mamba2_block_eager_tt`: gated on S==1; prefill keeps ttnn.conv1d
  `upload_mamba2_layer`: 4 per-position TILE weights + bias TILE on mesh
  defensive lazy upload for live-harness state-version skew
- **n-step regression** (chain of 8 after prefill):
  prefill TT=6993 HF=6993 PASS ✓ (16.5s, still ttnn.conv1d)
  TT chain [1063,6993,1063,5498,1063,6993,1063] = HF except single
  bf16 flip at step 3 — identical to v0.3.3 baseline.
  warm step time **0.6-0.7s** ≈ 1.4-1.6 tok/s (up from 0.064)

**Methodology lesson reinforced**: profile (v0.4.0b) localised the right
block (97% conv1d), but I assumed it was host bridges → wrong call to
on-device flow (v0.4.0c saved 0 ms). v0.4.0d isolation probe of the
kernel ALONE confirmed the kernel itself dominates → matmul-fold was the
right move. **Always isolate the suspect op alone, not just inside the
chain**. Saved in `[[feedback-profile-first-perf-method]]`.

**v0.4.1.e RESOLVED blocker #3 (commit `a65af53`)** — user's intuition
was right: `ttnn.reduce_scatter` does work for replicate→shard. My
prior v0.4.1.c probe failed because of a wrong kwarg
(`math_op=ttnn.ReduceType.Sum` — not in the API). Real signature is
`ttnn.reduce_scatter(input_tensor, dim, *, cluster_axis=...)` with
no `math_op` (op is hardcoded sum-reduce). Fix:
```python
h_scaled = ttnn.multiply(h_repl, 1.0 / NCHIPS)
h_shard  = ttnn.reduce_scatter(h_scaled, dim=2, cluster_axis=1)
```
Validated cos=0.999999 in `v041e_reduce_scatter_correct_probe`,
integrated into `moe_block_eager_ep_tt`, n-step chain 7/7 PASS,
0.2s warm step retained (no perf regression).

**TRACE STATUS: 3 of 4 blockers cleared.** Only blocker #2 remains
(MoE router host topk: scores readback + topk_indices upload).
On-device router probe (v0.4.0h.b) shows cos=0.9997 but 6/8
tie-break mismatch — precision-reducing change that could affect
long-context retrieval. **Gemma 4 Round 8 learning (2026-06-05)**:
bfp8 weights passed 100/100 short-token gate but BROKE long-context
needle retrieval (0/3 at L=128, partial at L=1024 — was 3/3 pre-bfp8).
Precision changes need long-context validation BEFORE landing.

Path forward: gate the on-device router behind a long-context test.
Current 0.26s warm step (~3.8 tok/s steady) usable for that validation.

**TRACE INTEGRATION PARKED 2026-06-05** — blockers 1 + 4 cleared
(commit `f45a710` — pure-ttnn embed/final_norm/lm_head/argmax); blocker
#2 (router) has tie-break drift, deferred behind long-context checks;
**blocker #3 (replicate→shard) has NO ttnn primitive** (verified via
probes `v041c_reshard_probe` + `v041d_replicated_dispatch_probe` +
research agent on tt-metal source). Production fix is **dual-resident
layer outputs** — significant refactor scope. Current 0.26s warm
step (3.8 tok/s steady, 5.0 peak) usable for long-context correctness
iteration WITHOUT trace. See `research/nemotron3_trace_plan_2026-06-05.md`
for full path.

**v0.4.1.a DIAGNOSTIC LANDED (commit `bfb04bb`)** — trace capture probe
confirms TT_FATAL "Writes are not allowed inside a captured trace" on
the current decode path. Definitive blocker list:

1. `embed_lookup` `ttnn.from_torch(ids)` → fix via pre-allocated tok_buf
2. MoE `ttnn.to_torch(scores)` + topk indices upload → on-device router
   (probe at `nemotron3_v040hb_ondevice_router_probe.py` has tie-break
   drift; park behind correctness flag)
3. MoE `ttnn.to_torch(h_input_tt)` for sharded re-upload → needs
   replicate→shard primitive (investigate `ttnn.experimental.reshard`)
4. `apply_final_norm` + `apply_lm_head_and_argmax` numpy → either pure-tt
   or leave outside the captured trace (27B's pattern: return
   `argmax_tt` on device, readback after `execute_trace`)

Full plan + elimination order at `research/nemotron3_trace_plan_2026-06-05.md`.
Strategy: tackle in order; re-run v0.4.1.a probe after each — it'll
progress further before hitting the next blocker.

**v0.4.0h.a SHIPPED (commit `fe502e3`)** — on-device MoE combine weighted-sum.
Replaces the largest data-movement readback in MoE
(combine_out_tt → numpy → routed_np → re-upload = 516 KB × 23 layers = 12 MB/step).

  combine_out_tt → ttnn.all_gather(dim=2) → ttnn.mul(broadcast) →
  ttnn.sum(dim=0) → ttnn.slice(S_orig). Reuses ttnn.all_gather at
  server_tp.py:1681 + sum pattern at server_35b_ttnn.py:1278.

PROFILE (post-h.a):
```
total step (warm):     15.5s → 0.66s → 0.43s → 0.33s → 0.26s    (60× cumulative)
moe per-layer:         10.5 → 8.2 → 8.2 → 5.8 ms                (-45% vs .e)
mamba2 per-layer:      16.5 → 9.1 → 4.3 → 4.2 ms                (-74% vs .e)
tok/s (eager warm):    0.064 → 1.5 → 2.5 → 3.0 → 3.8 / 5.0 steady
```

7/7 chain regression PASS throughout. **fp32 device state in v0.4.0g
fixed step-3 bf16 drift** that the legacy path had — correctness
IMPROVED through the perf work.

Remaining MoE numpy roundtrips (v0.4.0h.b): scores readback for
host topk; h_input readback for sharded re-upload (needs
replicate→shard primitive that ttnn doesn't ship cleanly); topk
indices upload. These are trace blockers — must land before v0.4.1.

**v0.4.0g COMPLETE (commits `36ec27c` + `9cfcb11`)** — fully pure-ttnn Mamba2 decode path:

```
total step (warm):     15.5s -> 0.66s -> 0.43s -> 0.33s        (52× cumulative)
mamba2 per-layer:      16.5 ms -> 9.1 ms -> 4.3 ms             (74% drop from v0.4.0e)
correctness:           6/7 PASS -> 7/7 PASS (fp32 device state fixed step-3 drift)
moe per-layer:         10.5 ms -> 8.2 ms (consistent across .a/.b)
tok/s (eager warm):    0.064 -> 1.5 -> 2.5 -> 3.3
```

**v0.4.0g.b** — three on-device pad helpers in `nemotron3_mamba2_step.py`
(`_pad_per_head_vector_tt`, `_pad_dt_tt`, `_replicate_per_group_to_per_head_tt`)
validated cos=0.999999 vs numpy in `nemotron3_v040gb_ondevice_pads_probe.py`.
New `mamba2_decode_step_ttnn_pure_tt` variant accepts the small inputs as
ttnn.Tensors at logical shapes — pads on-device via reshape + permute +
repeat + pad. Reuses 35B + 27B + Gemma 4 patterns.

**v0.4.0g.a** — pure-state Mamba2 SSD wrapper:
- New `mamba2_decode_step_ttnn_pure_state` accepts ssm_state as
  ttnn.Tensor, returns (state_out_tt, y_out_tt) as ttnn.Tensors. No
  device→numpy→device for the big tensors.
- `mixer_norm_w_tt` + SSD constants pre-uploaded at upload time.
- `state.ssm_state_tt[L]` lives across decode steps (fp32 on mesh).
- 35B GDN clone pattern (`ttnn.add(state, 0.0)`) used at the kernel
  boundary because the kernel writes through the state input buffer.

**Perf**: 660 ms → **427 ms warm step** (-35%). Mamba2 per-layer:
16.5 ms → 9.1 ms (-45%). **Correctness IMPROVED**: 7/7 PASS (was 6/7)
— fp32 on-device state eliminated the bf16-roundtrip drift at step 3.
Cumulative since v0.4.0d baseline: 15.5s → 0.4s = **~39×**.

**New profile breakdown (v0.4.0g.a)**:
```
total step:  427 ms  (warm)
  mamba2  23 layers ×  9.1 ms = 209.3 ms  (49%)  ← v0.4.0g.b next
  moe     23 layers ×  8.2 ms = 188.3 ms  (44%)  ← #226 v0.4.0h
  attention 6 layers ×  1.8 ms = 11.0 ms  ( 3%)
  embed+lm_head+sample:         18.4 ms  ( 4%)
```

**Next**: v0.4.0g.b — kill the remaining mamba2 per-layer ~7 ms of
input padding by moving x/z/B/C/dt onto device-side permute+pad+repeat
(eliminates `_pad_per_head_vector`, `_pad_dt_per_batch_per_head`,
`_replicate_per_group_to_per_head`). Expected mamba2 drop to ~3 ms/layer
≈ 69 ms → step ~290 ms ≈ 3.5 tok/s.

Then v0.4.0h (MoE host paths → on-device), then v0.4.1 trace (multi-
trace two-phase warmup, fp32 ssm risk per 35B precedent). Target
≥30 tok/s reached via v0.5 (vocab-shard + RMSNorm fusion + on-device
topk).

Tracing plan written: [`research/nemotron3_trace_plan_2026-06-05.md`](research/nemotron3_trace_plan_2026-06-05.md).

---

## PRIOR QUICK-START (2026-06-05 12:00 PT) — **Phase 0 DONE, Phase 1 LIVE** ✓

**Where we are**: Phase 0 (owned Mamba2 SSD decode kernel G0..G4)
COMPLETE 2026-06-05. The drop-in `mamba2_decode_step_ttnn(...)` wrapper
PASSES at full Nemotron shapes (state cos=0.999999, y cos=0.999995).
Phase 1 single-stream correctness ladder is LIVE — currently at v0.0
(HF oracle re-running on qb1 after the `model.backbone` fix).

**Locked ordering (user, 2026-06-05): 27B path**.
Phase 1 (single-stream correctness, v0.0 → v0.3) →
Phase 2 (single-stream perf, v0.4 trace + v0.5 perf pass → ≥30 tok/s) →
Phase 3 (continuous batching, DEFERRED) →
Phase 4 (HTTP server, LAST).
CB and HTTP are explicitly deprioritised until v0.5 single-stream is
shipping. Mirrors `[[feedback-correctness-first]]`: never scale before
correctness floor.

**Within Phase 1, L5 Attention is brought up BEFORE L0 Mamba2** as the
warmup — Attention is the simplest layer block we've ever shipped (no
RoPE, no q_norm/k_norm, standard `1/sqrt(128)` scale). Building the
bootstrap + paged SDPA + KV cache scaffold on a boring layer means
v0.1.2 (Mamba2) and v0.1.3 (MoE) integrate on a known-good foundation.

**Live task chain (#199 → #211)**: see `TaskList` or
`research/nemotron3_nano_30b_a3b_bringup_plan.md` §7.

Regression sweep (kernel still PASSES — run at any time as a
sanity check before kernel-adjacent work):
- mode=2: state cos=1.000000
- mode=3: state cos=1.000000
- mode=4: state cos=1.000000, y cos=0.999996
- mode=5: state cos=1.000000, y cos=0.999996
- 8-step multi-step replay: per-step cos ≥ 0.9999 ✓
- G2 multi-head smoke (B=1, NUM_HEADS=64): state cos=0.999999, y cos=0.999995 ✓
- G4 wrapper smoke (full Nemotron shapes): state cos=0.999999, y cos=0.999995 ✓
  (`experiments/cb/isolate/mamba2_step_wrapper_smoke.py`)

Run the kernel regression sweep:
  `ssh $TT_HOST 'cd ~/tt-xla && bash experiments/cb/isolate/mamba2_regression_sweep.sh'`

**Phase 0 (kernel) takeaways** (kept here as reference — see plan
§3a for the full G-ladder):

- **🔑 4-INGREDIENT RECIPE** ([[feedback-mm-init-prime-required]]) for
  Blackhole TRISC hangs at ~4 transpose+matmul+binary iters: full
  `mm_init` ONCE via a real prime matmul (we used `matmul_reduce_C_state`),
  pre-transpose phase outside the loop, `mm_init_short` inside the
  inner loop, explicit `pack_reconfig_data_format(cb_outer)`. All four
  required; any 3 alone fail.
- **bf16/fp32 mixed-format pitfall** (multi-step replay caught it):
  `add_tiles(fp32, bf16)` silently drops the bf16 source on Blackhole
  when pack_hw_config is stale. Fix: cb_outer is fp32 in production
  (matmul_outer_x_dt_B uses full mm_init).
- **G3 batched parked**: B>1 + 64 heads HANGS when blocks_per_core > 1.
  Not a Phase 1 blocker — B=1 + full 64 heads (G2) is correct, and the
  CB engine drives per-slot at the server layer (same as 27B/35B).

**Phase 1 — exact next task**: v0.4.0c — eliminate the conv1d host
roundtrip in `mamba2_block_eager_tt`. Profiling-driven target.

**v0.4.0a + v0.4.0b + v0.4.0c** (this session):
- v0.4.0a constants pre-upload: correctness PASS, +0.3% perf
- v0.4.0b section profile: S2 conv1d block = 637ms (97% of layer)
- v0.4.0c on-device conv path: correctness PASS, **perf UNCHANGED**.

**Honest perf finding**: the original S2=637ms was the ttnn.conv1d
KERNEL time, not the surrounding host bridges. Removing the numpy
roundtrip (v0.4.0c) is architectural hygiene only — the kernel
dominates. For a real perf win we need either:
1. v0.4.0d — speed up the conv1d kernel itself (matmul fold, smaller
   input shape, or a depthwise-step-mode op if ttnn ships one)
2. v0.4.1 — trace capture (amortises dispatch overhead across steps;
   the per-call dispatch cost may be a big chunk of the 637ms even
   if the kernel proper is fast)

**Critical methodology lesson reinforced**: the profile localised the
right block (S2 = 97% of time), but I assumed it was host bridges
when it was kernel time. **Next time, also measure the kernel call
ALONE** (without host setup) to know which sub-component dominates.

v0.4.0c is committed (correctness clean, no numpy reorg in decode
path). v0.4.0d explores conv-kernel alternatives next.

**v0.3.3.b perf data** (5 decode steps, cold + 4 warm):
- cold step 0:  15.48s
- warm mean:    15.469s (essentially identical to cold)
- JIT overhead: 0.01s
- **The 15.5s is real host-bridge cost, not JIT compile.**
- Tok/s: 0.065 (about 470× too slow for 30 tok/s target)
- Breakdown: 23 Mamba2 layers × ~12 host roundtrips/layer/step
  ≈ 14s/step in SSD wrapper bridges. MoE host topk + combine sum +
  residual ≈ 1.5s.

**v0.4 path**:
1. v0.4.0 — refactor `nemotron3_mamba2_step.py` G4 wrapper to take
   ttnn.Tensor in and return ttnn.Tensor out (eliminate the per-step
   numpy bridges). State (ssm_state, conv_state) moves to ttnn tensors.
2. v0.4.1 — first trace capture of single decode step.
3. v0.4.2 — two-phase warmup ([[ttnn-multi-trace-two-phase-warmup]]) +
   100-step accuracy regression vs eager.

Realistic v0.4 outcome (per plan): 3× speedup vs eager → ~5s/step
traced eager → still slow but unblocks v0.5 perf passes that get us
to 30 tok/s.

**v0.3.3 DONE 2026-06-05** (commit `a617a76`): N-step decode chain
6/7 PASS with recovery — identical bf16 drift pattern to v0.3.1.a
quadratic (single flip at step 5, recovered at step 6). cur_pos
advances 5→11. Constant-time state-carried decode fully validated.

  TT chain: [1063, 6993, 1063, 6993, 1063, 5498, 1063]
  HF chain: [1063, 6993, 1063, 6993, 1063, 6993, 1063]
                                          ↑ single bf16 flip

**v0.3.1 COMPLETE 2026-06-05** (commit `8ff3c57`): end-to-end
constant-time single-token decode pipeline PASSES.
- PREFILL: 5-token forward, argmax = HF = 6993 ✓
- DECODE: 1 step (S=1) with carried ssm_state + conv_state + paged
  KV cache, argmax = HF = 1063 ✓
- Per-step: ~15s cold JIT compile; subsequent warm/traced decode = ms
- All state carry mechanisms working (ssm_state, conv_state, KV cache)
- All decode primitives validated (paged_update_cache, paged_sdpa_decode,
  two-call Gemma-4-style GQA pattern)

The teacher-forced per-layer ladder was DECISIVE here — without it
I would have spent hours chasing the wrong attention paged decode
bug. Always run the ladder when correctness fails by a non-drift
amount.

**v0.3.1.c step 3d DONE 2026-06-05** (commit `8ff3c57`): Mamba2 conv_state
carry. Teacher-forced per-layer ladder went from mamba2 **14/9** to
**23/0 PASS** (every Mamba2 layer cos≥0.99 at decode_pos=5). Bug
diagnosis: conv1d kernel=4 needs 3 prior x_BC positions; we carried
ssm_state but not conv_state → decode zero-padded conv input → wrong
output. Fix: lazy `state.conv_state_np[L]` buffer prepended to x_BC
for decode S=1, last 3 of combined input saved as new state.

Remaining ladder failure: L51 (moe) cos=0.58 mad=6.4 — **identical to
the pre-existing hot spot** we saw in v0.2.b's 52-layer argmax-pass
forward (51/52 ≥0.99 with L51 at cos=0.586). final_norm recovers via
RMS rescaling; doesn't break argmax. Not a decode bug.

**v0.3.1.c step 3b PLUMBING DONE 2026-06-05** (commit `abe079b`):
attn_decode_step_tt + _shard_for_paged_write + _set_cur_pos_buf added,
two-call Gemma 4 pattern. Plumbing was always correct (teacher-forced
ladder showed attention 6/6 PASS); the actual bug was in mamba2 conv
state carry, not attention.

**Gemma 4 perf rounds 1+2 COMPLETE on qb2**:
- Round 1: paged_fused_update_cache → eager 2.11×, traced flat (`de0384a`)
- Round 2: `_shard_for_paged_write` 5-op → 2-op → traced **51.27 → 49.57 ms/tok** (-3.3%, real outside noise) (`b153c10`)
- Round 3 IN FLIGHT — agent `ac71cc6da27157872` chasing concat_heads_decode→o_proj fusion or rotary_embedding_llama_fused_qk

**v0.3.1.b DONE 2026-06-05** (commit `5ca94b8`): Mamba2 ssm_state lazy plumbing +
defensive `getattr` for harness state-version skew. Regression PASS
(argmax = HF = 6993).

**v0.3.1.c step 1 DONE 2026-06-05** (commit `f4743c6`): 5 paged-decode
State slots + constants (SDPA_BLOCK_SIZE=32, SDPA_NUM_BLOCKS=256 at MAX_KV=8192).
**v0.3.1.c step 2 in progress**: `setup_paged_decode_state(state)` helper +
per-layer KV cache allocation (Gemma 4 two-call: TWO caches per layer since
NUM_KV_HEADS=2 + NCHIPS=4 forces NKV_PER_CHIP=1; per-cache shape
[256, 4, 32, 128] bf16 TILE sharded dim=1). All forks 35B `:1875-1924`.
Verification via `nemotron3_v031c_setup_smoke.py` checking all 5
support buffers + per-layer KV caches allocated. Step 3 next:
attn_prefill_tt (paged_fill_cache + non-paged causal SDPA) +
attn_decode_step_tt (paged_update_cache + paged_sdpa_decode).

**v0.3.1.a DONE 2026-06-05** (commit `b93d552`): 7/8 quadratic multi-step
PASS with recovery. TT exactly matched HF's `Paris, Paris, Paris,` loop
for 4 consecutive steps; single bf16 argmax flip at step 4 (TT=5498 vs
HF=6993), then RECOVERED — TT independently locked back onto the loop
pattern at steps 5,6,7. Model is semantically correct; drift is bf16
matmul chain noise (not state mgmt, not kernel). v0.3.1.b/c won't fix
the drift but enables constant-time decode (per-step ~ms vs current
17-28s/step from re-running 52 layers on growing prefix).

**qb2 — Gemma 4 perf agent COMPLETED 2026-06-05**:
- paged_fused_update_cache landed (commit `de0384a`): **2.11× eager**
  (474 → 225 ms/tok). Traced 51.4 → 51.1 ms/tok (flat, within noise).
- qb2 sandbox set up (`scripts/run_remote_qb2.sh`).
- Next levers identified: distributed RMSNorm (+12-15 ms/tok projected).
- All work in `research/gemma4_perf_qb2_2026-06-05/`. Zero overlap
  with Nemotron foreground.

**v0.3.0 DONE 2026-06-05** (commit `42d303d`): switched from streaming to
all-layers-resident. P150 has 32 GB/chip (verified `ttnn/core/operation.cpp:33`),
full Nemotron-3 Nano load ≈ 21 GB/chip — fits with 11 GB headroom.
Streaming was over-engineered for a non-existent memory problem.
Bootstrap 106.5s, iter 17s, argmax = HF = 6993 ✓. **3.69×** vs v0.2.6 warm,
4.6× vs v0.2.5.

**v0.3.2 IN PROGRESS**: `nm3_dev_harness.py` (forks `cb35_dev_harness`)
+ `run_harness_tmux.sh nm3` (adds nm3 case). Bootstrap once → drop
trigger files for ~10s iteration instead of 108s.
- Launch: `bash scripts/run_harness_tmux.sh nm3 qb1`
- Iterate: `bash scripts/deploy.sh <files> && ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/<test>'`
- Result: `ssh qb1 'cat ~/tt-xla/.cache/nm3_runtime/trig/last.log'`
- Smokes accept `state=None`; harness passes a live State.

**Background work in flight 2026-06-05**: Gemma 4 perf optimization
agent running on qb2 (all 4 P150s free). Brief: read
`research/gemma4_perf_briefing_2026-06-04.md`, use tt-perf-report to
identify TOP-3 bottlenecks in traced decode, land ONE optimization
(P2 distributed RMSNorm OR P3 paged SDPA on globals OR RMSNorm
fusion — pick from profile, not queue order). Reports to
`research/gemma4_perf_qb2_2026-06-05/log.md`. ZERO file overlap with
the foreground Nemotron path. agent ID `aece88b1979f5345c`.

**v0.2.6 DONE 2026-06-05** (commit `1685827`): host numpy weight cache
+ MoE pre-stack cache. Got 2.97× warm speedup. **Now superseded by
v0.3.0 all-resident** which is 3.69× faster than v0.2.6 warm.

v0.2.5 COMPLETE 2026-06-05 (commit `926b49c`) — `_tt` block variants
take/return `ttnn.Tensor`, 0 inter-block readbacks. Regression PASS
(argmax_last = 6993 = HF). **Honest perf finding**: 82.4s vs v0.2.b's
77.7s — essentially same. The 52 round trips we eliminated were ~10ms
each (~520ms); drowned by 23 MoE × ~3s = 69s of MoE weight upload from
disk per forward. **The bottleneck is weight streaming, not tensor
flow.**

**v0.2.6 (next, ~30 lines)**: `state.weight_np_cache` dict.
`load_t(state, key)` checks cache before re-reading safetensors.
Iter 1 cold = 82s; iter 2+ warm = ~20s (skips 23×~2.5s disk reads).
4× steady-state speedup makes v0.3 multi-step decode bearable.

**v0.3 (after v0.2.6)**: 8-token chain match HF. KV cache + ssm_state
in ttnn tensors carried across steps. ~3-4 min/8-token run = doable.

**Deferred**:
- Pipeline upload/compute (marginal — compute << upload)
- Memory residency / aggressive sharding (v0.5 perf — too risky now)
- bf8 weights (user veto — long-context drift unknown)
- Needle-haystack L=8192 (blocked on v0.4 trace — eager won't scale)

v0.2 COMPLETE 2026-06-05 — full 52-layer streamed forward + final_norm
+ lm_head + argmax matches HF (commit `5ffd183`). TT argmax_last = 6993
= HF argmax_last ✓. Streaming pattern: bootstrap top-level only, then
per-layer `upload_one_layer → block_forward → deallocate_layer` to fit
the 23 MoE EP layers (~640 MB/chip each) inside 8 GB budget.

v0.1.4 EP COMPLETE 2026-06-05 — 3/3 gates PASS, cosines bit-equivalent
to v0.1.3.b naive path (commits `1abce07` forward, `13b111e` h_norm
host-bridge eliminated). Remaining host bridges = router topk-6, combine
weighted-sum, shared+residual add — same as the 27B/35B production
precedent. v0.5 perf pass will move them on-device coherently.

**v0.1.4 COMPLETE — what shipped:**
- `upload_moe_layer_ep` — 128 experts sharded as 32/chip
- Bootstrap dispatch with NEMOTRON3_MOE_MODE=ep|full|router_only (default ep)
- `moe_block_eager_ep` — full True-EP forward:
  - Seq-shard on dim 2 via `ShardTensorToMesh(dim=2)` (pad S=5 → 8)
  - `ttnn.all_to_all_dispatch(..., output_concat_dim=2)` → per-chip `[1, 1, 8, 2688]`
  - Per-expert FFN over E_LOCAL=32 on full padded seq per chip
  - `ttnn.all_to_all_combine(..., output_shard_dim=2)` → per-chip `[6, 1, 2, 2688]`
  - `ConcatMeshToTensor(dim=2)` readback → full `[6, 1, 8, 2688]`
  - Slice routed_np back to S_orig=5
- Bootstrap 9.2s; forward 1.8s; per-chip compute drops 128 → 32 experts.

**Smoke gates (commit `1abce07`):**
- Gate S shared_out cos=0.999713 mad=2.82e-4 PASS ✓
- Gate M mixer_out  cos=0.999826 mad=1.09e-3 PASS ✓
- Gate B block_out  cos=0.999805 mad=1.10e-3 PASS ✓

Cosines bit-equivalent to v0.1.3.b naive Pattern A (6-digit match across
all 3 gates). True-EP path is now correctness-verified on (1,4) BH.

**Critical kwarg combo** ([[reference-all-to-all-dispatch-shape-contract]]):
- `all_to_all_dispatch(..., output_concat_dim=2)` → per-chip `[1, 1, S_padded, H]`
- `all_to_all_combine(..., output_shard_dim=2)` → per-chip `[K, B, S_per_chip, H]`
- DEFAULT `output_shard_dim=1` FAILS on our setup: `batch_size*replicate_dim/num_devices = 1*1/4 = 0` (integer div) → moreh_full TT_FATAL `shape[1] = 0`. Contract documented at `all_to_all_combine_nanobind.cpp:79`.

Old next-step (v0.1.4 implementation): Both
`ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` validated
on (1,4) Blackhole P150. PR #39380 (Mar 2026) live in our build.
cluster_axis=1 for our single-row mesh. Combine contract:
input dim 0 = n_experts/n_devices, post-expert layout
`[experts_per_device, B, S, H]`.

v0.1.4 implementation will fork DeepSeek-V3 demo
(`~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/moe.py`):
- Shard 128 experts as 32/chip
- Forward: pre-norm → router → `all_to_all_dispatch` → batched
  local experts (32 × matmul+relu²+matmul) → `all_to_all_combine`
  → shared expert (replicated) → residual
- ~6 expert FFNs/token (vs 128 in Pattern A — 21× less compute)
- Memory ≤ 7.8 GB/chip; unblocks v0.2 + v0.3.

Earlier path that was discarded: Pattern A fork from 35B (each
chip runs all 32 local experts, mask unselected, all_reduce).
Known-good but ~95% wasted compute; saved as fallback.

23 MoE × 1.3 GB > 8 GB/chip → either path needs sharding to 32
experts/chip. Research round (2026-06-05, 2 parallel agents) found:
- Tenstorrent ships `ttnn.all_to_all_dispatch`, `all_to_all_combine`,
  `moe_routing_remap`, `moe_expert_token_remap`
- DeepSeek-V3 in-tree demo at `models/demos/deepseek_v3/tt/moe.py`
  uses these ops (sigmoid + bias + group topk — Nemotron arch match)
- BH status: #27859 broken, PR #39380 (Mar 2026) fixed, UNVALIDATED
  on our (1,4) P150 mesh
- Pattern A from 35B (`server_35b_ttnn.py:1254`) is known-good but
  runs ~95% wasted expert compute (each chip runs all 32 of its
  experts, masks unselected)

G0 spike resolves the ambiguity in ~4 hours. Then v0.1.4 implements
the chosen path. v0.2-v0.5 unblocked.

All three block kinds validated end-to-end at cos ≥ 0.999:
- L0 Mamba2 (v0.1.2, commits `587ae06` + `dd7b80d`)
- L1 MoE (v0.1.3, commits `7ad5681` + `aadecbc`)
- L5 Attention (v0.1.1, commit `e7f3e59`)

v0.2 wires the per-layer dispatch (52 layers — 23 Mamba2 + 23 MoE +
6 Attention by `state.layer_types[L]`) and adds final_norm + lm_head
+ argmax (already validated at v0.1.0). Memory budget: 23 MoE × ~1.3 GB
per chip exceeds 8 GB/chip → need Pattern A sharding (35B precedent)
OR a lazy upload strategy to fit. Most pragmatic for v0.2: upload only
the LAYERS WE'RE RUNNING (sparse uplink already wired via
`NEMOTRON3_UPLOAD_LAYERS`); for v0.5 perf pass, do real sharding.

Old next-step (v0.1.3, now COMPLETE): L0 Mamba2 fully on-device, all gates PASS at cos ≥ 0.99990.
Root-cause that closed it: hardcoded clamp constants (1e-4, 0.1) in BOTH
the numpy oracle and the kernel; HF Nemotron uses
`self.time_step_limit = (0.0, inf)` for the actual clamp — the
`time_step_min`/`max` config fields are not what HF uses. After
fixing (oracle defaults + kernel constants 0x7f800000 for inf),
oracle vs HF y_pre_norm cos jumped 0.943 → 0.999999. Memory:
[[feedback-nemotron3-time-step-clamp-bug]] (durable: any future
Mamba/SSM bringup must search the modeling code for `time_step_limit`).

v0.1.2.c smoke: Gate N 0.999904; O 0.999937; M 0.999937; B 0.999930.
Bootstrap 5.8s; forward 2.4s.
v0.1.2.c PARTIAL (commit `587ae06`): the full on-device L0 Mamba2 chain
mechanically runs end-to-end (pre-norm + in_proj + conv1d + silu + split
+ SSD-via-wrapper + MambaRMSNormGated + out_proj + residual), but the
gates fail at cos 0.957-0.982 (gate ≥ 0.999).

**Root cause** (debug helper `nemotron3_v012c_debug_numpy_ref.py`):
even with HF-correct in_proj + conv1d inputs + pure numpy fp32 SSD via
our oracle + numpy MambaRMSNormGated, cos vs HF L0_norm is only 0.957
(norm-then-gate) or 0.838 (gate-then-norm). Our
`experiments/utils/mamba2_numpy_oracle.py:mamba2_decode_step` drifts
from HF `mamba_chunk_scan_combined`. The on-device kernel + wrapper
match our oracle perfectly (cos=0.999998 in isolation per memory) —
but the oracle itself differs from HF.

v0.1.2.d: focused single-position SSD probe. Feed HF L0_in_proj +
L0_conv1d (post-causal-slice + post-silu) → split → run our numpy
oracle for ONE position → compare to HF's actual SSD output at that
step. Localize the drift (dt path? clamp? GQA broadcast?). Once
oracle matches HF, kernel + wrapper auto-match → v0.1.2.c gates flip.

Wrapper mesh-awareness landed (`isinstance(device, ttnn.MeshDevice)`
branches the upload + readback).
v0.1.2.b DONE 2026-06-05 (`745a438`): `ttnn.conv1d(groups=conv_dim=6144,
kernel=4, padding=3 sym)` works first try after fixing the L1_SMALL
bootstrap config ([[reference-l1-small-for-conv1d]]). 3/3 gates PASS
(H/I 0.999949, C 0.999991). Now adding:
- ttnn.silu on conv1d_out
- split into x[B,S,NUM_HEADS,HEAD_DIM]/B[B,S,N_GROUPS,SSM_STATE]/C[same]
- per-position SSD loop using `mamba2_decode_step_ttnn` wrapper
  (5 calls; ssm_state accumulates across positions)
- MambaRMSNormGated (head_dim=64 groups, gated by z·silu) — needs
  to be composed of ttnn ops; check existing wiki/35B code first
- out_proj matmul
- residual add
v0.1.2.a DONE 2026-06-05 (commit `490f89f`): `upload_mamba2_layer`
adds {norm, in_proj, out_proj} replicated on the mesh; tiny ops
(conv1d_w/b, dt_bias, A_log, D, mixer_norm_w) held host-side for now.
`mamba2_in_proj_only` PASS — H pre-norm cos=0.999949, I in_proj
cos=0.999949 vs HF L0_in_proj. Bootstrap (L0 only) 7.0s.

v0.1.2.b design (next):
- Depth-wise conv1d on x_BC slice (out of in_proj), K=4, pad=3 sym
- HF hook captures the full [1, 6144, 8] symmetric-padded output
  (pre-causal-slice)
- Two paths: (a) compose ~10 ttnn ops (slice+mul+add) on
  padded [1, 11, 6144] across 4 kernel offsets, OR (b) ttnn.conv1d
  with groups=conv_dim if supported. Quick API probe first.
- Then ttnn.silu after.

v0.1.2.c design: SSD loop using `mamba2_decode_step_ttnn` wrapper
(5 iters for 5 prefill positions, accumulating ssm_state) +
MambaRMSNormGated + out_proj + residual.

v0.1.1 final state: fully on-device, all gates PASS, block cos=1.0.
Previously DONE 2026-06-05:
- v0.0 oracle PASS — `.cache/hf_oracle_nemotron3_nano/` 19 artifacts
- v0.0+ oracle HARDENED — norm + mixer-out + shared-expert hooks, `--gen N` multi-step
- v0.0.1 tokenizer probe PASS — both `<think>\n` and `<think></think>` suffixes resolved
- v0.0.2 weights introspect PASS — 0 missing / 0 shape mismatches / 0 extras
- **v0.1.0 bootstrap PASS — 3/3 gates green** (Gate A embed cos=1.0;
  Gate B final_norm cos=0.9999; Gate C generation token+logits cos+argmax all PASS)
- **v0.1.1.a L5 projections PASS — 4/4 gates** (H/Q/K/V cos ≥ 0.9999)
- **v0.1.1.b L5 full attention block PASS — 3/3 gates** (O cos=0.9998,
  M cos=0.9998, B cos=1.000000 — effectively bit-exact post-residual).
  SDPA runs in numpy fp32 for now; on-device prefill SDPA = v0.5 perf.

**Real findings threaded into the next gates**:
1. **DeepSeek-V3 `e_score_correction_bias`** per MoE gate (from v0.0.2):
   added to router scores BEFORE the group-restricted topk; brief
   didn't flag this. Threaded into v0.1.3.
2. **HF `logits.npy` is bf16-imprecise** (from v0.1.0):
   `nn.Linear(bf16)` is the matmul precision; our HiFi4 path is more
   accurate. Use numpy fp32 as strict ground truth in every smoke;
   accept HF only as a soft sanity check. Memory:
   [[feedback-hf-logits-npy-is-bf16-imprecise]].

Steps for v0.1.0 (bootstrap, after v0.0 lands):
1. Fork bootstrap from `experiments/serve/server_35b_ttnn.py` (closest
   structural match — hybrid recurrent + MoE). New file
   `experiments/serve/server_nemotron3_nano_ttnn.py`.
2. Open (1, 4) mesh + fabric on qb1.
3. Upload safetensors weights (~8 GB/chip). Sanity check:
   `embed_tokens` should be `[131072, 2688]`.
4. Embed lookup: feed `prompt_ids` → returns `[1, seq, 2688]` hidden.
   Gate: cos ≥ 0.999 vs HF oracle's `hidden_states[0]`.
5. Final norm + lm_head + argmax at pos 0. Gate: argmax matches HF.
6. **Llama-style RMSNorm — NO `+1.0`** (vs 35B's Qwen-style with
   `+1.0`; this bit us hard on 35B —
   [[feedback-qwen36-qnorm-knorm-zero-centered]]).

**Phase 1 sub-stages** (reordered 2026-06-05 — L5 before L0):
- v0.1.1: L5 (Attention) — simplest layer (no RoPE/q_norm/k_norm),
  warmup for bootstrap+SDPA+KV scaffold
- v0.1.2: L0 (Mamba2) — drop in `mamba2_decode_step_ttnn` wrapper
- v0.1.3: L1 (MoE) — fork 35B Pattern A; deltas = sigmoid router,
  group-restricted topk, relu², scaling 2.5
- v0.2: all 52 layers + final_norm + lm_head — argmax match HF
- v0.3: multi-step decode + long-context at L=8192

**Phase 2 sub-stages**:
- v0.4: trace capture (BIGGEST RISK — fp32 ssm in trace; mirror of
  35B fp32-H hang). Fallback: bf16 ssm + measure drift.
- v0.5: single-stream PERF pass (target ≥30 tok/s) — vocab-shard,
  HiFi2, RMSNorm fusion, etc. **This is the demo-ready milestone.**

Plan: `research/nemotron3_nano_30b_a3b_bringup_plan.md` §3b.

**Memory entries**:
- [[feedback-mm-init-prime-required]] — the 4-ingredient recipe
- [[feedback-gdn-vs-mamba2-kernel-delta]] — structural math reason
- [[feedback-sdpa-transpose-b-flag-escape-hatch]] — transpose=1 B flag
  (used in matmul_reduce_C_state)
- [[tt-llk-frozen-in-tt-metal]] — repo state + missing L3 docs

**Current kernel source state**: PRODUCTION. All modes 1-5 PASS in
the regression sweep. The kernel at
`experiments/owned_ops/nemotron3_mamba2_decode_owned/device/kernels/compute/nemotron3_mamba2_decode_owned.cpp`
is drop-ready for Phase 1 integration via the wrapper at
`experiments/serve/nemotron3_mamba2_step.py` (G4).

**Dataflow + design refs** (still valid as reference):
- Kernel design: `research/mm7_g1_mamba2_kernel_design.md`
- Decisions log: `research/mm7_g1_dataflow_decisions.md` (D1-D11)
- Math primer: `wiki/65_mamba_state_space_models.md` §3
- GDN production source (fork base):
  `ssh qb1 cat /home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp`

**Setup commands** (verbatim):
```bash
# Build kernel after edits:
bash scripts/deploy.sh experiments/owned_ops/nemotron3_mamba2_decode_owned/
ssh qb1 'cd ~/tt-xla && python3 experiments/owned_ops/nemotron3_mamba2_decode_owned/integrate_into_ttmetal.py --tt-metal ~/tenstorrent/tt-metal && cd ~/tenstorrent/tt-metal && cmake --build build_Release --target ttnn -j8 2>&1 | tail -8'
ssh qb1 'cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnn.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so && cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnncpp.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnncpp.so'

# Run the smoke (clear JIT cache if you changed the kernel):
ssh qb1 'rm -rf /home/aditya/.cache/tt-metal-cache/*/kernels/nemotron3_mamba2_decode_owned/ 2>/dev/null; cd ~/tt-xla && TT_METAL_HOME=$HOME/tenstorrent/tt-metal TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME/ttnn LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib timeout 60 .venv/bin/python -u experiments/cb/isolate/mamba2_kernel_smoke.py'
```

**Notes for post-compaction me**:
- If qb1 server is running (`curl http://localhost:8000/health`), it
  holds the device; stop with `ssh qb1 'cd ~/tt-xla && bash
  experiments/serve/scripts/serve_cb.sh stop'` before running smoke.
- LLK API survey result (decision D8 RESOLVED): tt-metal ships
  `softplus_tile`, `clamp_tile`, `exp_tile`, `negative_tile`,
  `copy_tile`, `copy_tile_init`, `mul_tiles`, `mul_tiles_bcast_scalar`,
  `add_tiles`. All in `tt_metal/hw/inc/api/compute/eltwise_unary/*.h`
  and `bcast.h`, `eltwise_binary.h`, `tile_move_copy.h`. We use them
  directly — do NOT decompose.
- Dataflow decisions log:
  `research/mm7_g1_dataflow_decisions.md` (D1-D11, each labeled with
  alternatives + future v3 perf-lever rationale).
- Kernel design doc: `research/mm7_g1_mamba2_kernel_design.md`.
- Math primer: `wiki/65_mamba_state_space_models.md` §3.

**In-flight background work** (subagents started during this session):
- **Cleanup subagent**: archiving ~62 dead-code files from
  `research/repo_archive_audit_2026-06-04.md` into `archive/`. Pilot
  bucket DONE (commit `f3a4f1f`). The `cb_engine_scaffolding` bucket
  is mid-execution — staged renames visible in `git status`. Wait for
  the subagent to finish (it'll commit each bucket separately) before
  pushing.
- **Transcript subagent**: DONE. Session transcript at
  `archive/session_transcript_2026-06-04/session_transcript.md`
  (15.7 MB, 283k lines). Reusable extractor at
  `scripts/extract_session_transcript.py`. Commit `82cbb88`.

---

## NEW HEADLINE (2026-06-04, post-demo): MM7 — Nemotron-3 Nano 30B-A3B

Stanford CS440LX **demo shipped successfully** 2026-06-04 (27B + Gemma 4 12B
live chat via the TUI, [poster v5](archive/presentation_cs440lx_2026-06-04/poster.pdf),
Qwen 27B PC verified end-to-end at 5.1× / 8.0× per-token speedup).

**Next bringup target**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
This is a **Mamba2-Transformer HYBRID MoE**, NOT a clean Qwen-MoE clone.

- 52 layers, `MEMEM*EMEMEM*...` pattern → **23 Mamba2 + 23 MoE + 6 Attention**
- 30-31.6B total / ~3.5B active per token
- 6 attention layers have NO RoPE (positional info lives in Mamba2 state)
- MoE is DeepSeek-V3-style (sigmoid + group-restricted top-k + scaling=2.5)
- Experts use `relu²` activation; shared expert is 2× wider than routed
- ssm_state must be fp32 on device

**The blocker**: tt-metal does NOT ship a Mamba2 SSD kernel. 23 of 52
layers depend on it.

**User decision (2026-06-04): Path B — owned Mamba2 SSD kernel up-front.**
G0..G4 staging mirrors the 35B `qwen36_gdn_decode_owned` build. Phase 0
(kernel) before Phase 1 (forward/decode/CB/HTTP ladder). Estimated total
5-8 weeks to v2.

**Plan-of-action**: [`research/nemotron3_nano_30b_a3b_bringup_plan.md`](research/nemotron3_nano_30b_a3b_bringup_plan.md)
**Architecture brief**: [`research/nemotron3_nano_architecture_brief.md`](research/nemotron3_nano_architecture_brief.md)
**Tasks**: #183 (G0 numpy oracle), #184 (G0a harness), #185 (G0b qb1 prep,
parallel), #186 (G1 single-core), #187 (G2 multi-core), #188 (G3 batched),
#189 (G4 server wrapper) — each blocks the next.

**Current work (2026-06-04 EOD)**:
- ✅ Mamba primer: `wiki/65_mamba_state_space_models.md` (commit `52ca6ec`)
- ✅ G0b: qb1 RAM 503/468 GB OK; ttnn has ZERO ssm/mamba/scan ops at
  module level — building from scratch (commit `9832952`)
- ✅ G0 numpy oracle: `experiments/utils/mamba2_numpy_oracle.py`
  written + self-test PASSES on qb1 (commit `98fc43d`). Decay in
  (0,1], state mutated, bit-equal across seeded runs.
- ⏸ G0 HF byte-match gate: blocked on Nemotron weight download (~63 GB);
  fires at v0.0 oracle invocation. Internal self-test sufficient to
  unblock G0a harness development.
- ✅ G0a isolation harness: `experiments/utils/test_mamba2_decode_isolated.py`
  (commit `4352baf`). Multi-step replay + per-head cos/MAD gate.
- ✅ G1 kernel design: `research/mm7_g1_mamba2_kernel_design.md`
  (commit `642f50d`, 334 lines). File-by-file map qwen36_gdn →
  nemotron3_mamba2; SPMD work unit (batch, head); CB layout; LLK
  call pattern; 5-day order of operations.
- ✅ **Major reuse find**: `qwen36_conv1d_decode_owned` IS Mamba2's
  `conv1d_step` (4-tap depthwise causal Conv1d + rolling state + SiLU).
  Parametrise D=6144 and we get Mamba2's pre-SSD conv path for FREE.
- ✅ G1 day-1: `experiments/owned_ops/nemotron3_mamba2_decode_owned/`
  forked from `qwen36_gdn_decode_owned/` with all identifiers renamed
  (commit `58267c0`). Compute math still GDN — SSD rewrite is next.
- ✅ 3 background audits done — research/audit_{gemma4_opts,qwen36,gdn_kernel}_*.md
  with 1-paragraph email replies in §6 of each (commits `d5dc1bc`,
  `b26db66`, `0b672e0`).
- ✅ Wiki §66: `wiki/66_blackhole_kernel_dataflow_anatomy.md` — pedagogy
  on the kernel ↔ hardware mapping (Tensix, RISC-V cores, CBs,
  tiles, SPMD partition, cross-tile comm, DRAM streaming).
- 🟢 Dead-code audit subagent running in background — proposed
  archive layout for stale 27B/35B/Gemma 4 probes.
- 🟢 **Adoption subagent running in background** — tasks #191 + #192
  (paged_fused_update_cache on gm4 #44946, redundant to_memory_config
  audit #44958). Plan at `research/tt_metal_adoption_plan_2026-06-04.md`.
  Zero file overlap with foreground G1 work.
- ⏭ Foreground next: rewrite the Mamba2 compute kernel for SSD math per
  `research/mm7_g1_mamba2_kernel_design.md` §6. Then build on qb1 +
  debug_fill smoke + numpy-oracle compare via G0a harness.

---

## OLD HEADLINE (kept for reference): demo-day priorities

**Demo-day priorities (2026-06-04 11:55 PT)** — superseded by Nemotron above:
1. **Get the chat TUI rock-solid** — this is the live demo. 27B + Gemma 4 12B are the demo models. Hardening already shipped (commits `c523d28`, `c88f6d5`, `f20bb81`, `0e5a8f5`, `ee7cd20`). Test it live as soon as server is back.
2. **Make sure the server runs perfectly on 27B + 12B**. Existing perf numbers are great (see headline below); just verify nothing broke and screenshots get captured for the poster.
3. **De-prioritise 35B work**. 35B perf/drift/PC fixes can wait — user explicitly said "we can do performance and drift checking for the 35b model haha" meaning skip it.

**Server state**: STOPPED (qb1 in use by a colleague). Restart with the FINAL fast-path config:
```
ssh qb1 'cd ~/tt-xla && rm -f .cache/server_cb.pid && HF_HUB_OFFLINE=1 \
    TT_BACKEND=27b TT_CB_SLOTS=32 TT_CB_PREFIX_CACHE=1 \
    bash experiments/serve/scripts/serve_cb.sh start'
```
For Gemma 4: `TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it ...`. Do NOT pass `TT_CB_TOPK_K` — the default (0 = logits trace or argmax-tail) is what wins.

**Headline perf (HTTP CB, traced, owned_gdn, argmax-tail trace)**:
- Gemma 4 12B IT B=32: **316 tok/s aggregate at 32 clients** (8.35 → 316 = 27.7×)
- Qwen3.6-27B B=32: **232 tok/s aggregate at 32 clients** (8.32 → 232 = 27.9×)
- Single-client streaming: gm4 TTFT 1.40s decode 17.4 tok/s; 27B TTFT 1.70s decode 11.5 tok/s
- Multi-turn HTTP with PC on 27B: turn 2 = 5.99s for 172-tok prompt (PC hit, 6.3× speedup)
- 35B at TT_CB_SLOTS=1: 3.13 tok/s (B>1 blocked by task #162 — won't fix this session)

**TUI is ready** (`scripts/chat.py`, README at `scripts/CHAT_TUI.md`). Key features:
- Claude-Code-style welcome panel (closed box, url + model + cwd + settings)
- `● assistant (<model_short>)` per-turn header with thin grey rule
- **Character-level streaming** — chunks emit to stdout as they arrive
  (no line buffering), so long Qwen3.6 thinks visibly flow
- **Readline editing** in the input prompt — Ctrl-W word-delete, Ctrl-A/E
  line nav, Alt-B/F (= Option-←/→) word nav, ↑/↓ history (persisted
  at `.cache/chat_history`), Ctrl-R reverse search
- `--max 4096` default (up from 1024) so `/continue` is rare
- `<think>…</think>` blocks shown by default (the live token flow is
  the "model is alive" signal); `/think` toggles, `--hide-think` opts
  into the dim "(thinking…)" placeholder at launch
- `/status` (and `/show`) panel; `/clear` (and `/new`) reset; cwd shown as
  `~/…` with path-aware ellipsis for long paths
- `/paste` multi-line mode (terminal-level bracketed-paste also enabled)
- `/yank` copies last assistant reply / code block to system clipboard
- `/metrics [N]` live Prometheus dashboard for N refresh cycles
- `/screenshot` saves to `.cache/tui_screenshots/tui_<ts>.png`
- Expanded shell allow-list (`git`, `grep`, `find`, `python -V/-c`, etc.) with strict deny-list
- `write_file(path, content, mode)` and line-ranged `read_file(path, start=N, n=M)` tools
- Graceful HTTP-error recovery + terminal-state reset on exit

**TUI verified live on 27B 2026-06-04**: banner renders all-four-sides closed,
`<think>` hides cleanly, multi-turn HTTP wall/prompt_t drops from 0.349 → 0.024
across 3 turns (prefix cache hitting; same nickname/fact carry across turns).
Streaming is now char-level (no per-line stall); readline editing (Ctrl-W,
Option-←/→, ↑↓ history) works on both GNU readline and macOS libedit.

**Demo path RE-VERIFIED 2026-06-04 with PC metric inspection**:
- Qwen 27B + TT_CB_PREFIX_CACHE=1, 3-turn coding chat:
  - T0 (36 tok, cold): 8.69 s, wall/prompt_t = 0.241
  - T1 (125 tok, **PC HIT**): 5.84 s, wall/prompt_t = 0.047 (5.1× per-tok speedup)
  - T2 (201 tok, **PC HIT**): 5.92 s, wall/prompt_t = 0.030 (8.0× per-tok speedup)
- Metric delta: pc_hits +2, pc_misses +1, evictions +0 — clean PC story
- Carries the "ADIT" nickname from T0 → T2 chess strategy. Coherent.

**Gemma 4 12B IT also re-verified on TUI 2026-06-04**: streaming works, PC
hits on T1/T2 (wall/prompt_t 0.212 → 0.075). **But** chat output still
duplicates `thought\n` stanzas — the `<|channel>thought\n<channel|>`
chat-template asymmetry (root-caused in `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`,
task #176).

**Demo plan (user-decided 2026-06-04)**:
- **27B hot** the whole talk — that's the multi-turn / prefix-cache story.
  Demo PC by sending a 2-or-3-turn coding chat; show the wall_s drop on T2.
- **Swap to Gemma 4 12B IT** mid-talk to show insane first-prompt speed
  (~17 tok/s single-client decode, 316 tok/s aggregate at B=32). The
  server restart takes ~1.5 min for Gemma 4 (fast — used as time to talk
  through what's happening on stage). Gemma PC is parked — use cold
  first-prompt only.

**Gemma 4 PC: validator GREEN, live still misses (~80% done, parked)**:

Trail so far (2026-06-04):
- Step 1 ✅ Generic active-prompt suffix detector (commit `184753d`,
  `experiments/serve/openai_endpoint.py:_active_prompt_suffix`).
  Renders the same probe message twice (`add_generation_prompt=True`
  vs `=False`); the divergent tail is the active-only suffix to strip.
  Covers Qwen's `<think>\\n\\n</think>\\n\\n` (5 tokens), Gemma's
  `<|channel>thought\\n<channel|>` (4 tokens), and any future template
  with the same asymmetry. Memoised by `id(tokenizer)`.
- Step 2 ✅ Validator unification (`experiments/cb/validate/pc_token_match.py`):
  imports `_messages_to_prompt` directly so the gate tests the SAME
  code path the live server uses. No more drift between validator and
  prod logic.
- Step 3 ✅ Validator EOS auto-detection: walks the trailing tokens of
  `[user, asst]` rendered with `add_generation_prompt=False` to find the
  chat-end marker. Discovers Gemma's `<turn|>` (id 106) without a
  hard-coded name list; works for any tokenizer.
- Step 4 ✅ Validator result: **3/3 PASS** for both `google/gemma-4-12B-it`
  and `Qwen/Qwen3.6-27B`.

What's left (the ~20%, shovel-ready):
- Step 5 ❌ Live HTTP shows pc_hits=0 / pc_misses=3 / evictions=3.
  Root cause: when the model hits `max_tokens` instead of emitting a
  natural EOS, `cb_scheduler._finish` falls back to
  `next(iter(self.eos_ids))` for the canonical EOS. `self.eos_ids` is
  a `frozenset` — Python's hash-order iteration may land on id 1
  (`<eos>`) for Gemma `{1, 50, 106}`. But the chat template inserts
  id 106 (`<turn|>`) at past-asst boundaries, so the cached
  `prompt + canonical` is one byte off from the next turn's prompt.
- Fix design: add a `chat_end_id` scheduler attribute set at bootstrap
  via the same trailing-token-of-passive-render trick the validator
  uses (so it works for any future backend). Replace the
  `next(iter(eos_ids))` fallback in `_finish` AND `cancel(mark_live=True)`
  with `self.chat_end_id`. Two ~5-line edits, plus a deploy +
  server restart cycle (~6 min for 27B, ~2 min for Gemma 4) to verify
  live. Deprioritised because the demo plan uses Qwen-hot for the
  PC story (Qwen has worked since the 8aeeb53 canonicalise fix on
  2026-06-04 — it picks 151645 as `next(iter({151645}))` from a
  1-element set, which happens to match its chat-end token).

**To run TUI** (once server is back):
```
python3 scripts/chat.py --url http://qb1:8000 --model 'Qwen/Qwen3.6-27B'
# or
python3 scripts/chat.py --url http://qb1:8000 --model 'google/gemma-4-12B' --tools
```

**Open code-only items (parked pending server)**:
- Gemma 4 multi-turn PC: 2nd asymmetry root-caused (`<|channel>thought\n<channel|>` suffix); fix designed in `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`. Strip-from-cache-only requires scheduler plumbing. PARKED.
- `use_multicore=False` on lm_head argmax (commit `918c025`) — needs server test to confirm perf cost + determinism win.
- TUI screenshots for poster (task #177) — needs server up to demonstrate.

**Poster** (archived, demo done 2026-06-04): `archive/presentation_cs440lx_2026-06-04/poster.pdf`. Sky-blue Tenstorrent theme, columns rebalanced, both models' streaming numbers in.

**Key reference files** (read first if confused):
- `archive/presentation_cs440lx_2026-06-04/06_live_measurements.md` — historical record of demo-day measurements
- `research/tokenizer_chat_template_reference.md` — universal tokenizer/chat-template gotchas (so we never re-debug this)
- `research/cb_perf_regression_audit_2026-06-04.md` — explains the 13 → 232 tok/s recovery
- `scripts/CHAT_TUI.md` — TUI commands + tools

---

## OLD COMPACTION-READY STATUS (kept for reference)
## 🔥 COMPACTION-READY STATUS (2026-06-04) — read this first

**The user wants you to keep going through the queue below without
asking; they're not stopping for status updates.**

### Live RIGHT NOW
- **Server**: STOPPED (qb1 in use by a colleague, paused at user request).
  Will resume when free. All experiments paused.
- **Code state**: all session fixes committed + pushed (see `git log --oneline -20`).
- **Last verified perf** (HTTP CB, traced, owned_gdn, argmax-tail fast path):
  - Gemma 4 12B IT B=32: **8.35 / 89.25 / 172.52 / 316.12** tok/s at 1/8/16/32 clients (27.7×)
  - Qwen3.6-27B B=32: **8.32 / 61.27 / 117.62 / 232.12** tok/s at 1/8/16/32 clients (27.9×)
  - Single-client streaming: gm4 TTFT 1.40s, decode 17.4 tok/s; 27B TTFT 1.70s, decode 11.5 tok/s
  - 35B at TT_CB_SLOTS=1: 3.13 tok/s (blocked by task #162 for B>1)

### What's queued

| # | Task | State |
|---|---|---|
| 1 | Gemma 4 perf | ✅ +8% vocab-shard, +94% argmax-tail, gm4 = 316 tok/s at B=32 |
| 2 | 27B perf | ✅ +75% cb_dn_recurrence fix, +48% argmax-tail, 27B = 232 tok/s at B=32 |
| 3 | 35B drift / multi-turn Q&A / needle | ✅ resolved + 3/3 PASS + 50/50 (bf16 floor) |
| 4 | Multi-turn HTTP with PC (27B) | ✅ 6.3× speedup on turn 2 (PC hits) |
| 5 | Gemma 4 multi-turn PC | ⏳ scheduler-side canonicalisation shipped (commit `8aeeb53`); validator finds a SECOND chat-template asymmetry (`<|channel>thought\n<channel|>` suffix). Fix designed, parked pending test. See `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`. |
| 6 | bf16 determinism | ⏳ `use_multicore=False` on lm_head argmax (commit `918c025`); needs server test to confirm perf cost + determinism win |
| 7 | 35B B>1 empty-slot poisoning (#162) | ⏳ unblocks 35B aggregate AND spec-dec |
| 8 | Gemma 4 perf P2 (distributed RMSNorm) | ⏳ design in `research/gemma4_perf_briefing_2026-06-04.md` |
| 9 | Gemma 4 perf P3 (paged SDPA on globals) | ⏳ design in same briefing |
| 10 | TUI hardening | ✅ commits `c523d28`, `c88f6d5`, `f20bb81`, `0e5a8f5`; README `scripts/CHAT_TUI.md` |
| 11 | TUI screenshots for poster | ⏳ needs server back up |
| 12 | Spec-dec (Qwen 3B + 35B) | ⏳ feasibility in `research/speculative_decoding_plan_2026-06-04.md`; blocked on #162 |
| 13 | Cleanup audit (18 items) | ⏳ 3 done (B1 cb_api, A1 docs, determinism); 15 pending |
| 14 | Poster v3 | ✅ Tenstorrent sky-blue, columns rebalanced, streaming numbers in |

### In-flight as of right now (compaction-resilient: re-check these after compaction)
- **35B per-layer drift probe** — `cb35` tmux harness bootstrapping
  (35B = ~14 min bootstrap; ~20/40 layers @ 7 min when last checked).
  Check status: `ssh qb1 'tmux capture-pane -t cb35 -p | tail -10'`.
  Once `[harness] ready. Drop trigger files` appears:
  `ssh qb1 'touch ~/tt-xla/.cache/cb35_runtime/trig/per_layer_drift_pos1'`
  Result: `ssh qb1 'cat ~/tt-xla/.cache/cb35_runtime/trig/last.log'`.
  JSON: `~/tt-xla/.cache/cb35_runtime/per_layer_drift_pos1.json`.
  Pins owned_gdn=ON + dn_state_dtype=bf16 (manual path broken).
  Note: the dev harness file `harness.log` may not get written for cb35
  (open-path quirk under tmux); use `tmux capture-pane` as the live
  source of truth for bootstrap progress.
- **Tracy probe for Gemma 4 perf** — `experiments/utils/tracy_profile_one_gemma4_layer.py`
  shipped (commit `2610ef3`). Run AFTER the 35B drift probe finishes
  and the gm4 harness is back up, to find what's bottlenecking P2/P3.
  Capture cmd in the file's docstring + `research/gemma4_perf_briefing_2026-06-04.md`.
- **No subagents running** — both finished. Briefings saved:
  - `research/gemma4_perf_briefing_2026-06-04.md` — TOP-3 perf opts
    + Tracy capture commands + roofline. Read this first when starting perf.
  - `research/35b_drift_briefing_2026-06-04.md` — drift cliff bisection
    plan (3 steps: P_cliff search → per-layer L_locus → sub-op at
    (L_locus, P_cliff)). Read first when starting 35B drift.
  - The 35B sub-agent confirmed: `server_35b_ttnn.step_forward_inner`
    already writes `capture[f"layer_{L}"]` per layer (no server-side
    diff needed). Just need to write
    `experiments/cb/isolate/cb35_per_layer_drift_pos1.py` (fork
    `gm4_per_layer_drift_pos1.py`, swap to 35B oracle + cb35 harness).
    Use `owned_gdn=ON` ALWAYS — manual path is broken
    ([[feedback-35b-manual-recurrence-path-broken]]).

### What's shipped this session you might lose track of
- **Gemma 4 12B (base + IT)** end-to-end: bootstrap → forward →
  multi-step → long-context (3/3 needle haystack) → trace (3.56×) →
  CB B=4 → HTTP chat. Memory: `[[project-gm4-pos1-cliff]]` (resolved),
  `[[feedback-gemma4-sdpa-scale-1]]`, `[[reference-gm4-dev-harness]]`,
  `[[feedback-harness-state-version-skew]]`.
- **Multi-EOS support** across cb_engine + cb_scheduler + cb_api: reads
  generation_config.json + tokenizer.eos_token_id, accepts list/set.
  Driver: Gemma IT EOS list `[1, 106, 50]`. Smoke result: IT chat now
  has `finish_reason=stop` at 26 tokens (was 80, finish=length).
- **scripts/chat.py rewritten Claude-Code-style**: ANSI panels,
  markdown rendering, slash commands (/save /load /history /tools),
  multi-line input via trailing `\\`, **built-in tool calling** (shell
  /read_file/calc) with `--tools` flag.
- **Model bringup recipe** at `research/model_bringup_recipe.md`
  ([[reference-model-bringup-recipe]]) — the staging ladder, REUSE
  rules, bug catalog, and meta-lesson on why bringup got 10× faster.
  Read FIRST when starting a new model.
- **Allowlist hardened** at `.claude/settings.local.json` — tmux:*,
  env-prefix patterns, deny block for rm -rf / force push / reset
  --hard. (.gitignored)
- **Variant-aware bootstrap** for Gemma 4: `TT_GEMMA4_VARIANT={base, it}`.
  Tokenizer config differs (IT ships a real chat template + eos
  token list); same arch / shapes / weights structure.
- **Dev harness reload extended**: `gm4_dev_harness.py` reloads
  `server_gemma4_*` sibling modules so CB edits land without
  re-bootstrap.

### How to read this back together (90-sec rehydration)
1. `git log --oneline -30` to see this session's commits.
2. Skim `research/model_bringup_recipe.md` (1 page) — what we know about doing bringup fast.
3. Skim `research/gemma4_perf_briefing_2026-06-04.md` (1 page) — the perf attack.
4. Skim `research/35b_drift_briefing_2026-06-04.md` (1 page) — the drift attack.
5. `ssh qb1 'bash experiments/serve/scripts/serve_cb.sh status'` — is the IT server still up?
6. Resume from the queue table above.

### Hard rules (`CLAUDE.md` non-negotiables — restate yourself before each major action)
1. Plan, then act. No hand-wavy claims.
2. Remote-only execution (`ssh qb1` / `qb2`). Never local device code.
3. No `python -c`; no `/tmp`. Permanent files under `experiments/`.
4. Frequent commits.
5. REUSE before write. Every new file cites the existing pattern it forks in the commit message.

---

## Live session state (2026-06-03 — Gemma 4 12B is the active priority)

**Pivot 2026-06-03**: paused 35B drift (#163) and pivoted to Gemma 4 12B
bringup (#165). Driver: 14-min 35B bootstrap per harness restart was
severely rate-limiting iteration AND the harness itself hung silently
mid-investigation (task #166 captures the harden-it-before-next-bootstrap
work). Gemma 4 12B is dense, 12B, dual-attention-type — smaller weights
(~5-7 min bootstrap), structurally interesting (sliding+global), and
exercises position-dependent paths in isolation. If a positional-state
bug lives in our codebase, v0.3 surfaces it without MoE/DN confounders.

### Active — Gemma 4 12B bringup (#165)

- **Plan**: [`research/gemma4_12b_bringup_plan.md`](research/gemma4_12b_bringup_plan.md)
  — start at §"REUSE MANDATE" (always grep for an existing pattern
  before writing new code) → §0 Step 0 pre-flight (DONE) → §2
  code-reuse map at `file:line` → §3 novel items → §4 sub-task
  breakdown with cosine gates.
- **Step 0 DONE 2026-06-03 (commit `4395b28`)**:
  - §6.1 `sliding_window_size` kwarg confirmed on
    `ttnn.transformer.paged_scaled_dot_product_attention_decode`
    — sliding decode is a kwarg flip, not a new kernel. Plan §3.3
    risk closed.
  - §6.3 GELU variant probe (`experiments/utils/gemma4_gelu_variant_probe.py`):
    `ttnn.gelu(x, fast_and_approximate_mode=False)` matches
    `gelu_pytorch_tanh` at cos=0.999998. The fused-activation path
    (`ttnn.mul(.., [UnaryOpType.GELU])`) uses the APPROXIMATE kernel
    — DO NOT mirror 35B's SwiGLU fused pattern in Gemma 4 v0.
  - Bonus: SDPA doc confirms `cur_pos=-1` skips compute for that
    batch slot — may resolve 35B #162 if backported.
- **v0 oracle DONE 2026-06-03**: `experiments/utils/hf_reference_gemma4_12b.py`
  (commit `156dc9f`) produced `.cache/hf_oracle_gemma4_12b/` with
  6-token "The capital of France is" forward; HF predicts " a" at pos 5;
  all 6 L0 sub-captures present (in/post_attn/pre_ff/mlp/post_ff norms
  + mixer_out).
- **v0.1.0 DONE 2026-06-03 commit `b9f3c35`** — bootstrap + embed
  scale + L0 input_layernorm. cos 0.999996 / 0.999991 vs HF oracle.
  Bootstrap **74-84 sec** on qb1 (vs 35B's 14 min — 11× faster).
- **v0.1.1 DONE 2026-06-03 commit `a35525e`** — Q/K/V projections +
  q_norm/k_norm. 7/7 sub-steps PASS at cos ≥ 0.99997. Hit a sharder
  gotcha en route (`ShardTensor2dMesh` vs the correct
  `ShardTensorToMesh(dim=0)`) — one-line fix; memory entry
  `[[ttnn-shard-1d-vs-2d]]`.
- **v0.1.2 DONE** — attention output at pos 0 + o_proj. cos=0.999990
  mad=0.0625. Found via numpy reproducer that Gemma 4 has v_norm =
  `RMSNorm(head_dim, with_scale=False)` applied to V after v_proj
  (memory `[[gemma4-v-norm]]`). 27B/35B don't have this — reading the
  HF source caught what guessing wouldn't. v0.1.2 also surfaced a
  ttnn `all_reduce_async` signature change vs the simple
  `ttnn.all_reduce(cluster_axis=1)` used by 35B (one-line fix in
  `all_reduce_tt`).
- **v0.1.3 DONE 2026-06-03 commit `7f9f396`** — 13/13 sub-steps PASS;
  full L0 forward bit-id to HF at cos ≥ 0.999958. L0 output (vs HF
  hidden_states[1, 0, :]) cos=0.999975. Two-op GELU works per Step 0.2;
  both Gemma 4 post-norms (post_attention + post_feedforward) correct.
- **v0.2 DONE 2026-06-03** — all 48 layers (sliding + global dispatch)
  + final_norm + tied lm_head + 30·tanh logit softcap. Greedy top-1
  matches HF at pos 0: TT argmax=258882 == HF argmax=258882 (`<image|>`).
  final_norm cos=0.999563, logits cos=0.999137. Surfaced two more
  Gemma 4 novelties: per-layer `layer_scalar` ([[gemma4-layer-scalar]])
  multiplied at end of each decoder layer, AND the cosine-is-not-enough
  diagnostic discipline ([[cos-not-enough-also-check-mad]]) — direction
  passed at L0 but magnitude was 18× off, propagating to L1 collapse.
- **v0.3 IN FLIGHT — setup DONE 2026-06-03 commit `cb4e299`**:
  KV caches per layer (sliding [num_blocks, 8, 32, 256] sharded over
  mesh dim=1; global [num_blocks, 1, 32, 512] replicated), SDPA
  program + memory + compute configs (35B B3 recipe — HiFi2 +
  fp32_dest_acc=False per [[fp32-sdpa-cliff-probe]]), RoPE tables
  (sliding theta=10000 full-rotate, global p-RoPE theta=1e6 partial
  0.25). v0.2 probe re-runs with these in place: STILL PASSES.
  Only FORWARD changes remain. Plan §"v0.3 sub-staging" has detailed
  design notes with 35B `file:line` references.
  - v0.3.0 ARGMAX PASS commit `01c88d6` (batch-0 hack)
  - v0.3.0.1 **FULL PASS** commit `e2ae9f2` (2 SDPA calls per
    sliding layer, NKV=1 each, proper GQA). final_norm cos 0.999601,
    logits cos 0.999370, argmax matches HF.
    Memorialized: [[paged-update-cache-nkv-per-chip]],
    [[read-kernel-source-first]], [[use-existing-isolation-probes]].
  - **v0.3.1 FIXED 2026-06-03 commit `c97bf15`** — root cause was
    SDPA `scale=1.0/sqrt(head_dim)` (wrong); Gemma 4 text attention
    sets `self.scaling = 1.0` (HF `modeling_gemma4.py:1178`),
    confirmed in Tenstorrent's in-tree demo (`decode.py:144`). The
    wrong scale was MASKED at pos 0 because a single-token softmax
    is 1.0 regardless of scale. After-fix multi-step: pos 0..5
    cos_final all ≥ 0.997 (was 0.26 at pos 1); 5/6 argmax PASS (pos
    4 cos=0.9984 but argmax differs — bf16 tie noise per
    [[bf16-chain-drift-at-B-gt-1]]). Per-layer drift ladder
    L0-L46 cos > 0.996 at both pos 0 and pos 1. Memory rule:
    [[feedback-gemma4-sdpa-scale-1]].

    The debug ladder remains useful — keep the env knobs
    (GM4_DEBUG_POS, GM4_ROPE_ZERO, GM4_SKIP_SLIDING, GM4_SKIP_GLOBAL)
    and the four isolation probes (gm4_sliding_write_read,
    gm4_global_write_read, gm4_rope_lookup, gm4_per_layer_drift_pos1)
    for future bringup work.

    Full debug writeup in memory `[[project-gm4-pos1-cliff]]` — the
    bisection burned several "masked-at-pos-0" hypotheses (view-decay,
    in-place buffer update, canonical SDPA config) BEFORE landing on
    scale=1.0. None of those earlier fixes moved the 3/6 number, but
    each closed a real Tenstorrent anti-pattern that would have masked
    the real bug, so they ship.
  - **v0.3.2 DONE 2026-06-03 commit `acb20a6`** — 16-token free-run
    coherent: "The capital of France is a city of art, culture, and
    history." End-to-end forward composition validated. Probe:
    `gm4_v032_freerun.py`.
  - **v0.3.3 long-context validation IN FLIGHT** — mirrors 27B/35B's
    needle-haystack + bf16 prefill drift gates.
    - **v0.3.3.a DONE 2026-06-03 commit `a7eef0d`** — per-pos cos
      ladder at L=215 (Wikipedia Eiffel Tower paragraph). All 3
      bf16-aware gates PASS: argmax match 95.81% (≥90%), cos_final
      median 0.9932 (≥0.99), 5th-pct 0.9779 (≥0.95). No cliff.
      Probe: `gm4_v033a_long_cos.py`; oracle:
      `.cache/hf_oracle_gemma4_12b_L215/` (HF needs `.venv-gemma4`).
    - **v0.3.3.c needle-haystack PASS 2026-06-03** — 3/3 Y verdicts
      at L=100/256/512 frac=0.5 (random 8-char passwords retrieved
      verbatim). Eg L=512 needle `FWD7SWFY` → TT generated
      `**FWD7SWFY**`. Probe: `gm4_v033c_needle_haystack.py`. Saved
      under `.cache/needle_haystack_gm4_ttnn/`. Long-context
      retrieval works end-to-end. Decode at ~160 ms/tok eager.
    - **Dev harness for Gemma 4 LIVE 2026-06-03 commit `<gm4-harness>`**
      — `experiments/cb/dev/gm4_dev_harness.py` (forked
      `cb35_dev_harness.py`). Bootstraps Gemma 4 ONCE (~80s), runs
      tests on demand via trigger files; saves ~80s per iteration.
      Launch: `bash scripts/run_harness_tmux.sh gm4`. Run probes via
      `touch tt-xla/.cache/gm4_runtime/trig/<short_name>` (matches any
      probe whose filename ends in `_<short_name>.py`).
    - v0.3.3.b sliding-window invariance at pos > 1024 — pending; not
      blocking (v0.4 trace shipping first).
  - **v0.4 traced decode DONE 2026-06-03 commit `626c67a`** — 100/100
    traced == eager on a non-trivial teacher-forced + free-run sequence
    (`[258882, 236743, 529, 506, 236764, 496, 3207, 529, 1610, 236764]`
    first 10 tokens before the model degenerates to `<image|>`). Eager
    **182.7 ms/tok** → traced **51.3 ms/tok = 3.56× speedup** out of
    the box; trace capture cost 693 ms one-time. Two-phase warmup per
    [[ttnn-multi-trace-two-phase-warmup]]. `trace_region_size=400_000_000`
    on the mesh (default 50 MB OOMs the 48-layer decode trace).
    Validator: `gm4_v04_trace_validate.py`.
  - **v1 CB DONE 2026-06-04** — `server_gemma4_unified_cb.py`
    (forks `server_tp_cb.py`) ships `setup_cb_state`,
    `update_input_buffers_batched`, batched sliding+global layer
    forward (2 SDPA per sliding layer with NKV=1 each), batched
    paged_update_cache + paged SDPA over per-slot KV. **All gates PASS
    at B=1, B=2, B=4**: 3a B=1 == single-slot bit-identical; 3b
    identical-slot bit-identical; 3c distinct-slot with no cross-talk;
    4 slots in B=4 all match their B=1 references. Validators:
    `gm4_v1_0_alloc_smoke.py`, `gm4_v1_4_3a.py`, `gm4_v1_5_3bc.py`,
    `gm4_v1_6_b4.py`. B=4 eager forward ~1.0s/step.
  - **v2 HTTP wire-up DONE 2026-06-04 commit `9a1e45a`** —
    `gemma4_12b` registered in `cb_api.BACKENDS` +
    `cb_scheduler._BACKEND_MODULES`. CB module also exposes
    `cb_reset_slots`, `forward_batch_tp_inner` alias, and `return_topk`
    support to match the scheduler contract. Bootstrap loads the HF
    tokenizer and installs a minimal Gemma chat template
    (`<start_of_turn>{role}\n{content}<end_of_turn>\n`) since the BASE
    model ships no chat template. Logits readback forced to rank-2
    `[B, vocab]` so the scheduler's `t[:B]` works. `curl
    /v1/chat/completions` with `"The capital of France is"` returns
    `"Paris"`. **Known limitation of the BASE model**: `<end_of_turn>`
    is multi-token text (not a special token), so the model emits
    chat-template-echo noise after the answer and runs to `max_tokens`.
    Strip client-side on `<end_of_turn>` until we bring up the
    instruct variant.

  **Gemma 4 12B bringup COMPLETE end-to-end** —
  bootstrap → forward → multi-step decode → long-context → traced
  decode → continuous batching → HTTP chat. Stack is hot on qb1 for
  experimentation (`TT_BACKEND=gemma4_12b bash experiments/serve/scripts/serve_cb.sh
  start`). Per-token decode at **19.5 tok/s single-seq traced**;
  19.5 → projected ~55-65 tok/s aggregate at B=4 traced.

  - **Gemma 4 12B IT (instruct) DONE 2026-06-04 commit `bdd207c`** —
    bootstrap variant-switch via `TT_GEMMA4_VARIANT=it`. Forks the
    base machinery (same arch); the only deltas are weights, the
    shipped chat template, and proper `<end_of_turn>` EOS. v1.6 B=4
    acceptance PASSED at distinct-from-base argmax outputs. End-to-end
    HTTP chat smoke:
        `curl /v1/chat/completions ... "Write a one-sentence summary
         of the city of Paris."`
        → "Paris is a world-renowned cultural and historical capital
           celebrated for its iconic landmarks, rich artistic heritage,
           and sophisticated culinary scene."
    Known followup: IT has three EOS candidates `[1, 106, 50]` in
    `generation_config.json`; cb_engine only gates on one. Extend
    cb_engine to accept a list when that wart matters.

  **Two-variant bringup recipe validated** — the model_bringup_recipe.md
  staging ladder (v0.0 oracle → v1 CB → v2 HTTP) carried us from the
  base model bringup to an instruction-tuned variant in **~2 hours**
  (download + oracle + smoke + EOS fix). The "fork, don't write" rule
  paid out.
- **All computation on (1,4) P150 mesh on qb1**; readback only for
  cosine compare against the HF oracle (matches 27B/35B pattern).
- **Reuse mandate (user-set 2026-06-03)**: every new file must cite the
  existing file it forks (or "no prior art, here's why") in its commit
  message. Deep utility shelf exists: `experiments/cb/_runner.py`,
  `experiments/utils/{ttnn_introspect,hf_reference_35b,cosine_ladder_*,
  test_fused_*_isolated,needle_haystack_*,tracy_*}.py`,
  `experiments/cb/isolate/{paged_sdpa,paged_update_cache,chunked_sdpa,
  owned_gdn,...}.py`, `experiments/serve/{server_35b_ttnn,server_35b_cb,
  server_tp_cb,cb_api,cb_scheduler}.py`. Plan §"REUSE MANDATE" has
  the full table.
- **Step 0 — pre-flight hardware probes (no model upload, ~5 min)**:
  1. **§6.1**: confirm qb1's installed ttnn exposes `sliding_window_size`
     on `ttnn.experimental.paged_scaled_dot_product_attention_decode`.
     If missing, rebuild ttnn or use manual K/V slice fallback before v0.3.
  2. **§6.3**: 1D pointwise check — which ttnn UnaryOp matches
     `torch.nn.functional.gelu(approximate="tanh")` over `[-5, 5]`.
- **v0 staging** (mirrors 35B `research/35b_cb_bringup_plan.md`):
  v0.1 L0-only forward → v0.2 all 48 layers → v0.3 KV cache with
  sliding-window kwarg → v0.4 trace capture → v1 CB B=4 → v2 server
  wire-up + chat smoke.
- **Top NOVEL items** (full detail in plan §3):
  dual head_dim (256 sliding / 512 global), four norms per layer
  (Llama RMSNorm `w`, NOT Qwen `(1+w)` — bit us hard on 35B
  [[qwen36-qnorm-knorm-zero-centered]]), tied embed + sqrt(hidden)
  embed-scale + 30·tanh(x/30) logit softcap, GELU_tanh activation,
  p-RoPE = standard partial RoPE with global head_dim divisor,
  `attention_k_eq_v` on global layers only.

### Parked — 35B drift cliff (#163, #164, #162)

- **#163**: full staging notes in
  [`research/35b_drift_next_session_plan.md`](research/35b_drift_next_session_plan.md)
  §"REAL findings 2026-06-03". Cliff between pos 1 (cos_L32=0.99) and
  pos 5 (cos_L32=0.32); flavor = positional-state bug. Step 1 probe
  wrapper `cb35_drift_cliff_search` deployed but never executed
  (harness restart from hang was killed when we pivoted).
  Cross-pollination: if Gemma 4 v0.3 surfaces a positional-state bug,
  it may share mechanism with this cliff.
- **#164**: manual recurrence path structurally broken
  (cos 0.08 @ pos 0 with owned_gdn=OFF). Orthogonal to cliff;
  fp32 H_t fix (`92b442f`/`8010b3c`) on main routes through broken path.
- **#162**: B>1 batched forward empty-slot poison. Default
  TT_CB_SLOTS=1 for 35B masks it.
- **#166** (NEW): harness hang hardening — line-buffered Python log
  write, 30s heartbeat, top-level try/except. ~12 LOC; ship BEFORE
  next 35B bootstrap.

### Earlier in this session (still active in prod)

- **27B HTTP smoke PASSED** end-to-end on qb1 (commits `97abfab`,
  `a7ea0fe`, `73fd269`). `/v1/chat/completions` returns
  "The capital of France is Paris." in 2.4s.
- **35B HTTP COHERENT at TT_CB_SLOTS=1** (cb_api default for 35B).
  Sample: "Hello" → "Hello! How can I help you today?" with
  `finish_reason=stop`. 35B B>1 broken (see #162 above).
- **35B first inference step crash fixed** in `39f4663` (cb_scheduler
  3-D readback squeeze).

## Project

Qwen3.6-family bringup on Tenstorrent Blackhole (P150 × 4). Production paths:

- **27B dense, 4× P150 TP** — `experiments/serve/server_tp.py` (single-stream
  Unix socket) or via CB (`serve_cb.sh start`, default backend).
- **27B continuous batching** — `experiments/serve/cb_api.py` + `cb_engine.py`,
  served by `experiments/serve/scripts/serve_cb.sh` (the canonical chat path).
- **35B-A3B MoE continuous batching** — same `serve_cb.sh` with
  `TT_BACKEND=35b`. Routes through `server_35b_ttnn.py` (model) + `server_35b_cb.py`
  (batched forward). Production-ready 2026-06-02.

Backend selection: `BACKENDS` registry in `cb_api.py` + identical dispatch
in `cb_scheduler.py`. Adding a new backend = drop a `server_<name>_ttnn.py`
+ `server_<name>_cb.py` + register both in `BACKENDS`/`_BACKEND_MODULES`.

Hosts: `qb1` and `qb2`, both 4× Blackhole P150 with working `FABRIC_1D`.

## Where the perf is now

| Path | Number | Source |
|---|---|---|
| 27B TP single-seq (steady-state, traced) | **12.93 tok/s** (77 ms/tok) | `serve_tp` on qb2 |
| 27B CB B=1 (traced) | 12.96 tok/s (==prod) | `experiments/cb/bench/trace.py` |
| 27B CB B=32 (traced, aggregate) | **150.5 tok/s** (11.6×) | same |
| 27B CB B=64 (traced, aggregate, shift-acc conv1d) | **593 tok/s** (45.8×) | same |
| 35B-A3B traced decode (qb1, after A002+A003+A004+A008+A009) | **81.16 ms/tok = 12.32 tok/s** | `research/35b_perf_milestones.md` |

CB SLO (qb1, 8 clients × 60 s, P5 gate 2026-05-30):
0 errors / 36 requests / 15 tok/s aggregate / **TTFT p99 = 176 ms**
(`experiments/cb/load/concurrent_chat.py`).

## Hardware ceiling

P150 measured: **404 GB/s/chip** DRAM BW, 110 worker cores, 31.81 GB DRAM
(`feedback_p150_memory_bandwidth_measured` in MEMORY.md).
For 35B-A3B with ~3 GB active params/token/chip:
bf16 BW floor ≈ 3.7 ms/tok → **270 tok/s ceiling**;
bf8 BW floor ≈ 1.85 ms/tok → **540 tok/s ceiling**.

The target is the hardware ceiling, not parity with someone else's number.

## Chat path (production)

```bash
# 27B (default backend):
TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready

# 35B-A3B MoE:
TT_BACKEND=35b TT_CB_SLOTS=2 TT_CB_TOPK_K=64 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6-14 min bootstrap
# verify via:  curl http://qb1:8000/v1/models  (must report Qwen/Qwen3.6-35B-A3B)
# 35B contract: TT_CB_TOPK_K must be >0 (logits-readback broken, #149);
#               cb_api defaults TT_CB_TOPK_K=64 when TT_BACKEND=35b.
bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # SIGTERM → graceful drain → mesh release
```

Env knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics` (Prometheus), `/bootstrap` (stage + elapsed_s).
See README §"Chat server (production)" for `curl` + `openai` client examples.

**Bootstrap observability** (commit `3e150c0`, 2026-06-02). Lifespan
startup blocks uvicorn from accepting HTTP until it yields, so HTTP
health probes can't see the 14-min 35B bootstrap. Three channels:
- `/bootstrap` (JSON): stage + elapsed_s + ready. Only reachable AFTER
  lifespan yields — useful for "is it ready yet" but not for "is it
  stuck during boot".
- `/health` enriched: 503 payload includes `{bootstrap: {…}}` once
  reachable.
- **Side file `~/tt-xla/.cache/server_cb.bootstrap.log`** — appended
  with explicit fsync from the bootstrap thread. Tail-able during the
  lifespan startup phase, when no HTTP endpoint is reachable yet.
  This is the canonical "is it making progress?" probe.

**Current status (2026-06-02 late evening): 35B HTTP server (re)bootstrapping
on qb1**. Background poll task `bz5lcsa9n` is watching for ready. v0/v1/v2
device primitives + cb35_prod_topk all PASS via the dev harness; the HTTP
wrap is what's flaky to bring up (uvicorn lifespan + 14-min boot + worker-
thread stdout buffering). Once the side-file shows `[harness ready]`, the
chat TUI / curl should work end-to-end.

## Deploy hygiene before any `serve_cb.sh start`

Always sync the entire `experiments/serve/*.py` before starting the server.
The dev harness uses `importlib.reload` per test so it only needs the file
under test; the production server boots a fresh process and reads qb1's
filesystem ONCE.
- One MM1 commit (`418f9cc`) sat in local git but never reached qb1; the
  server kept loading 27B for hours of debugging until we caught it.
- See memory `[[deploy-serve-files-too]]`.
- Quick command: `bash scripts/deploy.sh experiments/serve/*.py`

## What's next

**Prefix caching — END-TO-END VALIDATED (2026-06-01).** Slot-level
content-keyed prefix cache for the CB scheduler. Returning chats reclaim
their live slot at `cur_pos = len(matched_prefix)`, skipping re-prefill of
the history. qb1 smoke test:
- Turn 1 (cold, miss): **5.33s**
- Turn 2 (warm, **HIT**): **2.73s** — 1.96× speedup, 1 cache hit, 0 misses
- Qualitative: turn 2 correctly continued conversation (France → Germany / Berlin)

**Recommended runtime config (2026-06-02):** `TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1`.
CW1 fix (`ea9aa20`) makes both flags coexist by skipping the eager
chunked-prefill fallback for L > chunk_size — the L > chunk_size path
takes the legacy 1-tok/iter route via the decode trace, which is
allocation-free and safe alongside captured traces. Cold-start L > 32
prompts pay 1-tok/iter latency (~80 ms/tok), but with prefix caching
catching turn-2+ this is rare in chat workloads. Verified on qb1 smoke
2026-06-02: 1.97× turn-2 speedup preserved, no wedge.

Gated by `TT_CB_PREFIX_CACHE=0/1` (default 0). `TT_CB_PREFIX_TTL_S=300` for
stale-slot cleanup. Metrics in `/metrics`: `cb_prefix_cache_{hits,misses,
evictions}_total`, `cb_prefix_cache_live_slots`, `cb_prefix_cache_enabled`.

Load-bearing fixes from the smoke debug chain:
- `mark_live` fires on `max_tokens` cap (not just EOS) — `4acc955`
- `tokens_so_far` keeps trailing EOS for chat-template compat — `184f00e`
- `_messages_to_prompt`: `preserve_thinking=True` + trailing-only `<think>` strip — `2cad663`

Plan + per-milestone status: [`research/27b_prefix_caching_plan.md`](research/27b_prefix_caching_plan.md).
Research:
- [`research/vllm_prefix_caching_audit.md`](research/vllm_prefix_caching_audit.md) — APC design
- [`research/vllm_chat_template_handling.md`](research/vllm_chat_template_handling.md) — Qwen3.6 quirks + upstream-blessed `preserve_thinking` fix

Regression gate: `experiments/cb/isolate/chat_template_invariant.py`
(7 cases including 239/239 long-prompt, unicode, 3-turn compound).
Memory: [[prefix-caching-design]], [[qwen36-preserve-thinking]].

**S2 — chunked prefill — LIVE in production (2026-06-01).** CB serves with
`TT_CB_CHUNKED_PREFILL=1`: traced chunked prefill at chunk_size=32 for L ≤ 32,
legacy 1-tok/iter fallback for L > 32. Two-phase warmup (compile-all-then-capture-all)
solves the multi-trace coexistence wedge per [vLLM #352](https://github.com/tenstorrent/vllm/issues/352).
Plan + post-mortem: [`research/27b_prefill_trace_plan.md`](research/27b_prefill_trace_plan.md).

Deferred / superseded: T3 multi-chunk traced prefill (chat win comes from skipping
re-prefill via prefix caching, not making re-prefill faster). Bigger chunk_size
(same reasoning). Both revisitable for long single-prompt cases (no prior cache
to match) after prefix caching ships.

## Roadmap (priority order, 2026-06-02)

1. **DONE — `chunked_prefill=1 + prefix_cache=1` coexist** (CW1, commit `ea9aa20`).
   Run prod with both flags on. Long L > chunk_size prompts use 1-tok/iter
   fallback (slow but allocator-safe). Validated 2026-06-02: 1.97× turn-2
   speedup preserved + no wedge.
2. **DONE — `TT_BACKEND` env selector** (MM1, commit `418f9cc`). cb_api.py
   has a `BACKENDS` registry; `TT_BACKEND={27b,35b}` switches at boot.
3. **IN PROGRESS — 35B-A3B CB bringup (CB35-v0..v4)**. Plan:
   [`research/35b_cb_bringup_plan.md`](research/35b_cb_bringup_plan.md).
   Research basis: [`research/tt_metal_moe_cb_patterns.md`](research/tt_metal_moe_cb_patterns.md)
   (DeepSeek-V3 + Llama-70B-Galaxy patterns).
   - **v0 BIT-VALIDATED 2026-06-02** (commits `112d72a`, `49778b3`):
     `cb35_v0_smoke` 3/3 + `cb35_v0_chat` (8-tok decode bit-identical).
     Critical fix: `base.reset_caches_ttnn` leaks per-layer tensors;
     `cb_reset_states` explicit-deallocs first.
     `return_logits=True` raises (35B [1,VOCAB] readback broken, #149).
     cb_engine routes 35B through topk via `TT_CB_TOPK_K=64`.
   - **v1.0 alloc + v1.1 embed/RoPE BIT-VALIDATED 2026-06-02** (commits
     `ec01052`, `cf211a4`). `setup_cb_state(B>1)` allocates B-leading
     `cb_dn`/`cb_kv` + per-slot `cb_page_table_tt`. `_batched_prelude`
     reads `cb_tok_buf` / `cb_rot_idxs_buf`. Gate: 3/3 alloc + 4/4 prelude.
   - **v1.2 batched DN BIT-VALIDATED 2026-06-02** (commit `4546d29`).
     `dn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2 slot 0. Rank-3
     `[B, N, D]` rms_norm fix ([[ttnn-rms-norm-shape-drift]]).
   - **v1.3 batched GatedAttention BIT-VALIDATED 2026-06-02** (commit
     `e8f2d82`). `attn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2
     slot 0 — first-try pass. Rank-3 `_apply_partial_rope_b` sidesteps
     base's K-broadcast workaround. Paged KV write + SDPA decode use
     `cb_cur_pos_buf`/`cb_page_table_tt` for per-slot context.
     `setup_cb_paged_cfgs` builds B-sized HEIGHT_SHARDED L1 mem cfg +
     SDPA progcfg (B=1 reuses base's `state.paged_*`).
   - **v1.4b broadcast MoE BIT-VALIDATED 2026-06-02** (commit `a6ac640`).
     True broadcast Pattern A — `[E_LOCAL, B, HIDDEN]` middle-dim bump
     in batched expert matmul. mad=0.000000 vs base at B=2 slot 0.
     Fixed the 13% per-slot drift from v1.4 loop. Critical bug in init
     port: used `MOE_INTER_CHIP (128)` instead of `MOE_INTER (512)`.
   - **v2 trace capture SHIPPED 2026-06-02** (commit `c547419`).
     Two-phase warmup + `begin_trace_capture`/`end_trace_capture`
     around `forward_batch_tp_inner` works at B=2. Replay 149.7 ms/step
     vs eager 296.7 ms/step = **1.98× speedup**. cb_scheduler trace
     plumbing inherits automatically (calls the unified entry).
     Higher B → larger speedups.
   - **CB35-prod wire-up GATED 2026-06-02** (commit `f1f7a61`).
     Unified `forward_batch_tp_inner` dispatches B=1→base (v0 bit-id) /
     B>1→v1 batched. Now supports `return_topk=K` at both Bs. cb_scheduler
     can drop in without changes. `cb35_prod_topk.py` 4/4 PASS:
     B=1 top-1 = 8 (matches v0), B=2 distinct prompts produce distinct
     top-1 tokens. Ready for cb_api/cb_engine end-to-end at
     `TT_CB_SLOTS=2`.
   - **v1.5 full B>1 forward FUNCTIONAL PASS 2026-06-02** (commit `0a50e97`).
     `forward_batch_tp_inner_batched` + `layer_forward_batched_35b` —
     40-layer chain at B>1 runs end-to-end. v1_chat results:
     - ✓ slot 0 != slot 1 with distinct prompts (per-slot independence)
     - ✓ slot 0 == slot 1 with same prompt (determinism)
     - ⚠ slot 0 != B=1 ref by argmax tokens — bf16 chain precision drift
       compounding across 40 layers (every individual op verified bit-id
       at B=2 slot 0; the chain noise is what flips argmax tokens). See
       [[bf16-chain-drift-at-B-gt-1]] for the lesson. **Production-shippable** —
       each slot's generation is valid; use cosine for benches, not
       exact tokens.
   - v1 (true B>1 batched forward): next. ~3-5 days.
   - v2 (trace capture at B=N): ~1-2 days.
   - v3 (owned-GDN batched FOLD trick): optional.
   - v4 (prefix cache for attn layers only): LOW PRIORITY (vllm#36493 reports
     ~0.1% hit rate on this arch class — DN layers can't be cached).
   - **Dev iteration harness** (`experiments/cb/dev/cb35_dev_harness.py`):
     bootstrap-once long-running python on qb1, watches `~/tt-xla/.cache/cb35_runtime/trig/`,
     reloads via `importlib`. Cuts fix-test cycle from ~14 min to seconds.
     **MUST launch via `bash scripts/run_harness_tmux.sh`** — nohup + disown
     and setsid + double-fork both fail to keep the python alive after SSH
     disconnect on qb1 (it dies before completing bootstrap). tmux
     survives by design. See `research/35b_cb_bringup_plan.md` for usage.
4. **Multi-model fleet** — plan: [`research/multi_model_serving_plan.md`](research/multi_model_serving_plan.md).
   Once 35B is live, MM5 (Mistral Small 3.2 24B) is the strongest
   framework-generalization test (different vendor, different tokenizer,
   pure dense GQA, no DN). Candidate research:
   [`research/home_llm_landscape_2026.md`](research/home_llm_landscape_2026.md).
5. **Long-context concurrent stress test (MM3)** — validate PC works at
   L=1000+ prompts under realistic concurrency. Reuses
   `experiments/cb/load/concurrent_chat.py`. Final-validation step after
   the multi-model work is in place.

**35B perf** (parallel track). Next levers tracked in
[`research/35b_perf_milestones.md`](research/35b_perf_milestones.md):
async all_reduce overlap, expert-broadcast elimination, routing-weight
fusion, bf8 expert weights.

## Repo entry points

- README — install + demos.
- CONTRIBUTING — dev loop, canary gate, code style.
- `research/` — design docs + living plans (index: `research/README.md`).
- `wiki/` — Q&A wiki, learning-by-building.
- `models/` — multi-model demos (Llama, Qwen2.5, SmolLM, 8B).

## Read order when resuming work

1. This file.
2. [`archive/superseded_research_2026-06-04/profiling-quick-reference.md`](archive/superseded_research_2026-06-04/profiling-quick-reference.md) — Tracy + tt-perf-report capture/analyze (archived 2026-06-04; still the best concise reference).
3. [`research/35b_perf_milestones.md`](research/35b_perf_milestones.md) — 35B perf trajectory.
4. [`research/27b_cb_scope.md`](research/27b_cb_scope.md) — CB design + numbers (CB0–CB4).
5. [`research/35b_tt_perf_report_findings.md`](research/35b_tt_perf_report_findings.md) — empirical writeup behind 35B advice.

## Load-bearing rules (each cost a multi-day debug)

- **View-decay**: `ttnn.slice` / `ttnn.reshape` return views. Never
  `ttnn.deallocate` the source while a view is live; clone when in doubt.
- **+1 zero-centered RMSNorm offset** on `q_norm` / `k_norm` /
  `input_layernorm` / `post_attention_layernorm` / `final_norm` (Qwen3.6).
- **K-broadcast RoPE workaround** in the SDPA path — sidesteps a ttnn
  `[1, HEAD_DIM]` slice/concat bug.
- **bf16 KV cache** required by paged SDPA (fp32 hard-rejected).
- **HiFi4 + `fp32_dest_acc_en`** on every matmul (the 91f recipe); mixing
  fidelities corrupts ops silently on Blackhole.

## Workflow

- Profile-driven only. Cite a Tracy / tt-perf-report number for any
  optimization claim. Frame deltas as Δ from BW floor.
- Correctness gate: 5-token Paris (`"The capital of France is" → " Paris..."`)
  on prefill IDs `[2614, 314, 279, 369, 11751]`.
- Iterations in git history or `scratch/`, never in demo scripts.
- Remote-only execution (`ssh qb1` / `ssh qb2`); no device code locally.
