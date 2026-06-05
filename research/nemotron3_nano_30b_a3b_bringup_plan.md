# Nemotron-3 Nano 30B-A3B → CB chat server — bringup plan (revised 2026-06-04)

**Target model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

**Architecture brief**: [`research/nemotron3_nano_architecture_brief.md`](nemotron3_nano_architecture_brief.md)
(authoritative; all shape constants and code-path claims sourced there).

> **Bringup posture (revised after research)**: this is the **biggest
> single departure** from the 27B / 35B / Gemma 4 lineage. Nemotron-3
> Nano is a **Mamba2-Transformer hybrid MoE** (NOT a pure transformer),
> with 23 Mamba2 SSD layers, 23 DeepSeek-V3-style MoE layers, and 6
> RoPE-free GQA Attention layers, dispatched in an interleaved pattern.
> **tt-metal does not ship a Mamba2 SSD kernel** — that's the gating
> blocker for everything downstream. Realistic timeline is **3-4 weeks
> (manual ttnn Mamba2 for v0..v2)** → **6-8 weeks total (with an owned
> Mamba2 kernel at v3 perf pass)**, not days.

------------------------------------------------------------------------

## 0. Locked decision: Path B — Owned kernel up-front

**Decision (2026-06-04, user)**: Path B. Build the owned Mamba2 SSD
decode kernel from scratch (G0..G4 staging per `[[build-kernels-from-scratch]]`)
BEFORE the v0..v2 forward / decode / CB / HTTP ladder. Total estimated
6-8 weeks to v2.

Rationale: Path B frontloads the kernel work so by the time the ladder
starts, every Mamba2 forward call is already running at production-grade
perf. No intermediate "manual ttnn composite" detour that would have to
be ripped out at v3. Trades intermediate demo visibility for cleaner
integration and one less rewrite.

**Implication**: every stage below assumes the owned kernel exists.
**Phase 0 (kernel build, §3a)** runs FIRST; **Phase 1 (forward/ladder,
§3b)** starts only after G4 lands.

------------------------------------------------------------------------

------------------------------------------------------------------------

## 1. TL;DR (architecture, sourced from §1 of the brief)

- **Class**: `NemotronHForCausalLM` (`model_type=nemotron_h`); requires
  `trust_remote_code=True` for HF AutoModel oracle. Permissive
  Nvidia open license; no HF gating.
- **52 layers, `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`
  dispatch**: 23 Mamba2, 23 MoE, 6 Attention. Attention sits at
  indices `[5, 12, 19, 26, 33, 42]`. No 'mlp' (dense FFN) layers.
- **30-31.6B total, ~3.5B active per token**. 13 safetensors shards ≈
  63 GB on disk; 32 GB bf16 weights = 8 GB/chip on a (1,4) mesh.
- **Mamba2 mixer**: 64 heads × 64 head_dim = 4096 d_inner, ssm_state=128,
  n_groups=8, conv_kernel=4, expand=2, chunk_size=128, **fp32 SSM
  state**.
- **Attention**: 32 Q heads, 2 KV heads (16:1 GQA), head_dim=128,
  **NO RoPE applied** (config fields present but modeling does NOT
  consume them — positional info lives entirely in Mamba2 state).
  Standard SDPA scale = 1/sqrt(128).
- **MoE**: 128 routed + 1 shared, top-6 routed, sigmoid router with
  group-restricted top-k (n_group=8, topk_group=1), `routed_scaling_factor=2.5`,
  `relu²` activation, shared expert 2× wider (3712 vs 1856).
- **Norm**: Llama-style RMSNorm (`y = x/rms * w`, NO `+1.0`, NO bias);
  one pre-norm per layer; no post-norm; no layer_scalar.
- **Vocab 131072** (smallest of any model we've shipped — clean power
  of 2 for vocab-shard), `tie_word_embeddings=False`, **no logit softcap**.
- **Context** `max_position_embeddings=262144` (256K). Our current
  `MAX_POS=8192` cap will need a bump for the full claim, but v0 ships
  at 8K.
- **Tokenizer**: ChatML (`<|im_start|>{role}\n…<|im_end|>\n`), same
  family as 27B/35B. EOS frozenset `{2, 11}` = `{</s>, <|im_end|>}`.
  Active-prompt suffix is `<|im_start|>assistant\n<think>\n` (thinking
  on) or `<|im_start|>assistant\n<think></think>` (thinking off) —
  similar shape to Qwen3.6's. Our `_active_prompt_suffix` should cover.

**Source**: all the above quoted verbatim from §1-§5 of the brief.

------------------------------------------------------------------------

## REUSE MANDATE (`[[feedback-reuse-mandate]]`)

The REUSE map is now grounded in the brief's §6, not guesses. Confidence
levels are the brief's; restated here for completeness.

### Reused verbatim — zero changes (HIGH confidence)

| Existing | Role for Nemotron |
|---|---|
| `experiments/serve/cb_engine.py`, `cb_scheduler.py`, `cb_metrics.py`, `cb_api.py`, `openai_endpoint.py`, `live_slot_store.py`, `protocol.py` | Production CB stack — model-agnostic |
| `experiments/serve/openai_endpoint.py:_active_prompt_suffix` | Active-prompt-suffix detector (post `184753d`) — Nemotron's ChatML `<think>` suffix has the same shape Qwen had |
| `experiments/serve/scripts/{deploy.sh,serve_cb.sh}` | Deploy + lifecycle unchanged |
| `experiments/utils/ttnn_introspect.py`, `hf_download.py`, `npz_inspect.py`, `syntax_check.py` | Helpers — call as-is |
| `experiments/cb/isolate/paged_sdpa.py` | Attention is the **simplest** we've shipped (no Q-gate / q/k_norm / RoPE), just `scale=1/sqrt(128)` |
| `experiments/cb/isolate/paged_update_cache.py` | KV cache write/read — verify NKV=2 / 4 chips contract, may need replication trick (see §7.2 of brief) |
| `experiments/cb/isolate/chat_template_invariant.py`, `chat_template_inspect.py`, `chat_template_roundtrip.py` | Chat-template gates |
| `experiments/cb/validate/pc_token_match.py` | Prefix-cache regression gate — works on any HF tokenizer |
| `scripts/chat.py` (TUI) | Demo client — backend-agnostic |
| `scripts/stress_multiturn_http.py`, `stress_ttft_decode.py`, `stress_concurrent_chat.py` | Stress tests — backend-agnostic |
| `experiments/serve/server_tp.py:1680-1687` P22 vocab-sharded LM head + on-device argmax | Reuse verbatim with VOCAB=131072 (clean /4 split = 32768/chip) |
| Two-phase warmup pattern (`[[ttnn-multi-trace-two-phase-warmup]]`) | Universal trace discipline |

### Fork from 35B (HIGH structural confidence; MEDIUM math delta)

| Existing | Fork target for Nemotron | Delta |
|---|---|---|
| `experiments/serve/server_35b_ttnn.py` (2040 LOC) | `experiments/serve/server_nemotron3_nano_ttnn.py` | Bootstrap skeleton + per-layer dispatch carries over. New: Mamba2 layer math; sigmoid router + group restriction; relu² activation; shared expert 2× wider. 52 layers vs 40, vocab 131072 vs 248320, hidden 2688 vs 5120. |
| `experiments/serve/server_35b_cb.py` (967 LOC) | `experiments/serve/server_nemotron3_nano_cb.py` | Batched CB wrapper; same shape with new layer types |
| `experiments/utils/hf_reference_35b.py` | `hf_reference_nemotron3_nano.py` | Drop DN hooks; add **Mamba2 layer hooks** (in_proj, conv1d, ssm_state, out_proj per-step); keep attn + MoE hooks. Use `trust_remote_code=True`. Needs CPU RAM check (~62 GB needed for AutoModel bf16) |
| `experiments/utils/cosine_ladder_35b.py` | `cosine_ladder_nemotron3_nano.py` | Swap n_layers 40→52, oracle path |
| `experiments/cb/dev/cb35_dev_harness.py` | `cb_nemotron3_dev_harness.py` | Fork (already hardened #166); expect ~10-12 min bootstrap (between Gemma 4's 80s and 35B's 14 min) |
| `experiments/cb/isolate/cb35_per_layer_drift_pos1.py` | `cb_nemotron3_per_layer_drift_pos1.py` | Per-layer drift probe at pos>0 |
| `experiments/cb/isolate/cb35_needle_haystack.py` | `cb_nemotron3_needle_haystack.py` | Long-context retrieval gate (push to ≥8K) |
| 35B's `moe_forward_ttnn_pattern_a_batched` (`server_35b_ttnn.py:1225`) | New `moe_forward_ttnn_nemotron3` | ~150-200 LOC delta: sigmoid not softmax; group-restricted topk pre-mask; routed_scaling_factor=2.5 multiply; relu² in expert FFN; shared expert 2× wider (different program config) |

### NEW — Mamba2 is fresh kernel territory (LOW reuse from owned-GDN)

| Component | LOC est. | Notes |
|---|---|---|
| `experiments/cb/isolate/mamba2_decode_composite.py` | ~200 | Manual ttnn composite for SSD recursion. v0 path. |
| `experiments/cb/isolate/mamba2_conv1d_step.py` | ~100 | Per-step causal Conv1d-step with rolling `[B, conv_dim, 4]` state. ttnn has Conv1d, not the per-step caching form. |
| Mamba2 cache plumbing in `MeshServerState` / `State` | ~50 | Conv state + SSM state allocation per Mamba layer; fp32 SSM state |
| **35B's `qwen36_gdn_decode_owned` kernel** | — | **NOT directly reusable**. The plumbing pattern (recurrent state in/out, kernel-side fp32 acc) is transferable to a new `nemotron3_mamba2_decode_owned` kernel; the per-step math is NOT. Path B / v3 owned kernel only. |

### Reuse summary

- **Reused as-is**: ~70-80% of CB + tokenizer + chat-template + utility infra.
- **Forked from 35B with delta**: bootstrap, MoE Pattern A, dev harness, oracle, ladders.
- **Net new**: Mamba2 composite (v0), Mamba2 owned kernel (v3).

------------------------------------------------------------------------

## 2. Memory cross-references (read before any code)

The high-leverage memories for a Mamba2 + MoE bringup:

- `feedback_reuse_mandate.md` — fork-don't-write rule
- `feedback_two_phase_warmup.md` — multi-trace capture discipline
- `feedback_perf_workflow.md` — 6-step perf workflow
- `feedback_correctness_first.md` — never optimise past cosine < 0.99
- `feedback_validate_against_ground_truth.md` — HF oracle, not weaker TT path
- `feedback_gemma4_sdpa_scale_1.md` — model-specific SDPA scale. **Nemotron is `1/sqrt(128)` (standard)**, NOT Gemma's `1.0`.
- `feedback_qwen36_qnorm_knorm_zero_centered.md` — the `(1+w)` rule. **Nemotron uses Llama-style RMSNorm (no `+1`)**, same as Gemma 4.
- `feedback_ttnn_slice_view_decay.md`, `feedback_ttnn_list_rebinding_leaks.md`, `feedback_ttnn_rms_norm_shape_drift.md` — universal ttnn footguns
- `feedback_paged_update_cache_nkv_per_chip.md` — NKV=2 per chip-pair; see §7.2 of brief for the sharding decision
- `feedback_read_kernel_source_first.md` — read TT_FATAL assertions before forking into a new shape regime
- `feedback_use_existing_isolation_probes.md` — fork an isolation probe before iterating in a full forward
- `feedback_cb_backend_dispatch_holes.md` — grep every `27b`/`35b`/`gemma4_12b` literal in `experiments/serve/` when adding `nemotron3_nano`
- `feedback_deploy_serve_files_too.md` — `deploy.sh experiments/serve/*.py` before server restart
- `feedback_use_dev_harness_for_iteration.md` — never restart `serve_cb.sh` per fix; use the dev harness. **Mandatory from day 1 here** because bootstrap is 10-12 min.
- `feedback_harness_state_version_skew.md` — new State fields need `getattr(state, "X", None)` lazy-init
- `feedback_35b_dn_h_state_drift_lever.md` — **fp32 recurrent state inside a trace caused a 30+ min hang on 35B**. Nemotron's fp32 SSM state hits the same risk surface. See §7.3 of brief.
- `feedback_show_thinking_traces.md` — TUI default (Nemotron also emits `<think>` blocks; user wants them visible)

------------------------------------------------------------------------

## 3a. Phase 0 — Owned Mamba2 SSD kernel (G0..G4)

This phase runs FIRST per the Path B decision. Each G-stage gates the
next; the kernel ladder mirrors the 35B `qwen36_gdn_decode_owned` build
which took G0..G4 in ~10 days (`feedback_owned_decay_gate_shipped`).
Mamba2 SSD has a similar structure (recurrent state, per-head loop,
fp32 accumulator) but different math; reuse the build *pattern*, not
the math.

### Per-step Mamba2 SSD math (the kernel implements this)

Pseudocode for one decode step, per head h ∈ [0, 64), per token:

```
# Read state and inputs
ssm_state[h, :, :]  ∈ fp32, shape [head_dim=64, ssm_state=128]
x[h, :]             ∈ bf16, shape [head_dim=64]
dt[h]               ∈ bf16, scalar
A_log[h]            ∈ bf16, scalar  (learned)
B[h, :]             ∈ bf16, shape [ssm_state=128]  (per-group, broadcast)
C[h, :]             ∈ bf16, shape [ssm_state=128]
D[h]                ∈ bf16, scalar  (learned)

# Discretization (fp32)
dt_eff = softplus(dt + dt_bias[h]).clamp(time_step_floor, time_step_max)
A      = -exp(A_log[h])

# Update SSM state (fp32 accumulator)
# ssm_state_new[d, s] = exp(dt_eff * A) * ssm_state[d, s] + dt_eff * B[s] * x[d]
ssm_state_new[d, s] = exp(dt_eff * A) * ssm_state[d, s] + dt_eff * B[s] * x[d]

# Output projection (fp32 reduce, bf16 result)
y[d] = sum_s (C[s] * ssm_state_new[d, s]) + D[h] * x[d]

# y is then gated by z and norm'd outside the kernel (MambaRMSNormGated
# uses head_dim=64 groups; can be a separate ttnn call).
```

The kernel signature is:
```
mamba2_decode_owned(
    x:          [B, num_heads=64, head_dim=64]      bf16 input
    z:          [B, num_heads=64, head_dim=64]      bf16 gate (passed through to norm-gated outside)
    dt:         [B, num_heads=64]                    bf16 dt
    B_tensor:   [B, n_groups=8, ssm_state=128]      bf16 B matrix
    C_tensor:   [B, n_groups=8, ssm_state=128]      bf16 C matrix
    ssm_state:  [B, num_heads=64, head_dim=64, ssm_state=128]   fp32 in/out (mutated)
    A_log:      [num_heads=64]                       bf16 learned
    dt_bias:    [num_heads=64]                       bf16 learned
    D:          [num_heads=64]                       bf16 learned
)  →  y: [B, num_heads=64, head_dim=64]              bf16 output
```

Note: B and C are *grouped* (8 groups, 8 heads per group), so each
group's B/C broadcasts across its 8 heads. The kernel must implement
the per-group broadcast.

### G-stage ladder

| Stage | Adds | Gate | Reference |
|---|---|---|---|
| **G0** | Read existing tt-metal SSM-adjacent ops (`ttnn.experimental.*`, look for ssm/mamba/recurrence/scan). Read `state-spaces/mamba` mamba_ssm Triton kernel for math reference. Numpy oracle: pure-numpy single-token SSD step matching HF `modeling_nemotron_h.py:NemotronHMamba2Mixer.forward` at the `mamba_chunk_scan_combined` call site. | numpy oracle byte-match vs HF eager forward at L0 step 0, all 64 heads; doc what tt-metal exposes vs needs to be authored | Fork pattern from `[[dnk-g0-read-owned-gdn-source-plan-batching]]` |
| **G0a** | Isolation harness: `experiments/cb/isolate/mamba2_decode_oracle.py`. Generates random inputs, runs numpy oracle, returns expected outputs. Used as ground truth by all subsequent stages. | runs end-to-end on host; produces deterministic outputs |  fork from `test_pattern_a_moe_np.py` |
| **G1** | Single-core full-chain Mamba2 step in `tt-metal/ttnn/cpp/ttnn/operations/...`. Single-tile (B=1, single head) program-factory. Implements the per-head SSD recursion (discretization → state update → output reduce). Reads/writes fp32 ssm_state. | cos ≥ 0.999 + MAD bit-close to numpy oracle for B=1, single head | Fork pattern from owned_gdn G1; share circular-buffer + LLK conventions |
| **G2** | Multi-core: shard `num_heads=64` across Tensix cores. Each core processes its heads independently (no cross-head reduce; the reduce is per-head over `ssm_state` dim). | cos ≥ 0.999 at full B=1, num_heads=64; throughput ≥ 8× G1 | Fork pattern from owned_gdn G2 |
| **G3** | Batching: assert `state_logical[0] == B`; program-factory handles the leading batch dim. Mirror `qwen36_gdn_decode_owned`'s batched form. (35B's hard-assert batch=1 is exactly what we DON'T want to repeat.) | cos ≥ 0.999 at B=1..32, num_heads=64 | Fork pattern from `[[dnk-g1-batch-the-kernel-assert-plus-program-factory]]` |
| **G4** | Integrate into `experiments/serve/server_nemotron3_nano_ttnn.py:mamba2_forward_ttnn` (which exists by then, see Phase 1). Two-phase warmup + trace capture works. | end-to-end smoke + cos vs eager bit-close | Fork pattern from `[[dnk-g2-integrate-batched-owned-gdn-into-cb-dn-step-plus-measure]]` |

### Tasks (Phase 0) — tracked in the project task list

- **Task #183** — G0 numpy oracle + tt-metal SSM survey — **IN PROGRESS**:
  oracle written (`experiments/utils/mamba2_numpy_oracle.py`, commit
  `98fc43d`) + self-test ✓ on qb1; tt-metal SSM survey ✓ (no matches,
  build from scratch confirmed). HF byte-match gate still open
  (pending Nemotron weight download).
- **Task #184** — G0a isolation harness — **DONE** (commit `4352baf`):
  multi-step replay + per-head cos/MAD gate. Self-test PASSES; kernel-
  compare path stubbed via `--kernel-callable` for G1 to wire in.
- **Task #185** — G0b qb1 RAM + tt-metal SSM survey — **DONE**:
  qb1 has 503/468 GB RAM (plenty); ZERO ttnn SSM ops at module level.
- **Task #186** — G1 single-core (blocked by #184)
- **Task #187** — G2 multi-core (blocked by #186)
- **Task #188** — G3 batched (blocked by #187)
- **Task #189** — G4 server wrapper + Phase 1 unblock (blocked by #188)

### Phase 0 progress (2026-06-04 EOD)

- [x] **G0b** — qb1 RAM + ttnn SSM survey done. No tt-metal Mamba2
      primitive exists; building from scratch.
- [-] **G0** — numpy oracle written + internal self-test PASS. HF
      byte-match gate still open (pending Nemotron weight download
      to qb1).
- [x] **G0a** — isolation harness (multi-step replay + per-head
      cos/MAD + kernel-compare hook via `--kernel-callable`). Commit
      `4352baf`.
- [-] **G1** — in progress (task #186, latest commit `e96b9a9`):
  - ✅ Kernel design doc: `research/mm7_g1_mamba2_kernel_design.md`
  - ✅ Dataflow decisions log (10 D-numbered choices, each a future
    optimization lever): `research/mm7_g1_dataflow_decisions.md`
  - ✅ Conv1d reuse discovered: `qwen36_conv1d_decode_owned`
    parametrised at D=6144 IS Mamba2's `conv1d_step`. Zero new
    conv code.
  - ✅ Day-1 fork: `experiments/owned_ops/nemotron3_mamba2_decode_owned/`
  - ✅ Day-1.5 compute kernel skeleton (commit `e96b9a9`): full
    file header documenting the math + SPMD + fp32 acc + debug_mode
    pattern. CB layout pinned (15 CBs, cb_x..cb_y). debug_mode=1
    (fill_one smoke) implemented for first-build gate. Six TODO
    blocks for the math helpers with explicit LLK call sequences.
  - [x] Day-3: LLK API survey done — softplus/clamp/exp/negative ship
    as first-class SFPU primitives (D8 RESOLVED). compute_decay +
    mul_decay_state_to shipped; D11 introduced split-helper approach.
  - [x] Day-3.5 (commit pending push): finalize_decay_with_dt_eff body
    landed. Three-stage pipeline:
      1. Stage 1: dt_eff = clamp(softplus(dt + dt_bias), floor, max)
         → cb_dt_B (reused as scratch).
      2. Stage 2+3: decay = exp(A * dt_eff) → overwrites cb_decay via
         queue-pop-after-push semantics.
    compute_decay simplified to one tile_regs cycle (A only). D11
    marked RESOLVED with the scratch-CB pattern documented. debug_mode=2
    output now: state_out = decay * state_in (matches oracle's
    decay-only term; full math at debug_mode=5).
  - [ ] Day-3.75: program_factory + reader/writer for the new 15-CB
    Mamba2 layout (currently still GDN's 18-CB layout — kernel won't
    actually run until factory updated). First build + debug_mode=1
    smoke on qb1.
  - [ ] Day-4: compute_dt_B + add_outer_input → debug_mode=3 (state correct).
  - [ ] Day-4: C_state_reduce + add_skip → debug_mode=4..5 (full math).
  - [ ] Day-5: build, oracle compare via G0a harness; ship.
- [ ] **G2..G4** — sequential, gated on G1.

Phase 0 timeline estimate: **3-5 weeks** depending on how many of the
G-stages hit unexpected snags. Each stage gates the next; do NOT
parallelise the G-ladder.

------------------------------------------------------------------------

## 3b. Phase 1 — Forward / decode / CB / HTTP ladder

This phase runs AFTER G4 lands. Each stage gates the next.

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v0.0** | `hf_reference_nemotron3_nano.py` with `trust_remote_code=True`, Mamba2 layer hooks, attn + MoE hooks. CPU RAM check on qb1 host (need ≥62 GB free for AutoModel bf16; if not, layer-stream via `accelerate.load_checkpoint_and_dispatch`). | `prompt_ids.npy`, `hidden_states.npy[53, S, 2688]` exist; layer count = 52; per-layer-type tag matches `hybrid_override_pattern` | PENDING |
| **v0.0.1** | On-qb1 tokenizer verification (file truncated in research). `tail -50 ~/.cache/.../tokenizer_config.json` → `tokenizer_class`, `model_max_length`, inline `chat_template` (if any) | tokenizer_class confirmed; no inline template overrides | PENDING |
| **v0.1.0** | Bootstrap on (1,4): mesh + fabric + safetensors weights upload + embed lookup + final_norm | bootstrap < 15 min; embed lookup cos ≥ 0.999 vs HF; weights /chip ≤ 9 GB | PENDING |
| **v0.1.1** | **L0 (Mamba2) — eager manual ttnn composite**: in_proj → split → conv1d_step → silu → split B/C/x → dt softplus + clamp → SSD recursion (per-head loop) → MambaRMSNormGated → out_proj | cos ≥ 0.999 on L0 output vs HF oracle (single position, single step) |
| **v0.1.2** | **L1 (MoE) — sigmoid router + group top-6 + relu² experts + shared expert + scaling**: fork from `moe_forward_ttnn_pattern_a_batched`. Test routing in isolation first via `_moe_router_topk` probe. | cos ≥ 0.999 on L1 output vs HF |
| **v0.1.3** | **L5 (Attention) — paged SDPA, NO RoPE, GQA repeat_kv**: simplest attention block we've shipped. q/k go from q_proj/k_proj directly into SDPA. NO RoPE table allocation. | cos ≥ 0.999 on L5 output vs HF |
| **v0.2** | All 52 layers (per-layer dispatch via `state.layer_types[L]`) + final_norm + lm_head + argmax | argmax matches HF at pos 0 (single token) |
| **v0.3** | Multi-step decode: KV cache for 6 attn layers + Mamba2 state (conv + ssm fp32) for 23 mamba layers, eager mode | TT tokens 0..5 match HF token-for-token |
| **v0.3.3** | Long-context smoke at L=8192 (fork `cb_nemotron3_needle_haystack.py`) | password retrieved ≥75% at L=8192 |
| **v0.4** | **Trace capture — fp32-in-trace risk check** ([[35b-dn-h-state-drift-lever]]). Two-phase warmup, then `begin_trace_capture` → forward → `end_trace_capture`. If trace hangs >5 min, fall back to bf16 SSM state + measure drift. | 100 traced steps == 100 eager token-for-token |
| **v1.0-v1.6** | Continuous batching: setup → batched embed → batched attention (paged SDPA over per-slot KV) → batched MoE → batched Mamba2 (per-slot conv + ssm state) → end-to-end at B=4 | 3a/3b/3c gates PASS at B=4 |
| **v2** | HTTP wire-up: register in `cb_api.BACKENDS` + `cb_scheduler._BACKEND_MODULES`, tokenizer + chat template | `curl /v1/chat/completions` returns sensible text; multi-turn coherence 3/3 |
| **v3** (perf) | **Owned `nemotron3_mamba2_decode_owned` kernel** (G0-G4 staging per `[[build-kernels-from-scratch]]`). Optional but high-leverage given 23/52 layers go through this path. | step_ms ≥ 30% lower than v0.4 baseline; correctness unchanged |

------------------------------------------------------------------------

## 4. Backend wiring (exact edits at v2)

Per `[[feedback-cb-backend-dispatch-holes]]`, register the new backend
in BOTH dicts. Grep every `"27b"`, `"35b"`, `"gemma4_12b"` literal in
`experiments/serve/` before deploying. Deploy the WHOLE
`experiments/serve/` glob with `scripts/deploy.sh`.

```python
# experiments/serve/cb_api.py:50 — BACKENDS dict
BACKENDS = {
    "27b":            ("server_tp",                       "Qwen/Qwen3.6-27B"),
    "35b":            ("server_35b_ttnn",                 "Qwen/Qwen3.6-35B-A3B"),
    "gemma4_12b":     ("server_gemma4_unified_ttnn",      "google/gemma-4-12B"),
    "nemotron3_nano": ("server_nemotron3_nano_ttnn",      "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
}
```

```python
# experiments/serve/cb_scheduler.py:50 — _BACKEND_MODULES dict
_BACKEND_MODULES = {
    "27b":            ("server_tp",                   "server_tp_cb"),
    "35b":            ("server_35b_ttnn",             "server_35b_cb"),
    "gemma4_12b":     ("server_gemma4_unified_ttnn",  "server_gemma4_unified_cb"),
    "nemotron3_nano": ("server_nemotron3_nano_ttnn",  "server_nemotron3_nano_cb"),
}
```

------------------------------------------------------------------------

## 5. Risk register (from brief §7 — restated with project framing)

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| Mamba2 SSD kernel does not exist on Blackhole | **CERTAIN** | **CRITICAL** | Path A — manual ttnn composite for v0. Plan §0 decision. |
| fp32 SSM state in trace → 30+ min Blackhole trace hang (35B fp32-H pattern) | medium | high | Validate fp32 path in eager (v0.3) before trace (v0.4). If hangs, fall back to bf16 + measure drift. |
| 2 KV heads on 4 chips — replication / sub-shard decision | medium | medium | Replicate for v0 (1.5 GB/chip budget for 256K context; fits). Revisit at v3 perf pass. |
| Squared ReLU (`relu²`) — fused vs split for precision | medium | low | Probe both compositions via `test_fused_*.py` pattern; prefer split for correctness, revisit fusion at perf pass (Gemma 4 lesson) |
| 30B model weights → ~10-12 min bootstrap | certain | low | Dev harness mandatory from day 1; never restart `serve_cb.sh` per fix |
| Tokenizer file end not captured in research | low | low | On-qb1 `tail -50` of `tokenizer_config.json` at v0.0.1 |
| Chat template active-prompt suffix has TWO branches (`<think>\n` vs `<think></think>`) | medium | medium | Our `_active_prompt_suffix` should cover; verify via `chat_template_invariant.py` test on both branches |
| Param count discrepancy (30B vs 31.6B) | resolved | n/a | Use 32 GB BF16 / 4 chips = 8 GB/chip for memory planning |
| NO RoPE — stale RoPE table application would silently degrade output | low (after audit) | high | Triple-check: q/k go directly q_proj→SDPA with NO intermediate transform. Memory entry predicted at `[[feedback-nemotron3-no-rope-silent-drift]]` |
| Custom modeling code requires `trust_remote_code=True` for HF oracle | certain | low | Set the flag in `hf_reference_nemotron3_nano.py`. Our TT path reads safetensors directly, doesn't need it. |

------------------------------------------------------------------------

## 6. Timeline (Path B — locked)

### Phase 0 — Owned Mamba2 SSD kernel (3-5 weeks)

- G0 read + numpy oracle: 3-4 days (math + research + numpy match against HF)
- G0a isolation harness: 0.5 day
- G1 single-core: 5-7 days (program factory + LLK + first cos gate)
- G2 multi-core: 3-5 days (sharding 64 heads across cores + cross-core reduce avoidance)
- G3 batched: 3-5 days (mirror owned_gdn batched form; B=1..32 gate)
- G4 server integration prep: 1-2 days (kernel ready to call from Python; can't fully integrate until §3b lands the server skeleton)

Subtotal: **~3-5 weeks**

### Phase 1 — Forward / decode / CB / HTTP (2-3 weeks)

Faster than the Path A estimate because Mamba2 is already a single-call
kernel rather than a 200+ ms composite. Bootstrap waits and CB integration
dominate.

- v0.0 + v0.0.1 (oracle + tokenizer verification): 1 day
- v0.1.0 (bootstrap on (1,4)): 1 day (~10-12 min upload * iterations)
- v0.1.1 (Mamba2 L0 via owned kernel call): 1-2 days (integration smoke)
- v0.1.2 (MoE L1, fork from 35B): 1-2 days
- v0.1.3 (Attention L5, simplest we've shipped): 0.5 day
- v0.2 (full forward via per-layer dispatch): 1-2 days
- v0.3 (multi-step decode): 1-2 days
- v0.3.3 (long context smoke at 8K): 0.5 day
- v0.4 (trace capture, fp32-in-trace risk check): 1-3 days
- v1.0-1.6 (CB at B=4): 2-3 days
- v2 (HTTP wire-up + chat smoke): 0.5 day
- buffer / unknowns: 2-3 days

Subtotal: **~2-3 weeks**

### Grand total: 5-8 weeks to v2

Aligned with the brief's §9 estimate. The headline number to track
internally is "Phase 0 G4 lands" as the gating milestone; everything
after is much more predictable.

------------------------------------------------------------------------

## 6.5. Parallel adoption sidecar (NEW 2026-06-04)

While Phase 0 (kernel) runs in the foreground, the qb1 hardware has
idle windows between kernel iterations that we're using to land
**tt-metal adoption wins** from the 3 audits. Tracked separately in
[`research/tt_metal_adoption_plan_2026-06-04.md`](tt_metal_adoption_plan_2026-06-04.md).

- **NOW results (commit `058bedd`)**: task #191 BLOCKED on input-overlap
  shard contract (reverted; bumped to NEXT-F #198 — needs ~30 LOC
  disjoint K/V mem_cfg in `setup_state`); task #192 AUDIT-ONLY (our
  gm4 already clean — no redundant `to_memory_config` calls found).
  See `research/tt_metal_adoption_plan_2026-06-04.md` §2a for the
  detailed findings.
- **NEXT (post-Nemotron G4)**: tasks #193..#198 — GDN bake-off,
  RMSNorm fusion (+12-15 ms/tok), chunk-outer 2048-tok prefill,
  masked fixed-bucket prefill, wider deltanet_recurrence op,
  disjoint K/V mem_cfg for paged_fused (#198, unblocks +1.6 ms/tok).

Zero file overlap with Phase 0 (subagent touches `experiments/serve/
server_gemma4_unified_*` only; main agent touches
`experiments/owned_ops/nemotron3_mamba2_decode_owned/` only).

------------------------------------------------------------------------

## 7. Concrete next steps (Path B — start here)

**Phase 0 (kernel) first.** The Phase 1 forward / oracle work also has
infrastructure prep that can run in parallel with G0..G2, but the
dependency edge is G4 → v0.1.1.

### Immediate (this week)

1. **Task #182.G0 — Read + numpy oracle**
   - Read `mamba-ssm` source at `state-spaces/mamba` (specifically
     `mamba_ssm/modules/mamba2.py` and the chunk_scan_combined kernel).
   - Search tt-metal for any SSM-adjacent ops via
     `experiments/utils/ttnn_introspect.py` (look for "ssm", "mamba",
     "selective", "recurrence", "scan").
   - Read Nemotron-H paper (arXiv 2504.03624) for the canonical SSD
     formulation.
   - Write `experiments/utils/mamba2_numpy_oracle.py`: pure-numpy
     single-token SSD step that bit-matches HF
     `NemotronHMamba2Mixer.forward` at the post-conv1d split point.
   - Gate: numpy oracle's output bit-matches HF eager forward at L0
     step 0, all 64 heads.

2. **Task #182.G0a — Isolation harness**
   - Fork `experiments/utils/test_pattern_a_moe_np.py` →
     `experiments/utils/test_mamba2_decode_isolated.py`.
   - Generates random fp32 ssm_state, bf16 x/z/dt/B/C, runs both numpy
     oracle and (later) ttnn kernel; reports cos + MAD per head.
   - Gate: runs end-to-end on host; deterministic outputs.

3. **Task #182.G0b — qb1 host RAM check (for parallel oracle work)**
   - `ssh qb1 'free -g'` — confirm we have ≥70 GB free for HF AutoModel
     bf16. If not, plan layer-streaming via `accelerate.load_checkpoint_and_dispatch`.
   - Read tail of tokenizer_config.json on qb1 to confirm
     `tokenizer_class` and inline chat_template absence.
   - Gate: oracle / tokenizer infrastructure clear for v0.0 when Phase 1 starts.

### Subsequent (gated)

4. **Task #182.G1 — Single-core kernel** (after G0a passes).
5. **Task #182.G2 — Multi-core.**
6. **Task #182.G3 — Batched.**
7. **Task #182.G4 — Server-side Python wrapper.**

Each task gates the next. **Do NOT spawn parallel implementation tasks
inside Phase 0** — Mamba2 is novel kernel territory.

### Phase 1 starts when G4 lands

At that point, v0.0 (oracle), v0.0.1 (tokenizer), v0.1.0 (bootstrap),
and so on per §3b.

------------------------------------------------------------------------

## 8. Open questions for the user

1. ~~Path A / B / C~~ **RESOLVED 2026-06-04: Path B — owned kernel up-front**.
2. **Long-context priority** — is the 256K claim a v2 must-have, or
   can we ship v0..v2 at 8K and push long-context to a separate workstream?
3. **fp32-in-trace fallback** — if the Blackhole trace hang reproduces
   with fp32 SSM state (per `[[35b-dn-h-state-drift-lever]]`), is
   bf16 SSM state + measurable drift acceptable as a v0 compromise?
4. **G-stage scheduling** — should G0 (research + numpy oracle) start
   immediately, or do we want to triangle on the Mamba2 paper + tt-metal
   SSM survey results before committing?

------------------------------------------------------------------------

## 9. Memory entries predicted during bringup (will populate)

- `feedback_nemotron3_no_rope_silent_drift` — stray RoPE application
  degrading cosine without crashing
- `feedback_mamba2_ssm_fp32_state_in_trace` — Blackhole trace behaviour
  with fp32 recurrent state (mirror of 35B fp32-H)
- `feedback_nemotron3_moe_sigmoid_group_router` — diff from softmax
  routing; how it interacts with our Pattern A batched matmul
- `feedback_nemotron3_relu_squared` — fused vs split decision
- `feedback_nemotron3_thinking_template_double_branch` — `<think>\n`
  vs `<think></think>` active-prompt variants vs our `_active_prompt_suffix`

------------------------------------------------------------------------

## Sources

All architecture claims trace to
[`research/nemotron3_nano_architecture_brief.md`](nemotron3_nano_architecture_brief.md)
§2 (source table). Recipe + REUSE mandate from
[`research/model_bringup_recipe.md`](model_bringup_recipe.md). Reference
plans:
- [`research/35b_cb_bringup_plan.md`](35b_cb_bringup_plan.md) — close MoE precedent
- [`research/gemma4_12b_bringup_plan.md`](gemma4_12b_bringup_plan.md) — close hybrid-dispatch + Llama-RMSNorm precedent
