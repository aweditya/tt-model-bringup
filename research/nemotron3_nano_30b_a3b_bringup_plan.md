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

## 0. Decision required before any code lands

> *(User decision; surface during the next sync)*

**Path A — Ship-correct-first** (recommended; 3-4 weeks to v2):
1. Manual ttnn composite for Mamba2 SSD recurrence (per-head loop with
   `ttnn.mul`/`ttnn.add`/`ttnn.exp`). Slow (~200+ ms/tok projected for
   23 layers) but unblocks the entire ladder.
2. Ship v0..v2 (HTTP chat works, end-to-end), measure perf, demonstrate
   the recipe generalises to a hybrid.
3. Defer Mamba2 owned-kernel work to a v3 perf pass (1-3 additional
   weeks).

**Path B — Owned-kernel-up-front** (6-8 weeks to v2):
1. G0-G4 owned `nemotron3_mamba2_decode_owned` kernel build first
   (per `[[build-kernels-from-scratch]]`).
2. Then ladder + ship.
3. Risk: kernel work can stall the ladder; no end-to-end signal until
   late.

**Path C — De-scope to a non-Mamba2 model** (no Mamba2 needed):
- Pick a different next target (e.g. `mistralai/Mistral-Small-3.2-24B`,
  the original MM5 target). Trades architecture interest for shipping
  speed.

**Recommendation**: **Path A**. The recipe's lesson is "correctness gates
first, then perf". Path A also produces the most learning per week and
keeps the working server demoable throughout. Path B sacrifices visibility
for performance front-loading and risks deep stalls.

This decision lives at the top of this plan because every downstream
stage depends on it. **Resolve before v0.0.**

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

## 3. Bringup ladder (refined from the brief's §9)

Each stage gates the next. **Don't move on until the gate passes.**
Track the gate verbatim in the row.

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

## 6. Timeline (revised)

The brief is explicit (§9): **3-4 weeks to v2 if Mamba2 manual composite
is acceptable; 6-8 weeks if an owned kernel is required upfront.**

Path A breakdown:
- v0.0 + v0.0.1 (oracle + tokenizer): 1 day
- v0.1.0 (bootstrap): 1 day (mostly waiting for upload)
- v0.1.1 (Mamba2 L0 composite): **3-5 days** (new code, no precedent)
- v0.1.2 (MoE L1): 1-2 days (fork 35B's pattern A + routing diffs)
- v0.1.3 (Attention L5): 0.5 day (simplest attention we've shipped)
- v0.2 (full forward): 1-2 days (dispatch + integration)
- v0.3 (multi-step decode): 1-2 days
- v0.3.3 (long context smoke): 0.5 day
- v0.4 (trace, with fp32-in-trace risk): 1-3 days (depending on whether the 35B hang reproduces)
- v1.0-1.6 (CB): 2-3 days
- v2 (HTTP wire-up): 0.5 day
- buffer / unknowns: 3-5 days

**Total Path A: 15-25 working days (3-5 weeks)** depending on Mamba2
composite complexity + trace risk.

Path C (de-scope, e.g. Mistral Small 3.2 24B): ~4-6 days, comparable to
Gemma 4's 36-hour bringup but with a slightly bigger model.

------------------------------------------------------------------------

## 7. Concrete next steps (in order)

1. **Sync with user on §0 decision** — Path A / B / C. **Block until
   resolved.**
2. (If Path A) **Create task #182.1** "v0.0 — HF oracle with Mamba2 hooks":
   - Probe qb1 host CPU RAM (`free -g`); if <70 GB free, use layer-streaming
   - Fork `hf_reference_35b.py` → `hf_reference_nemotron3_nano.py`
   - Add Mamba2 layer hooks (forward hooks on `in_proj` / `conv1d` / `ssm_state` / `out_proj`)
   - Add MoE hooks (router_logits, topk_idxs, topk_weights, per-expert outputs, shared_expert_out)
   - Gate: oracle dir exists with `hidden_states.npy` shape `[53, S, 2688]`
3. (If Path A) **Create task #182.2** "v0.0.1 — tokenizer verification on qb1":
   - `tail -50 ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/.../tokenizer_config.json`
   - Confirm `tokenizer_class` (expected `PreTrainedTokenizerFast`)
   - Confirm no inline `chat_template` overrides the `.jinja` file
4. (If Path A) **Create task #182.3** "v0.1.0 — bootstrap on (1,4)":
   - Fork `server_35b_ttnn.py:bootstrap()` skeleton
   - Implement per-layer-type dispatch table from `hybrid_override_pattern`
   - Upload weights via existing sharded-upload helpers
   - Gate: embed lookup cos ≥ 0.999

Each subsequent task is gated on the prior. **Do NOT spawn parallel
implementation tasks** — Mamba2 is novel; one stage at a time.

------------------------------------------------------------------------

## 8. Open questions for the user

1. **Path A / B / C** (above).
2. Tolerance for v0 perf — is "200+ ms/tok manual composite Mamba2"
   acceptable as a v0 demo target, or do we need to push for the owned
   kernel before any demo?
3. Long-context priority — is the 256K claim a v2 must-have, or can
   we ship v0..v2 at 8K and push long-context to a separate workstream?
4. If we discover Mamba2 needs fp32-in-trace and the Blackhole hang
   reproduces, is bf16 SSM state + measurable drift an acceptable v0
   compromise?

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
