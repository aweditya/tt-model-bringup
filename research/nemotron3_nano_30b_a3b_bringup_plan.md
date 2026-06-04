# Nemotron-3-Nano-30B-A3B → CB chat server — bringup plan (started 2026-06-04)

**Target model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

**Status**: planning phase. Architecture details pending —
see `research/nemotron3_nano_architecture_brief.md` (research subagent
in flight as of 2026-06-04). This doc has the structural plan (recipe
ladder, REUSE map, backend wiring) and will be updated with the
shape-specific constants once the brief lands.

This is a living plan. Update inline as gates pass / blockers surface.

------------------------------------------------------------------------

## 0. Why this model, why now

Stanford CS440LX demo (27B + Gemma 4 12B + 35B-A3B) shipped 2026-06-04
([poster v5](../presentation/poster.pdf), TUI verified live). MM7 is
the next-model exercise: prove the bringup recipe generalises to a
**different vendor's** A3B-style MoE. Nemotron is the first NVIDIA
model in this repo; it stresses parts of the recipe that were
Qwen/Google-specific (tokenizer family, chat template, MoE routing
convention, special-token handling).

Concrete success criterion: end-to-end HTTP chat via
`TT_BACKEND=nemotron_nano` with a multi-turn coherence test passing
3/3, on qb1 (1×4 P150 mesh), reusing ≥80% of the existing stack.

------------------------------------------------------------------------

## REUSE MANDATE (user-set, durable — `[[feedback-reuse-mandate]]`)

**Before writing any new file: grep the repo. Before writing any new
function: grep for an existing helper. Every PR cites the existing
pattern it forks.**

The two prior MoE/non-trivial bringups (35B-A3B and Gemma 4 12B) seeded
a deep shelf. The first-cut reuse table below — to be refined once the
architecture brief lands — assumes Nemotron is "A3B-style MoE with
top-k routing, SwiGLU experts, GQA/MQA attention". Lines marked
**[arch-dependent]** need confirmation from the brief.

### Likely reusable as-is — zero changes

| Existing | Role for Nemotron |
|---|---|
| `experiments/serve/cb_engine.py`, `cb_scheduler.py`, `cb_metrics.py`, `cb_api.py`, `openai_endpoint.py`, `live_slot_store.py`, `protocol.py` | Production CB stack — model-agnostic |
| `experiments/serve/openai_endpoint.py:_active_prompt_suffix` | Active-prompt-suffix detector (post commit `184753d`) — handles whatever asymmetry Nemotron's chat template has |
| `experiments/serve/scripts/{deploy.sh,serve_cb.sh}` | Deploy + lifecycle — unchanged |
| `experiments/cb/_runner.py` | `project_root()` / `log()` / `bootstrap_*_cb()` factory — fork `bootstrap_nemotron_cb` |
| `experiments/utils/ttnn_introspect.py`, `hf_download.py`, `npz_inspect.py`, `syntax_check.py` | Utility helpers — call as-is |
| `experiments/cb/isolate/paged_sdpa.py`, `paged_update_cache.py` | Primitive SDPA + KV cache probes — model-agnostic |
| `experiments/cb/isolate/chat_template_invariant.py`, `chat_template_inspect.py`, `chat_template_roundtrip.py` | Chat-template gates — reuse as-is |
| `experiments/cb/validate/pc_token_match.py` | Prefix-cache regression gate — works on any HF tokenizer |
| `scripts/chat.py` (TUI) | Demo client — backend-agnostic |
| `scripts/stress_multiturn_http.py`, `stress_ttft_decode.py`, `stress_concurrent_chat.py` | Stress tests — backend-agnostic |
| Two-phase warmup pattern ([[ttnn-multi-trace-two-phase-warmup]]) | Trace capture discipline — universal |

### Likely fork base — 35B (MoE) is the closer reference, Gemma 4 12B for hybrid attention if needed

| Existing | Fork target for Nemotron | Why |
|---|---|---|
| `experiments/serve/server_35b_ttnn.py` (2040 LOC) | `experiments/serve/server_nemotron_nano_ttnn.py` | Closest architecture match — A3B-style MoE; swap shape constants + MoE routing |
| `experiments/serve/server_35b_cb.py` (967 LOC) | `experiments/serve/server_nemotron_nano_cb.py` | Batched-forward CB wrapper for the MoE path |
| `experiments/utils/hf_reference_35b.py` | `hf_reference_nemotron_nano.py` | Numpy oracle (per-layer hidden + L0 sub-captures); prune Qwen-specific hooks |
| `experiments/utils/cosine_ladder_35b.py` | `cosine_ladder_nemotron_nano.py` | Per-position cos against HF; swap layer count + oracle path |
| `experiments/cb/dev/cb35_dev_harness.py` | `cb_nemotron_dev_harness.py` | tmux dev harness — fork after #166 hardening (already merged) |
| `experiments/cb/isolate/cb35_per_layer_drift_pos1.py` | `cb_nemotron_per_layer_drift_pos1.py` | Per-layer drift probe at pos>0 |
| `experiments/cb/isolate/cb35_needle_haystack.py` | `cb_nemotron_needle_haystack.py` | Long-context retrieval gate |
| `experiments/utils/test_pattern_a_moe_np.py` + `test_pattern_a_moe_tt.py` | reuse + fork: `test_pattern_a_moe_nemotron.py` | Numpy oracle for routed-expert FFN — shape-swap; **arch-dependent** on whether Nemotron uses SwiGLU |
| `experiments/utils/test_batched_expert_matmul_isolated.py` | reuse-as-is | Batched expert matmul kernel probe — kernel-shape agnostic |

### Likely NEW (model-specific) — to be confirmed by the architecture brief

| Component | Estimated LOC | **arch-dependent on:** |
|---|---|---|
| `MeshServerState` / `State` constants (HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, N_EXPERTS, N_ACTIVE, MOE_INTER, etc.) | ~50 | All shape fields from `config.json` |
| Routing math: top-k vs sigmoid-gating, normalisation, aux loss handling | ~80 | Nemotron's `router_*` config |
| Norm offset (`+1.0` vs raw) per `[[feedback-qwen36-qnorm-knorm-zero-centered]]` | inline | Whether Nemotron uses Llama-style or Qwen-style RMSNorm |
| SDPA `scale` constant | inline | Whether Nemotron does anything Gemma-style (`scale=1.0`); see `[[feedback-gemma4-sdpa-scale-1]]` |
| RoPE variant (full vs partial vs YaRN scaling) | ~30 | Nemotron's `rope_scaling` config |
| Chat-template peculiarities (special EOS list, active-prompt suffix) | inline + endpoint | `chat_template.jinja` |

------------------------------------------------------------------------

## Memory cross-references (read before any code)

Run `ls ~/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-model-bringup/memory/feedback_*.md` for the full list. The high-leverage ones for an MoE bringup:

- `feedback_reuse_mandate.md` — fork-don't-write rule
- `feedback_two_phase_warmup.md` — multi-trace capture discipline
- `feedback_perf_workflow.md` — 6-step perf workflow (profile → hypothesize → isolate+correct → e2e eager → trace A/B → long context)
- `feedback_correctness_first.md` — never optimise past cosine < 0.99
- `feedback_validate_against_ground_truth.md` — HF oracle, not weaker TT path
- `feedback_qwen36_qnorm_knorm_zero_centered.md` — the `(1+w)` trap. Inspect Nemotron RMSNorm before assuming Llama-style.
- `feedback_gemma4_sdpa_scale_1.md` — model-specific SDPA scale. Check Nemotron's HF source for `self.scaling`.
- `feedback_ttnn_slice_view_decay.md` — view-decay; don't dealloc source while view is live
- `feedback_ttnn_list_rebinding_leaks.md` — list/dict of ttnn tensors needs explicit dealloc before rebind
- `feedback_ttnn_rms_norm_shape_drift.md` — rank-3 rms_norm input; folding to rank-2 drifts in bf16
- `feedback_paged_update_cache_nkv_per_chip.md` — NKV>1 per chip needs split SDPA calls
- `feedback_read_kernel_source_first.md` — read TT_FATAL assertions before forking into a new shape regime
- `feedback_use_existing_isolation_probes.md` — fork an existing probe before iterating in a full forward
- `feedback_cb_backend_dispatch_holes.md` — when adding a new backend to `BACKENDS`, grep every `27b`/`35b` literal in `experiments/serve/`
- `feedback_deploy_serve_files_too.md` — `deploy.sh experiments/serve/*.py` before every server restart
- `feedback_use_dev_harness_for_iteration.md` — never restart `serve_cb.sh` per fix; use the dev harness
- `feedback_harness_state_version_skew.md` — new State fields need `getattr(state, "X", None)` lazy-init
- `feedback_macos_python_libedit_readline.md` — for the validator (local Python)
- `feedback_show_thinking_traces.md` — user-set TUI default
- `feedback_qb1_tmux_for_long_running.md` — tmux, not nohup

------------------------------------------------------------------------

## 1. Stages — to be filled in once the architecture brief lands

Following `research/model_bringup_recipe.md` §2, the staging ladder is:

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v0.0** | HF oracle: `hf_reference_nemotron_nano.py` → `prompt_ids.npy`, `hidden_states.npy` (per-layer), `argmax.npy`, L0 sub-captures | Oracle dir exists; layer count + HIDDEN match config | PENDING |
| **v0.1.0** | Bootstrap on (1,4): mesh, fabric, weights upload, embed lookup, L0 input_layernorm | cos ≥ 0.999 on `embed_scaled` + `in_norm` vs HF | PENDING |
| **v0.1.1-3** | L0 sub-ops one-by-one (q/k/v_proj → q/k/v_norm? → RoPE → SDPA → o_proj → MoE FFN) | cos ≥ 0.999 at each sub-op vs HF | PENDING |
| **v0.2** | All N layers + final_norm + lm_head + softcap? | argmax matches HF at pos 0 | PENDING |
| **v0.3** | KV cache + paged SDPA + multi-step decode | argmax matches HF for tokens 0..5 | PENDING |
| **v0.3.3** | Long context (cos ladder at L=200+, needle haystack) | argmax match ≥ 90%; median cos ≥ 0.99 | PENDING |
| **v0.4** | Trace capture (two-phase warmup) | 100 traced steps == 100 eager token-for-token | PENDING |
| **v1.0-v1.6** | Continuous batching: setup → batched embed/RoPE → batched attention → batched MoE → end-to-end at B=4 | 3a/3b/3c gates PASS at B=4 | PENDING |
| **v2** | HTTP wire-up: register in `cb_api.BACKENDS` + `cb_scheduler._BACKEND_MODULES`, tokenizer + chat template | `curl /v1/chat/completions` returns sensible text | PENDING |

Each row will get an explicit fork-base file (e.g. `forks from server_35b_ttnn.py:bootstrap`) and a concrete acceptance gate once we have the config.

------------------------------------------------------------------------

## 2. Backend wiring — exact edits required at v2

Per `[[feedback-cb-backend-dispatch-holes]]`, register the new backend
in BOTH places, grep all hardcoded `27b`/`35b` literals, and
deploy the WHOLE `experiments/serve/` glob.

```python
# experiments/serve/cb_api.py:50 — BACKENDS dict
BACKENDS = {
    "27b":           ("server_tp",                    "Qwen/Qwen3.6-27B"),
    "35b":           ("server_35b_ttnn",              "Qwen/Qwen3.6-35B-A3B"),
    "gemma4_12b":    ("server_gemma4_unified_ttnn",   "google/gemma-4-12B"),
    "nemotron_nano": ("server_nemotron_nano_ttnn",    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
}
```

```python
# experiments/serve/cb_scheduler.py:50 — _BACKEND_MODULES dict
_BACKEND_MODULES = {
    "27b":           ("server_tp",                "server_tp_cb"),
    "35b":           ("server_35b_ttnn",          "server_35b_cb"),
    "gemma4_12b":    ("server_gemma4_unified_ttnn", "server_gemma4_unified_cb"),
    "nemotron_nano": ("server_nemotron_nano_ttnn",  "server_nemotron_nano_cb"),
}
```

------------------------------------------------------------------------

## 3. Open questions (to resolve before v0.0)

1. **License + access** — is the model gated? Need `HF_TOKEN`? `huggingface-cli login` on qb1?
2. **Tokenizer family** — is this a SentencePiece (T5/Llama-style) or BPE (Qwen/Gemma-style) tokenizer? Vocab size?
3. **Chat template** — does it have an active-prompt asymmetry like Qwen's `<think>` or Gemma's `<|channel>thought\n<channel|>`? Our generic `_active_prompt_suffix` detector should cover it but worth confirming.
4. **MoE routing convention** — top-k softmax (Qwen), sigmoid + top-k + normalise (35B-A3B), top-k-of-sigmoid-gates (Mixtral), or something Nvidia-specific?
5. **Attention shape** — GQA? MQA? Standard MHA? Head dim? KV head count?
6. **RoPE variant** — full RoPE? Partial RoPE (like Qwen3.6 with `partial_rotary_factor`)? YaRN-scaled? What `rope_theta`?
7. **Activation** — SwiGLU is the safe default; some Nvidia models use GeGLU or plain GELU. Check `act_fn`.
8. **Norm offset** — Llama-style RMSNorm (raw `x/rms*w`) or Qwen-style (`x/rms*(1+w)`)?
9. **SDPA scale** — `1/sqrt(d_k)` (most) or something model-specific (Gemma 4: 1.0)?
10. **Any recurrence layers?** — full transformer, or hybrid (mamba2 / RetNet / DeltaNet)? `config.json` `model_type` is the smoking gun.
11. **`transformers` version requirement** — Nvidia models sometimes need a recent `transformers` (≥4.45 or trunk). Plan a `.venv-nemotron` if so.

All of these go in `research/nemotron3_nano_architecture_brief.md` —
the research subagent is fetching them now.

------------------------------------------------------------------------

## 4. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Model gated; needs HF login | medium | Detect at v0.0; document `HF_TOKEN` requirement |
| Custom MoE routing (not top-k softmax) | medium | Brief should surface; fork `_moe_router_topk` if needed |
| Recurrence layer (mamba2 / DeltaNet variant) | low-medium | If yes, this becomes a 1-2 week bringup; if pure transformer MoE, ~3-5 days. Brief will tell. |
| Tokenizer is a custom Nvidia format | low | Detect at v0.0; fallback to T5/Llama SentencePiece path |
| Chat template emits novel asymmetric special tokens | medium | `_active_prompt_suffix` generic detector should cover; validator gates it |
| Weights don't fit (1,4) mesh in bf16 | low | 30B bf16 ≈ 60 GB / 4 chips = 15 GB/chip. Chip DRAM is 32 GB. Comfortable headroom. |
| Active 3B suggests cheap decode but MoE routing per-token adds dispatch overhead | low | We have the Pattern A batched broadcast variant in 35B that's trace-friendly |

------------------------------------------------------------------------

## 5. Timeline estimate (working assumption)

Based on the recipe's track record (27B = 3 weeks, 35B = 2 weeks,
Gemma 4 = ~36 hours), and the assumption Nemotron is closer to "Qwen-MoE
clone with different shapes" than "novel architecture":

- v0.0 oracle: 0.5 day
- v0.1.x L0 sub-op gates: 1 day
- v0.2 full forward: 0.5 day
- v0.3 multi-step decode: 0.5 day
- v0.3.3 long context: 0.5 day
- v0.4 trace capture: 0.5 day
- v1.0-1.6 CB: 1-2 days
- v2 HTTP wire-up + chat smoke: 0.5 day

**Total**: 4.5-6 days if the architecture is "Qwen-MoE-like". Could
extend to 1-2 weeks if Nemotron has a recurrence layer or novel routing.

------------------------------------------------------------------------

## 6. Next concrete step (right now)

1. **AWAIT** research subagent output → `research/nemotron3_nano_architecture_brief.md`.
2. Fill in §1 (stage table) gates + §3 (open questions) with verbatim
   answers from the brief.
3. Refine §REUSE map — mark each row as "reuse-as-is", "fork", or
   "new" based on the architecture diff.
4. Spawn task #182.1 (v0.0 oracle) once the brief is signed off.

If the brief flags a major divergence (e.g. mamba2 recurrence layer or
custom kernel territory), pause to re-scope and triangle with user
before any code lands.
