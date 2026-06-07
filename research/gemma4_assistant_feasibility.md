# Phase 0.A — Gemma4UnifiedAssistantForCausalLM ttnn feasibility verdict

Status: **GO — outcome (a)**. Read 2026-06-07.

## TL;DR

`google/gemma-4-12b-it-assistant` is implementable in ttnn cleanly, with
a *scope reduction* from the design doc. Key findings:

1. **The centroid masked-embedding path is DISABLED for 12B**
   (`use_ordered_embeddings: False` in the actual config). 12B uses a
   standard `lm_head` Linear(1024, 262144). No top-k softmax approximation
   needed. The "centroid" stuff is the E2B-asst path only.
2. **The drafter is PARALLEL, not autoregressive Leviathan-style.** One
   forward → K candidates emerge from K-token sliding context. No
   autoregressive draft loop, no rejected-draft KV rewind machinery.
3. **The drafter shares the target's KV cache** (one tuple per
   `layer_type` = sliding/full). Target server must EXPOSE its
   last-layer KV per step to the drafter — a real API addition.
4. **Requires transformers 5.10.0.dev0+** for HF oracle. Current install
   is 5.9.0 (has `Gemma4AssistantForCausalLM` for E2B, not the unified
   12B variant). Need `pip install -U transformers` before v0.0 oracle.

## What the drafter actually does

From `Gemma4UnifiedAssistantForCausalLM.forward` in transformers main:

```
inputs_embeds:           [B, L, 2*3840=7680]   # concat(target_h_t, target_h_t-1)
   ↓ pre_projection:     Linear(7680, 1024)
   ↓ 4-layer Gemma4 transformer (with shared_kv_states injection):
       layer_types = ["sliding", "sliding", "sliding", "full"]
       Each layer cross-attends to target's KV cache for that layer_type
       (NO autonomous KV cache; use_cache=False)
   ↓ last_hidden_state:  [B, L, 1024]
   ↓ post_projection:    Linear(1024, 3840)    # back to target hidden_size
   ↓ standard lm_head:   Linear(1024, 262144)  # use_ordered_embeddings=False
   ↓ logits:             [B, L, 262144]
```

Inputs:
- `inputs_embeds`: target's hidden state for last 2 tokens, concat'd
- `shared_kv_states`: dict `{"sliding_attention": (K, V), "full_attention": (K, V)}`
  — the target's KV cache at its last sliding attn layer + last full attn
  layer. Target produces these once per step.

Outputs:
- `logits`: K-candidate next-token distribution over vocab
- `last_hidden_state` (post-projected): for next-step's `t-1` slot

## Actual 12b-it-assistant config (verified from HF)

```
Top-level:
  architectures: ['Gemma4UnifiedAssistantForCausalLM']
  model_type: gemma4_unified_assistant
  backbone_hidden_size: 3840              # = target Gemma 4 12B hidden_size
  use_ordered_embeddings: False           # ← centroid path DISABLED
  num_centroids: 2048                     # ← unused (use_ordered=False)
  centroid_intermediate_top_k: 32         # ← unused
  tie_word_embeddings: True               # lm_head ties to embed_tokens
  dtype: bfloat16
  transformers_version: 5.10.0.dev0

text_config (the inner Gemma4 transformer):
  hidden_size: 1024
  num_hidden_layers: 4
  num_attention_heads: 16
  num_key_value_heads: 8
  num_global_key_value_heads: 1
  head_dim: 256
  intermediate_size: 8192
  layer_types: [sliding, sliding, sliding, full]
  sliding_window: 1024
  vocab_size: 262144                      # ≡ target's vocab
  rms_norm_eps: 1e-06
  attention_k_eq_v: True
  rope_parameters:                        # ≡ target's RoPE settings
    full:    proportional, theta=1e6, partial_rotary_factor=0.25
    sliding: default, theta=1e4
```

Param count estimate:
- model.embed_tokens (tied with lm_head): 262144 × 1024 = 268M
- 4 layers × (4 attn matmuls + 3 MLP matmuls + 4 norms) ≈ 100M
- pre_projection: 7680 × 1024 = 7.9M
- post_projection: 1024 × 3840 = 3.9M
- **Total ≈ 380M params** (~0.4B as Google quotes), ~760MB in bf16.

Fits on a single P150 trivially. Even sharing the (1,4) qb2 mesh with
the 12B target (~24GB) is no problem.

## Scope diff vs the original plan-of-action

| Plan-of-action assumed | Reality (post-feasibility) | Impact |
|---|---|---|
| Autoregressive Leviathan drafter (call K times for K tokens) | Parallel drafter (one forward → K logits via sliding window over K positions) | **simpler**, no draft-loop bookkeeping |
| Drafter has own KV cache | Drafter is stateless; consumes target's KV | **smaller**, ~0 KV memory for drafter |
| Centroid embedding (gemma4_assistant E2B pattern) | Standard `lm_head` matmul (12B has use_ordered_embeddings=False) | **simpler**, no topk + scatter + masking dance |
| ~3 days for unusual arch (centroid reverse-engineer) | 4 Gemma 4 layers fork from existing 12B server | **faster bringup**, ~2 days |
| Spec-dec accept/reject walk = Leviathan Algorithm 1 | Accept walk over K parallel candidates from drafter's logits | conceptually simpler; same math |
| Drafter sample on host (DeepSeek-V3 constraint) | Same — host sample for greedy accept walk | no change |

Net: **estimate drops from ~7 days to ~5 days build**. The big win is
the parallel-drafter pattern (no autoregressive bookkeeping) AND no
centroid implementation needed.

## What this changes in the integration plan

### Phase 1 — drafter bringup (revised, ~2 days)

Forks `[[reference-model-bringup-recipe]]`. Drafter forks ~80% of
`server_gemma4_unified_ttnn.py` (we already have Gemma 4 layer code).

| Stage | Adds | Time |
|---|---|---|
| v0.0 | `pip install -U transformers` (need 5.10.0+); HF oracle artifacts (5-prompt) | 1 h |
| v0.1 | Bootstrap drafter on (1,4) qb2 mesh; embed_tokens + pre_projection + lm_head | 3 h |
| v0.2 | 4 Gemma 4 layers (forks 12B per-layer code) — but with `shared_kv_states` injection | 6 h |
| v0.3 | Full forward + argmax matches HF on 5 prompts | 4 h |
| v0.4 | Trace capture (single forward at B=1 + B=K to verify the parallel-K shape) | 4 h |

### Phase 2 — target KV exposure (revised, ~0.5 days)

NEW requirement: target server must produce the last-layer sliding +
full attn KV per step. Currently it doesn't expose this — KV stays
internal to attn_decode_step. We need:

| Step | Adds | Time |
|---|---|---|
| 2.A | Add `state.shared_kv_for_drafter = (K_sliding_last, V_sliding_last, K_full_last, V_full_last)` populated in `attn_decode_step_tt` for the last sliding + last full layer | 3 h |
| 2.B | Verify HF parity: target's shared_kv_for_drafter matches HF's `shared_kv_states` shape + content | 1 h |

### Phase 3 — spec-dec scheduler (revised, ~1 day)

Forks DeepSeek-V3 `_MtpDecodeLoopResult` + accept walk (NOT Leviathan
Algorithm 1 — different pattern since drafter is parallel).

| Step | Adds | Time |
|---|---|---|
| 3.A | `spec_dec_scheduler.py` skeleton — drives target step + drafter step + B=K+1 verify | 3 h |
| 3.B | Accept-walk: compare drafter's K logits vs target's K+1 verify logits, emit longest accept prefix + 1 correction | 4 h |
| 3.C | Bench α + tok/s at K∈{3,5,7} | 1 h |

### Phase 4 — HTTP wire-up (unchanged, ~0.5 days)

## Revised total: ~5 days build (was ~7 days)

| Phase | Original | Revised | Saving |
|---|---|---|---|
| 0 (feasibility + determinism) | 0.5 d | 0.5 d | 0 |
| 1 (drafter bringup) | 3 d | 2 d | -1 d |
| 2 (verify trace + KV exposure) | 1 d | 0.5 d | -0.5 d |
| 3 (scheduler) | 1.5 d | 1 d | -0.5 d |
| 4 (HTTP) | 0.5 d | 0.5 d | 0 |
| Buffer | 1 d | 1 d | 0 |
| **Total** | **7.5 d** | **5.5 d** | **-2 d** |

## Open risks (revised)

1. **`shared_kv_states` shape contract**: HF docstring says
   `(batch_size, num_heads, q_len, head_dim)` (line 209 of modeling). We
   need to confirm our target server's KV cache layout matches (it
   probably does — same Gemma 4 family — but `num_heads` ≠
   `num_kv_heads` matters for GQA).
2. **`pre_projection` input shape**: model takes `[B, L, 2*3840]`,
   meaning concat of 2 target hidden states. Need to confirm WHICH two
   (last + previous? or last + last-shifted? or something else?). Likely
   need to read `Gemma4UnifiedAssistant`'s generate-assistance hook
   in `GenerationMixin.assisted_decoding`. Low risk — once we have
   one HF run working, the format is observable.
3. **`tie_word_embeddings: True`**: lm_head shares weights with
   embed_tokens — we should load embed_tokens.weight once and view it as
   lm_head.weight too. Cheap (same trick as 27B).
4. **Determinism A+B+D port** still recommended for ~5% α floor.
5. **`use_bidirectional_attention: vision`** in text_config — odd field
   for a text drafter; probably inert for token-only path. Verify at
   v0.2.

## Greenlights

- ✅ outcome (a) confirmed: cleanly implementable in ttnn
- ✅ scope DROPPED ~2 days (parallel drafter + no centroid)
- ✅ all ops are standard ttnn (matmul + rms_norm + embedding + softmax)
- ✅ memory fits trivially
- ✅ tt-metal MTP precedent at DeepSeek-V3 `tt/generator.py` 100% applicable

## Action items before Phase 1

1. `.venv/bin/pip install --upgrade "transformers>=5.10.0"` — required for the HF oracle. (Confirm 5.10.0 is on PyPI; if dev0 only, install from main.)
2. Update `research/gemma4_mtp_plan_of_action.md` with the parallel-drafter scope (this doc supersedes some of it).
3. Port determinism A+B+D from 35B to Gemma 4 12B (Phase 0.B, ~2 h, independent value).
4. When user greenlights kickoff, free qb2 by pausing the gm4 perf agent's tmux session.

## Sources

- `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py` (local 5.9.0, 241 lines) — read in full
- `transformers/models/gemma4_assistant/configuration_gemma4_assistant.py` (local 5.9.0, 109 lines) — read in full
- `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4_unified_assistant/modeling_gemma4_unified_assistant.py` (main branch) — read first ~200 lines
- `https://huggingface.co/google/gemma-4-12b-it-assistant/blob/main/config.json` (fetched via `hf_hub_download`)
- `research/gemma4_mtp_design.md` (eb014f5) — original feasibility scoping
- `research/gemma4_mtp_plan_of_action.md` (8a982d5) — phased build plan to update
