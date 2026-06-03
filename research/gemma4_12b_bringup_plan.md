# Gemma 4 12B Unified — bringup plan-of-action

Released 2026-06-03. Companion to the higher-level scoping in
[`gemma4_12b_scoping.md`](gemma4_12b_scoping.md). This document is a
fresh-investigator handoff: every architecture number is sourced; every
code-reuse mapping is `file:line`; every sub-task has a validation gate
with a concrete cosine target.

------------------------------------------------------------------------

## REUSE MANDATE (user-set, durable)

**Before writing any new file, grep the repo for a forkable pattern.
Before writing any new function, grep for an existing helper. Never
re-implement what already ships.**

The 27B + 35B bringups have produced a *deep* utility shelf:

| Existing | Use for Gemma 4 |
|---|---|
| `experiments/cb/_runner.py` `project_root()` / `log()` / `bootstrap_27b_cb()` | New `bootstrap_gemma4_cb()` sister function — same shape, ~20 LOC |
| `experiments/utils/ttnn_introspect.py` | Step 0.1 §6.1 — call as-is over ssh, NO new code |
| `experiments/utils/test_fused_swiglu_isolated.py` + `test_fused_binary_activations_isolated.py` | Pattern for Step 0.2 §6.3 GELU variant probe — fork shape, swap unary list |
| `experiments/utils/hf_reference_35b.py` | Fork → `hf_reference_gemma4_12b.py`; PRUNE MoE/DN hooks, ADD `--hook-rope-layer` |
| `experiments/utils/cosine_ladder_35b.py` + `cosine_ladder_hf_ref.py` + `cosine_ladder_aggregate.py` | Fork → `cosine_ladder_gemma4_12b.py`; swap layer-count + oracle |
| `experiments/cb/dev/cb35_drift_ladder.py` | Fork → `cb_gm4_drift_ladder.py` for harness-callable cosine ladder |
| `experiments/cb/dev/cb35_dev_harness.py` | Fork → `cb_gm4_dev_harness.py`; AFTER #166 hardening lands |
| `experiments/cb/isolate/paged_sdpa.py` | Reuse-or-fork for sliding+global SDPA isolation |
| `experiments/cb/isolate/paged_update_cache.py` | Reuse for KV cache write/read isolation tests |
| `experiments/cb/isolate/chunked_sdpa.py` | Reference for prefill chunked-SDPA pattern (v0.2+ if needed) |
| `experiments/serve/server_35b_ttnn.py` | Fork base for `server_gemma4_unified_ttnn.py`; STRIP MoE/DN, KEEP hybrid dispatch |
| `experiments/serve/server_35b_cb.py` + `server_tp_cb.py` | Fork base for CB shape (v1) |
| `experiments/serve/cb_api.py` + `cb_scheduler.py` `BACKENDS` / `_BACKEND_MODULES` dicts | EDIT, don't fork — add `"gemma4_12b": (…)` entries |
| `experiments/serve/openai_endpoint.py` `_messages_to_prompt` | Reuse as-is (tokenizer-driven; tokenizer ships with the model) |
| `experiments/utils/needle_haystack_*.py` | Reuse for v0.4 sliding-window correctness test |
| `experiments/utils/syntax_check.py` | Local Python-parse helper (no qb1 round-trip) |
| `experiments/utils/full_layer_tp_probe.py` + `tp_attn_traced_probe.py` | Pattern for full-layer / TP-attn isolation probes |
| `experiments/utils/p22_vocab_sharded_lm_head_probe.py` | Reference for vocab-sharded lm_head (with the tied-embed twist) |
| `experiments/utils/paged_vs_nonpaged_sdpa_latency.py` | Pattern for sliding-window SDPA perf isolation (post-correctness) |
| `experiments/utils/tracy_profile_traced_decode.py` + `run_tracy_probe.sh` | Reuse for traced-decode perf (post-v0.4) |
| `experiments/utils/hf_download.py` | Use as-is for Gemma 4 12B safetensors fetch |
| `experiments/utils/npz_inspect.py` | Use as-is for capture-dict inspection during debug |
| `experiments/serve/scripts/deploy.sh` + `serve_cb.sh` | Use as-is; deploy.sh auto-globs `experiments/serve/*.py` |

**Decision rule before any LOC**: state which existing file you are
forking (or "no existing pattern, here's why") in the commit message.
If reviewer can't see the prior art cited, the PR doesn't ship.

**Memory cross-references** (search by name in `~/.claude/.../memory/`):
- `feedback_cb_backend_dispatch_holes` — when adding `gemma4_12b` to
  `BACKENDS`, grep every `import server_tp` AND every `27b`/`35b` literal
  in `experiments/serve/` before deploying.
- `feedback_deploy_serve_files_too` — `deploy.sh` on the WHOLE
  `experiments/serve/` glob; one file stuck in local git burned hours.
- `feedback_use_dev_harness_for_iteration` — once Gemma 4 v0.1 is up,
  switch all probes to harness triggers; do NOT restart `serve_cb.sh`
  per experiment.
- `feedback_qwen36_qnorm_knorm_zero_centered` — the `(1+w)` rule that
  bit us on 35B. For Gemma 4: DO NOT add `+1.0` (Llama-style RMSNorm).

------------------------------------------------------------------------

## 0. SCHEDULING — ACTIVE 2026-06-03

This is **task #165** in the project task tracker. Status: **in_progress**.

**Pivot rationale (2026-06-03)**: 35B drift work (#163) was rate-limited
by the 14-min weight-upload bootstrap per harness restart, and the
harness itself hung silently mid-investigation (see `[[feedback-cb35-dev-harness-hung-2026-06-03]]`).
Pivot to Gemma 4 12B because:
- Iteration cycle: ~5-7 min bootstrap (vs 35B's 14 min) cuts every
  fix-test cycle ~2x.
- Architecture: dense + dual-attention-type hybrid is a cleaner
  vehicle for the same sliding+global mechanism that may sit behind
  the 35B cliff.
- Cross-pollination: Gemma 4's sliding/global hybrid forces us to
  exercise position-dependent paths in isolation — if there's a
  positional-state bug in our codebase (RoPE, paged KV write/read),
  v0.3 will surface it FAST with no MoE/DN confounders.

**Parked, not abandoned**: 35B drift investigation #163 stays at pending
with full staging notes in [`research/35b_drift_next_session_plan.md`](35b_drift_next_session_plan.md)
(§"REAL findings 2026-06-03"). Step 1 = linear-search pos 0-7,
Step 2 = per-layer cos at P_cliff, Step 3 = sub-op probe. Pick back up
after Gemma 4 v0.4 or sooner if the cliff turns out to be a shared
mechanism. Tasks #164 (manual-path repair) and #162 (B>1 empty-slot)
also parked.

**Bringup sequencing**: v0..v4 mirrors the 35B plan
(`research/35b_cb_bringup_plan.md`). Single-slot bit-validated forward
→ batched B>1 → traced → prod wire-up. Concrete sub-tasks in §4.

**Step 0 — hardware-probe pre-flight DONE 2026-06-03**.

### 0.1 sliding_window kwarg — POSITIVE ✓

`ttnn.transformer.paged_scaled_dot_product_attention_decode` exists on
qb1's installed ttnn AND exposes `sliding_window_size (int, optional)`
in its kwargs (verified via `bash scripts/run_remote.sh --no-reset
experiments/utils/ttnn_introspect.py paged_scaled_dot_product_attention_decode
ttnn.transformer --doc`). Sliding-window decode is a **kwarg flip**,
not a new kernel. §3.3 risk dissolved.

Bonus from the doc: `cur_pos (List of int, optional)` documents *"If a
position is given as (-1), compute for the corresponding index in the
batch is skipped."* — exactly the empty-slot semantics 35B's #162 is
fighting. We may be able to use `cur_pos=-1` to safely batch empty
slots in Gemma 4 v1 instead of carrying 35B's masked-multiply reset
pattern forward.

### 0.2 GELU variant — POSITIVE with gotcha ⚠

Probe: `experiments/utils/gemma4_gelu_variant_probe.py` (forked from
`experiments/utils/test_fused_swiglu_isolated.py`, x ∈ [-5, 5], N=4096,
bf16 round-trip on (1,1) mesh).

| Variant | max_abs vs torch tanh GELU | cos |
|---|---|---|
| A: `ttnn.gelu(x, fast_and_approximate_mode=False)` | **1.57e-2** | **0.99999803** |
| B: `ttnn.gelu(x, fast_and_approximate_mode=True)` | 2.64e-2 | 0.99999316 |
| C: `ttnn.mul(x, ones, input_tensor_a_activations=[UnaryOpType.GELU])` | 2.64e-2 | 0.99999316 |

The variants A and B differ algorithmically; A is closer to both torch
references (tanh and exact). The fused-activation path (C) gives the
SAME numbers as B — **the kernel's fused `UnaryOpType.GELU` is the
fast/approximate variant**, NOT the exact GELU. This matters because
35B's SwiGLU shipped `ttnn.mul(silu_gate, up, activations=[SILU])`
fused; Gemma 4 CANNOT mirror that pattern without losing precision.

**Decision for Gemma 4 v0**: split into two ops in the MLP forward:

```python
gelu_gate = ttnn.gelu(s_gate, fast_and_approximate_mode=False)
s_mid    = ttnn.mul(gelu_gate, s_up)
ttnn.deallocate(gelu_gate)
```

Two dispatches per MLP per layer (96 dispatches/token at 48 layers)
instead of one — accept the perf cost for correctness. Post-v0.4 we
can evaluate the fused path as a perf lever; 2.6e-2 max_abs over 48
layers is the same magnitude as 35B's chain drift
[[bf16-chain-drift-at-B-gt-1]] and may compound to argmax flips.

The bf16 round-trip noise (~1.5e-2 max_abs even for variant A) is
expected — same magnitude as 35B observations. Cosine is the bench
metric, not max_abs.

### Net effect on the plan

- §3.3 sliding-window SDPA: ~~1 day~~ → effectively 0; kwarg flip.
- §3.6 GELU_tanh: ~~0.5 day~~ → 1 hour; two-op call shape decided.
- §6.1 BLOCKER-RISK: closed.
- §6.3 risk: closed; left a perf note for later.
- New micro-finding for #162: sliding-window kernel supports
  `cur_pos=-1` to skip empty slots. Try this when Gemma 4 v1 hits
  multi-slot — may resolve 35B's poison too if backported.

### Step 0.3 — separate venv setup DONE 2026-06-03

User-mandated isolation: qb1's main `.venv` has `transformers 5.9.0`
which does not recognize `gemma4_unified` (model released today). To
keep 27B/35B prod pristine, Gemma 4 work runs in a sibling venv.

- One-shot setup: `bash scripts/setup_venv_gemma4.sh qb1` (idempotent;
  creates `~/tt-xla/.venv-gemma4`, installs transformers from git plus
  torch / numpy / safetensors / huggingface_hub / accelerate /
  sentencepiece, verifies `AutoConfig.from_pretrained("google/gemma-4-12B")`
  loads).
- Verified config matches plan §1.1: model_type=`gemma4_unified`,
  text_model_type=`gemma4_unified_text`, n_layers=48, hidden_size=3840,
  vocab_size=262144.
- Run pattern for Gemma 4 HF reference / oracle scripts:
  `ssh qb1 'source ~/tt-xla/.venv-gemma4/bin/activate && cd ~/tt-xla && python -u experiments/utils/hf_reference_gemma4_12b.py'`.
- `scripts/run_remote.sh` hardcodes `.venv/bin/python`; OK for ttnn-only
  probes (Step 0.1/0.2), not for HF-reference scripts. Decision: do
  not edit `run_remote.sh` (keep it dumb); invoke the venv inline for
  HF-dependent scripts. Will revisit if we end up running enough
  Gemma 4 HF probes to justify a `PY_VENV` override.

### Step 0.4 — HF oracle (in progress at commit time)

Forked `experiments/utils/hf_reference_35b.py` →
`experiments/utils/hf_reference_gemma4_12b.py` per REUSE MANDATE:
- KEEP verbatim: argparse, log, save layout, forward call.
- REMOVE: DN sub-hooks (`in_proj_qkv/z/a/b`, `conv1d`, etc.), MoE
  router hooks, `--hook-dn-layer` flag.
- ADD: `pre_feedforward_layernorm` + `post_feedforward_layernorm`
  hooks (the 4-norm structure §1.5), `attn_layer_type` per-layer
  dump in meta.json, `--hook-rope-layer` flag, `get_text_layers()`
  helper that tolerates the gemma4_unified
  `model.model.language_model.layers` path AND fallbacks.

Outputs under `.cache/hf_oracle_gemma4_12b/`:
`prompt_ids.npy`, `hidden_states.npy` `[49, seq, 3840]`,
`logits.npy` `[seq, 262144]` (post-softcap), `final_norm.npy`,
`argmax.npy`, `L0_*.npy` (6 sub-step captures including the two
new pre/post feedforward norms), `meta.json` (with `layer_types`,
`head_dim`, `global_head_dim`, `sliding_window`,
`final_logit_softcapping`, `tie_word_embeddings`,
`hidden_activation`, `base_attr_path`).

First-run cost: ~24 GB download from HF Hub (~5-10 min) + ~3-5 min
CPU bf16 load + ~30 sec forward. Subsequent runs skip the download.
Log goes to `.cache/oracle_runs/gemma4_12b_smoke.log`.

Validation use: v0.1 cosine-ladder probe loads
`hidden_states.npy[1+L]` and compares vs the TT capture from
`server_gemma4_unified_ttnn.step_forward_ttnn(..., capture=cap)`.

### v0.1.0 — bootstrap + embed scale + L0 input_layernorm (in progress)

Fork shape: `experiments/serve/server_gemma4_unified_ttnn.py` (NEW;
forked from `server_35b_ttnn.py` per REUSE MANDATE). v0.1.0 scope:

- Open mesh (1,4).
- Read config from `~/.cache/huggingface/.../config.json` directly
  (not via `AutoConfig`) so the SERVER doesn't depend on
  `transformers >= git_main`. Tokenizer deferred to v0.2.
- Upload embed (replicated, ROW_MAJOR, bf16).
- Upload final_norm with Gemma 4 Llama-style `w` (NO `+1.0`).
- Upload all 48 layer weights: each layer has the FOUR norms
  (input/post_attention/pre_feedforward/post_feedforward; all `w` not
  `(1+w)`), Q/K/V/o projections, q_norm/k_norm, MLP (gate/up/down).
  Sliding layers use NUM_KV_HEADS=8, head_dim=256 (sharded across 4
  chips → 2 KV heads/chip). Global layers (8 of 48) currently upload
  K-replicated across chips; the (1, 1·512) shape is plan §6.8 — v0.3
  will resolve.
- `step_forward_v01(state, tok_id, capture)`:
  `embed(tok) → ·sqrt(HIDDEN) → rms_norm(input_layernorm_w)`.
  Returns two capture entries: `embed_scaled`, `in_norm`.

Validator: `experiments/cb/isolate/gm4_v01_L0_cos.py`. Loads
`.cache/hf_oracle_gemma4_12b/`, runs the forward at
`tok_id=prompt_ids[0]`, cosines `embed_scaled` vs
`hidden_states[0, 0, :]` and `in_norm` vs `L0_in_norm[0, :]`.

Gate: both cos ≥ 0.999.

Expected: bootstrap ~3-5 min (12B bf16, ~6 GB/chip). Forward < 1 sec.
If gate PASSES we have a correct bootstrap + correct embed scale +
correct RMSNorm convention. v0.1.1 then adds q/k/v_proj + q_norm/k_norm
projection sub-steps.

### v0.1 STAGED sub-task breakdown (extends §4)

| Stage | Adds | Validation gate | Status |
|---|---|---|---|
| v0.1.0 | bootstrap + embed scale + L0 input_layernorm | cos ≥ 0.999 on `embed_scaled` + `in_norm` | **DONE 2026-06-03 commit `b9f3c35`** — cos 0.999996 / 0.999991 |
| v0.1.1 | q/k/v_proj + q_norm/k_norm at L0 | cos ≥ 0.999 on `q_norm_out`, `k_norm_out`, `v_proj_out` (vs HF attn sub-hooks) | **DONE 2026-06-03 commit `a35525e`** — 7/7 PASS at cos ≥ 0.99997. Hit a sharder gotcha en route (memory `[[ttnn-shard-1d-vs-2d]]`). |
| v0.1.2 | attention at pos 0 (sliding) + v_norm + o_proj | cos ≥ 0.999 on `mixer_out` | **DONE 2026-06-03** — cos 0.999990 mad 0.0625; found Gemma 4 has v_norm RMSNorm(with_scale=False) NOT in 27B/35B, memory `[[gemma4-v-norm]]`. Numpy reproducer `experiments/cb/isolate/gm4_v012_oproj_sanity.py` isolated the math-model bug from any TT impl issue. |
| v0.1.3 | post_attention_layernorm + residual_1 + MLP + post_ff_norm + residual_2 | cos ≥ 0.999 on all 4 remaining sub-steps + L0 output | IN FLIGHT |
| v0.2 | all 48 layers (sliding + global dispatch) + final_norm + lm_head + softcap | greedy top-1 matches HF at pos 0..4 | |
| v0.3 | KV cache + paged SDPA with `sliding_window_size=1024` | 8-tok generation matches HF token-for-token | |
| v0.4 | Trace capture | 100-step traced == eager | |

### Bootstrap & runtime (verified)

- Bootstrap: **74-84 seconds** total on qb1 from a cold ttnn process
  (vs 35B's 14 min — 11× faster, as the plan §0 estimate predicted).
- 48 layer-weight uploads break down as: layer 10 at ~15s, 20 at ~30s,
  30 at ~46s, 40 at ~60s, all done at ~72s. Linear pace ≈ 1.5 sec/layer.
- Embed lookup + sqrt-scale + L0 forward (through Q/K/V proj + norms)
  completes in < 1 sec post-bootstrap. JIT cache hot.
- All computation on (1,4) P150 mesh; only readback for cosine compare.

------------------------------------------------------------------------

## 1. Verified facts (sourced)

### 1.1 Architecture (from config.json, `huggingface.co/google/gemma-4-12B/resolve/main/config.json`, fetched 2026-06-03)

The repo registers as `Gemma4UnifiedForConditionalGeneration` (NEW class,
distinct from the Gemma 4 family's `Gemma4ForConditionalGeneration` used
by E2B/E4B/26B-A4B/31B). `model_type = "gemma4_unified"`. Text submodel
type = `gemma4_unified_text`.

| Field | Value | Notes |
|---|---|---|
| `text_config.hidden_size` | **3840** | per-chip on (1,4) mesh = 960 |
| `text_config.num_hidden_layers` | **48** | |
| `text_config.num_attention_heads` | **16** | per-chip on (1,4) = 4 Q heads |
| `text_config.num_key_value_heads` | **8** | sliding-layer KV (per-chip = 2 KV heads) |
| `text_config.num_global_key_value_heads` | **1** | global-layer KV (per-chip = ???; see §7.1) |
| `text_config.head_dim` | **256** | sliding layers |
| `text_config.global_head_dim` | **512** | full-attention layers (NEW vs 27B/35B) |
| `text_config.intermediate_size` | **15360** | MLP up/gate per-chip = 3840 |
| `text_config.vocab_size` | **262144** | NEW high (27B=152064/248320, 35B=248320) |
| `text_config.sliding_window` | **1024** | |
| `text_config.tie_word_embeddings` | **true** | embed table === lm_head transposed (NEW) |
| `text_config.final_logit_softcapping` | **30.0** | `logits = 30·tanh(logits/30)` (NEW) |
| `text_config.hidden_activation` | **"gelu_pytorch_tanh"** | NOT SiLU (NEW) |
| `text_config.attention_k_eq_v` | **true** | global layers only — see §1.2 |
| `text_config.num_kv_shared_layers` | **0** | no cross-layer KV sharing (simpler than E2B/E4B) |
| `text_config.hidden_size_per_layer_input` | **0** | NO Per-Layer Embeddings (simpler than E2B/E4B) |
| `text_config.use_double_wide_mlp` | **false** | |
| `text_config.enable_moe_block` | **false** | DENSE model |
| `text_config.rms_norm_eps` | **1e-6** | matches Qwen3.6 |
| `text_config.max_position_embeddings` | **131072** | **128K**, not 256K as the blog claims |
| `text_config.use_bidirectional_attention` | **"vision"** | irrelevant for text-only |
| top-level `dtype` | **"bfloat16"** | |

### 1.2 Layer schedule (verbatim from `text_config.layer_types`, 48 entries)

Pattern: **5 sliding + 1 full, repeating 8x = 48 layers**. The 0-indexed
**full_attention layer positions are: 5, 11, 17, 23, 29, 35, 41, 47**.
Final layer (L47) is full_attention — confirms the model card's
"final layer is always global" claim. All other 40 layers are
sliding_attention.

This dispatch map is **exactly** the shape the 35B layer-type loop already
handles (`server_35b_ttnn.py:1660` `state.layer_types[L]` switch);
literally a `lt == "sliding_attention"` vs `lt == "full_attention"` swap.

### 1.3 RoPE configuration (verbatim from `text_config.rope_parameters`)

```
sliding_attention:
  rope_type:  "default"
  rope_theta: 10000.0
  # partial_rotary_factor implicit at 1.0 → full head rotated

full_attention:
  rope_type:  "proportional"   # p-RoPE
  rope_theta: 1000000.0
  partial_rotary_factor: 0.25
```

**p-RoPE formula (verified from `transformers/modeling_rope_utils.py`
`_compute_proportional_rope_parameters`):**

```
rope_angles = int(rope_proportion * head_dim // 2)   # 0.25 * 512 // 2 = 64
inv_freq_rotated = 1.0 / (base ** (
    torch.arange(0, 2*rope_angles, 2, dtype=torch.int64).float() / head_dim
))                                                   # base=1e6, head_dim=512
inv_freq = torch.cat([inv_freq_rotated,
                      torch.zeros(nope_angles)])     # zero-pad to head_dim//2
```

In plain words: p-RoPE is **standard partial RoPE** (rotate first
`partial_rotary_factor` fraction of head dims), parametrized by
`head_dim_key="global_head_dim"` so the divisor in the exponent is the
**global** head_dim (512), not the sliding head_dim (256). The exponent
formula remains `i/head_dim`; the "proportional" name refers to the
proportional split, NOT a frequency-scaling trick. **The existing
`server_35b_ttnn._apply_partial_rope` (server_35b_ttnn.py:746) already
implements this exact operation** for `partial_rotary_factor=0.5`,
`head_dim=256`. We just need the cos/sin tables baked at the right
`head_dim` and `rope_theta`.

### 1.4 RMSNorm convention (verified from
`transformers/models/gemma4_unified/modeling_gemma4_unified.py
Gemma4UnifiedRMSNorm.forward`)

```python
def _norm(self, hidden_states):
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    return hidden_states * torch.pow(mean_squared, -0.5)

def forward(self, hidden_states):
    normed_output = self._norm(hidden_states.float())
    if self.with_scale:
        normed_output = normed_output * self.weight.float()
    return normed_output.type_as(hidden_states)
```

This is the **Llama** convention (weight = `nn.Parameter(torch.ones(dim))`,
multiplied directly), **NOT the Qwen3.6 `(1+w)` zero-centered convention**
we ship for 27B/35B. This is a TWO-CHARACTER GOTCHA waiting to happen:
35B took several days to root-cause the missing `+1.0` on q_norm/k_norm
(memory: `feedback_qwen36_qnorm_knorm_zero_centered.md`, fix raised
top-1 82/97 → 95/97). For Gemma 4: **do not add `+1.0`** at weight upload.
ttnn.rms_norm with `weight=w` (no `+1.0`) is exactly correct.

### 1.5 Decoder block structure (verified from `Gemma4UnifiedTextDecoderLayer.forward`)

Gemma 2/3 pattern — **four** RMSNorms per layer (not two like Qwen3.6):

```
residual = h
h = input_layernorm(h)
h, _ = self_attn(h)
h = post_attention_layernorm(h)      # NEW: post-norm BEFORE residual add
h = residual + h
residual = h
h = pre_feedforward_layernorm(h)
h = mlp(h)
h = post_feedforward_layernorm(h)    # NEW: post-norm BEFORE residual add
h = residual + h
```

Both 27B `gated_attn_step_tp` and 35B `layer_forward_ttnn`
(server_35b_ttnn.py:1057) only do TWO norms per layer
(`input_layernorm` + `post_attention_layernorm` pre-residual). The
**double-norm pre-and-post** pattern is novel for this codebase.

### 1.6 MLP (verified from `Gemma4UnifiedTextMLP.forward`)

```python
down_proj(act_fn(gate_proj(x)) * up_proj(x))
# where act_fn = ACT2FN["gelu_pytorch_tanh"]
```

Gated SwiGLU shape with **GELU(gate) instead of SiLU(gate)**. ttnn has
`UnaryOpType.GELU` (and `gelu_pytorch_tanh` approximation — we should
verify which exact variant maps to which UnaryOp; see §6 risks).
Otherwise the shape exactly matches our existing
`s_mid = ttnn.mul(s_gate, s_up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`
pattern in `server_35b_ttnn:1162` — drop-in swap of `SILU` for the
GELU equivalent.

### 1.7 Embedding scaling (verified from `Gemma4UnifiedTextScaledWordEmbedding`)

```python
return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
# embed_scale = hidden_size ** 0.5 = sqrt(3840) ≈ 61.97
```

The embed lookup is multiplied by `sqrt(hidden_size)` AT THE EMBEDDING.
27B and 35B do NOT do this (`server_tp.py:1644` and
`server_35b_ttnn.py:1638` both feed the raw embedding directly to the
layer stack). We need an extra `ttnn.multiply` (scalar) after
`ttnn.embedding`, OR fold the scale into the embed table at upload
(cheaper, no runtime cost). The latter requires `tie_word_embeddings`
handling — see §1.8.

### 1.8 Tied word embeddings

`tie_word_embeddings: true` means the lm_head weight IS the transposed
embed table. The HF safetensors **may not contain a separate
`lm_head.weight`** entry — if it doesn't, lm_head must be derived from
`model.language_model.embed_tokens.weight` (transposed). We will not
know until we run `safe_open(...).keys()` on the checkpoint shards. The
35B path uses `load_t(key_to_shard, "lm_head.weight")` unconditionally
(`server_35b_ttnn.py:1772`) — that pattern will break. See §3.1 for
the audit step.

If we fold `embed_scale = sqrt(3840)` into the embed table, we must NOT
reuse the scaled table as lm_head — the unscaled weights must be used
for the lm_head matmul. Concretely: upload TWO copies, or upload the
unscaled embed and scale at runtime.

### 1.9 Self-attention sub-pattern (verified from `Gemma4UnifiedTextAttention.forward`)

```
q = q_proj(h).view([..., n_q_heads, head_dim])
q = q_norm(q)
q = apply_rotary_pos_emb(q, cos, sin)
k = k_proj(h).view([..., n_kv_heads, head_dim])
k = k_norm(k)
k = apply_rotary_pos_emb(k, cos, sin)
v = v_proj(h)                if not use_alternative_attention else (key_states)
# attention scaling = head_dim ** -0.5
```

Q-norm and K-norm are present (matches Qwen3.6 family). Note: K is
normed and ROPE'd; V is the raw projection (or = K when
`attention_k_eq_v` is set on global layers — see §1.10).

### 1.10 `attention_k_eq_v` scope

```python
self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
```

`attention_k_eq_v=true` applies **ONLY to non-sliding layers** (the
8 global layers — L5/11/17/23/29/35/41/47). For those layers:
`v_proj is None` and `value_states = key_states` post-norm-post-rope.

**Sliding layers always project K and V independently**, even when the
flag is set. This MUST be respected per-layer in the bootstrap and
forward — same dispatch shape as our existing `lt == "linear_attention"`
branch but on a different axis.

------------------------------------------------------------------------

## 2. Code reuse map (file:line → role)

| Gemma 4 concern | Existing pattern to fork | Reuse confidence |
|---|---|---|
| Bootstrap skeleton (mesh + fabric + cfg + tokenizer + safetensors enumeration) | `server_35b_ttnn.py:1720-1755` `bootstrap()` | HIGH — copy verbatim, swap MODEL_ID + skip MoE upload |
| Per-layer dispatch on `state.layer_types[L]` | `server_35b_ttnn.py:1660-1684` for-loop | HIGH — swap `linear_attention` → `sliding_attention`, `full_attention` stays as full_attention name |
| Pre-allocated index buffers (`tok_buf`, `cur_pos_buf`, `rot_idxs_buf`) | `server_35b_ttnn.py:1407-1410`, `server_tp.py:99-105` | HIGH — same scheme |
| On-device embed + cos/sin lookup via `ttnn.embedding` | `server_35b_ttnn.py:1636-1655`, `server_tp.py:1644-1663` | HIGH — but TWO cos/sin tables (one per RoPE config) |
| Partial RoPE rotation (rotate first 25%, passthrough rest) | `server_35b_ttnn.py:746-769` `_apply_partial_rope` | HIGH — algebra identical; tune `n_heads`, `ROTARY_DIM`, `HEAD_DIM` constants per call site |
| Paged KV cache + `paged_update_cache` + `paged_scaled_dot_product_attention_decode` | `server_35b_ttnn.py:842-870`, `server_tp.py` `gated_attn_step_tp_paged` | HIGH for global layers; **sliding wants the `sliding_window_size` kwarg already present in tt-metal** — see §3.3 |
| Vocab-sharded lm_head + on-device argmax + `all_gather` + slice + multicore argmax | `server_tp.py:1680-1687` | HIGH for the dense bottom of the forward |
| Continuous-batching registration via `BACKENDS` dict | `cb_api.py:50-58`, `cb_scheduler.py:50-58` | HIGH — register `"gemma4_12b": ("server_gemma4_unified_ttnn", "google/gemma-4-12B")` |
| CB batched forward shape (`forward_batch_tp_inner`, `setup_cb_state`, `cb_reset_slots`, `update_input_buffers_batched`) | `server_tp_cb.py` (CB1/CB2/CB4 validated) | MEDIUM — sliding window changes per-slot cur_pos→K-window math (see §3.4) |
| HF oracle pattern | `experiments/utils/hf_reference_35b.py` | HIGH — swap model id, drop MoE-specific hooks |
| Cosine ladder | `experiments/utils/cosine_ladder_35b.py` + `cosine_ladder_hf_ref.py` | HIGH — same JSON shape, swap n_layers from 40 → 48 |
| Dev harness | `scripts/run_harness_tmux.sh` + `experiments/cb/_runner.py` (commit `28dc985`) | HIGH — register a new harness name like `gm4` |
| OpenAI HTTP path + chat template | `cb_api.py` + `openai_endpoint.py` `_messages_to_prompt` | HIGH — relies on `state.tok.apply_chat_template`; Gemma 4 ships its own template |

### 2.1 Concrete constants table (Gemma 4 12B vs 27B vs 35B, per-chip on (1,4) mesh)

| Constant | 27B (server_tp) | 35B (server_35b_ttnn) | **Gemma 4 12B (target)** |
|---|---|---|---|
| `HIDDEN` (full) | 5120 | 2048 | **3840** |
| `HIDDEN_PER_CHIP` | 1280 | 512 | **960** |
| `NUM_LAYERS` | 64 | 40 | **48** |
| `NUM_Q_HEADS` | 16 (=4/chip) | 16 (=4/chip) | **16 (=4/chip)** |
| `NUM_KV_HEADS_SLIDING` | n/a (4 GQA flat) | n/a (DN) | **8 (=2/chip)** |
| `NUM_KV_HEADS_GLOBAL` | (4) | n/a (DN) | **1 (=??? — see §7.1)** |
| `HEAD_DIM_SLIDING` | 256 | 256 | **256** |
| `HEAD_DIM_GLOBAL` | (256) | (256) | **512** (NEW) |
| `ROTARY_DIM_SLIDING` | 128 | 128 | **256** (full of 256) |
| `ROTARY_DIM_GLOBAL` | n/a | n/a | **128** (0.25 of 512) |
| `INTERMEDIATE` | 17408 (=4352/chip) | 6144/expert (MoE) | **15360 (=3840/chip)** |
| `VOCAB` | 152064 (padded 248320) | 248320 | **262144** |
| `MLP activation` | SiLU(gate)·up | SiLU(gate)·up | **GELU_tanh(gate)·up** |
| `final_norm` convention | (1+w) | (1+w) | **w** (Llama-style) |
| `q_norm/k_norm` convention | n/a | (1+w) | **w** (Llama-style) |
| `final_logit_softcapping` | none | none | **30.0** (NEW) |
| `tie_word_embeddings` | false | false | **true** (NEW) |
| `embed_scale` | 1.0 | 1.0 | **sqrt(3840)** (NEW) |
| `attn_output_gate` (Qwen quirk) | yes | yes | **NO** |

------------------------------------------------------------------------

## 3. What's actually NOVEL (ranked by risk)

### 3.1 (P0) Tied word embeddings + checkpoint key audit — **2 hours, low risk**

The 35B path hard-codes `lm_head.weight` (`server_35b_ttnn.py:1772`).
For Gemma 4 the checkpoint may only contain
`model.language_model.embed_tokens.weight`. **Action**: run `safe_open`
on one shard, dump `f.keys()`, and conditionally derive lm_head from
embed when `tie_word_embeddings=True`. **Validation**: pos-0 cos vs HF.
**Risk**: forgetting the transpose convention (embed is `[V, H]`, lm_head
matmul wants `[H, V]`).

### 3.2 (P0) Embedding scale + final logit softcap — **2 hours, low risk**

* **Embed scale**: multiply embed lookup by `sqrt(3840)`. Cheapest path:
  fold into the embed table at upload time (`embed_w_np *= sqrt(3840)`)
  — but only safe if lm_head is loaded SEPARATELY (i.e., not tied). For
  tied embeddings: scale at runtime (`x_tt = ttnn.mul(x_tt, scale)` or
  bake into the *first* `input_layernorm` weight). Decide based on §3.1
  audit.
* **Logit softcap**: insert `logits = softcap * tanh(logits / softcap)`
  BEFORE the argmax. On-device: `ttnn.mul(ttnn.tanh(ttnn.div(logits,
  30.0)), 30.0)`. **CRITICAL** for sampling correctness — argmax of
  `tanh(x)` ≠ argmax of `x` only when ties occur on saturated values,
  but sampling distributions are very different. Bench: insert at end
  of `forward_token_tp_inner` between vocab-shard slice and argmax.

**Validation gate**: logits cos ≥ 0.9999 vs HF at pos 0.

### 3.3 (P1) Sliding-window paged SDPA — **1 day, LOW risk** (kernel exists)

**De-risking finding**: `tt-metal/ttnn/cpp/ttnn/operations/transformer/
sdpa_decode/sdpa_decode.cpp` lines 45, 105, 163, 231 expose
`std::optional<uint32_t> sliding_window_size` on
`paged_scaled_dot_product_attention_decode` and its variants. The
nanobind (`sdpa_decode_nanobind.cpp:46-72`) exposes it to Python as
`sliding_window_size=<int>`. The kernel-level masking
(`device/kernels/dataflow/dataflow_common.hpp:715-772`) handles BOTH
causal and centered sliding-window cases.

**Risk reduction**: sliding-window SDPA is NOT a new kernel — it is a
**kwarg flip** on the existing op. The original scoping doc estimated
"2-3 days" because it assumed a manual K/V slice or new kernel; both
unnecessary.

**Action**: call `paged_scaled_dot_product_attention_decode(...,
sliding_window_size=1024)` on sliding-layer dispatch, and call without
that kwarg on global-layer dispatch.

**Open question to verify on hardware**: does the qb1 ttnn build
include this kwarg surface? Run `ttnn_introspect`
(`experiments/utils/ttnn_introspect.py`) to confirm the Python signature
exposes `sliding_window_size`. If it does NOT (older ttnn build),
**rebuild ttnn from a tt-metal main >= 2025-XX with the sliding-window
sdpa_decode patch** (see §6 risk).

**Validation gate**: per-layer cos ≥ 0.999 at pos 0..1023 (within
window) AND pos 1024..2047 (token at pos K should attend to
[K-1023..K]). Compare directly to HF oracle's `model.layers[L].attn`
output snapshot.

### 3.4 (P1) Dual head_dim (256 sliding, 512 global) — **1-2 days, MEDIUM risk**

This is the meaningful novel issue. The KV cache must be allocated
TWICE per layer-type:
- Sliding layers: `[NUM_BLOCKS, n_kv_heads=8, BLOCK_SIZE=32, head_dim=256]`
- Global layers: `[NUM_BLOCKS, n_kv_heads=1, BLOCK_SIZE=32, head_dim=512]`

Per-chip (NCHIPS=4):
- Sliding: 2 KV heads × 256 dim
- Global: see §7.1 — 1 KV head over 4 chips is the open question

**Memory budget for KV cache at MAX_POS=8192** (initial bringup cap):
- Sliding × 40 layers: `8192 * 8 * 256 * 2(K+V) * 2(bf16) * 40 = 1.0 GB`
- Global × 8 layers (assume 1 KV head per chip in worst case): `8192
  * 1 * 512 * 1(KV) * 2(bf16) * 8 = 64 MB`
  (`attention_k_eq_v` halves it.)
- Total ≈ 1.1 GB / 4 chips = ~275 MB/chip. Fits trivially.

**Risk**: paged_update_cache assumes a uniform `head_dim` per cache call,
which is fine because we'll never mix; but the trace machinery captures
across all layers in one trace. Verify trace_region_size handles
heterogeneous KV cache shapes — same trace just touches different
sub-tensors per layer.

**Validation gate**: KV cache at L5 and L0 each independently bit-equal
to HF after warmup; full forward cos ≥ 0.999 at pos 0..15.

### 3.5 (P1) Decoder layer's four norms — **0.5 day, LOW risk**

Add `post_attention_layernorm` and `post_feedforward_layernorm` to the
weight loader; insert two more `ttnn.rms_norm` calls into
`layer_forward_ttnn`. Both are POST the sub-block, PRE the residual add.

Llama-style weights (no `+1.0`) — easy to upload but **easy to confuse
with the Qwen3.6 +1 convention** during the port. Reviewers MUST check
the rms_norm weight upload for any accidental `+ 1.0`.

**Validation gate**: per-sub-block cos ≥ 0.9999 (norm is a pointwise
deterministic op; should bit-match).

### 3.6 (P1) GELU_tanh activation — **0.5 day, LOW risk**

`gelu_pytorch_tanh` ≡ `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
x^3)))`. ttnn has `UnaryOpType.GELU` (exact) and `gelu_approximate`
(tanh-approx). Verify which (or both) match `gelu_pytorch_tanh` via a
1D micro-probe (`experiments/utils/ttnn_introspect.py` shows the
available enum); use the matching one as the second arg to
`ttnn.mul(s_gate, s_up, input_tensor_a_activations=[ttnn.UnaryOpType.<X>])`.

**Validation gate**: MLP cos ≥ 0.9999 at one position on random input
vs HF MLP.

### 3.7 (P1) `attention_k_eq_v=true` on global layers — **0.5 day, LOW risk**

For global layers only, skip the v_proj matmul; alias V := K (after
K's RoPE and K_norm). Slightly cheaper (saves one matmul + 0.5 KV-cache
storage on global layers). The architecture conditional is just
`if not is_sliding_layer:`.

**Validation gate**: pos 0 cos ≥ 0.9999 on a global layer.

### 3.8 (P2) Long-context 128K — **DEFER to a separate workstream**

`max_position_embeddings = 131072` but our infra caps at MAX_POS=8192
(server_tp.py:53). Initial bringup at MAX_POS=8192 mirrors 27B/35B
prod. Long-context push is a separate task — and it shares mechanism
with task #163 (35B drift cliff), so we should not invest until that's
root-caused.

------------------------------------------------------------------------

## 4. Sub-task breakdown

Each sub-task: prerequisites, files to touch, validation gate (cosine
vs HF oracle at what positions, target value), estimated effort.

### v0.1 — Skeleton, single layer, no KV cache (~2 days)

* **Prereqs**: task #163 root-caused; v0 plan reviewed.
* **Files to create**:
  - `experiments/serve/server_gemma4_unified_ttnn.py` (NEW, fork
    `server_35b_ttnn.py` shape; ~800 LOC initial)
* **Files to touch**: none yet (CB integration in later subtask)
* **Forward stages**: tokenize → embed×sqrt(H) → L0 forward
  (input_layernorm → q/k/v_proj → q_norm/k_norm → manual SDPA on
  sliding window, no KV cache yet → post_attention_layernorm → residual
  → pre_feedforward_layernorm → mlp(GELU_tanh-gated) →
  post_feedforward_layernorm → residual) → repeat L=0..0 only.
* **Validation gate**: per-sub-step cos ≥ 0.9999 vs HF Layer 0 hook at
  prompt "The capital of France is" (5 tokens). Reuse
  `experiments/utils/hf_reference_35b.py` hook pattern (`--hook-attn-layer
  0`) ported to Gemma 4.
* **Done when**: L0 output array matches HF L0 output cos ≥ 0.9999.

### v0.2 — All 48 layers, no KV cache, prompt-only (~1 day)

* **Prereqs**: v0.1 done.
* **Forward stages**: full 48-layer ladder; sliding layers use the
  PROMPT (full seq) K/V — no cache, no paged SDPA. Final norm
  (Llama-conv) + lm_head + softcap + argmax.
* **Validation gate**: greedy next-token argmax matches HF at positions
  0..4 of the 5-token prompt; per-layer cos ≥ 0.999 throughout.
* **Done when**: 5/5 argmax PASS + full per-layer ladder above
  threshold.

### v0.3 — Decode KV cache (single-slot, sliding + global) (~1-2 days)

* **Prereqs**: v0.2 done; confirmed `sliding_window_size` kwarg exists
  on the installed ttnn build (§6 risk).
* **Files**: same module; add `setup_kv_caches`, `paged_update_cache`
  per layer-type, `paged_scaled_dot_product_attention_decode` with
  `sliding_window_size=1024` on sliding layers.
* **Validation gate**:
  - 8-token generation matches HF greedy at positions 0..7 token-for-token.
  - cos ≥ 0.999 at pos 0..7 final logits.
  - At pos 1024 (1 token past window): sliding layers attend ONLY to
    [pos 1..1024]; verify by injecting a probe at pos 0 that should
    have decayed by pos 1024.
* **Done when**: greedy generation matches HF for at least 100 tokens
  on the prompt "The capital of France is", AND the sliding-window
  decay test passes.

### v0.4 — Trace capture single-slot, eager comparison (~1 day)

* Mirrors 35B's B17 trace work
  (`feedback_b16i_full_ondevice_35b.md`).
* `forward_token_gemma4_inner(state)` reads only from
  `state.tok_buf/cur_pos_buf/rot_idxs_buf`; `update_input_buffers`
  writes them out-of-trace.
* **Validation gate**: traced output bit-identical to eager output for
  100 consecutive steps; perf is irrelevant at this stage.
* **Done when**: 100-step traced generation == 100-step eager
  generation (token-for-token).

### v1 — Continuous batching B=4 (~2-3 days)

* **Prereqs**: v0.4 done; #162 (35B empty-slot poison) decided
  (fix-or-defer) — if deferred, default `TT_CB_SLOTS=1` for Gemma 4
  too, until we have a known-good multi-slot story.
* **Files to create**: `experiments/serve/server_gemma4_unified_cb.py`
  (mirror `server_35b_cb.py` / `server_tp_cb.py`).
* Functions to implement: `setup_cb_state`, `cb_reset_states`,
  `cb_reset_slots`, `update_input_buffers_batched`,
  `forward_batch_tp_inner`. The CB shape question for sliding is:
  **per-slot windows**. cur_pos differs per slot; `sliding_window_size`
  is a single compile-time int; the kernel masks based on `cur_pos -
  window_size` per slot. Verify on hardware.
* **Validation gate** (mirrors `cb_validate_27b.py`):
  - 3a: B=1 batched_forward equals single-slot forward (logit cos 1.0)
  - 3b: identical inputs in two slots produce identical outputs
  - 3c: distinct inputs in two slots stay isolated (no cross-talk)
* **Done when**: 3/3 PASS at B=4.

### v2 — Server wire-up + chat smoke (~1 day)

* **Files to touch**:
  - `experiments/serve/cb_api.py` lines 50-58 → add `"gemma4_12b":
    ("server_gemma4_unified_ttnn", "google/gemma-4-12B")` to `BACKENDS`.
  - `experiments/serve/cb_scheduler.py` lines 50-58 → add
    `"gemma4_12b": ("server_gemma4_unified_ttnn",
    "server_gemma4_unified_cb")` to `_BACKEND_MODULES`.
  - Audit `cb_api.py:282-296` for any 27B/35B-specific defaults
    (`TT_CB_SLOTS`, `TT_CB_TOPK_K`) and add a `gemma4_12b` branch.
    Memory: `feedback_cb_backend_dispatch_holes.md` — grep `27b` /
    `35b` matches in `experiments/serve/`; add coverage.
* **Validation gate**:
  - `curl /v1/models` returns `google/gemma-4-12B`.
  - `curl /v1/chat/completions` with `{"messages":[{"role":"user",
    "content":"Hello"}]}` returns coherent text in < 5 sec.
  - 100-token completion of "The capital of France is" matches the v0.3
    HF-validated path (same first 8 tokens).
  - `/health` shows ready=true.
* **Done when**: `TT_BACKEND=gemma4_12b ./scripts/serve_cb.sh start`
  fully boots and chat works end-to-end through the HTTP API.

------------------------------------------------------------------------

## 5. HF oracle plan

**Memory budget**: AutoModelForCausalLM on CPU + bf16 = ~24 GB for
12B (vs 35B's ~70 GB). qb1 has ~478 GB available → trivial.

**Where to cache on qb1**: `.cache/hf_oracle_gemma4_12b/` (top of
`tt-xla/` per project convention; matches 35B oracle dirs).

**Prompts (priority order)**:
1. **5-tok smoke**: `"The capital of France is"` — 5 token ids, dump
   per-layer hidden_states (`[49, 5, 3840]`) and logits.
2. **Chat-template smoke**: `apply_chat_template([{"role":"user",
   "content":"Hello, are you a language model?"}])` — measures the
   chat formatting path that production serves through.
3. **Long-context probe (DEFERRED)**: 85-tok mathematical ladder
   (mirrors 35B's `cb35_drift_long_*` probe). Not needed for v0; saved
   for the drift investigation that will inevitably come.
4. **Needle haystack at L=2048**: validates sliding-window correctness
   (a needle injected at pos 0 should be UNRETRIEVABLE by an attention
   query at pos 2048 if window=1024 — that's a STRONG correctness
   signal that we're not accidentally running full-attention).

**What to capture**: same shape as `hf_reference_35b.py`
(`meta.json`, `prompt_ids.npy`, `hidden_states.npy`,
`logits.npy`, `final_norm.npy`, `argmax.npy`), plus a new
**`attn_layer_type.json`** dump that records which layers in the HF
trace ran sliding vs full (defensive: HF could in principle re-order
or hot-swap; we want to crash early if our `state.layer_types` doesn't
match HF's).

**Hooks**: port `--hook-attn-layer` from `hf_reference_35b.py:64-71`:
sub-step capture of `q_proj/k_proj/v_proj/q_norm/k_norm/o_proj/
input_layernorm/post_attention_layernorm/pre_feedforward_layernorm/mlp/
post_feedforward_layernorm`. Drop the `--hook-dn-layer` flag (no
DeltaNet path in Gemma 4). Add `--hook-rope-layer N` to dump cos/sin
tables actually used (sliding vs proportional).

**Driver**: new file `experiments/utils/hf_reference_gemma4_12b.py`
(~250 LOC, fork `hf_reference_35b.py` and prune MoE / DN hooks).

------------------------------------------------------------------------

## 6. Risks + open questions

### 6.1 (BLOCKER-RISK) Does qb1's installed ttnn expose `sliding_window_size`?

The kwarg is present in tt-metal main C++ (verified at
`experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/
sdpa_decode/sdpa_decode_nanobind.cpp:46-72`). qb1's ttnn build was last
patched 2026-05-28 (memory:
`feedback_owned_decay_gate_shipped`). Whether THAT build includes the
sliding-window kwarg depends on the upstream commit qb1's tt-metal
sits at — unverified.

**Action before v0.3 starts**: run
`PYTHONPATH=... .venv/bin/python -u experiments/utils/ttnn_introspect.py
ttnn.experimental.paged_scaled_dot_product_attention_decode` on qb1;
look for `sliding_window_size` in the Python signature. If missing, the
fallback (re-build ttnn from tt-metal main + sliding-window
sdpa_decode patches; needs ~30 min build on qb1) is acceptable but
adds calendar risk.

### 6.2 Multimodal-token poisoning of the text-only forward

`config.json` declares `image_token_id=258880`, `audio_token_id=258881`,
`boi_token_id=255999`, `eoi_token_id=258882`, `boa_token_id=256000`,
`eoa_token_index=258883`, `video_token_id=258884`. These IDs are
INSIDE the 262144 vocab. If the tokenizer or chat template ever emits
one of these on a text-only prompt (because the user typed a
`<|image|>` literal, or because the chat template injects an image
placeholder when it shouldn't), the embedding lookup will return the
real multimodal embedding (which was trained to be replaced by raw
image patches at inference) and produce garbage.

**Action**: before bringup ships to users, write a defensive guard at
`update_input_buffers`/`_messages_to_prompt` that asserts the prompt
token ids do NOT include any of the multimodal special-token IDs. Fail
loudly with HTTP 400.

### 6.3 GELU variant mismatch

`gelu_pytorch_tanh` is the tanh approximation. ttnn exposes
`ttnn.UnaryOpType.GELU` (exact erf-based) and likely
`gelu_approximate`. We must verify which produces bit-equiv to HF's
`gelu_pytorch_tanh`. Numerical diff: ~1e-4 in the worst case — large
enough to drift cos through 48 layers.

**Action**: micro-probe pre-v0.2. Compare ttnn variants pointwise on
`[-5, 5]` against `torch.nn.functional.gelu(approximate="tanh")`.

### 6.4 KV-cache layout heterogeneity in a single trace

We've never captured a trace where some layers touch a head_dim=256 KV
cache and others touch a head_dim=512 KV cache. Should "just work"
(trace is a sequence of ops; each `paged_scaled_dot_product_attention_decode`
binds its own tensors), but the program cache will have 2x entries.
Worst case: trace_region_size of 800 MB
(`server_tp.py:189`) is insufficient. **Action**: bump to 1200 MB on
first capture attempt; reduce later.

### 6.5 Logit softcap and on-device argmax interaction

`logits = 30 * tanh(logits / 30)` is monotonic, so argmax is invariant.
BUT for sampling (top-k, top-p, temperature), the distribution shape
matters. We MUST apply softcap before sampling. For trace_compatibility:
ttnn has `ttnn.tanh` and `ttnn.mul` with scalar; trivial to insert.

### 6.6 Tied embeddings vs vocab-sharded lm_head

Vocab-sharded lm_head (`server_tp.py:1680-1687`) expects an upload that
shards the lm_head weight along the vocab axis. If lm_head is tied
(== embed.T), uploading the embed table replicated AND uploading
lm_head sharded means storing the same 1 GB of weights twice. Memory
fine; ugly. Alternative: a custom ttnn path that all-gathers the per-
chip hidden, then runs an embedding-shaped lookup. Defer optimization;
upload twice for v0.

### 6.7 Bootstrap time

35B bootstrap is ~14 min, 27B is ~6 min. 12B should land at ~5-7 min
(smaller weights). Original scoping doc estimate ~10-12 min was
defensive; if it's faster, all the better.

### 6.8 Open: `num_global_key_value_heads = 1` per-chip on (1,4) mesh

The full-attention layers have ONE KV head TOTAL. On a (1,4) mesh, we
either (a) replicate that KV head to all 4 chips and let each chip do
the same matmul, or (b) keep KV on chip 0 and broadcast Q/results. The
27B paged-SDPA path assumes KV is sharded over chips; (a) is the
natural extension but breaks the "K_PER_CHIP = NUM_KV/NCHIPS=0" math.
**Decision needed** before v0.3 — likely (a) (replicated, all-chips
compute the same global attention; cheap because head_dim=512 × 1 head
= 64 KB/token).

------------------------------------------------------------------------

## 7. Note on the existing `research/archive/gemma4_architecture.md`

There is a 2026-04-22 archive doc covering Gemma 4 E2B/E4B/26B-A4B/31B.
It is **stale for our task**:
- It does NOT cover the 12B Unified SKU (released 2026-06-03).
- 12B has `num_kv_shared_layers=0` and `hidden_size_per_layer_input=0`
  — none of the E2B/E4B PLE+KV-sharing complexity.
- 12B is dense (`enable_moe_block=false`), not MoE.
- 12B's `head_dim=256` sliding / 512 global agrees with the archive
  pattern; sliding_window=1024 also agrees with 26B-A4B/31B.

The archive doc remains useful as **cross-reference** for the
dual-head-dim semantics, dual RoPE config semantics, and Gemma 4
family invariants. **Do not delete**; cite for §1.3 family context.

------------------------------------------------------------------------

## 8. What is explicitly NOT in scope for this plan

- Image, audio, video input. Strictly text-only. The unified
  multimodal projection layers are NEVER loaded.
- Long context > 8192 tokens. We cap MAX_POS=8192 for v0..v2; long
  context push is task #168 (not yet scheduled).
- Performance optimization. Get correctness first; perf is a separate
  workstream (analogous to 35B's A002+A003+A004+A008 perf session
  AFTER bringup).
- Owned kernels (analog of `qwen36_gdn_decode_owned`). 12B has NO
  linear-attention; nothing to fuse on the DN side. Owned MLP/SDPA
  kernels are a perf concern, not a bringup one.

------------------------------------------------------------------------

## 9. Appendix: file inventory after v2

NEW files (rough LOC estimates):

```
experiments/serve/server_gemma4_unified_ttnn.py     ~1100 LOC  (fork of 35b_ttnn minus MoE+DN)
experiments/serve/server_gemma4_unified_cb.py       ~600 LOC   (mirror of 27b_cb / 35b_cb)
experiments/utils/hf_reference_gemma4_12b.py        ~250 LOC   (fork of hf_reference_35b)
experiments/utils/cosine_ladder_gemma4_12b.py       ~150 LOC   (fork of cosine_ladder_35b)
experiments/cb/dev/cb_gm4_drift_bf16.py             ~50 LOC    (probe wrapper)
experiments/cb/validate/cb_gm4_validate_27b.py      ~80 LOC    (3a/3b/3c validators)
```

EDITED files (minimal):

```
experiments/serve/cb_api.py        BACKENDS dict + per-backend defaults
experiments/serve/cb_scheduler.py  _BACKEND_MODULES dict
scripts/run_harness_tmux.sh        register 'gm4' harness session
```

Total NEW code: ~2.2k LOC. Comparable to the 35B bringup volume.
