# Qwen3.6-35B-A3B correctness plan

Living plan doc. Update as we progress. If context compacts, read this first.

## Goal

Validate that on-device 35B-A3B decode produces correct output across
positions, not just at position 0. The B16/B17 milestone proves position-0
is bit-perfect through all 40 layers and that decode runs in trace mode at
production shape — but we have not measured what happens to the per-position
cosine over 100+ positions.

The 27B precedent says this matters: bf16 prefill noise creates a cliff at
position 129 (top-1 collapses, cos → 0.85). Fix was switching the SDPA
decode `compute_kernel_config` to HiFi2 + no fp32_dest_acc (variant B3 in
`feedback_fp32_sdpa_cliff_probe.md`). Same mechanism is highly likely to
manifest in 35B-A3B since both share dense-attention layers with bf16 KV.

Success criterion: a teacher-forced cosine ladder over 100 positions with
no cliff (cos ≥ 0.99 throughout, top-1 match ≥ 95%). If that holds, extend
to 500 to replicate the 27B P21 bar.

## Decisions (2026-05-22)

- **Anchor**: 100-position teacher-forced cosine ladder first, then react.
  Cheapest macro signal. Mirrors the 27B P21 workflow that found the cliff.
- **Prompt**: 100 tokens of real Wikipedia/book text. Synthetic
  `range(100, N)` IDs produce degenerate baselines that look like the
  change-under-test broke things — see `feedback_fp32_dest_acc_matmul_blackhole_broken.md`.
- **Probe shape**: standalone script `experiments/utils/cosine_ladder_35b.py`.
  Don't add an endpoint to `server_35b_ttnn.py` yet — promote later once the
  probe shape stabilizes (same pattern as 27B's `cosine_ladder`).
- **Granularity**: final hidden + per-layer (all 40 decoder layers + embed
  output = 41 captures). HF oracle already saves this; TT side needs new
  capture path. Final cos is macro signal; per-layer points at layer-of-first-divergence.
- **Host**: qb1 (35B weights cached at `~/tt-xla/.cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B`).
  qb2 doesn't host MoE work today.

## Steps

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Write this plan doc | done | |
| 2 | Pick + tokenize prompt | done | `experiments/utils/ladder_prompt.txt` — 85 Qwen tokens (Wikipedia "Mathematics" opening) |
| 3 | Read server_35b_ttnn.py forward path | done | `step_forward_ttnn(state, tok, pos, capture=dict)` already exposes per-layer hidden states |
| 4 | Generate HF oracle | done | `.cache/hf_oracle_35b_100tok/` on qb1; 13s wall (weights mmap-cached) |
| 5 | Write probe | done | `experiments/utils/cosine_ladder_35b.py` |
| 6 | 10-pos smoke | done | top-1 9/10; first divergence pos 0 L32 |
| 7 | 85-pos full | done | **top-1 69/85 = 81.2%, median cos 0.94, drift concentrated in 10 AT layers** |
| 8 | Analyze cliff / layer-of-first-divergence | done | no cliff; gradual layer-depth drift; L03=AT first to slip; L39 worst (cos 0.70 at pos 16); see `feedback_35b_a3b_attn_layer_drift.md` |
| 9 | Decide next probe | done | **swap manual attention for paged SDPA + B3 compute_kernel_config** — same intervention as 27B P21 |

## Findings (post 85-pos ladder)

**The drift mechanism is not a cliff; it's monotonic per-AT-layer accumulation.** Embed + L00-L02 (all DN) are bit-perfect. From L03 onwards (the first AT layer), every full-attention block adds visible cosine drop. By L39 the worst-position cos is 0.70.

**Root cause hypothesis (high confidence):** `server_35b_ttnn.py:attn_forward_ttnn` does manual `matmul → softmax → matmul` with no compute_kernel_config passed. ttnn's default fidelity on Blackhole is HiFi4 + fp32_dest_acc_en=True — exactly the config the 27B P21 probe identified as buggy for attention numerics (`feedback_fp32_sdpa_cliff_probe.md`). 27B's fix was paged SDPA + B3 (HiFi2, no fp32_dest_acc). Same intervention should apply here.

## SDPA swap — LANDED 2026-05-22 (default flipped to sdpa)

Goal was: replace manual attention with `ttnn.transformer.paged_scaled_dot_product_attention_decode` + B3 config. Done. Implementation behind `state.attn_mode` flag; default flipped to `"sdpa"` on 2026-05-22.

**Result vs success criterion:**

| metric | manual baseline | SDPA warmed | success criterion |
|---|---|---|---|
| top-1 match (85 pos) | 81.2% | 80.0% | ≥ 95% — **MISSED** |
| median cos_final | 0.9384 | 0.9531 | ≥ 0.99 — **MISSED** |
| L39 min cos | 0.7000 | 0.7808 | ≥ 0.90 — **MISSED** |
| pos 0 cos_final | 0.9922 | **0.9999** | — **bit-perfect, win** |
| first divergence layer | L32 | **L40** | — drift starts later, win |

Modest improvement, but the criterion was set assuming SDPA would close the gap. It didn't. SDPA is still net-positive (per-step precision + structural cleanup + well-trodden 27B path) so it stays as default.

**Gotcha:** first SDPA-mode forward call has a JIT-compile race that corrupts pos 0 (DN layers L00-L05 cos 0.55-0.81, L10-L30 all uniform 0.9991). Warmup + `ttnn.synchronize_device` fixes it. Implemented in `cosine_ladder_35b.py`; same warmup likely needed for any new server lifetime exercising SDPA for the first time. See `feedback_35b_a3b_sdpa_swap_result.md`.

## Conclusion: 35B drift is NOT the 27B mechanism

The 27B P21 fix was specific to a buggy compute kernel config on SDPA decode. The 35B never used that kernel, so swapping in the well-configured SDPA only nibbles at the edges. The dominant drift mechanism for 35B is something else. Candidates:

1. **Bf16 noise via the K-broadcast workaround**. RoPE-broadcast still required even in SDPA mode (the `[1, HEAD_DIM]` ttnn slice/concat bug from `feedback_qwen36_attn_rope_single_row_ttnn_bug.md` stands). 4× redundant bf16 ops in the RoPE step per AT layer × 10 AT layers = lots of compounded quantization.
2. **MoE router instability**. 35B-A3B uses top-k expert routing. A single ULP at the gate logits can flip experts, cascading to very different outputs. Has not been measured yet.
3. **GatedDeltaNet recurrent state drift**. 30 DN layers each carrying conv1d + recurrent state. State quantization compounds across positions. Has not been measured.
4. **Smaller hidden dim** (2048 vs 27B's 5120). Less averaging means more bf16 noise per op. Architecture-level, can't fix in code.

## Next probe (chosen) — superseded by SDPA investigation below

Originally planned MoE router stability probe. We pivoted to sub-step capture inside the SDPA path instead. That investigation concluded incorrectly (see below).

## CORRECTNESS INVESTIGATION SUMMARY — 2026-05-22/23

### Investigation timeline

1. **Cosine ladder baseline** (manual + SDPA): top-1 81% / 80%, median cos 0.94 / 0.95, drift concentrated in 10 AT layers (especially L34-L39). Real signal.
2. **Phase 1 drift attribution**: per-AT-layer drop is roughly flat vs position (H2 per-step noise dominant; H5/H6 accumulating noise rejected).
3. **Phase 2 K-broadcast ablation**: broadcast workaround is bit-inert (HF/TT same numerics with or without). REJECTED H2 (RoPE-broadcast as noise source).
4. **Needle-haystack at L=100**: HF retrieves verbatim, TT generates `"N/A\nassistant\n<think>..."`. **Confirmed: drift is user-facing.**
5. **Per-layer ladder on the needle prompt**: identified pos 91 (chat boundary) + L34-L39 as catastrophic. Real signal.
6. **Sub-step capture inside L31**: q_proj_full cos 0.99+ at all positions; post_gate cos drops to 0.66 at pos 91. CONCLUDED: "bf16 SDPA kernel is the noise source." Started scoping a custom fp32 SDPA kernel project.
7. **fp32 KV cache test**: hard-rejected by paged SDPA decode kernel (same as 27B feedback note from months ago).
8. **HF bf16-softmax monkey-patch test**: HF retrieves even with bf16 softmax. REJECTED the "fp32 softmax is the differentiator" hypothesis.
9. **27B needle at L=500 regression**: **PASSES** (verdict Y, retrieved `7TTJ3PCK` in 39s on qb2 prod TP server). Same SDPA kernel, same B3, same bf16 KV. **INVALIDATES the "bf16 SDPA is the bottleneck" conclusion.**

### What's actually true vs what we believed

| Claim | Status |
|---|---|
| 35B-A3B TT fails needle retrieval at L=100 | TRUE — verified |
| HF can retrieve the same needle | TRUE — verified |
| Per-layer cos drops are real in TT's 35B path | TRUE — measurement is real, interpretation may be wrong |
| Drift is concentrated in late layers at chat-boundary positions | TRUE — verified via ladder |
| "bf16 SDPA precision is the bottleneck" | **FALSE** — 27B works at L=500 with same kernel |
| "fp32 softmax is the differentiator vs HF" | **FALSE** — HF retrieves with bf16 softmax |
| "Custom fp32 SDPA kernel project is the fix" | **NOT JUSTIFIED** — no evidence the kernel is the problem |

### Open hypotheses (next session)

- **H12 — 35B-specific TT plumbing bug**: per-chip GQA chip mapping (2 chips per KV head, different from 27B's 1:1), RoPE-broadcast interaction, or some other 35B integration mistake. The kernel is fine; our wiring of it may not be.
- **H13 — Sub-step reassembly bug**: my per-chip → flat reassembly in the sub_capture code may not match HF's layout. The cosine 0.66 at pos 91 might be a SHAPE/ORDER comparison artifact rather than a precision delta.
- **H14 — Chat-template token handling**: TT's failure mode (leaking `assistant\n<think>` tokens) is highly structural. Maybe TT mishandles specific chat-template token IDs (248045, 248046, 248068, etc.) in a way that breaks attention.
- **H15 — HF baseline broken specifically for 35B**: less likely; HF retrieves cleanly.
- **H16 — 35B intrinsically more precision-sensitive at chat-boundaries**: needs more investigation, but only if H12-H14 are ruled out.

### Methodology lessons

1. **27B is a free control** — for any future correctness work on 35B (or any other model), the first sanity check should be: does the sibling model exhibit the same symptom? If not, the bug is model-specific.
2. **Verify sub-step capture reassembly with a known-correct ground truth** before trusting cosine numbers.
3. **Don't propose big-bore fixes** (kernel projects, model rewrites) without first ruling out plumbing/integration bugs AND confirming the diagnosis against a sibling-model counterexample.
4. **Per-layer cosine drops + sub-step localization are useful diagnostic tools but only relative to a verified baseline.** They don't independently prove a mechanism without cross-checks.

### Cheap next-session probes (in order)

1. **35B needle retrieval without chat template** (~15 min). Same haystack + needle + question but raw text continuation. If 35B retrieves raw but fails chat-rendered, the chat-template handling is the bug, not generic attention.
2. **Audit `server_35b_ttnn.py:upload_attn_layer` and `attn_forward_ttnn_sdpa`** vs `server_tp.py:gated_attn_step_tp` line-by-line. Look for per-chip layout divergence, GQA chip mapping mismatches, RoPE-broadcast interactions.
3. **Verify sub-step reassembly** with a tiny known-correct test (identity matrix in, check reassembled output matches expected). Confirms the cos 0.66 isn't a comparison artifact.

### Artifacts (current state)

All ladder JSONs + sub_capture npz + HF oracles are on qb1 under `.cache/sanity_2026_05_22/` and `.cache/hf_oracle_35b_needle100/`. Probe code is permanent at:
- `experiments/utils/cosine_ladder_35b.py` (with --capture-attn-layer, --sdpa-variant, --kv-cache-dtype flags)
- `experiments/utils/cosine_ladder_35b_analyze.py`
- `experiments/utils/cosine_ladder_35b_drift_attribute.py`
- `experiments/utils/cosine_ladder_35b_attn_sub_compare.py`
- `experiments/utils/cosine_ladder_35b_diff.py`
- `experiments/utils/needle_haystack_35b_ttnn.py`
- `experiments/utils/needle_haystack_35b_hf.py`
- `experiments/utils/needle_haystack_35b_hf_softmax_test.py`

The SDPA path + B3 config + sub_capture infrastructure in `server_35b_ttnn.py` is net positive (structural alignment with 27B + reusable diagnostic infra) and stays as default. The K-broadcast workaround can be removed (proven bit-inert) but no urgency.

## Risks / things to track

- **Same bf16 SDPA HiFi4 cliff as 27B** (most likely failure mode). Fix is
  known: `compute_kernel_config_sdpa = HiFi2 + math_approx_mode=False +
  fp32_dest_acc_en=False + packer_l1_acc=False`. If 35B-A3B's attention path
  uses the same config object, the cliff WILL be there.
- **MoE router instability** (35B-A3B-specific). Logit noise can flip
  top-k expert ranks; a single ULP at the gate output can route to a
  different expert. The per-layer cosine ladder won't directly show this
  unless a router flip cascades. Separate probe candidate if the cosine
  ladder is clean but generation is still wrong.
- **RoPE broadcast workaround** (`feedback_qwen36_attn_rope_single_row_ttnn_bug.md`).
  We broadcast K AND V from [1, HEAD_DIM] to [NQ_PER_CHIP, HEAD_DIM] before
  RoPE+cache. Math holds at pos 0; does it hold at pos 99 when cache is
  full of broadcasted noise? Per-layer cosine will surface this if it's
  the issue.
- **B17 trace path may diverge from eager**. The smoke ran 522 ms/tok
  pre-trace. If we want production-shape numbers, the ladder should use
  the traced decode path. But the trace might also have its own
  drift signature. Start with eager teacher-forced to control variables.

## Pointers

- HF oracle: `experiments/utils/hf_reference_35b.py` →
  `.cache/hf_oracle_35b/{prompt_ids,hidden_states,logits,argmax,final_norm,meta}.{npy,json}`.
- TT server: `experiments/serve/server_35b_ttnn.py` (uncommitted
  comment-only diff at session start, no functional change).
- Smoke entry: `experiments/utils/decode_smoke_35b_ttnn.py`.
- B16 memory: `feedback_b16_coherent_text_on_device.md`,
  `feedback_b16i_full_ondevice_35b.md`.
- B17 memory: trace work via commits `71df77b`, `8517fac`, `7269d21`,
  `8a77bbc`.
- 27B precedent: `feedback_fp32_sdpa_cliff_probe.md` (the fix recipe).
- 27B drift root: `feedback_bf16_prefill_drift_cliff.md`.

## Followups (after this work)

- If the cliff is the same SDPA HiFi4 mechanism, audit
  `server_35b_ttnn.py` for the `compute_kernel_config_sdpa` (or wherever
  attention is configured) and apply the B3 variant.
- Promote `cosine_ladder_35b.py` to a `cosine_ladder` endpoint on
  `server_35b_ttnn.py` once the probe shape is proven (so future runs
  don't pay bootstrap cost).
- MoE router stability probe (35B-A3B-specific).
- Long-context needle-haystack at L=500+ on 35B (matches the 27B
  daily-driver gate).
