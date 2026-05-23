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
| 1 | Write this plan doc | in_progress | |
| 2 | Pick + tokenize a 100-tok real-text prompt | pending | stable English passage; save in plan so reruns match |
| 3 | Read server_35b_ttnn.py forward path | pending | find the per-step function + how to capture per-layer states |
| 4 | Generate HF oracle for chosen prompt | pending | `experiments/utils/hf_reference_35b.py`; ~15 min, ~70 GB RAM on qb1 |
| 5 | Write `experiments/utils/cosine_ladder_35b.py` | pending | permanent probe; teacher-forced loop; JSON output |
| 6 | Smoke at 10 positions | pending | validates probe before 100-pos commit |
| 7 | Full 100-position run | pending | ~30-50 min wall on qb1 |
| 8 | Analyze cliff / layer-of-first-divergence | pending | feedback memory + plan doc update |
| 9 | Decide next probe | pending | extend 500 / B3 SDPA variant / MoE router stability |

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
