# Qwen3.6-35B-A3B State Dict Findings — 2026-05-21

Second addendum to `qwen36_35b_a3b_implementation_plan.md` (2026-05-19).
Audit done via `experiments/utils/qwen36_35b_a3b_state_dict_audit.py` (run on
qb2; model snapshot at `~/.cache/huggingface/hub/.../995ad96e...`).

## Confirmed structural prefix

All language-model tensors live under `model.language_model.*` (multimodal
nest). Plan's reference to `model.layers.{i}.*` is wrong; correct is
`model.language_model.layers.{i}.*`.

Top-level text-only tensors:
```
lm_head.weight:                                shape=[248320, 2048], bf16
model.language_model.embed_tokens.weight:      shape=[248320, 2048], bf16
model.language_model.norm.weight:              shape=[2048], bf16
```

Vision tower exists under `model.visual.*` (hundreds of keys, skip for
text-only). MTP head under `mtp.*` (one block — same structure as a
language_model layer).

## Layer 0 DeltaNet — Mamba2-style selective SSM (PLAN WAS WRONG)

The plan said "DN config bit-identical to 27B" — that's incorrect. Layer 0
DN is structurally different:

```
linear_attn.A_log:               shape=[32], bf16     # log of state-transition decay
linear_attn.dt_bias:             shape=[32], bf16     # selective dt bias
linear_attn.conv1d.weight:       shape=[8192, 1, 4]   # depthwise conv on intermediate
linear_attn.in_proj_a.weight:    shape=[32, 2048]     # → A (per-head)
linear_attn.in_proj_b.weight:    shape=[32, 2048]     # → B (per-head)
linear_attn.in_proj_qkv.weight:  shape=[8192, 2048]   # fused q+k+v: 16·128 + 16·128 + 32·128 = 2048+2048+4096
linear_attn.in_proj_z.weight:    shape=[4096, 2048]   # gating, 32·128 = 4096 (matches V dim)
linear_attn.norm.weight:         shape=[128]          # post-DN norm on head_dim
linear_attn.out_proj.weight:     shape=[2048, 4096]   # back to hidden
```

This is a **Mamba2 / selective SSM** architecture, NOT the GatedDeltaNet
(GDN) of Qwen3.6-27B. The defining signals:
- `A_log` + `dt_bias` — Mamba's per-head selective state-transition params
- `in_proj_a` and `in_proj_b` projecting hidden → small A/B params (the
  "selective" pattern: A and B vary per-token via these projections)
- Single fused `in_proj_qkv` for the q/k/v of the linear attention, plus a
  separate `in_proj_z` for the gating value
- `mamba_ssm_dtype: float32` in config

The 27B's `owned_gdn_decode_owned` + `owned_decay_gate` kernels DO NOT
carry over. The new DN block needs a from-scratch implementation as a
Mamba2-style selective recurrence.

**Plan §2 reuse map row "deltanet_step_tp UNCHANGED" is wrong.** Re-categorize as
REPLACED.

## Layer 0 MoE — experts FUSED into single tensors (good news)

```
mlp.gate.weight:                       shape=[256, 2048], bf16    # router
mlp.experts.gate_up_proj:              shape=[256, 1024, 2048]    # fused gate||up for all 256 experts
mlp.experts.down_proj:                 shape=[256, 2048, 512]     # all 256 experts
mlp.shared_expert.gate_proj.weight:    shape=[512, 2048]
mlp.shared_expert.up_proj.weight:      shape=[512, 2048]
mlp.shared_expert.down_proj.weight:    shape=[2048, 512]
mlp.shared_expert_gate.weight:         shape=[1, 2048]            # scalar sigmoid gate
```

Big win vs the 90b reference: experts ship as ONE [256, …] tensor each, not
256 separate tensors. This is exactly what `owned_moe_expert_decode` wants
for indexed gather. The shapes also confirm the plan: `moe_intermediate=512`,
`gate_up_proj=[256, 2*INTER, HIDDEN] = [256, 1024, 2048]` (gate and up
concatenated along the 2nd axis), `down_proj=[256, HIDDEN, INTER] = [256,
2048, 512]`.

Layernorms: `input_layernorm.weight: [2048]`, `post_attention_layernorm.weight:
[2048]`. RMS-norm with no bias, matches 27B.

## Layer 3 (first full-attention layer) — standard GQA shapes

```
self_attn.q_proj.weight:  shape=[8192, 2048]   # 16·512=8192 (Q heads × head_dim … but head_dim=256?)
self_attn.k_proj.weight:  shape=[512, 2048]    # 2·256=512 (KV)
self_attn.v_proj.weight:  shape=[512, 2048]
self_attn.o_proj.weight:  shape=[2048, 4096]   # 4096=16·256 (Q heads × head_dim, then gated by attn_output_gate)
self_attn.q_norm.weight:  shape=[256]          # per-head_dim QK norm (matches 27B QK-rms_norm pattern)
self_attn.k_norm.weight:  shape=[256]
```

Wait — `q_proj=[8192, 2048]`. With head_dim=256 and n_q_heads=16: 16·256
= 4096, not 8192. So `q_proj` projects to **double** the expected size,
which matches `attn_output_gate=True`: the Q projection includes both Q
heads and a gating projection (the gate is computed by the same
matmul that produces Q).

Similarly o_proj input is 4096 = 16·256, but the attn output before o_proj
is 4096 (after gating, the Q-dim half is gated by the gate-dim half via
sigmoid, then o_proj produces hidden=2048).

This matches the 27B's gated-attention pattern (`attn_output_gate=True`). The
TP layout from `gated_attn_step_tp` for 27B applies — modulo the 2 KV heads /
4 chips question (plan §7 risk 5).

## Revised reuse map

| Component | Plan said | Actual status | Why |
|---|---|---|---|
| Backbone tensor prefix | `model.layers.*` | `model.language_model.layers.*` | Multimodal nest |
| DeltaNet block | UNCHANGED (reuse owned_gdn) | **REPLACED** (Mamba2 SSM, new kernel) | Different math (A_log/dt_bias/in_proj_a/b vs gated-delta) |
| Gated attention | MINOR (head shape constants) | MINOR (head shape constants) | Plan was right |
| MoE FFN | NEW | NEW | Plan was right |
| Embed/LM head/RoPE | UNCHANGED (vocab 248320) | UNCHANGED | Confirmed |
| MTP head | UNCHANGED | Same arch as a regular block (full self_attn + MoE) | Different from 27B which had its own MTP block layout |

## Implications for bringup order

1. **The Mamba2 DN block is now part of "the new work"** — it's not just the
   MoE FFN that's new. Effort estimate should grow accordingly.
2. **Start the G0 numpy oracle with Mamba2 DN first** (not MoE) — that's the
   harder unknown. If we get the recurrence right, MoE is straightforward by
   comparison.
3. **The owned_moe_expert_decode kernel design is unchanged** (fused expert
   tensors play nicely with indexed gather, as the plan anticipated).
4. **Re-confirm the day-1 checklist:**
   - [x] Download weights (done, 67 GB)
   - [x] Audit config (done — see `qwen36_35b_a3b_config_audit_2026_05_21.md`)
   - [x] Audit state dict (this doc)
   - [ ] **Next:** write Mamba2 DN block numpy oracle (G0, layer 0) — `experiments/91x_qwen36_35b_dn_numpy_oracle.py`
   - [ ] Then MoE block numpy oracle (G0, layer 0) — `experiments/91y_qwen36_35b_moe_numpy_oracle.py`
   - [ ] Then layer-0 full block (DN + MoE residual) — `experiments/91z_qwen36_35b_layer0_numpy.py`
   - [ ] Then trace-safe router probe on (1,4) mesh
