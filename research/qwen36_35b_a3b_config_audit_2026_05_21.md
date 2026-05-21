# Qwen3.6-35B-A3B Config Audit — 2026-05-21

Addendum to `qwen36_35b_a3b_implementation_plan.md` (2026-05-19). Live audit
of the HF config + state dict before starting bringup. Three findings the
plan didn't anticipate.

## Plan expectations confirmed (no surprises)

| Field | Plan | Actual config |
|---|---|---|
| `num_hidden_layers` | 40 | 40 |
| `hidden_size` | 2048 | 2048 |
| `num_experts` | 256 | 256 |
| `num_experts_per_tok` | 8 | 8 |
| `moe_intermediate_size` | 512 | 512 |
| `shared_expert_intermediate_size` | 512 | 512 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 2 | 2 |
| `head_dim` | 256 | 256 |
| DeltaNet: `linear_num_value_heads` / `linear_num_key_heads` | 32 / 16 | 32 / 16 |
| DeltaNet: `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | 128 / 128 |
| `linear_conv_kernel_dim` | 4 | 4 |
| `partial_rotary_factor` | 0.25 | 0.25 |
| `vocab_size` | 248320 | 248320 |
| MTP head | exists | `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=False` |
| `attn_output_gate` | (implicit from Qwen3.6) | True |
| `full_attention_interval` | "3 GDN + 1 GA" pattern | 4 (every 4th is full_attention) |
| `rms_norm_eps` | 1e-6 | 1e-6 |

## Three new findings (not in plan)

### A. Model is MULTIMODAL (vision-language)

Architecture: `Qwen3_5MoeForConditionalGeneration`. Top-level config splits
into `text_config` + `vision_config` with `image_token_id=248056`,
`video_token_id=248057`, vision_start/end token IDs. Model has a vision
tower we aren't using.

**Impact:** All the MoE/DN/attn params live under `text_config.*`, NOT at
top level. State-dict prefixes likely have `language_model.` or `model.`
nesting we need to discover from `safe_open` (deferred to first weight
shard download).

**Decision:** text-only inference. Skip vision tower entirely. Server doesn't
need to handle image tokens since the user prompt path is text-only.

### B. MRoPE — multimodal interleaved RoPE (not standard partial RoPE)

`rope_parameters: {'mrope_interleaved': True, 'mrope_section': [11, 11, 10],
'partial_rotary_factor': 0.25, 'rope_theta': ...}`

The 32 rotary frequencies (head_dim × partial_rotary_factor / 2 = 256 ×
0.25 / 2 = 32) are split into 3 sections of 11/11/10 for
temporal/height/width axes. With `mrope_interleaved=True`, those positions
interleave in the head_dim layout (not contiguous chunks).

**For text-only inference:** the temporal axis carries the token position;
height and width are uniformly 0 across all positions. MRoPE then degenerates
to a per-section rotation where sections 2 and 3 see zero rotation (identity).
**Net effect for text:** equivalent to standard partial RoPE with the
temporal frequencies only (11 freqs active), other 21 freqs identity.

**Risk:** the interleaved layout means the index pattern that picks freqs is
different from 27B's contiguous-section layout. Need a small numpy probe
against HF to confirm the freq-picking math before integrating.

**Existing 27B RoPE infrastructure** in `server_tp.py` assumes contiguous
partial RoPE. Will need extension (probably a new helper, not a server-wide
rewrite). Defer this concrete decision until G1.

### C. DeltaNet SSM state is fp32, not bf16

`mamba_ssm_dtype: float32` in the text_config. Our 27B server stores the
SSM state at bf16. For 35B-A3B, the reference uses fp32 for the recurrence
state.

**Impact:** owned_gdn_decode kernel and owned_decay_gate kernel both pass
the SSM state through their compute path. Need to:
1. Check whether the kernels currently accept fp32 state input (probably no
   — they're typed bf16 in `experiments/owned_ops/qwen36_gdn_decode_owned/*`).
2. Either widen the kernels to support fp32 state, OR cast bf16 → fp32 at
   the boundary and run a fp32 SSM update outside the kernel (numerical
   parity check needed; might lose the owned-kernel speedup).

**Decision:** keep this as a G2 (mesh integration) risk. The G0 numpy oracle
naturally uses fp32 (numpy default), so it gets the reference math right by
construction. G2 has to confront the kernel dtype mismatch.

## Plan section §7 risk re-evaluation

The plan flagged 7 risks. Status after this audit:

| Risk | Status |
|---|---|
| 1. Trace-safe dynamic top-K | Still open. Day-1 checklist item. |
| 2. `ttnn.sparse_matmul` on Blackhole | Still open (fallback only) |
| 3. bf4 vs bf8 routed experts | Still open; start with bf8 per plan |
| 4. B=1 sparsity efficiency | Resolved theoretically; verify in G2 |
| 5. **2 KV heads / 4 chips** | Still open — need decision before G4 |
| 6. MTP head shape | Still open; defer to post-G4 |
| 7. Bootstrap weight-load time | Estimated 10-15 min cold (matches our 27B 17 min cold) |
| **NEW: MRoPE freq-section interleaving** | Open; add to G1 probe scope |
| **NEW: fp32 SSM in owned kernels** | Open; G2 integration risk |
| **NEW: Multimodal state-dict prefix discovery** | Open; resolves on first shard download |

## Day-1 status update

Plan day-1 checklist:
- [x] Re-read implementation plan and 91w numpy oracle template (this doc + `experiments/91w_qwen36_27b_prefill_numpy_ref.py`, plus `experiments/90b_moe_numpy_reference.py` for MoE pattern)
- [ ] Download Qwen3.6-35B-A3B weights — **IN PROGRESS** (started 2026-05-21 07:56 UTC on qb2; ~14 GB at +2 min, total ~70 GB; ETA ~10-15 min)
- [ ] Write `experiments/91x_qwen36_35b_moe_numpy_oracle.py` — pending download completion
- [ ] Trace-safety probe for router ops on (1,4) mesh — pending
