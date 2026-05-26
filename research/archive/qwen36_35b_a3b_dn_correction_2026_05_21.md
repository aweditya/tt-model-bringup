# Qwen3.6-35B-A3B DN Architecture — Correction to State-Dict Findings (2026-05-21)

Correction to `qwen36_35b_a3b_state_dict_findings_2026_05_21.md` based on
reading the HF reference: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:359`
(class `Qwen3_5MoeGatedDeltaNet`).

## What I got wrong

The state-dict findings doc claimed Layer 0 was a "Mamba2 selective SSM."
This was wrong. The class name is `Qwen3_5MoeGatedDeltaNet` and the
recurrence uses `chunk_gated_delta_rule` / `recurrent_gated_delta_rule` —
the **same gated delta rule** as Qwen3.6-27B.

I was misled by:
- `A_log[32]` parameter (mamba uses this name; so does GDN apparently)
- `dt_bias[32]` parameter (same)
- Split `in_proj_a/b/qkv/z` tensors (27B used one fused `in_proj`)

But reading the forward shows the math IS gated delta rule:

```python
beta = b.sigmoid()                                                  # gating
g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)          # decay
core_attn_out, state = chunk_gated_delta_rule(q, k, v, g, beta, ...)
core_attn_out = norm(core_attn_out, z)                              # RMSNormGated with z
out = out_proj(core_attn_out)
```

That's identical to 27B's `gated_attn_step_tp` math (modulo the decay
parameterization).

## What's the same as 27B

- **Recurrence math** (`chunk_gated_delta_rule` / `recurrent_gated_delta_rule`).
  Our `owned_gdn_decode_owned` kernel implements this and DOES carry over.
- **Q/K/V dims**: 16 K heads / 32 V heads / head_dim 128 (same as 27B).
- **`z` gating**: `z = in_proj_z(x)` of shape `[B, T, v_dim]`, then
  `core_attn_out = RMSNormGated(core_attn_out, z)`. Same.
- **Conv1d**: depthwise on `[k|k|v]` channels, kernel=4 (same as 27B).
- **out_proj**: `[hidden, value_dim]` linear (same).
- **`beta = sigmoid(b)`**: same as 27B.

## What's different vs 27B

1. **Decay parameterization.**
   - 27B: single `decay_proj.weight` projecting `x → per-head decay`, then
     `softplus → exp` (our owned_decay_gate kernel fuses this).
   - 35B: TWO learnable parameters `A_log[32]` and `dt_bias[32]` plus a
     small projection `in_proj_a[32, 2048]` going `x → a[32]`, with formula
     `g = -exp(A_log) * softplus(a + dt_bias)`.

   **Impact:** `owned_decay_gate` kernel needs a REWRITE. New computation
   takes (a, A_log, dt_bias) → g. Conceptually one op family deeper than
   27B's decay_gate (extra `* -exp(A_log)` mul-with-replicated-parameter
   step at the end). Effort ~3 days for a new owned kernel, faster if we
   port the fusion pattern from owned_decay_gate.

2. **Split in_projs (a/b/qkv/z) vs single fused in_proj.**
   - 27B: `in_proj.weight[a+b+qkv+z, hidden]` then slice. One matmul.
   - 35B: 4 separate matmuls (in_proj_a/b/qkv/z). Slightly more dispatches.
     Could fuse at load time by concatenating the weight tensors and using
     a single matmul + slice (matches 27B layout exactly).

   **Impact:** small. Optimize at server-integration time (B14), not a
   correctness concern.

3. **`use_qk_l2norm_in_kernel=True`** is passed to the recurrence. 27B
   computed QK l2-norm OUTSIDE the kernel (our QK RMS-norm fusion lives at
   `Qwen3_5MoeRMSNorm` level). 35B fuses it INTO the recurrence kernel.
   **Impact:** our owned_gdn_decode_owned kernel either (a) ALREADY does
   l2-norm internally (probably not), or (b) needs the l2-norm step added
   to its compute path. Audit during B7/B10.

4. **`beta = sigmoid(b)` uses a single small projection** `in_proj_b[32, 2048]`
   producing per-head beta. 27B used a beta_proj of similar shape. Same idea.

## Implications for the block plan

**Block B0 changes:**
- The reference IS HF's `Qwen3_5MoeGatedDeltaNet.forward()`. Don't reimplement
  the recurrence — just capture HF's output for a fixed input + weights.
- Save (input, conv_state, recurrent_state, output, output_state) to npz
  for use as the cosine reference in B7/B10.

**Block B7 (single-chip ttnn) re-scoped:**
- Reuse `owned_gdn_decode_owned` for the recurrence (no kernel change).
- Build a NEW `owned_a3b_decay_gate_decode` kernel for the new decay formula
  `g = -exp(A_log) * softplus(a + dt_bias)` (or keep it manual via stock
  ttnn ops for G1; own it only if dispatch overhead bites).
- The `in_proj_a/b/qkv/z` split is fine to keep as 4 matmuls in G1; fuse
  later if needed.

**B16 (owned kernels) re-scoped:**
- New: `owned_a3b_decay_gate_decode` (decay formula refactor) — small,
  derived from owned_decay_gate. ~3 days.
- Existing: `owned_gdn_decode_owned` carries over with possible l2-norm
  augmentation. Audit at B10.
- New: `owned_moe_expert_decode` (per plan). ~14 days.

## GDN kernel dataflow principle (user observation 2026-05-21)

"Everything we learned from the gated delta net stuff is super useful as a
principle of mapping dataflow to hardware." Concrete patterns from
`owned_gdn_decode_owned` that port directly:

1. **Per-slot indexed weight reads** in the dataflow kernel — applies to MoE
   expert decode (the K=8 selected experts' weight slabs).
2. **Tile-fused per-head math** in the compute kernel — applies to both
   the new decay-gate kernel and MoE expert SwiGLU + weighted accumulate.
3. **Output gather + reduce in writer** — applies to MoE's weighted sum
   over K=8 expert outputs.

The owned_gdn_decode_owned + owned_decay_gate Q&A in `feedback_*` memory
notes is the de-facto reference for the new owned kernels' dataflow
design.
