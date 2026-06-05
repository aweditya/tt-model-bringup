# Qwen3.6-35B-A3B Incremental Block Plan — 2026-05-21

Third addendum to `qwen36_35b_a3b_implementation_plan.md` (2026-05-19),
written after `qwen36_35b_a3b_config_audit_2026_05_21.md` and
`qwen36_35b_a3b_state_dict_findings_2026_05_21.md`. Supersedes plan §5
"Staged validation gates" with a finer-grained block-by-block recipe.

**Principle (user directive 2026-05-21):** validate blocks in isolation;
then layer them together; only then attempt coherent text generation. Each
block is one commit. Each commit includes:
- A self-contained numpy/HF oracle (block-only, no broader integration)
- Cosine-comparison gate
- Notes on what was learned

No inline scripts. All probes live in `experiments/utils/` or
`experiments/` per CLAUDE.md non-negotiable #6.

## Stage map

```
G0: numpy oracle, no device                ── pure CPU math validation
G1: single-chip ttnn (qb1)                 ── stock ttnn ops, no mesh, no owned kernels
G2: TP mesh ttnn (qb2)                     ── (1,4) mesh, sharded weights, _tp_all_reduce
G2.5: owned kernels                        ── only if G2 is correct and slow
G3: multi-layer cosine ladder              ── 500-position teacher-forced
G4: server integration + coherent decode   ── handle_generate_tp returns "Paris"
G5: long context (needle-haystack)         ── L=460, 1024, 1990, 4000, 7312
```

Each block below has G0/G1/G2 sub-gates. Start G0 for **all** blocks
before any G1 work — that way the numpy oracle is the ground truth that
every later stage compares against.

## Block ladder (in execution order)

### B0 — Mamba2 DeltaNet block (numpy, G0)

**File:** `experiments/91x_qwen36_35b_dn_numpy_oracle.py`

**Inputs:** `hidden_state[B, T, 2048]`, `ssm_state[B, n_v_heads=32, head_dim=128, head_dim=128]` (fp32 per `mamba_ssm_dtype`), `conv_state[B, n_v_heads*head_dim*2 + 2*n_k_heads*head_dim, conv_kernel=4]`.

**Computation (per layer, per token):**
1. `pre_norm = rms_norm(hidden_state, input_layernorm)`
2. Project: `q = in_proj_qkv[:2048] @ pre_norm[..., None]`, `k = in_proj_qkv[2048:4096] @ ...`, `v = in_proj_qkv[4096:] @ ...`, `z = in_proj_z @ pre_norm`, `a = in_proj_a @ pre_norm`, `b = in_proj_b @ pre_norm`
3. Conv1d: depthwise conv over `[q|k|v]` along seq dim with state buffer (causal, kernel=4)
4. Selective SSM update: `dt = softplus(a + dt_bias)`, `A = -exp(A_log)`, `dA = exp(dt * A)`, `dB = dt * b * x` … the standard Mamba2 selective recurrence
5. Output: `y = ssm_output * silu(z)`, `norm`, `out_proj`

**Validation:** Load HF `Qwen3_5MoeForCausalLM` (or its `language_model`
sub-module), run layer 0's `linear_attn.forward()` against synthetic input,
compare element-wise. Cosine ≥ 0.9999, max|Δ| ≤ 1e-4 (fp32 vs fp32).

**Risk:** I haven't built a Mamba2 selective SSM from scratch before. Reference:
HuggingFace `transformers/models/qwen3_5/modeling_qwen3_5_moe.py`
(`Qwen3_5MoeLinearAttn` class — read this BEFORE writing). Also
`mamba_ssm` python package implements the canonical recurrence.

**Estimated wall:** 1 day to write, 1 day to debug numerical edge cases
(state-init order, conv state layout, dt softplus precision).

### B1 — MoE FFN block (numpy, G0)

**File:** `experiments/91y_qwen36_35b_moe_numpy_oracle.py`

**Inputs:** `hidden_state[B, T, 2048]`, layer-0 MoE weights (FUSED: `gate.weight[256,2048]`, `experts.gate_up_proj[256,1024,2048]`, `experts.down_proj[256,2048,512]`, `shared_expert.{gate,up,down}_proj`, `shared_expert_gate.weight[1,2048]`).

**Computation (per token):**
1. `post_norm = rms_norm(hidden_state, post_attention_layernorm)`
2. Router: `logits = post_norm @ gate.weight.T`, `probs = softmax(logits, dim=-1)`, `top8_vals, top8_idxs = topk(probs, 8)`, `weights = top8_vals / top8_vals.sum(dim=-1, keepdim=True)` (norm_topk_prob=True)
3. Routed experts (loop K=8):
   ```
   for k in range(8):
       e = top8_idxs[k]
       gu = experts.gate_up_proj[e]  # [1024, 2048] — but [:512] is gate, [512:] is up
       gate = gu[:512] @ post_norm; up = gu[512:] @ post_norm  # OR they're interleaved — VERIFY against HF
       expert_out = (silu(gate) * up) @ experts.down_proj[e].T
       routed_out += weights[k] * expert_out
   ```
4. Shared expert: `s_gate = silu(post_norm @ shared.gate_proj.T)`, `s_up = post_norm @ shared.up_proj.T`, `s_pre = s_gate * s_up`, `shared_out = s_pre @ shared.down_proj.T`
5. Scalar gate: `g = sigmoid(post_norm @ shared_expert_gate.weight.T)`; `shared_out *= g`
6. Output: `routed_out + shared_out` (no further norm)

**Validation:** HF Qwen3.6-35B-A3B layer 0 `mlp.forward()` on synthetic
input. Cosine ≥ 0.9999 (fp32).

**Open question to verify in code:** how is `gate_up_proj[256,1024,2048]`
split into gate and up — is it the first/second half of dim 1, or
interleaved? **Must read HF source to confirm before writing the oracle.**

**Estimated wall:** 1 day (block is straightforward modulo the gate_up split).

### B2 — Full layer 0 (numpy, G0)

**File:** `experiments/91z_qwen36_35b_layer0_numpy.py`

**Inputs:** `hidden_state[B,T,2048]` + carryover state (DN ssm_state + conv_state).

**Computation:**
```
ssm_out, ssm_state, conv_state = B0(hidden_state, ssm_state, conv_state, layer0.linear_attn)
hidden = hidden_state + ssm_out                        # residual
moe_out = B1(hidden, layer0.mlp)
hidden = hidden + moe_out                              # residual
```

**Validation:** HF reference layer 0's `forward()`. Cosine ≥ 0.999 (compounded
B0+B1 noise).

**Why this matters:** validates the residual scaffolding works (which is
the only "glue" beyond the two block oracles). If this passes, B0 and B1
combine correctly.

### B3 — Full layer 3 (first full-attention layer; numpy, G0)

**File:** `experiments/91aa_qwen36_35b_layer3_numpy.py` (or fold into B2's file).

**Inputs:** same as B2 + KV cache state. Layer 3 swaps `linear_attn` for
`self_attn` (gated GQA, 16 Q / 2 KV / head_dim 256 / `attn_output_gate=True`).

**Computation:** standard gated GQA with QK-rms-norm, partial RoPE (factor
0.25), causal mask. The MoE block is identical to B1.

**Validation:** HF layer 3 forward. Cosine ≥ 0.999.

**RoPE caveat:** model uses MRoPE (`mrope_interleaved=True, mrope_section=[11,11,10]`).
For text-only inference (height/width axes are 0), MRoPE degenerates to
standard partial RoPE on the temporal-axis freqs. **Probe this in code:
run HF with text-only input and compare the rotated Q/K vs our standard
partial-RoPE output. If exact, ignore MRoPE; if not, implement the
interleaved-section freq picking.**

### B4 — Multi-layer chain (3 layers DN + 1 layer attn; numpy, G0)

**File:** `experiments/91ab_qwen36_35b_block4_numpy.py`

The repeating unit of the backbone is one "block" = 3× DN + 1× attn (per
`full_attention_interval=4`). Validate one full block end-to-end against
HF's `language_model.forward()` truncated to 4 layers.

**Validation:** cosine ≥ 0.997 after 4 layers (drift expected as depth
grows; 27B precedent showed cos ~0.998 per layer in fp32 numpy ref vs HF
fp32).

### B5 — Full 40-layer + embed + lm_head (numpy, G0)

**File:** `experiments/91ac_qwen36_35b_full_numpy.py`

**Computation:** embed → 40 layers (10 of these 4-layer blocks) → final
norm → lm_head. Greedy-decode 1 token after the prompt "The capital of
France is".

**Validation:**
- Compare full hidden-state output element-wise to HF: cosine ≥ 0.99
  (deeper drift is OK in fp32 numpy)
- Final argmax matches HF: top-1 token == "Paris" (or whatever HF
  generates)

**Why this is the G0 grand finale:** if this passes, the math is
end-to-end correct. Every subsequent stage (G1, G2, G2.5) compares
against this oracle, not against HF.

### B6 — Trace-safety probe (G2 precursor on qb2)

**File:** `experiments/utils/qwen36_35b_router_trace_probe.py`

**Test:** can `ttnn.softmax + ttnn.topk + ttnn.sum + ttnn.div` be wrapped
in `begin_trace_capture/end_trace_capture` on (1,4) mesh with replicated
weights? Dynamic top-K indices must flow as tensor data, not as graph
arguments.

**Validation:** capture trace once, execute on N=5 different inputs,
verify outputs match eager mode element-wise.

**Decision gate:** if trace fails → owned router kernel becomes higher
priority (G2.5 instead of G3). If trace succeeds → stock ttnn router is
sufficient for G2.

### B7 — Mamba2 DN single-chip ttnn (G1 on qb1)

**File:** `experiments/utils/g1_mamba2_dn_single_chip.py`

Port B0's numpy math to stock ttnn ops on a single P150. Real layer 0
weights from the HF cache (`safe_open`). Validate cosine ≥ 0.999 vs B0
oracle.

**Risk:** ttnn doesn't have a native selective-SSM op. Will need to
synthesize from `ttnn.matmul`, `ttnn.exp`, `ttnn.softplus`, `ttnn.mul`,
`ttnn.sum` — many dispatches, expected slow. Acceptable for G1 (correctness
over speed).

### B8 — MoE FFN single-chip ttnn (G1 on qb1)

**File:** `experiments/utils/g1_moe_single_chip.py`

Naive Python loop over top-8 expert indices, each running a dense ttnn
matmul. Plan §3.3 strategy (i). Slow (~50 µs × 24 dispatches × 8 experts =
~10 ms/token JUST in dispatch for MoE) but correct.

**Validation:** cosine ≥ 0.999 vs B1 oracle.

### B9 — Full layer 0 single-chip ttnn (G1)

Compose B7 + B8 with residuals. Validates the per-block ttnn ports
chain correctly.

### B10 — TP mesh: Mamba2 DN (G2 on qb2)

Move B7 to (1,4) mesh, shard DN's V-dim (32 heads → 8/chip), use
`_tp_all_reduce` after `out_proj`. Validate cosine ≥ 0.999 vs B0.

### B11 — TP mesh: MoE FFN (G2 on qb2)

Move B8 to (1,4) mesh, intermediate-dim-sharded experts per plan §3.2
layout (C): each chip holds all 256 expert weight slabs but only the
local INTER/4=128 of each expert's intermediate dim. One `_tp_all_reduce`
after routed `down_proj`; another after shared `down_proj`. Validate
cosine ≥ 0.999 vs B1.

### B12 — TP mesh: full layer 0 (G2)

Compose B10 + B11 + residuals on mesh. Validate.

### B13 — Multi-layer cosine ladder (G3)

Pull `experiments/utils/cosine_ladder_tp_compare.py` (from 27B work) and
adapt to 35B model. 500-position teacher-forced cosine vs HF. Confirm no
cliff (or identify where one appears).

### B14 — Server integration (G4)

Fork `experiments/serve/server_tp.py` to `server_tp_35b.py`. Replace:
- `deltanet_step_tp` → `mamba2_dn_step_tp` (from B10)
- `mlp_step_tp` → `moe_step_tp` (from B11)
- Constants (NQ_PER_CHIP=4 vs 6, NKV_PER_CHIP=0.5 → needs replication
  decision per plan §7 risk 5)
- MAX_POS stays at current shipped value (probably 8192)

Validate: `handle_generate_tp --prompt "The capital of France is" --max-tokens 16`
returns "Paris" or coherent text. This is the **first commit that produces
coherent text** — the user's goal.

### B15 — Long-context validation (G5)

Run `experiments/utils/needle_haystack_qb2_tp.py` against the new server.
Verify retrieval at L=460 and L=1024+ (we already have the infrastructure
from 27B work).

### B16 — Owned kernels (G2.5, only if perf demands)

Only after B14 ships and we have a real latency measurement:
- `owned_mamba2_dn_decode` — if Mamba2 DN dispatch overhead dominates
- `owned_moe_expert_decode` — if MoE expert dispatch overhead dominates
- Build G0→G4 ladder per `feedback_build_kernels_from_scratch.md` for each

## Effort estimate (revised)

| Stage | Estimate | Notes |
|---|---|---|
| B0 (Mamba2 DN numpy) | 2 days | New math, never done from scratch in this repo |
| B1 (MoE FFN numpy) | 1 day | Straightforward; gate_up split is the main unknown |
| B2-B5 (composition + full forward) | 2 days | Reuses B0+B1; debug residuals + final norm |
| B6 (trace probe) | 0.5 day | One probe |
| B7-B9 (single-chip ttnn) | 4 days | First ttnn implementation of all new blocks |
| B10-B12 (TP mesh ttnn) | 5 days | TP plumbing for Mamba2 DN is new; MoE TP per plan |
| B13 (cosine ladder) | 1 day | Reuses 27B infra |
| B14 (server integration) | 4 days | New server fork; trace integration |
| B15 (long context) | 1 day | Reuses 27B needle probe |
| B16 (owned kernels) | 14+ days | Defer until B14 measures the perf gap |
| **Total to coherent text (B14)** | **~20 days** | One full month of focused work |
| **Total to perf parity (B16)** | **~34 days** | Matches original plan estimate |

## Day-2 starting point (next commit)

1. **Read HF source** for `Qwen3_5MoeLinearAttn` (mamba2 selective SSM) —
   the exact selective recurrence formulas matter. Save findings in
   `research/mamba2_selective_ssm_notes_*.md` for the next agent.
2. **Read HF source** for `Qwen3_5MoeFFN` to nail down the gate_up_proj
   split (first-half/second-half vs interleaved).
3. **Write B0** (`experiments/91x_qwen36_35b_dn_numpy_oracle.py`). Gate
   on cosine vs HF layer 0.
4. **Commit B0** when green; that's the next milestone.

## What's NOT in this plan

- Vision tower / image inputs / video. Text-only.
- The owned_gdn/owned_decay_gate kernels — they don't apply to Mamba2 DN.
- KV cache layout changes for the gated attention — plan §7 risk 5
  (KV replication on 4 chips) is on the agenda for B12, not now.
- Speculative decoding (D' branch). Defer until B14 ships.
