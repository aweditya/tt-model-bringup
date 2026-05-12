# The Seven Bugs of Qwen3.6-27B — case studies

Found and fixed across session 2026-05-12 during the Branch III bringup. Each one cost real time. Each one has a detection recipe.

---

## Bug #1 — Missing `linear_attn.norm.weight` (DeltaNet RMSNormGated)

**Symptom**: Cosine ≈ 0 vs HF for the final hidden state. Generation produced `'FRFR...'` indefinitely.

**Where it hid**: `experiments/91f.load_layer_weights_all` for `linear_attention` layers. Our loader requested 13 weight keys per DeltaNet layer; safetensors had 14. The missing one was `linear_attn.norm.weight` — the per-head learned scale for the `Qwen3_5RMSNormGated` module applied *after* the DeltaNet recurrence and *before* the silu-gate.

**HF reference**: `transformers/models/qwen3_5/modeling_qwen3_5.py:540`
```python
core_attn_out = self.norm(core_attn_out, z)   # ← the missing op
```

**How we found it**: HF substep dump (`91o_hf_reference_layer0.py`) showed HF expects 14 weights per layer; our loader only loaded 13.

**Fix pattern**: Add the key to the loader, add a per-head RMSNorm call before the silu-gate. Note the formula: this norm uses standard `weight * x` (not the `(1+w)` form), see bug #4.

**Detection recipe** (add to `bringup_checklist.md`): enumerate every safetensors key for a representative layer of each type. Cross-check against loader. Missing one = silent bug. The `experiments/utils/weight_audit.py --diff-loader` mode catches this in one command.

---

## Bug #2 — Missing `self_attn.q_norm.weight`

**Symptom**: Detected only AFTER bug #1 fixed; cosine still bad in full_attention layers.

**Where it hid**: Same loader, full_attention path. Qwen3.5 applies a per-head RMSNorm to Q *before* RoPE.

**HF reference**: `Qwen3_5Attention.forward`:
```python
query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
```

**Detection recipe**: Same as bug #1.

---

## Bug #3 — Missing `self_attn.k_norm.weight`

**Symptom**: Same as #2.

**Where it hid**: Same loader. Per-head RMSNorm on K before RoPE.

**HF reference**: `Qwen3_5Attention.forward`:
```python
key_states = self.k_norm(self.k_proj(...).view(hidden_shape)).transpose(1, 2)
```

**Detection recipe**: Same as bug #1. These three all surface together once you start counting weights.

---

## Bug #4 — `Qwen3_5RMSNorm` uses `(1.0 + weight)`, not `weight`

**Symptom**: Even after loading all weights, cosine still wrong. Hidden states had wrong scale.

**Where it hid**: We assumed `ttnn.rms_norm(x, weight=w)` matches HF's RMSNorm. Standard RMSNorm is `(x / sqrt(mean(x²)+eps)) * weight`. But Qwen3_5RMSNorm uses `(x / sqrt(mean(x²)+eps)) * (1.0 + weight)` — the learned scale is parameterized as offset-from-1, with `weight init = zeros` (so initial behavior is identity scale).

**HF reference**: `Qwen3_5RMSNorm.forward`:
```python
output = self._norm(x.float())
output = output * (1.0 + self.weight.float())   # ← (1 + w), not w
```

vs Qwen3_5RMSNormGated which uses standard `w*x`.

**How we found it**: weight stats (`experiments/utils/weight_audit.py --stats`) showed `input_layernorm` weights had mean ≈ 0, while `linear_attn.norm` weights had mean ≈ 1. Two different parameterizations in the same model.

**Fix pattern**: At load time, pre-process: `w_loaded = 1.0 + raw_safetensors_weight` for keys that use the Qwen3_5RMSNorm formula. Then `ttnn.rms_norm(x, w_loaded)` produces the correct math.

**Detection recipe**: For every norm weight in the model, check the weight mean. ≈ 0 → offset-from-1 parameterization. ≈ 1 → standard parameterization. Don't assume.

---

## Bug #5 — `ttnn.repeat` is tile, not interleave (GQA broadcast)

**Symptom**: Generation produced garbage subwords; full_attention layers correct, DeltaNet wrong.

**Where it hid**: GQA broadcast. We had `n_k_heads=16, n_v_heads=48, N_REP=3`. To broadcast K from 16 heads to 48 to match V, we did:
```python
k = ttnn.repeat(k.reshape(16, 128), ttnn.Shape([3, 1]))
```
This produces `[h0, h1, ..., h15, h0, h1, ..., h15, h0, h1, ..., h15]` (TILE semantics, like `torch.tensor.repeat`).

**What HF does**: `torch.repeat_interleave(k, 3, dim=2)`, producing `[h0, h0, h0, h1, h1, h1, ..., h15, h15, h15]`.

Different head-to-head mapping → wrong recurrence math.

**How we found it**: empirical probe (`experiments/utils/repeat_semantics_probe.py`) with a tiny tensor that revealed the head ordering.

**Fix pattern**: Workaround via singleton-dim repeat (which is tile == interleave on a 1-wide axis):
```python
def gqa_interleave(t_flat, n_kh, d, n_rep):
    t = ttnn.reshape(t_flat, [n_kh, 1, d])
    t = ttnn.repeat(t, ttnn.Shape([1, n_rep, 1]))
    return ttnn.reshape(t, [n_kh * n_rep, d])
```

**Detection recipe**: Empirically probe any library op that has tile vs interleave semantics ambiguity. Use distinctive values per row (e.g., all-1s, all-2s) so the output pattern reveals the semantics.

---

## Bug #6 — ttnn LLK upstream bug: `int → sfpi::RoundMode`

**Symptom**: `trisc1 build failed. cannot convert 'int' to 'sfpi::RoundMode' [-Wtemplate-body]` whenever ttnn had to JIT-compile a new kernel variant for our shapes/dtypes. Crashed `91q` substep dump and any non-bf8 weight ablation.

**Where it hid**: ttnn's LLK headers (`tt_llk_blackhole/common/inc/sfpu/` and `hw/ckernels/blackhole/metal/llk_api/llk_sfpu/`). Someone refactored `sfpi::int32_to_float(x, rounding)` from `int rounding` to a strongly-typed `enum class RoundMode`, but didn't update the call sites — many SFPU kernels still pass literal `0`.

**HF reference**: N/A — this is a ttnn-side bug, not a model-side issue.

**How we found it**: Cleared the JIT cache, attempted a new kernel build, read the actual compiler error from the cache log.

**Fix pattern**: `experiments/utils/patch_ttnn_llk_roundmode.py --apply` — replaces all `_to_int{16,32}(..., 0)` and `_to_float(..., 0)` and `_to_fp16{a,b}(..., 0)` and `_to_uint{8,16}(..., 0)` with `..., sfpi::RoundMode::NearestEven)`. The enum value `NearestEven = 0` so the patch preserves whatever `int 0` would have produced. Backs up files to `~/tt-xla/.cache/ttnn_llk_backup/<timestamp>/`. Restore mode included.

182 sites across 73 files. Fully idempotent.

**Detection recipe**: When ttnn fails to JIT a kernel and the error mentions sfpi RoundMode, run the patch utility. If it's not yet patched, this is the first thing to check after a fresh ttnn install or wheel upgrade.

---

## Bug #7 — Missing Q-scaling `1/sqrt(head_k_dim)` in DeltaNet

**Symptom**: Generation produced English-ish prefix (`'is 1 0'`) then locked at `'0'` fixed-point. Per-layer cosine for full_attention was 0.9999, but DeltaNet layers compounded badly. Layer 2 cosine collapsed to 0.508 at position 2.

**Where it hid**: I read this line in HF's `torch_recurrent_gated_delta_rule` audit:
```python
scale = 1 / (query.shape[-1] ** 0.5)
query = query * scale
```
And dismissed it with: "scaling Q by a constant doesn't change cosine of the output, so this isn't the bug."

**Why I was wrong**: At very small magnitudes (layer 2's recurrence output has ‖·‖ ≈ 0.0001 per row), the downstream RMSNorm's `eps=1e-6` dominates the variance. When `eps` dominates, RMSNorm is **NOT scale-invariant** — it becomes essentially `output ≈ x × 1/sqrt(eps) = x × 1000`, where the scaling factor depends on `eps` not on the input magnitude. With our Q being 11.3× larger (no scale), our recurrence output is 11.3× larger, putting us in a different eps-vs-variance regime than HF. The per-row probe (`experiments/utils/norm_in_per_row_probe.py`) showed the exact 11.0-11.3× magnitude ratio = `sqrt(128)` = the scaling factor we skipped.

**HF reference**: `torch_recurrent_gated_delta_rule` line 326-327. Also same line in `torch_chunk_gated_delta_rule`.

**Fix pattern**: One line in DeltaNet, after L2-normalizing Q:
```python
q = ttnn.mul(q, 1.0 / (K_DIM ** 0.5))
```

**Impact on per-layer cosines**:
- Layer 0: 0.99662 → 0.99975
- Layer 1: 0.95195 → 0.99995
- Layer 2: 0.50766 → **0.99970** (the catastrophic case)

**Detection recipe**: Two parts.

1. **Don't dismiss audit lines**, ever. If the reference has a line you don't, add it. Argue from empirical ablation, not from analytical equivalence. See `feedback_dont_dismiss_audit_lines.md`.

2. **Per-row diagnostics** when global cosine is high but downstream is wrong. A constant magnitude ratio across all rows is the signature of a missing scaling factor. See `feedback_per_row_diagnostics.md`.

---

## Common patterns across all 7

1. **Five of the seven were visible in the HF source from the start.** Reading carefully on day 1 would have caught them. We didn't because:
   - We trusted our own numpy reference for validation (which had the same bugs)
   - We dismissed audit lines as analytically equivalent
   - Our isolation tests used small-magnitude inputs that didn't exercise the bug

2. **Two were ttnn-specific** (bugs #5, #6): semantics that don't match torch's named equivalents, plus an upstream incomplete-refactor bug.

3. **All seven were silent at the cosine ≥ 0.99 level** in our self-comparison. None of them would have been caught by "cosine ttnn-vs-numpy ≥ 0.99" gates. Only HF as authoritative oracle surfaced them.

The methodology in `wiki/debugging_methodology.md` and the bringup hygiene in `wiki/bringup_checklist.md` are the institutional knowledge that came out of finding all seven.
