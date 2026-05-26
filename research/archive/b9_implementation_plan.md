# Phase B'9 — fp32 Residual Stream Implementation Plan

**Goal**: Break Qwen3.6-27B out of the `'FR'` fixed-point by upgrading the residual stream from bf16 to fp32, eliminating cumulative bf16 rounding across 64 layers.

**Hypothesis (from B'8 diagnostic)**: 64 layers of bf16 residual adds compound ~3-5 in absolute error at ‖x‖≈150, distorting the lm_head landscape enough that top-1 to top-2 margins (~0.062 today) flip noisily — and worse, the top-5 candidates are all junk subwords. fp32 residual should restore both correct top-1 selection AND reasonable top-5 distributions.

## Design

### What changes

| Tensor | Before | After |
|---|---|---|
| Weights (bf8/bf16 storage) | unchanged | unchanged |
| KV cache (bf16) | unchanged | unchanged |
| **Hidden state `x` (residual)** | **bf16** | **fp32** |
| Matmul DEST accumulator | fp32 (via HiFi4) | fp32 (unchanged) |
| Matmul output cast | bf16 | **fp32** (natural via fp32 input) |
| Pointwise op outputs (silu, mul) | bf16 | **fp32** (natural propagation) |

### Why this is small

API probe (91k) confirmed: feed fp32 `x` in at the start of the layer, and every downstream op (`rms_norm`, `linear`, `add`) propagates fp32 to its output naturally — no `dtype=` kwarg needed. The implementation collapses to:

**Single change in `91h.py:206`:**
```python
# OLD
x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)
# NEW
x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
```

Plus a few defensive typecasts inside `91f` where bf16 sneaks in:

1. **`gated_attn_step`**: K/V get downcast to bf16 for the cache write (storage). The downcast is fine — but the K/V read from cache (after the numpy roundtrip) comes back bf16, and the SDPA input Q is fp32 → mixed dtype. **Fix**: explicit typecast Q to bf16 right before SDPA call; SDPA output may be bf16; typecast back to fp32 before the residual add.
2. **`deltanet_step`**: `conv_state` is bf16 (storage), `ssm_state` is fp32 (already correct). Conv1d via concat+mul+sum: produces fp32 if x is fp32. SSM recurrence: fp32. Output projection: fp32 in → fp32 out. Residual add: fp32+fp32 → fp32. **No change needed.**
3. **`mlp_step`**: gate_proj, up_proj, down_proj — fp32 in → fp32 out. Residual add: fp32+fp32 → fp32. **No change needed.**
4. **Final norm + lm_head**: `x` is fp32, weight bf16/bf8 — output fp32. **No change.**
5. **Argmax host**: takes fp32 logits, argmax. **No change.**

### What stays the same

- All ttnn kernels (matmul, rms_norm, silu, etc.) — same calls, just operating on fp32 tensors
- HiFi4 compute kernel config — unchanged, still fp32 DEST
- All weights on device (bf8/bf16) — unchanged
- KV cache layout and bf16 dtype — unchanged
- DeltaNet recurrent state H (already fp32) — unchanged
- The diagnostic harness (`91j`) — re-usable; we'll write a new `91m` that diff-tests against the bf16 baseline
- Prompt, tokenizer, embed lookup, sampling — unchanged

## Implementation files

1. **`experiments/91l_fp32_residual_generate.py`** (new) — copy of `91h`, with:
   - `x` uploaded as fp32 at every entry to forward
   - Typecasts around SDPA (Q to bf16 before, output to fp32 after)
   - Embed lookup uses fp32 row (small change in upload dtype)
   - All else inherited from `91f` (kernels untouched)

2. **`experiments/91m_fp32_diagnostics.py`** (new) — clone of `91j`, but uses `91l`'s forward. Runs the same 5-step instrumented decode and prints the same metrics. We compare the two output files side-by-side.

3. **`experiments/91f_qwen36_27b_full_ondevice.py`** (modify) — add typecast(Q → bf16) inside `gated_attn_step_ondevice` right before SDPA, and typecast(attn → fp32) right after. Verify SDPA accepts bf16 Q — it does (we've used this path all along). The fp32 typecast back recovers our higher-precision residual.

## Validation gates

1. **No crashes** — run prefill + 5 decode steps end-to-end (this is the diagnostic harness)
2. **Cross-step ‖x‖ spread expands** — if fp32 residual restores precision, the lm_head should see varied inputs across decode steps and produce varied outputs; the layer-32-onwards spread should grow from today's 4-12% (where the relative spread *shrinks* as we go deeper) to something less monotonically converging
3. **Top-5 logits become coherent** — at minimum, after a `"The capital of France is"` prompt, top-5 should include at least one of: `'Paris'`, `'a'`, `'the'`, `'France'`, or other common English continuations. Junk subwords like `'jadi'`, `'illac'` dropping out of the top-5 would be a big win.
4. **Top-1 margin grows** — currently 0.062; we expect this to grow to 0.5+ if the logit landscape is correctly resolved
5. **Decode does NOT lock to a fixed point** — different tokens picked across the 5 steps
6. **Performance check** — fp32 residual may cost ~10-20% per token. Acceptable if it fixes correctness. Document for later.

If gates 3-5 pass: **generate 60 tokens** and eyeball coherence. The prompt has a canonical answer (`Paris`), so we have a clear correctness target.

## Risk assessment

- **Risk: SDPA decode does NOT accept bf16 Q in some new code path**. Mitigation: the existing 91f code already uses bf16 Q through SDPA — no change to that call site, so this risk is zero by definition.
- **Risk: ttnn.linear(fp32 act, bf8 weight) is slower than bf16-out variant**. Mitigation: the probe runs in <2s. The matmul cost is dominated by weight read bandwidth, not output write. Expect ~5-10% perf hit on the whole forward, not 50%.
- **Risk: fp32 residual doesn't fix the symptom**. Mitigation: if logits still cluster at junk tokens after fp32 residual, the issue is more than residual stream — possibly bf8 weight precision or KV cache precision. We'd then escalate to bf16 KV cache + B'9.2 = fp32 RMSNorm internals. But based on the PJRT agent's per-op cosines (≥0.9999 everywhere), residual stream is by far the dominant source.

## What we are NOT doing

- Not touching the KV cache bf16 layout (its precision doesn't compound)
- Not touching bf8 weights (would 2× the storage; defer)
- Not rewriting trace capture (defer)
- Not touching the SDPA kernel itself
- Not changing DeltaNet recurrent state H (already fp32)
- Not committing to also fix paged_update_cache (B'6.5, separate)

## Sequence

1. Write `91l_fp32_residual_generate.py` (copy 91h, change embed upload dtype, add SDPA typecasts in 91f)
2. Write `91m_fp32_diagnostics.py` (clone 91j, point at 91l's forward)
3. Sync to qb2, run `91m` (10 min weight load + 30s diagnostic)
4. Compare to `b8_diagnostics.json` from yesterday's run
5. If gates pass → run `91l` with 60 tokens, eyeball coherence
6. If text is coherent → commit B'9 milestone, mark Branch III decode-quality solved
7. If text still degenerates → diagnose what else is contributing (likely bf8 lm_head)
