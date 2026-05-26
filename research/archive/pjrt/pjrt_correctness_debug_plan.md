# PJRT Correctness Debug Plan — Qwen2.5-0.5B layer-by-layer

Date: 2026-05-11
Branch: main
Last commit: `9270241 Engine fix: parse JAX packed-hex-bytes constant arrays (bf16 + fp32)`

## Goal

Localize WHERE the TT-PJRT Qwen2.5-0.5B run breaks. JAX CPU run is correct
(emits "Paris" and continues coherently). Device run emits a correct first
token then degrades. We do NOT fix the bug — we just identify it.

## Hypotheses (in priority order)

1. **Cumulative bf16 noise.** Every device tensor lives at bf16; 24 layers
   of accumulation may drift far enough from fp32 ground truth that the
   sampler chooses a different argmax for token 2.
2. **Per-op arithmetic bug.** A specific op (matmul, softmax, rms_norm,
   silu) gives a wrong answer regardless of precision.
3. **Engine plumbing bug.** Per-op cosine looks fine in isolation but
   layered together the shapes/broadcast/padding logic corrupts state.

## Two parallel investigations

### Investigation A — Layer-by-layer cosine for the full model

**Method.** For ONE decode step (the first one after the prompt-prefill
completes), instrument the JAX program so we can capture the residual
stream snapshot at each layer boundary. Then run it twice:

- CPU run (`jax.devices('cpu')[0]`) — fp32 ground truth.
- TT device run (`TT_PJRT_USE_DEVICE=1`) — bf16 on hardware.

Snapshots taken per layer i ∈ [0, 24):
- `x_after_attn_i`: residual after attention (x + o)
- `x_after_mlp_i`: residual after MLP (x + d)
- Plus final pre-norm and pre-logits

Compare CPU vs TT cosine + max-abs-error layer-by-layer. The output
of the script is a table of 24 rows × {after_attn, after_mlp} columns.

**Decision rule.**
- If cosine(layer 0 after_attn) < 0.99 → engine arithmetic bug; layer 0 is suspect.
- If cosine(layer 0 after_attn) > 0.99 but drops monotonically and falls
  below 0.99 at some later layer → cumulative bf16 noise.
- If cosine is fine all the way through but logits argmax disagrees → 
  precision in the final norm or lm_head.

The script that captures snapshots needs to either (a) be a re-instrumented
copy of `jax_qwen05b_pjrt.py` that returns ALL layer intermediates from
the JAX-jit'd function, or (b) run a sequence of 24 progressively-deeper
jit'd programs. (a) is simpler and matches how the engine sees the real
workload — pick (a).

The challenge: the current `decode_step_fn` returns only `(logits, k, v)`.
We'd need a "debug" version that also returns all the residuals. With
24 layers that's 48 extra outputs of shape (1,1,HIDDEN), totalling
~2 MB transfer per call. Acceptable.

Output: `research/pjrt_layer_by_layer.md` with the table.

### Investigation B — Independent bf16 op correctness

**Method.** Pick the four arithmetic ops that dominate the model:
1. Matmul `[1, 896] @ [896, 1024]` (Q proj-sized).
2. Softmax over `[1, 14, 100]` (attention scores).
3. RMS-norm over `[1, 1, 896]` (residual stream).
4. SiLU-gate `silu(a) * b` where a, b are `[1, 4864]` (MLP gate).

For each: generate fp32 numpy inputs, compute fp32 numpy reference,
then run the same input through the engine in device mode. Report
cosine + max-abs-error.

Constraint: ops must run inside a JAX program (otherwise we're not
testing PJRT). Use a tiny `jax.jit`-wrapped function per op.

Output: `experiments/qwen05b_op_correctness.py` + a small table appended
to `research/pjrt_layer_by_layer.md`.

## What I will NOT do

- Fix the bugs I find. Reporting only.
- Change the engine. Reporting only.
- Test on more than ONE decode step. The first divergence is enough.
- Touch any Phase A/B Branch III file.

## Workflow

1. Write `experiments/qwen05b_layer_debug.py` (snapshot per-layer with
   instrumented `decode_step`).
2. Write `experiments/qwen05b_op_correctness.py` (independent ops).
3. Run on qb1: A on CPU, A on device, B on device, B vs numpy fp32.
4. Tabulate results → `research/pjrt_layer_by_layer.md`.
5. Append diagnosis + recommendation to `research/pjrt_handoff_to_main.md`.

## Time budget

3-4h. Hard stop at 4h.

## ssh stability

qb1 sshd is flaky. Use a stability gate: `until ssh qb1 'echo ok' &&
ssh qb1 'echo ok' && ssh qb1 'echo ok'; do sleep 30; done`. Three
consecutive successes before declaring qb1 back.
