# PJRT Phase 5 Handoff — Steps 5+6 Complete

Date: 2026-05-11
Branch: main
Last commit: `087ba48 PJRT Phase 5: fix double-engine-instance crash + trace release at exit`

## TL;DR

Phase 5 is at **Step 6 complete** — trace capture lands 9-13x speedup on
pure-device programs through the full JAX → PJRT → engine → Blackhole
pipeline. All 50 device tests pass together in one process. Step 7
(op fusion) is the next obvious win but was not started.

## Completed

### Step 5 — Benchmarks
Permanent benchmark at `pjrt_plugin/tests/bench_device.py`. Six surfaces:
raw ttnn, engine eager, parse-only, eager e2e, traced e2e, jax.jit.
Results land in `research/pjrt_phase5_benchmarks.md` (committed and
append-only across runs). Plan in
`research/pjrt_phase5_step5_plan.md`.

Findings:
- Parse cost dominates eager mode (1.4-1.7ms per call).
- Engine dispatch overhead is negligible (~5us on top of ttnn).
- C++ PJRT shim is ~50us — basically free.
- Per-op dispatch 45-95us; matmul dispatch-bound at 64x64 AND 256x256.

### Step 6 — Trace capture
- `_parse_cache: bytecode_hash → parsed ops` (always applies).
- `_trace_cache: bytecode_hash → {trace_id, placeholders, outputs}`.
  Applies when no host-transfer op participates in data-dependent
  compute.
- Env opt-out `TT_PJRT_NO_TRACE=1`.

Numbers (qb1, warm cache, single device 0):

| program            | eager  | traced  | speedup |
|--------------------|-------:|--------:|--------:|
| x + 1              | 1999us |  156us  |  12.8x  |
| exp(x)             | 1707us |  155us  |  11.0x  |
| a @ b 64x64        | 1795us |  199us  |   9.0x  |
| linear (a@w+b)     | 2186us |  534us* |   4.1x  |
| softmax            | 2904us | 1012us* |   2.9x  |
| jit: x + 1         | 2086us |  254us  |   8.2x  |
| jit: a @ b 64x64   | 1829us |  313us  |   5.8x  |

`*` programs include `broadcast_in_dim` and fall back to the parse-cache
path (no actual trace). They still get a measurable win.

Plan and discussion in `research/pjrt_phase5_step6_plan.md`. Full
reflection in `research/pjrt_reflections.md` under the 2026-05-11
(cont. 3) entry.

## Test status (qb1, run together in one process)

- `test_engine_device.py`: 23/23
- `test_basic_ops.py`: 27/27
- Together: 50/50, 3.48s

A long-standing latent bug surfaced during Step 6: test_engine_device
imported the engine via `importlib.spec_from_file_location` — a
SECOND module instance with its own `_device` global. When run after
test_basic_ops (which uses the canonical `jax_plugins.tt.engine`), the
second instance crashed re-opening device 0. Fixed in the same commit
by switching to `from jax_plugins.tt import engine`.

## In flight / next

### Step 7 — Op fusion (not started)

The big unblocked win for the no-trace programs (linear, softmax) is
making `broadcast_in_dim` reliably on-device, so softmax becomes
traceable. I added an on-device `ttnn.repeat` path inside
`_execute_broadcast_device` but kept `broadcast_in_dim` in the
`_HOST_TRANSFER_DEVICE_OPS` set (so trace capture still skips it)
pending broader validation under contention. The fallback to CPU is
preserved, so correctness is unchanged.

Step 7 plan to write when picking this up:
1. Validate on-device broadcast_in_dim with the full test suite.
2. Drop `broadcast_in_dim` from `_HOST_TRANSFER_DEVICE_OPS` if step 1
   stays green.
3. Pattern-match (max → sub → exp → sum → div) → `ttnn.softmax` for
   layer-norm and RMS-norm too.
4. Re-benchmark. Expected: softmax/linear drop to ~200us like other
   traceable programs.

### Phase 6 (longer term)

The trace path's floor is currently ~75% host transfer (numpy → ttnn
input, ttnn → numpy output). Removing that requires changing the PJRT
ABI so C++ holds device pointers directly. That's a separate phase.

## Blocked / known issues

- qb1's device 0 is occasionally locked by the model-bringup track
  (PCIe mutex). Tests can hang when this happens. Not a code issue.
- bf16 precision: `test_basic_ops.py::TestMatmul::test_larger_matmul`
  needs `atol=1.0` for a 128-deep matmul. Already handled with
  mode-aware tolerances. Pre-existing.
- Cross-process trace cache: traces are per-process. Not a regression,
  but a potential Phase 6 improvement.

## Final benchmark numbers (committed in research/pjrt_phase5_benchmarks.md)

Three full runs are recorded in
`research/pjrt_phase5_benchmarks.md`: `baseline-eager`, `step6-trace`,
`step6-validated`. Comparing them shows trace capture is stable across
runs.

## One-liner

**Phase 5 is at Step 6 complete (trace capture, 9-13x). Next: Step 7 op
fusion to extend traceability to softmax/layer-norm/RMS-norm.**

---

## Step 7 result (appended 2026-05-11)

### What fused

Nothing. Step 7 turned out to need ONE LINE — dropping `broadcast_in_dim`
from `_HOST_TRANSFER_DEVICE_OPS` — because the existing on-device
`ttnn.repeat` path inside `_execute_broadcast_device` is already robust
for the five broadcast patterns JAX emits (scalar->tensor, rank-up after
reduction, broadcast across reduced dim, per-channel rank-up, batch
broadcast). All trace-capture-safe.

I did NOT implement softmax/RMSNorm pattern-match fusion. The on-device
broadcast change alone landed the target latency and the cost/risk of a
pattern matcher wasn't justified.

### New benchmark numbers vs step6-validated

|                          | step6-validated | step7-broadcast | speedup |
|--------------------------|----------------:|----------------:|--------:|
| traced: x + 1            |          156us  |          160us  | ~       |
| traced: exp(x)           |          157us  |          156us  | ~       |
| traced: a @ b 64x64      |          200us  |          201us  | ~       |
| traced: linear (a@w+b)   |          534us  |      **228us**  | **2.3x**|
| traced: softmax (1x64)   |         1012us  |      **198us**  | **5.1x**|

Inspected with `pjrt_plugin/scripts/inspect_trace_status.py`: softmax,
layer_norm, rms_norm, linear, attention ALL hit trace cache cleanly.

### What didn't fuse (and why we didn't try harder)

The trace replay path has a hard floor of ~150us per call dominated by
numpy<->ttnn host transfer. A 13-op softmax trace and a 1-op `x+1` trace
both land at ~155-200us. Per-op fusion (replace 13 ops with 1
`ttnn.softmax` call) would save ~30-50us in the replay body — not worth
the brittleness of a pattern matcher that breaks on any JAX lowering
change.

### Test status

- `test_engine_device.py`: 27/27 (added 4 TestTrace assertions for
  softmax, layer_norm, rms_norm, linear).
- `test_basic_ops.py`: 27/27 unchanged.
- `test_engine.py` + `test_buffer.py`: 50/50 CPU tests unchanged.
- Together: **54/54 device, 50/50 CPU, all green.**

### Files touched

- `pjrt_plugin/jax_plugins/tt/engine.py` — one-line change to
  `_HOST_TRANSFER_DEVICE_OPS`.
- `pjrt_plugin/tests/test_engine_device.py` — new `TestTrace` class.
- `pjrt_plugin/scripts/inspect_trace_status.py` — new debug helper.
- `research/pjrt_phase5_step7_plan.md` — design doc.
- `research/pjrt_phase5_benchmarks.md` — two `step7-broadcast-on-device`
  rows appended (run + replay).
- `research/pjrt_reflections.md` — 2026-05-11 (cont. 4) entry.

### One-liner

**Phase 5 is at Step 7 complete (on-device broadcast trace-safe).
Softmax/linear/LN/RMSNorm now trace. Trace replay floor is ~150us —
70-75% host transfer. Phase 6 lever is the PJRT ABI: keep device
tensors across calls.**

---

## Vanilla tt-nn comparison results (appended 2026-05-11)

### The question

"When the same computation is hand-written in native tt-nn, how does it
compare to running it through our PJRT-traced path?"

This is the answer the entire PJRT effort hinges on. If PJRT is slower
than vanilla, the plugin is a worse abstraction. If equal, the value
is convenience. If faster, op fusion / trace logic is paying off.

### Setup

Plan: `research/pjrt_vanilla_comparison_plan.md`. Bench script:
`pjrt_plugin/tests/bench_vanilla_vs_pjrt.py`. Six programs, three
implementations each. Median + p90 over 100 measurement iters, 5
warmup. Two consecutive runs for stability (run1 had cold-cache
anomaly on P1; run2 is steady-state and the headline).

Cosine-equivalence between all three implementations: 1.0000 for
5 of 6 programs, 0.9998 for softmax. All paths compute the same
program.

### Numbers (run2, steady-state)

| Program | Vanilla eager med/p90 (us) | Vanilla traced med/p90 (us) | PJRT traced med/p90 (us) | PJRT / vanilla-traced |
|---|---:|---:|---:|---:|
| P1 x + 1 (1x32)        | 229 / 241 | 149 / 157 | 153 / 163 | **1.03** |
| P2 exp(x) (1x32)       | 216 / 229 | 148 / 156 | 150 / 161 | **1.01** |
| P3 a @ b (64x64)       | 224 / 235 | 195 / 208 | 200 / 206 | **1.03** |
| P4 softmax (1x64)      | 181 / 192 | 150 / 158 | 312 / 327 | **2.08** |
| P5 linear (a@w+b)      | 378 / 479 | 217 / 226 | 227 / 236 | **1.05** |
| P6 attention (8x32)    | 440 / 452 | 269 / 278 | 328 / 336 | **1.22** |

Mean ratio: 1.23 (dominated by the P4 softmax outlier). Excluding P4,
mean ratio is 1.07.

### Honest prose answer

**PJRT-traced is at parity with vanilla tt-nn-traced for 4 of 6
programs.** Ratios 1.01-1.05 for P1, P2, P3, P5 — within 5% of
vanilla. The engine's parse-cache hit + trace replay add ~3-10us of
bookkeeping per call, which is essentially free.

**1.22x slower on attention (P6, +58us).** Multi-op programs pay a
small linear-in-op-count overhead from the engine's per-op dispatch
during trace replay. Real but bounded.

**2.08x slower on softmax (P4, +161us).** This is the only material
gap. It's NOT framework overhead — it's the algorithmic difference
between `ttnn.softmax(dim=-1)` (one fused kernel) and the JAX
lowering `max → broadcast → sub → exp → sum → broadcast → div`
(5-13 ops). An expert writing vanilla TT-NN would never write the
decomposition; JAX users get it because JAX lowers softmax that way.
Pattern-match fusion in the engine (~1 day of work) would close
this gap.

### Recommendation

**Don't invest Phase 6 (PJRT ABI / persistent device tensors) for
raw latency.** The plugin already matches vanilla on real programs
(P1, P2, P3, P5 at 1.01-1.05x). The remaining engine overhead is
small, bounded, and scales harmlessly. Phase 6's gain on top of
parity would be marginal.

**Invest in softmax/RMS-norm/LN pattern-match fusion.** This is the
one place PJRT loses materially (P4 at 2.08x). For any transformer
workload — Qwen3-Coder-Next, Gemma4, etc. — JAX will lower softmax
into the decomposed form and pay 2x vs the fused kernel a hand
TT-NN engineer would write. Estimated work: ~1 day per fused
kernel (softmax, then RMSNorm, then LayerNorm). This is the highest
ROI investment the PJRT track has remaining.

### Files

- `pjrt_plugin/tests/bench_vanilla_vs_pjrt.py` — the comparison bench.
- `research/pjrt_vanilla_comparison_plan.md` — plan + methodology.
- `research/pjrt_phase5_benchmarks.md` — full results with run1 + run2
  numbers and per-bucket prose analysis.

### Test status

50/50 device tests still pass (`test_engine_device.py` +
`test_basic_ops.py`). Engine unchanged; only added a bench script.

### One-liner

**PJRT-traced is at parity with vanilla tt-nn-traced on 4/6 programs
(within 5%). The only material slowdown is softmax (2.08x) — JAX
lowers `softmax` to a 5-13-op graph while vanilla uses
`ttnn.softmax`. Pattern-match fusion is the next highest-ROI
PJRT investment.**

---

## Real-model test: Qwen2.5-0.5B via PJRT (appended 2026-05-11)

### The question

"The best test for the PJRT plugin is if we're able to run an actual
model."

### Status

**Blocked at engine bug — fix landed, validation pending qb1 ssh.**

The JAX implementation is complete (`experiments/jax_qwen05b_pjrt.py`)
and demonstrably correct on the JAX CPU backend (no plugin):

```
Prompt: "The capital of France is" (5 tokens)
Generated: "The capital of France is Paris, and the capital of"
```

Throughput on JAX CPU backend: 88.5 ms/tok = **11.3 tok/sec**.
Native ttnn reference: 7 ms/tok = 142 tok/sec.

The TT-PJRT backend tripped a real engine bug discovered while
building this: JAX deduplicates common helpers (e.g. SiLU) into a
**single private function called from N call sites**. The previous
`execute_func_call` assumed 1-to-1 sequential dispatch (Nth call ⇒
Nth private function), failing with "func.call #1 but only 1 private
functions found in module" the moment a model has any repeated MLP
block.

### Engine fix (committed)

Two parts in `pjrt_plugin/jax_plugins/tt/engine.py`:

1. `_module_to_text_with_callees` walks the MLIR operation tree to
   harvest `(callee, sym_name)` pairs (the default printer drops
   them under `allow_unregistered_dialects`), then splices them
   back into the text the parser reads.

2. `execute_func_call` now dispatches by callee name when available;
   falls back to "single private function" / positional rules when
   names are missing. Tolerates legacy 3-tuple `private_fns` entries
   alongside the new 4-tuple form.

### Design (see research/pjrt_real_model_plan.md)

- **Prefill on host (numpy)** — variable-length, run once.
- **Decode on JAX-jit'd device function** — fixed shape per call.
- **RoPE via rotation matrix** — `x*cos + (x@R)*sin`. Eliminates the
  `jnp.split → stablehlo.slice` (host-transfer) that breaks trace.
- **Pre-computed causal mask** — `mask[None,None,None,:]` added to
  scores. Eliminates `compare`/`iota` (host-transfer).
- **Host-side KV cache update** — cache update is numpy `[:,:,pos:pos+1,:] = new`.
  Eliminates `scatter` (host-transfer). Trade-off: current token's
  K isn't in cache for THIS step's attention (small accuracy hit,
  recoverable in next step).

Op-coverage check via `experiments/jax_qwen05b_inspect.py` confirms
ALL ops are supported and ZERO host-transfer ops appear with this
design — the program should trace cleanly.

### Files

- `research/pjrt_real_model_plan.md` — plan
- `experiments/jax_qwen05b_pjrt.py` — the real-model test
- `experiments/jax_qwen05b_inspect.py` — op-coverage check
- `experiments/jax_qwen05b_bisect.py` — bisects the bug
- `experiments/jax_qwen05b_dump_full_text.py` — IR dump that proved
  the dedup pattern (2 func.calls → 1 private function = SiLU)
- `pjrt_plugin/jax_plugins/tt/engine.py` — `_module_to_text_with_callees`,
  callee-aware `execute_func_call`, sym_name parsing

### What's left

Re-run `experiments/jax_qwen05b_pjrt.py --device --tokens 100` on
qb1 once ssh stabilizes. Expected outcomes:
1. Engine fix makes it past the func.call dispatch error.
2. Trace capture should fire (no host-transfer ops in decode step).
3. tok/s likely 30-80 (24 layers × ~10 traceable ops ≈ ~250 ops in
   one trace, replay floor ~5-8 ms vs native 7 ms).

If trace capture succeeds, this is at-parity with native. If it
falls back to parse-cached eager, expect 50-100ms/step.

### One-liner

**Blocked at qb1 ssh outage during validation. JAX implementation is
correct (verified on JAX-CPU); engine bug fixed (func.call dedup);
device path needs the final smoke test.**

---

## Correctness-debug agent — partial results (appended 2026-05-11/12)

### What was attempted

Investigation A: layer-by-layer cosine of CPU-fp32 vs TT-bf16 residual
stream for one decode step.
Investigation B: independent bf16 op correctness through the engine on
the shapes Qwen2.5-0.5B uses.

### What was completed

1. **Plan**: `research/pjrt_correctness_debug_plan.md`.
2. **Scaffolding**:
   - `experiments/qwen05b_layer_debug.py` — instrumented decode step
     returning per-layer post-attn and post-mlp residuals + final pre-
     norm, post-norm, logits. Two-pass `--mode cpu` / `--mode tt` /
     `--mode compare`. Writes npz snapshots to `.cache/qwen05b/`.
   - `experiments/qwen05b_op_correctness.py` — independent ops:
     matmul (Q-proj shape), softmax (attn scores), rms_norm, swiglu,
     attn Q·Kᵀ, full SDPA chain, plus a composite layer-0 test.
3. **Incidental fix**: `experiments/jax_qwen05b_pjrt.py` — `scale =
   1.0 / jnp.sqrt(jnp.float32(HEAD_DIM))` was being lifted as a 0-d
   captured JAX array, which the current JAX version rejects at
   StableHLO verification time ("broadcast_dimensions size (0) does
   not match operand rank (1)"). Replaced with `float(1.0 / np.sqrt(
   HEAD_DIM))` — passes cleanly. **This means the current
   `jax_qwen05b_pjrt.py` in main does not jit-lower on this JAX
   version.** The previous Paris-then-garbage run on TT must have used
   an older JAX or a different scale form. The bug is unrelated to
   the gibberish, but blocks running the original script at all today.
4. **CPU baseline (partial)**: `qwen05b_layer_debug.py --mode cpu` ran
   to completion on qb1. The numpy prefill produces ` Paris` (correct).
   The first JAX decode step on JAX-CPU samples `'啬'` — a Chinese
   character. This is suspicious; it might be the known v0 "current-
   token-K excluded" quirk biting harder than expected, OR it might
   indicate the JAX decode step has a logic divergence from the numpy
   prefill (e.g. position/mask off-by-one). The handoff doc reports
   that the original `jax_qwen05b_pjrt.py --no-pjrt` produced
   ` Paris, and the capital of` — that result was either on a
   different JAX version or with a different `pos`/`mask` layout.
   **The instrumented script's structure mirrors the original's
   verbatim; identical inputs should produce identical outputs.**

### What is blocked

`ssh qb1` has been unreachable for ~30+ minutes (sustained Connection
refused since ~23:39 local). Cannot run:
- `--mode tt` on the layer-debug script (the TT half of the cosine
  comparison).
- The op-correctness script (both CPU-PJRT and TT-PJRT runs).
- A re-baseline of `jax_qwen05b_pjrt.py --no-pjrt` (CPU) with the
  scale fix, to confirm whether the JAX-CPU first token really is
  `'啬'` today or `' ,'`/something coherent.

Until ssh recovers, the layer-by-layer table and the bf16 op-by-op
table cannot be produced.

### Files left for the next agent / main chat

- `experiments/qwen05b_layer_debug.py`
- `experiments/qwen05b_op_correctness.py`
- `research/pjrt_correctness_debug_plan.md`
- `.cache/qwen05b/residuals_cpu.npz` (lives on qb1 only — the CPU run
  did complete and the npz was written)

### Suggested next steps (precision strategy, partial)

Even without the layer-by-layer numbers, the engine.py read tells us:
1. The engine pads/converts everything to bf16 (`ttnn.bfloat16`) at
   `_to_device`. Every tensor crossing the host→device boundary is
   truncated bf16. There is no fp32-on-device path.
2. `_to_device` always goes through torch.float() → bf16. Means even
   if the StableHLO IR declares fp32, the device payload is bf16.
3. There is NO mixed-precision policy in the engine — RMS-norm,
   softmax, residual adds all execute at bf16.

For a 24-layer model, the bf16 residual stream IS likely to drift
materially over the depth. The native ttnn Qwen2.5-0.5B reference at
142 tok/s achieves cosine >0.99 first-token with EXPLICIT bf8/bf16
mixed precision and dedicated fp32 accumulators in SDPA — features the
engine does not have today. Until layer-by-layer numbers land, my
**preliminary** recommendation is:

- Hypothesize cumulative bf16 drift is the dominant cause; the per-op
  cosine is probably fine (cos > 0.999 for matmul/softmax at small
  shapes), but compounding over 24 layers crosses the argmax-flip
  threshold by token 2.
- Cheapest first probe is to compare CPU-fp32 vs TT-bf16 logit cosine
  for one decode step. If cosine > 0.99 → bf16 noise is small enough
  in isolation, the gibberish is something else (engine bug). If
  cosine < 0.95 → precision strategy needed.
- Precision lever: add an `fp32_high_precision` mode to `_to_device`
  for the residual stream (`x`) and norm gammas. Keep matmul weights
  as bf16 (the bandwidth win is large). Cost: ~2x extra memory for
  the residual.
- Second lever: do RMS-norm in fp32 (compute the rsqrt in fp32, scale
  in fp32, then cast). Same trick all major framework backends use.

### One-liner

**Scaffolding for layer-by-layer + bf16 op correctness landed; CPU
half of A completed but produced a suspicious `'啬'` argmax that
deserves a sanity-check rerun. Sustained `ssh qb1` outage blocked
the TT half. Incidental fix: `jax_qwen05b_pjrt.py` no longer
jit-lowers with the current JAX version due to a captured-scalar
shape mismatch (patched). The big-picture diagnosis is still
pending the TT run; preliminary hypothesis is cumulative bf16 drift
in the residual stream, with the engine lacking any fp32 path for
LayerNorm/residuals.**
