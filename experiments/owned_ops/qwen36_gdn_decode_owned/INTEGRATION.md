# qwen36_gdn_decode_owned integration notes

This directory is an owned source drop for the fused single-device GDN decode
bring-up op. It updates state in place and returns `(state, out)`.

## Install

Install into a TT-Metal checkout:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/integrate_into_ttmetal.py \
  --tt-metal ~/tenstorrent/tt-metal \
  --source-dir ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned
```

Build and refresh the source-package extensions:

```bash
cmake --build ~/tenstorrent/tt-metal/build_tracy_gcc12_nodist --target ttnn -j8
cp ~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnn.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so
cp ~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnncpp.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnncpp.so
```

When validating against the rebuilt source tree on qb2, set:

```bash
export PYTHONPATH=~/tenstorrent/tt-metal/ttnn
```

Without this, the local virtualenv wheel can shadow the rebuilt source-package
extension and the experimental symbol will not be visible.

## Validation Gate

Do not run this from the local laptop. Run it on a TT host only after the op is
built and only when the persistent inference server is not holding the chips.

BF16 native ladder:

```bash
for shape in 32x32 32x128 128x32 128x128; do
  key_dim="${shape%x*}"
  value_dim="${shape#*x}"
  PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
    ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
      --device-id 0 \
      --slots 1 \
      --key-dim "$key_dim" \
      --value-dim "$value_dim" \
      --max-abs-diff-threshold 0.001 \
      --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_${shape}_bf16.json
done
```

Validated qb2 results from 2026-05-16:

```text
32x32:   state PCC 0.9999987949387302, max 0.000244140625; out PCC 0.9999946671314629, max 0.000030517578125
32x128:  state PCC 0.9999992937225979, max 0.00048828125;  out PCC 0.9999984056702879, max 0.00006103515625
128x32:  state PCC 0.9999992127972703, max 0.00048828125;  out PCC 0.9999985261461334, max 0.00006103515625
128x128: state PCC 0.999999113477798, max 0.00048828125;  out PCC 0.9999977235415836, max 0.0001220703125
```

Preallocated output validation:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --key-dim 32 \
    --value-dim 32 \
    --preallocate-output-fill -7.0
```

Passing this gate validates the single-device fused recurrence/output kernel
against the BF16-native CPU oracle ladder. It does not validate TP sharding,
real model tensors, or any decode speedup.

Optional pre-transposed K validation:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --slots 1 \
    --key-dim 128 \
    --value-dim 128 \
    --use-pretransposed-k \
    --max-abs-diff-threshold 0.001
```

Validated qb2 result from 2026-05-16:

```text
state PCC 0.999999113477798, max 0.00048828125
out PCC   0.9999977235415836, max 0.0001220703125
```

Optional compact-vector validation:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --slots 12 \
    --key-dim 128 \
    --value-dim 128 \
    --compact-vectors \
    --max-abs-diff-threshold 0.0015
```

Validated qb2 result from 2026-05-17:

```text
state PCC 0.9999989660533721, max 0.0009765625
out-first-row PCC 0.9999967841079703, max 0.0001220703125
```

For resident-server integration, compact q/k/value rows must be explicitly
zero-padded before calling the op. A plain `ttnn.reshape` to logical row 1 is
not a valid proof that padded tile rows are zero for real production tensors.

Optional native-IO validation:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --slots 12 \
    --key-dim 128 \
    --value-dim 128 \
    --native-io \
    --max-abs-diff-threshold 0.0015
```

Validated qb2 result from 2026-05-17:

```text
state PCC 0.9999989660533721, max 0.0009765625
out-first-row PCC 0.9999967841079703, max 0.0001220703125
```

## Microbench

The component-vs-fused microbench also supports trace capture/replay:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/benchmark_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --key-dim 128 \
    --value-dim 128 \
    --warmup 10 \
    --repeats 80 \
    --trace-warmup 10 \
    --trace-repeats 80 \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_microbench_trace_128x128.json
```

Validated qb2 2026-05-16 medians:

```text
sync-only:             0.016889534890651703 ms
component eager:       0.1550390152260661 ms
fused eager:           0.05385500844568014 ms
component execute_trace: 0.026614987291395664 ms
fused execute_trace:     0.021064537577331066 ms
```

Treat these as isolated component timings only.

For fused-kernel bottleneck ablations, run the slots=12 variant:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/owned_ops/qwen36_gdn_decode_owned/benchmark_qwen36_gdn_decode_owned.py \
    --device-id 0 \
    --slots 12 \
    --key-dim 128 \
    --value-dim 128 \
    --warmup 10 \
    --repeats 80 \
    --trace-warmup 10 \
    --trace-repeats 80 \
    --include-ablation-modes \
    --summary-json ~/tt-xla/.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_ablation_trace_slots12_128x128_20260516.json
```

Validated qb2 2026-05-16 trace medians:

```text
component chain:                  0.041463994421064854 ms
full fused op:                    0.025850022211670876 ms
mode1 skeleton/read/write/fill:   0.021854997612535954 ms
mode2 decay/write-state:          0.022180029191076756 ms
mode3 decay+prediction:           0.022369669750332832 ms
mode4 decay+prediction+delta:     0.0228099524974823 ms
mode5 update-without-output:      0.02554489765316248 ms
mode6 output-only:                0.021864427253603935 ms
```

Use this result to focus the next kernel work on the mode5 update path. Do not
convert this component microbench into a full-decode speedup claim.

Follow-up modes 7-9 split the update path further:

```text
mode7 transpose-only-after-delta: 0.02888450399041176 ms
mode8 transpose+outer-no-add:    0.029569491744041443 ms
mode9 update-state-out-only:     0.03016891423612833 ms
mode5 update-without-output:     0.03026460763067007 ms
```

The same-run interpretation is that K transpose is the largest visible
sub-cost. A direct `transpose_k_all` hoist wedged during correctness sync and
was reverted; do not reintroduce it without an isolated CB multi-pack test.

Follow-up optimization notes from 2026-05-16:

```text
pre-transposed-K run:
  fused internal-K execute_trace:       0.030665076337754726 ms
  fused pre-transposed-K execute_trace: 0.03065506462007761 ms

direct-state-output run:
  component chain execute_trace:        0.04136492498219013 ms
  fused execute_trace:                  0.02614909317344427 ms
  fused pre-transposed-K execute_trace: 0.026340014301240444 ms
```

The optional `k_col` path is correctness-safe but not a standalone measured win
because the standalone op still reads row K for prediction and adds K-col reads
for update. It is only useful if an upstream prepare/QKV path can produce or
keep row-K and column-K without extra DRAM traffic. The production fused path
now computes output directly from the state-output CB instead of packing a
duplicate internal `state_next` CB for the output matmul. Absolute trace medians
move with the sync floor, so compare within-run deltas and do not turn these
component timings into full-decode speedup claims.

Resident-server real-tensor integration status, qb2 2026-05-16:

```text
all 48 DeltaNet layers accepted the owned op
46/48 passed the original strict state/output max-diff gate
layer 0 sparse state tail: 8 / 786432 state elements > 0.001, output max 0.0009765625
layer 60 sparse state tail: 1 / 786432 state elements > 0.001, output max 0.0000152587890625
all layers: zero output elements > 0.001
20-token guarded decode: generated IDs matched exactly
```

The guarded production path is not a speed win and must remain disabled:

```text
manual execute_trace median:      82.2640989208594 ms
owned execute_trace median:       83.36730755399913 ms
manual update+execute median:     83.0496409907937 ms
owned update+execute median:      84.1441850643605 ms
manual full decode loop:          82.94731070054695 ms/tok
owned full decode loop:           83.76180131454021 ms/tok
```

Reason: production tensors are compact vectors/scalars, while this
correctness-first op expects repeated-row/repeated-column/full-scalar tiles. The
guarded path adds trace-time repeats to adapt the tensors. Next integration
work should remove that adapter cost with a compact-vector owned op contract or
a prepare/QKV path that creates the needed tiled inputs without extra traffic.

Resident-server compact-vector zero-pad status, qb2 2026-05-17:

```text
layer 0 compact zero-pad:
  output PCC 0.999999195685178, max 0.0009765625, num_gt_0_001 0
  state PCC 0.9999999792358177, max 0.00390625, num_gt_0_001 8 / 786432
20-token guarded decode: generated IDs matched exactly
manual execute_trace median:        82.22798747010529 ms
compact owned execute_trace median: 82.4081456521526 ms
manual full decode loop:            82.88642710540444 ms/tok
compact owned full decode loop:     83.05341660743579 ms/tok
```

This narrows the problem: the kernel is compatible with the rest of decode when
the input contract is satisfied, but the current adapter still costs more than
the fused recurrence saves. Keep the mode disabled until the adapter ops are
removed and same-session full decode is faster.

Resident-server native-IO status, qb2 2026-05-17:

```text
layer 0 native-IO:
  output PCC 0.999999195685178, max 0.0009765625, num_gt_0_001 0
  state PCC 0.9999999792358177, max 0.00390625, num_gt_0_001 8 / 786432
20-token guarded decode: generated IDs matched exactly
manual execute_trace median:          82.2632450144738 ms
native-IO owned execute_trace median: 80.33466408960521 ms
manual full decode loop:              82.90520827285945 ms/tok
native-IO owned full decode loop:     80.99161736899987 ms/tok
```

Native-IO removes q/k/value pad, alpha/beta full-tile repeat, and output slice
from the resident owned path. This is a measured one-prompt/20-token guarded
result, not a broad benchmark. Keep default production decode manual until the
standard multi-prompt/longer decode benchmark confirms it.

Standard benchmark follow-up, qb2 2026-05-17:

```text
probe_deltanet_owned_gdn_benchmark_tp --iters 6 --warmup 2 --max-tokens 64
artifact: .cache/qb2_tp_deltanet/results_owned_gdn_native_io_benchmark_5prompt_64tok_20260517.json
strict token identity: failed, 1 / 5 prompts matched all 64 generated IDs
all one-step argmax checks: matched
manual decode mean:      82.92718172015157 ms/tok
native-IO owned mean:   80.97573915947578 ms/tok
mean measured delta:     1.9514425606757868 ms/tok
mean measured delta pct: 2.3531987041836517 %
```

Do not promote `owned_gdn` as default. The measured timing delta is real for
the benchmark runs, but strict longer-decode identity fails. First divergence
steps across the five default prompts were 41, 43, 10, none through 64, and 14.
The 20-token benchmark still failed on the Python and hybrid-attention prompts.

Diagnostic endpoint `probe_deltanet_owned_gdn_divergence_tp` reads eager top-k
logits to explain mismatches. Artifacts:
`.cache/qb2_tp_deltanet/results_owned_gdn_divergence_python_20260517.json` and
`.cache/qb2_tp_deltanet/results_owned_gdn_divergence_hybrid_20260517.json`.
Python prompt eager decode matched through 16 tokens; the trace-divergence
step had an exact owned bf16 top-2 tie (`2784` and `2007` both `20.375`).
Hybrid prompt eager decode diverged at step 14 with tiny margins: manual chose
`16099` over `8751` by `0.125`, owned chose `8751` over `16099` by `0.25`.
Next correctness work should use a fixed teacher-forced token stream and track
manual-vs-owned recurrence/state/logit error accumulation before optimizing
the native-IO path further.

Teacher-forced diagnostic follow-up:

```text
endpoint: probe_deltanet_owned_gdn_teacher_forced_tp
artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_python_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_hybrid_20260517.json
```

The endpoint builds a manual-greedy teacher stream, then feeds both manual and
owned GDN exactly the same tokens/positions. This isolates numeric drift from
autoregressive branching.

Python prompt (`16` forced tokens) had no argmax difference and no forced-token
rank change. Worst per-step logit max abs was `0.59375`; worst logit PCC was
`0.999207043665397`. Final selected SSM state comparisons were close:
layer 0 PCC `0.9999915620386256`, layer 1 PCC `0.9999133964220422`, layer 2
PCC `0.9998986647201842`.

Hybrid prompt (`20` forced tokens) first differed at step `14`, the same
near-tie region as the eager diagnostic. Manual chose `16099` with logit
`23.5`; owned chose `8751` and gave token `16099` logit `23.25`. Overall
step-14 logits were close (`PCC=0.9993653486862898`, `max_abs=0.4375`).
There was an earlier non-argmax outlier at step `4` (`max_abs=7.14453125`,
`PCC=0.9613907258289726`), so the next diagnostic should report top absolute
logit-diff tokens per step and optionally per-layer SSM snapshots around
steps 3-5 and 13-14.

Conclusion remains: `owned_gdn` is not promotable as default yet. It is close,
but longer greedy decode can flip on bf16-scale near ties.

Performance attribution follow-up:

```text
artifacts:
.cache/qb2_tp_profile/profile_decode_tp_ops_manual_timed_mode2_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_timed_mode2_20260517.json

manual profiled TTNN calls:    4268
owned-GDN profiled TTNN calls: 3788
calls removed:                  480

manual DeltaNet_recurrence:    816 calls, 128.445 ms sync-bounded
owned DeltaNet_recurrence:     336 calls,  51.069 ms sync-bounded
owned custom GDN op itself:     48 calls,   9.540 ms sync-bounded
```

This sync-bounded eager profile is for attribution, not throughput. It explains
why the measured end-to-end owned-GDN win is real but modest: recurrence fusion
does remove a substantial part of the recurrence subgraph, but the full decode
body still has thousands of TTNN calls after fusion. Top remaining owned-profile
op counts include reshape `899`, slice `593`, linear `321`, rms_norm `305`,
add `304`, and mul `288`; large categories remain DeltaNet decay/gate, conv,
qkv repeat, matmuls, RoPE, attention plumbing, collectives, output gate, and
MLP plumbing.

Adapter-elimination follow-up, qb2 2026-05-17:

```text
rank-4 resident SSM artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_manual_rank4ssm_timed_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4ssm_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_benchmark_5prompt_64tok_20260517.json
```

Production DeltaNet SSM now lives in rank-4 custom-op shape and is sharded on
mesh dim `1`. This removes the SSM-to-4D recurrence reshape and the returned
state-to-3D copy-back reshape while preserving safe copy-back semantics.
Measured profile counts: manual `4268 -> 4172`, safe `owned_gdn` `3788 ->
3692`, both `reshape -96`. The short 20-token gate passed, but the standard
5-prompt/64-token owned-GDN gate still failed strict identity (`1/5` streams
matched). Do not promote owned GDN as default from this change.

```text
rank-4 q/k/v owned-mode artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4qkv_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_benchmark_5prompt_64tok_20260517.json
```

For owned recurrence mode, q/k/v are now produced in the native-IO op contract
shape and passed directly to the op. This removes the q4/k4/v4 recurrence
reshapes. Safe `owned_gdn` profile counts moved `3788 -> 3548` versus the
pre-shape-fix owned profile (`reshape 899 -> 659`). The 20-token guarded trace
passed, but the standard 5-prompt/64-token gate still failed (`1/5` streams
matched), so owned GDN remains experimental.

```text
owned_gdn_inplace artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_smoke_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_benchmark_5prompt_64tok_20260517.json
```

`owned_gdn_inplace` removes the redundant SSM copy/commit adapter around the
owned op. The profile changed `3788 -> 3644` profiled TTNN calls: `add -48`,
`reshape -48`, `copy -48`. It is not a default candidate yet. It passed short
exact-token checks but failed the standard 5-prompt/64-token gate, with only
1/5 prompt streams matching all generated IDs.

```text
native-softplus combination artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_softplus_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_softplus_timed_20260517.json
```

`native_softplus + owned_gdn_inplace` reduced profile counts to `3500` and cut
`DeltaNet_decay_gate` `480 -> 336`, but the 20-token generated stream diverged.
Do not promote native softplus in this path without a stricter teacher-forced
numeric equivalence result.

Stepwise isolation, qb2 2026-05-17:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_nativeio_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_compact_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_nativeio_seeded_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_seeded_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_seeded_pretransk_l0_20260517.json
```

The owned op now has a `--stepwise` real-tensor probe and a
`--seed-state manual_once` option for nonzero-state diagnostics. With zero
state, layer 0, production native-IO, the full output matched (`max_abs_diff
0.0009765625`) but returned state had eight BF16-quantum outliers
(`max_abs_diff 0.00390625`). Tiled and compact contracts showed the same
signature, so this is not a native-IO-only adapter issue.

With seeded nonzero state on the trusted tiled diagnostic path,
`state_scaled`, `prediction`, `delta`, and `state_next` all stayed high-PCC but
not strict-equivalent. The state update reached `max_abs_diff 0.015625`
with 20 values above `0.001`; full output reached `max_abs_diff 0.001953125`
with one value above `0.001`. Pretransposed K did not change the signature.

Conclusion: current failures are numerical-equivalence failures in the owned
kernel's rounding/materialization schedule, not remaining q/k/v shape adapter
compatibility. Native-IO debug modes 2-4 are not valid for seeded nonzero-state
substep diagnosis because they use older alpha handling; use tiled seeded mode
for intermediate substeps and native-IO only for production full-op validation.
