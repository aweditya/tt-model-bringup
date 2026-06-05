# qwen36_gdn_decode_owned

Owned TTNN source for the correctness-first fused Qwen3.6 GDN decode op:

```text
state_scaled = alpha * state
prediction   = k @ state_scaled
delta        = beta * (value - prediction)
state_next   = state_scaled + k[:, :, None] * delta[:, None, :]
out          = q @ state_next
```

The op updates `state` in place and returns `(state, out)`. This is not copied
as correctness ground truth from `tt-qwen-36`; that tree is only a TTNN/TT-Metal
wiring and dataflow reference.

## Contract

- `state`: tiled rank-4 FP32 or BF16 tensor `[1, slots, key_dim, value_dim]`
- `q`, `k`: tiled tensors `[1, slots, 32, key_dim]`, vector repeated across
  tile rows for bring-up
- optional keyword-only `compact_vectors=True`: `q`, `k`, and `value` may use
  a compact vector contract with only row 0 carrying data and the remaining tile
  rows zero. Logical row dim may be `1` or an explicitly zero-padded `32`.
  `alpha` and `beta` are still full scalar tiles. Do not assume a plain
  production `ttnn.reshape([1, slots, 1, dim])` has zero padded rows; the
  resident integration must prove or create that zero padding.
- optional keyword-only `native_io=True`: `q`, `k`, and `value` use compact
  row tiles with only row 0 semantically live, `alpha` and `beta` are
  `[1, slots, 1, 1]` scalar tiles, and output is flat `[1, slots * value_dim]`.
  The compute kernel performs row broadcast and scalar broadcast in L1.
- optional keyword-only `k_col`: tiled tensor `[1, slots, key_dim, 32]`, vector
  repeated across tile columns. This bypasses the compute-kernel K transpose
  but still leaves row-K reads for prediction in the standalone op.
- `value`: tiled tensor `[1, slots, 32, value_dim]`, vector repeated across
  tile rows for bring-up
- `alpha`, `beta`: tiled tensors `[1, slots, 32, 32]`, slot scalar repeated
  across the tile
- return: `(state, out)`, where `out` is `[1, slots, 32, value_dim]`

SPMD work unit:

```text
block = (slot, value_tile)
slot = block / value_tiles
value_tile = block % value_tiles
```

The reader loads all key tiles for the selected state value tile, repeated-row
Q/K tiles, one repeated-row value tile, and scalar alpha/beta tiles. The compute
kernel keeps the recurrence intermediates in L1 circular buffers, transposes K
inside the compute kernel with `transpose_wh_tile` unless `k_col` is supplied,
writes updated state tiles back to the input state buffer, and writes one
output tile. The production path computes `out = q @ state_next` directly from
the state output CB instead of packing a duplicate internal `state_next` CB.

## Status

- local Python harness syntax check passes
- qb2 TTNN Tracy build compiles and registers
  `ttnn.experimental.qwen36_gdn_decode_owned`
- BF16-native fused correctness ladder passed on qb2 on 2026-05-16 with
  `--max-abs-diff-threshold 0.001`
- 32x32 preallocated-output path also passed

Validated fused ladder:

```text
32x32:   state PCC 0.9999987949387302, max 0.000244140625; out PCC 0.9999946671314629, max 0.000030517578125
32x128:  state PCC 0.9999992937225979, max 0.00048828125;  out PCC 0.9999984056702879, max 0.00006103515625
128x32:  state PCC 0.9999992127972703, max 0.00048828125;  out PCC 0.9999985261461334, max 0.00006103515625
128x128: state PCC 0.999999113477798, max 0.00048828125;  out PCC 0.9999977235415836, max 0.0001220703125
```

Artifacts:

- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_32x32_bf16_native_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_32x128_bf16_native_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x32_bf16_native_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_bf16_native_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_32x32_prealloc_bf16_native_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_microbench_trace_128x128_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_ablation_trace_slots12_128x128_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_ablation_modes789_slots12_128x128_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_kcol_api_internal_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_kcol_api_pretransposed_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_pretransposed_k_slots12_128x128_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_direct_stateout_internal_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_direct_stateout_slots12_128x128_20260516.json`
- `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_direct_stateout_ablation_slots12_128x128_20260516.json`

## Microbench Result

For the 128x128 BF16 fixture on qb2, the dedicated harness reports:

```text
sync-only median:              0.016889534890651703 ms
component chain eager median:  0.1550390152260661 ms
fused op eager median:         0.05385500844568014 ms
component chain trace median:  0.026614987291395664 ms
fused op trace median:         0.021064537577331066 ms
```

This shows that most of the isolated eager gap is dispatch/intermediate
overhead. The traced-device replay gap is smaller: about `0.00555 ms` for this
single recurrence fixture. This is a component microbench, not a full-decode
speedup claim.

## Slots=12 Ablation Result

The benchmark supports internal fused-kernel ablation traces with
`--include-ablation-modes`. The slots=12 fixture better matches the local
Qwen3.6 recurrence slot count and gets above the one-slot trace floor.

Validated qb2 2026-05-16 correctness for the 128x128 BF16 slots=12 fixture:

```text
component out vs oracle: PCC 0.9999967841079703, max 0.0001220703125
fused state vs oracle:   PCC 0.9999989660533721, max 0.0009765625
fused out vs oracle:     PCC 0.9999967841079703, max 0.0001220703125
fused out vs component:  PCC 0.9999999999999998, max 0.0
```

Trace replay medians:

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

Interpretation: the component-chain trace pays about `0.01561 ms` more than
the fused trace on this synthetic recurrence fixture. Inside the fused op,
decay, prediction, delta, and output-only are close to the replay/read-write
floor; the first meaningful jump is mode5. Optimize the update path first:
internal K transpose, outer multiply, state_next/state_out production, and
temporary CB traffic. This remains a component microbench, not a full-decode
speedup claim.

Follow-up modes split the update path further:

```text
mode7 transpose-only-after-delta: 0.02888450399041176 ms
mode8 transpose+outer-no-add:    0.029569491744041443 ms
mode9 update-state-out-only:     0.03016891423612833 ms
mode5 update-without-output:     0.03026460763067007 ms
```

Within that same run, K transpose is the largest visible sub-cost. Outer
multiply and state update/write are smaller, and duplicate internal
`state_next` staging is negligible. A direct `transpose_k_all` hoist attempt
wedged during correctness sync and was reverted; treat pre-transposed K from
the prepare path as the cleaner next hypothesis.

## 2026-05-16 Bottleneck Follow-Up

Optional `k_col` input was added as a keyword-only API. Correctness passed for
both internal-transpose and pre-transposed-K paths on the 128x128 BF16 fixture:

```text
internal transpose: state PCC 0.999999113477798, max 0.00048828125; out PCC 0.9999977235415836, max 0.0001220703125
pre-transposed K:   state PCC 0.999999113477798, max 0.00048828125; out PCC 0.9999977235415836, max 0.0001220703125
```

Standalone pre-transposed K did not improve trace replay, because prediction
still requires row-K reads and the standalone op adds K-col reads:

```text
fused internal-K execute_trace:      0.030665076337754726 ms
fused pre-transposed-K execute_trace: 0.03065506462007761 ms
```

The production path was then changed to compute output directly from the
state-output CB, avoiding duplicate `state_next` packing for the output matmul.
Correctness still passed. A clean timing run showed:

```text
component chain execute_trace: 0.04136492498219013 ms
fused execute_trace:           0.02614909317344427 ms
pre-transposed-K trace:        0.026340014301240444 ms
```

A subsequent ablation run had a higher sync floor and reported fused
`execute_trace` `0.030564493... ms`, with mode9 update-state-out-only
`0.030194991... ms`. Treat the exact absolute medians as run-sensitive; the
same-run conclusion is stable: the fused op removes about 15 us vs the
component-chain trace, and the remaining fused-op time is close to the
skeleton/output/update floor. This remains a component microbench, not a
full-decode speedup claim.

## Real Tensor Integration Probe

The resident qb2 server has a guarded real-tensor endpoint:
`probe_deltanet_owned_gdn_real_tensors_tp`. It runs through the server socket,
constructs real q/k/v/decay/beta from the live DeltaNet path, broadcasts them
to this op's repeated-row/scalar-tile contract, and compares against the current
manual recurrence without mutating the resident SSM state.

Artifacts from 2026-05-16:

- `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_richdiff_20260516.json`
- `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensor_layer_sweep_20260516.json`
- `.cache/qb2_tp_deltanet/results_owned_gdn_trace_probe_20260516.json`

The first-token layer sweep accepted all 48 DeltaNet layers. The original
strict gate passed for 46/48. The two strict failures were sparse state tails:
layer 0 had 8 of 786432 state elements above `0.001`; layer 60 had 1 element
above `0.001`. All 48 layers had output max diff at or below `0.0009765625`,
with zero output elements above `0.001`.

A guarded production trace mode, `deltanet_recurrence_mode="owned_gdn"`, is
implemented but disabled by default. It matched one-step argmax and exactly
matched 20 generated token IDs on `"The capital of France is"`, but it was
slower in same-session timing:

```text
manual execute_trace median:       82.2640989208594 ms
owned-GDN execute_trace median:    83.36730755399913 ms
manual update+execute median:      83.0496409907937 ms
owned-GDN update+execute median:   84.1441850643605 ms
manual full decode loop:           82.94731070054695 ms/tok
owned-GDN full decode loop:        83.76180131454021 ms/tok
```

Do not promote this guarded mode. The current production wrapper has to add
q/k/value/decay/beta tile-broadcast/repeat work to satisfy the correctness-first
contract, which more than erases the fused-recurrence component win. The next
useful kernel target is a compact vector/scalar owned recurrence contract or a
prepare/QKV path that creates row-K, column-K, and scalar/vector tiles without
extra DRAM traffic or trace-time repeats.

## Compact-Vector Follow-Up

Compact q/k/value support was added on 2026-05-17. Synthetic slots=12
correctness passed with `--compact-vectors`:

```text
state PCC 0.9999989660533721, max 0.0009765625
out-first-row PCC 0.9999967841079703, max 0.0001220703125
```

Resident production integration has one important layout rule: direct
`ttnn.reshape` of real production vectors to `[1, slots, 1, dim]` is invalid for
this contract because padded tile rows can contain packed head data. That path
changed the model argmax and is not usable. Explicitly zero-padding q/k/value
rows to 32 fixed the real-tensor check:

```text
layer 0 compact zero-pad:
  output PCC 0.999999195685178, max 0.0009765625, num_gt_0_001 0
  state PCC 0.9999999792358177, max 0.00390625, num_gt_0_001 8 / 786432
```

Guarded 20-token decode also matched generated IDs exactly, but remained slower
than manual in same-session timing:

```text
manual execute_trace median:       82.22798747010529 ms
compact owned execute_trace median:82.4081456521526 ms
manual full decode loop:           82.88642710540444 ms/tok
compact owned full decode loop:    83.05341660743579 ms/tok
```

Conclusion: the compact owned recurrence is semantically compatible when the
adapter explicitly zero-pads rows, but it is still not a production win. The
remaining bottleneck is integration/contract shaping: q/k/value zero-pad,
alpha/beta full-tile repeat, and output slicing. Keep `owned_gdn` disabled by
default until those adapter ops are removed and a guarded full decode is faster.

## Native-IO Follow-Up

Native-IO mode removes the resident adapter ops that made compact zero-pad
slower: no q/k/value pad, no alpha/beta full-tile repeat, and no output slice.
It uses `unary_bcast<ROW>` to expand compact q/k/value row 0 inside L1 and
`mul_tiles_bcast_scalar` for alpha/beta scalar tiles.

Validated qb2 2026-05-17 synthetic slots=12 result:

```text
state PCC 0.9999989660533721, max 0.0009765625
out-first-row PCC 0.9999967841079703, max 0.0001220703125
```

Resident layer-0 real-tensor gate:

```text
output PCC 0.999999195685178, max 0.0009765625, num_gt_0_001 0
state PCC 0.9999999792358177, max 0.00390625, num_gt_0_001 8 / 786432
```

Guarded production trace/full decode on `"The capital of France is"` matched
one-step argmax and exactly matched 20 generated token IDs:

```text
manual execute_trace median:         82.2632450144738 ms
native-IO owned execute_trace median:80.33466408960521 ms
manual full decode loop:             82.90520827285945 ms/tok
native-IO owned full decode loop:    80.99161736899987 ms/tok
```

This is a measured one-prompt/20-token guarded result, not a broad benchmark.
Default production decode remains manual until this is run through the standard
multi-prompt/longer decode benchmark.

Broader resident-server benchmark, qb2 2026-05-17:

```text
command: probe_deltanet_owned_gdn_benchmark_tp --iters 6 --warmup 2 --max-tokens 64
artifact: .cache/qb2_tp_deltanet/results_owned_gdn_native_io_benchmark_5prompt_64tok_20260517.json
strict token identity: failed, 1 / 5 prompts matched all 64 generated IDs
all one-step argmax checks: matched
manual decode mean:      82.92718172015157 ms/tok
native-IO owned mean:   80.97573915947578 ms/tok
mean measured delta:     1.9514425606757868 ms/tok
mean measured delta pct: 2.3531987041836517 %
```

Do not promote `owned_gdn` from this result. The timing delta is measured, but
strict generated-token identity fails on the standard longer prompt set.
First divergence positions in the 64-token run were steps 41, 43, 10, none
through 64, and 14 across the five default prompts. A 20-token run still failed
on the Python and hybrid-attention prompts.

Top-k divergence diagnostics:

```text
artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_divergence_python_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_divergence_hybrid_20260517.json

Python prompt:
  eager manual and eager owned matched through 16 tokens; at the trace
  divergence step, owned eager had an exact bf16 top-2 tie:
  token 2784 = 20.375, token 2007 = 20.375.

Hybrid prompt:
  eager manual and eager owned diverged at step 14 with tiny margins:
  manual picked 16099 over 8751 by 0.125; owned picked 8751 over 16099 by 0.25.
```

Current interpretation: native-IO owned GDN is close enough to expose argmax
tie/near-tie sensitivity, but it is not yet strict-token-equivalent over longer
greedy decode. The next correctness step is teacher-forced comparison of
manual-vs-owned recurrence/state/logit error accumulation under the same token
stream.

Teacher-forced diagnostics, qb2 2026-05-17:

```text
artifacts:
.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_python_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_hybrid_20260517.json
```

Python prompt, 16 forced tokens:

```text
manual-vs-owned argmax diff: none
forced-token rank change:    none
worst logit max abs:         0.59375
worst logit PCC:             0.999207043665397
final selected state:
  layer 0 PCC 0.9999915620386256, max 0.25
  layer 1 PCC 0.9999133964220422, max 0.015625
  layer 2 PCC 0.9998986647201842, max 0.005126953125
```

Hybrid prompt, 20 forced tokens:

```text
first argmax diff:           step 14
forced-token rank change:    step 14
step 14 manual argmax:       16099, logit 23.5
step 14 owned argmax:        8751
step 14 owned logit[16099]:  23.25
step 14 logit PCC:           0.9993653486862898
step 14 logit max abs:       0.4375
final selected state:
  layer 0 PCC 0.9999943059140779, max 0.125
  layer 1 PCC 0.9999174040075741, max 0.015625
  layer 2 PCC 0.9998951026135912, max 0.00830078125
```

There is one non-argmax outlier in the hybrid forced run at step 4
(`max_abs=7.14453125`, `PCC=0.9613907258289726`). The current read is still
that owned native-IO has small bf16-level drift rather than gross recurrent
state corruption, but the next diagnostic should capture top absolute logit
diff tokens and per-layer state snapshots around hybrid steps 3-5 and 13-14.

Post-fusion profile attribution, qb2 2026-05-17:

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

This is an eager, per-op synchronized profile of the trace body, not a direct
trace replay timing. It is useful for attribution only. It shows that recurrence
fusion worked: the manual recurrence subgraph lost 480 TTNN calls and became
one owned op per DeltaNet layer plus adapter/state-shaping work. The reason the
end-to-end gain is only about 2 ms/token is that the owned path still has 3,788
profiled TTNN calls across the rest of decode: decay/gate, conv, qkv repeat,
matmuls, RoPE, attention plumbing, RMSNorm, collectives, output gate, MLP
plumbing, cache update, SDPA, and LM-head/io.

## Optimization Debt

- Q/K are reread for every value tile.
- In native-IO mode, q/k/value row broadcast and alpha/beta scalar broadcast
  happen inside the compute kernel. The older non-native paths still require
  repeated-row/full-scalar input tiles.
- State still writes to DRAM. Native-IO output is flat, but the older paths
  still write repeated-row output.
- `transpose_wh_init_short` is called per K tile in the correctness-first
  kernel unless `k_col` is supplied.
- Subtract, beta multiply, K transpose, outer multiply, and state add still use
  separate L1 CB stages.
- Standalone `k_col` is not a measured win unless upstream prepare/QKV can
  produce or keep both row-K and column-K without an extra DRAM read.
- The guarded native-IO path still has reshape/view adapter debt for q/k/v/beta
  and copies H before mutation. True rank-2 `[slots, dim]` q/k/v support needs
  row extraction from tiled head rows or an upstream prepare path that emits one
  compact tile per slot.

Single-device correctness numbers are not decode speedup claims. The native-IO
decode timing above is a guarded one-prompt/20-token measurement only.

## Adapter-Elimination Follow-Up

Rank-4 resident SSM shape fix, qb2 2026-05-17:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_manual_rank4ssm_timed_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4ssm_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_benchmark_5prompt_64tok_20260517.json
```

The resident production SSM is now allocated/reset as
`[1, N_V_HEADS, K_DIM, V_DIM]` sharded on mesh dim `1`, so local SSM already
matches the custom-op state contract `[1, NV_PER_CHIP, K_DIM, V_DIM]`. This
removes the production `H_4d` reshape and the safe copy-back reshape without
using in-place aliasing. Profile counts moved as expected: manual `4268 ->
4172` profiled calls and safe `owned_gdn` `3788 -> 3692`, both from
`reshape -96`. The 20-token guarded trace matched exactly, but the standard
5-prompt/64-token owned-GDN benchmark still failed strict token identity
(`1/5` prompts matched). This is a safe adapter cleanup, not a default-promotion
result.

Rank-4 q/k/v owned-mode shape fix, qb2 2026-05-17:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4qkv_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_benchmark_5prompt_64tok_20260517.json
```

When the recurrence mode is owned GDN, the server now emits q/k/v directly as
`[1, NV_PER_CHIP, 1, K_OR_V_DIM]` and passes them to the native-IO op without
the q4/k4/v4 call-site reshapes. Manual recurrence mode still uses the old
rank-2 q/k/v path. The 20-token guarded trace matched exactly. The owned
profile moved `3788 -> 3548` calls versus the pre-shape-fix owned path,
entirely from `reshape 899 -> 659` (`-240`). The longer 5-prompt/64-token
identity gate still failed (`1/5` streams matched), so this is an adapter
cleanup only.

qb2 artifacts, 2026-05-17:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_smoke_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_timed_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_benchmark_5prompt_64tok_20260517.json
```

`owned_gdn_inplace` passes the resident SSM view directly into the owned op and
skips the returned-state reshape/copy-back. It removed the intended adapter ops:
profiled TTNN calls dropped `3788 -> 3644`, with `add -48`, `reshape -48`, and
`copy -48`. It passed a 4-token smoke and a one-prompt/20-token exact-token
check, but failed the standard 5-prompt/64-token gate: only 1/5 prompt streams
matched all generated IDs. Keep it experimental; do not promote as default.

Native-softplus combination artifacts:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_inplace_softplus_trace_20tok_20260517.json
.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_softplus_timed_20260517.json
```

`native_softplus + owned_gdn_inplace` reduced profiled calls further to `3500`
and cut `DeltaNet_decay_gate` from `480` to `336` calls (`add -48`, `exp -48`,
`log -48`). It is correctness-rejected: the one-prompt/20-token generated IDs
diverged even though first-token argmax matched. Treat those counts as
attribution only.

## Rejected Strict-Reduce Attempt

qb2 2026-05-18: a production-only strict contraction variant tried to replace
`matmul_reduce` with tilewise column-vector multiply, `REDUCE_COL`, and explicit
top-row accumulation across key tiles. The motivation was to mimic the manual
TTNN recurrence's `mul + sum(dim=-2)` schedule more closely.

It is rejected:

```text
32x32 native-IO synthetic:
  state PCC 0.9999794459006638, max 0.0008544921875
  out PCC   0.9999908059586862, max 0.000030517578125

slots=12 128x128 native-IO synthetic:
  state PCC 0.9999768405609735, max 0.002227783203125
  out PCC   0.9999840993853149, max 0.00030517578125

resident layer-0 seeded native-IO:
  state PCC 0.9868074718878412, max 9.5
  out PCC   0.9894557125780156, max 0.392578125
```

Artifacts:

```text
.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_strict_32x32_native_20260517.json
.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_strict_slots12_128x128_native_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_strictreduce_fixed_stepwise_nativeio_seeded_l0_20260517.json
```

The active qb2 build was restored to the matmul-reduce production path after
this test. Do not report the strict-reduce variant as an improvement.

Restored-build sanity check:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_restored_matmul_nativeio_seeded_l0_20260517.json
state PCC 0.9999997911075303, max 0.015625
out PCC   0.9999971009389962, max 0.001953125
```

This matches the prior matmul-reduce signature and remains below the strict
promotion bar.

## Standalone Prediction Equivalence Probe (not ready)

qb2 2026-05-18: added a resident-server `component_prediction` comparison that
calls `ttnn.experimental.qwen36_gdn_prediction` on the same real layer-0
`state_scaled` and K tensors used by the full GDN stepwise probe.

Useful positive findings:

- The failed debug10 strict-reduce path was removed from the full owned GDN
  kernel and server probe. The full op launches again.
- `owned_copy_vs_input` is exact.
- `isolated_prediction_state_scaled` is exact, proving the isolated contraction
  probe receives the intended state tensor.
- The existing full owned GDN path remains at the known matmul-reduce signature:
  state PCC `0.9999997911075303`, state max diff `0.015625`; output PCC
  `0.9999971009389962`, output max diff `0.001953125`.

Rejected component attempts:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_nativek_nativeio_seeded_l0_20260518.json
component_prediction PCC -0.28762728422335415
component_prediction max_abs_diff 11.5625
```

The later narrowed debug harness fixed the gross orientation problem but still
does not meet the default-safe gate:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_component_mode10_kcol_nativeio_seeded_l0_20260518.json
component_kcol0: exact

.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_ttnn_expected_nativeio_seeded_l0_20260518.json
component_product0_vs_ttnn: PCC 0.9999995951668267, max_abs_diff 0.00390625

.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_ttnn_expected_nativeio_seeded_l0_20260518.json
component_reduce0_vs_ttnn: PCC 0.9999996063019564, max_abs_diff 0.0078125
```

Changing the TTNN expected path to match manual recurrence order (full
`state_scaled * k_col`, then first-tile slice/reduce) did not change the
result:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json
```

Mode 2 full strict prediction is wedge-prone/too slow in the resident server
path and required killing/restarting the server. The standalone strict-reduce
prediction component is therefore still not an equivalence fix. Do not
integrate it into the full GDN op until mode 11/12 are exact or the remaining
BF16 differences are theoretically justified and generated-token validation
passes.

Direct matmul contract check:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_matmul_contract_nativeio_seeded_l0_20260518.json
ttnn.matmul(k4, state_scaled) shape: [1,12,1,128] @ [1,12,128,128] -> [1,12,1,128]
prediction_matmul_vs_broadcast: PCC 0.999993423459592, max_abs_diff 0.0625
component_prediction_vs_matmul: PCC 0.9999933819833065, max_abs_diff 0.0625
full_state_next_vs_matmul: PCC 0.999995486148546, max_abs_diff 0.03125
full_out_vs_matmul: PCC 0.9999750303376052, max_abs_diff 0.005859375
```

The clean matmul formulation is shape-valid, but it is a different numerical
contract from the current broadcast-reduce recurrence. The owned kernel is not
made correct by changing the reference to matmul.
