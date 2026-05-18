# DeltaNet / GDN Custom Op Plan - 2026-05-15

Target: Qwen3.6-27B TP decode on qb2, `(1,4)` mesh, P25 path.

## Decision

Writing a Gated DeltaNet custom op is the right direction, but the first
implementation should not be greenfield. The reference tree already contains
the exact Qwen36 GDN-native split we want to evaluate:

- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prepare_decode/`
- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode/`

The fastest credible path is to port/expose these ops in the qb2 TTNN build,
then adapt our current `server_tp.py` DeltaNet path to feed their expected
rank-4 tiled layouts behind a guarded mode. Only after that passes tensor and
full-decode gates should we consider a new generalized op.

## Why This Is Worth Doing

The current DeltaNet decode path is the largest small-op envelope in the eager
profile proxy:

- DeltaNet recurrence: `816` TTNN calls/token.
- DeltaNet decay/gate: `480`.
- DeltaNet QKV repeat: `336`.
- DeltaNet conv: `288`.
- DeltaNet output gate: `240`.
- DeltaNet state update: `144`.

The true Tracy run shows all four chips active for the full `~82.3 ms` trace
replay window, so the remaining problem is not obvious host serialization. The
working model is an unfused batch-1 decode graph with many small kernels,
layout operations, and collectives.

## Arithmetic Intensity

Tool:

```bash
python3 experiments/utils/deltanet_gdn_arithmetic_intensity.py
```

Default estimate for the per-chip recurrence body with `12` value heads,
`K=128`, `V=128`, FP32 recurrent state, BF16 Q/K/V:

- FLOPs per slot: `115072`.
- FLOPs per chip/token: `1380864`.
- Minimum traffic per chip/token: `1588320` bytes.
- Arithmetic intensity: `0.869 FLOP/byte`.
- Lower bound at `512 GB/s`: `3.10 us`.
- Lower bound at `1 TFLOP/s`: `1.38 us`.

This is not a runtime prediction. It is a shape-level lower bound. The key
conclusion is that the recurrent body is low arithmetic intensity and small in
absolute math. A native op's value is reducing dispatch count, layout churn,
and intermediate tensor traffic. It should not be sold as making dense compute
faster.

## Reference Op Semantics

`qwen36_gdn_decode` computes the recurrent update:

```text
state_scaled = alpha * state
prediction   = k @ state_scaled
delta        = beta * (value - prediction)
state_next   = active * (state_scaled + k_col @ delta)
output       = q @ state_next
```

Important constraints from the reference implementation:

- `state`: device tensor, `TILE`, `FP32`, rank 4, logical `[1, slots, 128, 128]`.
- `q`, `k`, `value`: device tensors, `TILE`, rank 4, logical `[1, slots, 1 or 32, 128]`, padded `[1, slots, 32, 128]`.
- `alpha`, `beta`: `FP32`, rank 4, logical `[1, slots, 1 or 32, 1 or 32]`, padded `[1, slots, 32, 32]`.
- Q/K/V must be all interleaved or all L1 height-sharded consistently.
- Output sharding is not supported in the current reference op.
- The op updates/returns state and also returns output.
- `normalize_qk_l2` exists, but the reference model normalizes Q/K externally and passes `False`.

`qwen36_gdn_prepare_decode` prepares Q/K/V from the conv output:

- Input `conv_out` rank 4, logical `[1, 1, batch_rows, local_conv_dim]`.
- `head_dim=128`.
- `value_heads_per_device` divisible by `key_heads_per_device`.
- Interleaved conv output.
- Output Q/K/V are rank-4 tensors compatible with `qwen36_gdn_decode`.

## Current Server Mapping

Current `experiments/serve/server_tp.py::deltanet_step_tp` does:

1. input RMSNorm.
2. `w_in` linear.
3. manual slice into QKV/Z/A/B.
4. manual conv1d.
5. manual Q/K/V slice, repeat, reshape.
6. manual Q/K L2 RMSNorm.
7. manual softplus/decay/beta.
8. manual recurrence.
9. output RMSNorm and `silu(z)` gate.
10. output projection and all-reduce.
11. in-place recurrent/conv state update.

The first guarded native-op experiment should replace only steps 5-8. Keep the
existing input norm, projection, conv, output gate, output projection, all
reduce, and state update plumbing unchanged until equivalence is proven.

## Staged Implementation

### Stage 0 - Import/Build Check

On qb2, confirm whether the active TTNN build already exposes the ops:

```python
import ttnn
print(hasattr(ttnn.experimental, "qwen36_gdn_prepare_decode"))
print(hasattr(ttnn.experimental, "qwen36_gdn_decode"))
```

If missing, port the reference C++ op directories into the qb2 TTNN source tree
and wire the registrations/CMake for the Tracy-enabled build.

Status: done on qb2. The active Tracy TTNN build now exposes both symbols.

### Stage 1 - Synthetic Tensor Harness

Create a qb2-only harness that:

- Builds synthetic `state`, `q`, `k`, `value`, `alpha`, `beta` tensors with the
  exact rank/layout/dtype expected by the native op.
- Runs the Python TTNN recurrence equivalent.
- Runs `ttnn.experimental.qwen36_gdn_decode`.
- Compares `state_next` and `output`.

Gate:

- `PCC >= 0.9999` for `state_next`.
- `PCC >= 0.9999` for `output`.
- Log max absolute error, shape, dtype, layout, memory config, and seed.

Status: passed.

Artifact:

- `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_20260515_0658.json`

Measured gate:

- state PCC `0.9999923945472474`, max diff `0.0018737204372882843`.
- output PCC `0.9999854277820642`, max diff `0.0003914386034011841`.

Synthetic component timing:

- manual median `2.2546573309227824 ms`.
- native median `0.06879458669573069 ms`.

This timing is only the isolated recurrence body on one device. It is not a
full-decode or trace speedup claim.

### Stage 2 - Server Tensor Compatibility Harness

Add an endpoint or standalone server-side probe that captures one real
DeltaNet layer's tensors after conv/decay generation:

- Current `conv_out`.
- Current repeated/normalized `q`, `k`, `v`.
- Current `decay` as native-op `alpha`.
- Current `beta`.
- Current `dn["ssm"]` state.

Feed these into the native op after only the minimum reshapes/layout changes.
Compare native `state_next`/`output` against the existing recurrence branch.

Gate:

- `H_new` PCC `>= 0.9999`.
- recurrent output PCC `>= 0.9999`.
- No host readback inside the decode loop for the production mode.

Status: failing as of first real-tensor gate.

Artifact:

- `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_fp32cast_20260515_0743.json`

Measured result:

- native op symbols visible in resident server.
- native op accepts the mesh tensors.
- state PCC `0.08921031954116858`, max diff `237.2486114501953`.
- output PCC `0.21901998019382157`, max diff `2.4686734676361084`.
- gate failed.

Interpretation: do not integrate native GDN into decode yet. The single-device
synthetic gate proves the op math can match the manual recurrence, but the
resident mesh path does not. The next isolation step is a synthetic-on-mesh
server control. If that fails, the reference op is not MeshDevice-safe as
currently wired. If it passes, the bug is in the real tensor layout/order/dtype
adaptation from our server path.

### Stage 3 - Guarded Server Integration

Add a mode such as `deltanet_recurrence_mode=native_gdn` that swaps only the
recurrence body. Keep the old recurrence as default and as oracle.

Full-decode gate:

- Same prompt, same sampling settings, deterministic decode.
- Token IDs match for the standard probe.
- Full decode `ms/token` measured with sync-bounded server timing.
- Record same-session baseline and native-GDN timing.

No speedup claim unless this full decode timing moves.

### Stage 4 - Trace Replay Timing

Capture a temporary trace for the guarded native-GDN path and compare:

- trace capture success/failure;
- `update_input_buffers + execute_trace`;
- `execute_trace`;
- output readback;
- TTNN op count proxy before/after.

Gate:

- no fabric wedge;
- no trace replay regression;
- timing measured in same session against baseline.

### Stage 5 - Follow-On Fusion

Only after recurrence-native mode passes:

- try `qwen36_gdn_prepare_decode` to remove Q/K/V split/repeat plumbing;
- evaluate fusing Q/K L2 normalization if the native option matches exactly;
- consider output RMSNorm + `silu(z)` gate fusion;
- revisit conv fusion only if profile evidence still points there.

## Risks

- The reference op's output sharding is not supported. We may pay a layout cost
  before `w_out`.
- The reference model recomputes `recurrent_output = q @ recurrent_state` after
  native decode and discards the native output. Treat this as a warning: state
  update fusion may be easier to validate than output fusion.
- Our current `conv_out` is flatter than the prepare op's rank-4 expected
  shape. The first integration may need manual reshape/layout adaptation.
- Current `dn["ssm"]` is uploaded as `[N_V_HEADS, K_DIM, V_DIM]` sharded on
  dim 0 and reshaped to `[1, NV_PER_CHIP, K_DIM, V_DIM]`; native validation
  requires rank-4 `TILE` FP32 state with the exact logical/padded shape.
- Native alpha is `decay = exp(g)`, not the pre-exp `g`.
- Q/K/V rank, memory config, and sharding consistency are likely the main
  integration hazards.

## Contribution Angle

This is a useful contribution if we can make it reproducible:

- port/expose the Qwen36 GDN ops cleanly;
- add synthetic op tests;
- add a model-level decode equivalence probe;
- document supported layouts and the exact Qwen36 mapping;
- publish measured before/after op counts and same-session decode timing.

The contribution should be framed as reducing decode graph fragmentation for
Qwen36 Gated DeltaNet, with arithmetic-intensity evidence explaining why the
benefit comes from fewer kernels/intermediates rather than raw recurrence math.
