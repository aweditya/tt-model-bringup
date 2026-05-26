# GDN Custom Kernel Bring-Up - 2026-05-15

Read this after `research/ACTIVE_CONTEXT.md` when resuming GDN work.
Hardware mapping details live in
`research/gdn_kernel_hardware_mapping_2026_05_15.md`.

## Goal

Build an owned Qwen3.6 gated DeltaNet decode custom op, single-device first,
then adapt to the TP mesh only after single-chip correctness and component
performance are understood. The purpose is to fuse the skinny decode recurrence
and remove small TTNN kernels, layout churn, and intermediate traffic. It is not
a claimed full-decode speedup until measured in the server path.

## Non-Negotiables

- Device work only via `ssh qb1` or `ssh qb2`; never run local `ttnn`.
- qb2 is the multi-chip target. qb1 is unavailable until the user says it is
  free.
- Do not run a raw TTNN process on qb2 while the resident TP server owns chips.
- Validate every optimization. No guessed speedup numbers.
- Use sync-bounded timing only.
- Use `.cache/`, `research/`, and `experiments/`; no `/tmp`.
- Preserve compaction state in `research/ACTIVE_CONTEXT.md` plus this file.

## Kernel Contract

Initial local-shard decode contract:

- `state`: `[slots, 128, 128]`, normally 12 slots per chip for TP4.
- `q`: `[slots, 128]`
- `k`: `[slots, 128]`
- `value`: `[slots, 128]`
- `alpha`: `[slots]`
- `beta`: `[slots]`
- outputs:
  - updated `state_next`: `[slots, 128, 128]`
  - recurrence output `out`: `[slots, 128]`

Formula per slot:

```text
H_scaled  = alpha * H
prediction = k @ H_scaled
delta = beta * (value - prediction)
H_next = H_scaled + k_col @ delta
out = q @ H_next
```

Current server state note: `dn["ssm"]` is BF16 in the manual path, while the
ported reference native GDN op expects FP32 state. We need an explicit dtype
decision before production integration. If FP32 state is required, migrate
persistent state allocation and revalidate both quality and performance.

## Current Evidence

Important correction: `experiments/.refs/tt-qwen-36` is not ground truth. It is
a friend's implementation and is known to have GDN recurrence errors. Use it
only for implementation patterns: TTNN op layout, CMake/nanobind wiring,
reader/compute/writer structure, circular-buffer sizing, and dataflow idioms.
Correctness must be defined by our explicit recurrence formula, CPU oracle, and
model-level checks.

- Ported reference ops on qb2:
  - `ttnn.experimental.qwen36_gdn_prepare_decode`
  - `ttnn.experimental.qwen36_gdn_decode`
- Single-device synthetic native-GDN gate passed:
  `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_20260515_0658.json`.
- Real resident mesh tensor native-GDN gate failed:
  `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_fp32cast_20260515_0743.json`.
- Conclusion: the recurrence math is valid on one device, but mesh or real
  tensor layout/order adaptation is wrong. Do not integrate into production
  decode yet.
- Follow-up resident-server controls:
  - replicated synthetic mesh passed:
    `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_mesh_20260515.json`
    with state PCC `0.9999964644317922`, output PCC
    `0.9999903074626416`.
  - synthetic `ShardTensorToMesh(dim=0)` failed:
    `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_mesh_sharded_dim0_20260515.json`
    with state PCC `0.9927821563373678`, output PCC
    `0.9917728381167387`.
  - real tensors still fail in `fp32_cast` mode:
    `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_fp32cast_rerun_20260515.json`
    with state PCC `0.08921031954116858`, output PCC
    `0.21901998019382157`.
  - `current_dtype` real tensors are rejected by validation because state must
    be FP32:
    `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_current_dtype_rerun_20260515.json`.

Updated conclusion: the reference op is useful only as a design reference. It
passes some synthetic cases, but that does not make it correct. The owned
kernel must be implemented and unit-tested independently, starting on one
device and then treating mesh-sharded local shards as a first-class input
contract.

## Reference Dataflow To Reuse

Reference files:

- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode/device/qwen36_gdn_decode_program_factory.cpp`
- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode/device/kernels/dataflow/reader_qwen36_gdn_decode.cpp`
- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode/device/kernels/compute/qwen36_gdn_decode.cpp`
- `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode/device/kernels/dataflow/writer_qwen36_gdn_decode.cpp`

Useful mapping:

- `KEY_TILES = 4`, `VALUE_TILES = 4`.
- One work block is `(slot, value_tile)`.
- Total work per chip is `slots * VALUE_TILES`, so default TP4 is `12 * 4 = 48`
  work blocks.
- Reader stages:
  - 4 state tiles for the selected value tile,
  - 4 q tiles,
  - 4 k tiles,
  - 1 value tile,
  - alpha/beta scalar tiles,
  - optional active/k-row masks.
- Compute stages:
  - optional k mask and q/k L2 normalization,
  - `state_scaled = alpha * state`,
  - `prediction = k @ state_scaled`,
  - `delta = beta * (value - prediction)`,
  - `state_next_tile = state_scaled_tile + transpose(k_tile) @ delta`,
  - `out = q @ state_next`.
- Writer mutates the state tiles in place and writes one output value tile.

L1 residency target:

- Keep the four `state_scaled` tiles resident across prediction and update when
  possible.
- Keep `state_next_internal` resident for the final `q @ state_next` output.
- Use circular-buffer depth 2 for input streams to pipeline reader/compute/writer.
- Do not spill intermediate vectors back to DRAM unless component tests prove the
  L1 budget is too tight.

## Isolated Bring-Up Sequence

1. CPU oracle and deterministic fixture generation.
2. Scalar alpha state decay component: `state_scaled = alpha * state`.
3. Prediction component: `prediction = k @ state_scaled`.
4. Delta component: `delta = beta * (value - prediction)`.
5. Outer update component: `state_next = state_scaled + k_col @ delta`.
6. Output component: `out = q @ state_next`.
7. Full single-chip recurrence kernel using the same work-block mapping.
8. Optional q/k L2 normalization and masks.
9. Mesh wrapper and synthetic-on-MeshDevice control.
10. Resident server real-tensor gate.
11. Full decode trace/benchmark gate.

## Current Owned Source State

First component source tree:

```text
experiments/owned_ops/qwen36_gdn_decay_state/
```

Contents:

- TTNN public wrapper and primitive launcher.
- device operation validation/spec/topology/create-output scaffolding.
- program factory with SPMD block split across cores.
- reader data-movement kernel for state/alpha tile streams.
- compute kernel using scalar broadcast multiply.
- writer data-movement kernel for the scaled state tiles.
- nanobind scaffolding for `ttnn.experimental.qwen36_gdn_decay_state`.
- remote-only device correctness gate:
  `test_qwen36_gdn_decay_state.py`.
- integration notes:
  `INTEGRATION.md`.

Status:

- The source tree was copied into qb2's
  `~/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/`
  via `integrate_into_ttmetal.py`.
- TTNN CMake/nanobind wiring builds successfully in
  `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist`.
- Python import confirms `ttnn.experimental.qwen36_gdn_decay_state` is
  registered after refreshing the source-package `_ttnn.so` and `_ttnncpp.so`.
- Runtime correctness is not validated. The raw one-device correctness gate
  timed out twice on qb2 and required `tt-smi -r 0,1,2,3`.

Current failure mode: device run hangs or stalls before producing the summary
JSON. The harness now emits progress markers around device open, uploads, op
launch, synchronization, readback, and close so the next run can localize the
stage. Until this passes, the op is a compiled but unvalidated component.

## Validation Gates

- CPU component identities must be exact within float32 tolerance.
- TT component kernels must match the CPU oracle fixture before composition.
- Full single-chip kernel must pass state and output PCC/max-diff gates against
  the CPU oracle and a TTNN decomposed oracle.
- Mesh wrapper must first pass synthetic mesh tensors, then real resident server
  tensors.
- Production decode can only use the op behind a guard after real tensor
  equivalence and one-step argmax/generation gates pass.
- Any performance claim must identify whether it is component-only, trace-level,
  or full-decode measured.

## Arithmetic Intensity

`experiments/utils/deltanet_gdn_arithmetic_intensity.py` default estimate:

- `1.381 MFLOP/chip/token`
- `1.588 MB/chip/token` lower-bound useful traffic with FP32 state
- `0.869 FLOP/byte`

This is a low arithmetic-intensity recurrence. The realistic win is reducing
dispatch count, layout conversion, and intermediate memory traffic, not turning
the body into a large dense compute kernel.

## Risks To Keep Visible

- MeshDevice behavior differs from one-device tensors.
- Mesh-sharded dim-0 tensors are already proven to diverge from the manual
  recurrence with the reference native op.
- Real q/k/value slot order may not match reference op expectations.
- BF16 state may not be equivalent to a reference FP32-state kernel.
- Tile logical rows and padded rows can leak into transpose/L2 paths.
- In-place state update must not race output computation.
- Later multi-chip design needs communication/computation overlap, but only
  after single-chip recurrence correctness is locked.
