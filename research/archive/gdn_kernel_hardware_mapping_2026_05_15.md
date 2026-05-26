# GDN Kernel Hardware Mapping - 2026-05-15

This is the implementation map for our owned GDN recurrence kernel. It treats
`experiments/.refs/tt-qwen-36` as a source of TT-Metal/TTNN patterns only, not
as correctness ground truth.

## Execution Model

For one TT-Metal program on one Tensix core:

- reader data-movement kernel streams tiles from DRAM/L1 tensor buffers into
  circular buffers;
- compute kernel consumes circular buffers and emits result tiles to circular
  buffers;
- writer data-movement kernel consumes output circular buffers and writes tiles
  back to tensor buffers;
- circular buffers are the synchronization contract:
  `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`.

The host program is SPMD across cores: every participating core runs the same
reader/compute/writer binaries, with per-core runtime args giving a start block
and block count. The MIMD details inside the tile pipeline are handled by the
TT compiler/runtime split across data movement and compute processors.

## Ground Truth

The recurrence per local slot is:

```text
H_scaled  = alpha * H
prediction = k @ H_scaled
delta = beta * (value - prediction)
H_next = H_scaled + k_col @ delta
out = q @ H_next
```

Shapes for the first owned single-device kernel:

- `state`: `[slots, 128, 128]`, FP32 initially.
- `q`: `[slots, 128]`, FP32 initially for bring-up.
- `k`: `[slots, 128]`, FP32 initially for bring-up.
- `value`: `[slots, 128]`, FP32 initially for bring-up.
- `alpha`: `[slots]`, FP32.
- `beta`: `[slots]`, FP32.
- outputs:
  - `state_next`: `[slots, 128, 128]`
  - `out`: `[slots, 128]`

The CPU oracle is `experiments/utils/gdn_kernel_oracle.py`.

## Work Decomposition

Start with one work block per `(slot, value_tile)`.

For Qwen3.6 local TP4 shape:

- `slots = 12`
- `KEY_TILES = 128 / 32 = 4`
- `VALUE_TILES = 128 / 32 = 4`
- total work blocks per chip = `12 * 4 = 48`

Block index mapping:

```text
slot = block / VALUE_TILES
value_tile = block % VALUE_TILES
```

This is conservative and easy to validate because each block independently
updates four `state[:, value_tile]` tiles and emits one output tile. It repeats
q/k reads for each value tile, but it keeps L1 pressure low and exposes enough
parallel blocks for a first implementation.

Later alternative: one block per slot that keeps q/k resident once and processes
all four value tiles. That may save q/k traffic but raises L1 use and reduces
work-block count from 48 to 12; it is not the first bring-up target.

## Reader Contract

For each `(slot, value_tile)` block, reader stages:

- state tiles:
  - tile ids for `(key_tile, value_tile)` for `key_tile in 0..3`
  - four FP32 tiles
- q tiles:
  - four vector tiles for the slot
- k tiles:
  - four vector tiles for the slot
- value tile:
  - one vector tile for the selected value tile
- alpha and beta:
  - one scalar tile each

The reader owns layout-dependent address calculation. Compute should only see
ordered tile streams in CBs. This matters for the future mesh path: local
shard tile ids must be computed from the local tensor shard, not assumed from a
global tensor layout.

## Compute Contract

For each work block:

1. `state_scaled_tiles[key_tile] = alpha * state_tile[key_tile]`
2. `prediction_tile = sum_key_tiles(k_tile @ state_scaled_tile)`
3. `delta_tile = beta * (value_tile - prediction_tile)`
4. For each `key_tile`:
   - transpose or columnize `k_tile`
   - `outer_tile = k_col_tile @ delta_tile`
   - `state_next_tile = state_scaled_tile + outer_tile`
   - emit `state_next_tile` both for writer and for output matmul
5. `out_tile = sum_key_tiles(q_tile @ state_next_tile)`

Important: keep `state_scaled` and `state_next_internal` resident in L1 CBs
across the block. Do not materialize intermediate vectors to DRAM.

## Writer Contract

For each work block:

- write four `state_next` tiles back to the selected slot/value tile column;
- write one `out` tile for the selected slot/value tile.

The first implementation should avoid aliasing surprises: write `state_next` to
a separate output tensor. Only after correctness passes should we add in-place
state update semantics.

## Circular Buffer Plan

Initial FP32-only bring-up uses one page per FP32 tile = 4096 bytes.

| CB | Purpose | Tiles |
|---|---:|---:|
| state_in | 4 key tiles, double-buffered | 8 |
| q | 4 key tiles, double-buffered | 8 |
| k | 4 key tiles, double-buffered | 8 |
| value | selected value tile, double-buffered | 2 |
| alpha | scalar tile, double-buffered | 2 |
| beta | scalar tile, double-buffered | 2 |
| prediction | `k @ state_scaled` | 1-2 |
| delta | `beta * (value - prediction)` | 1-2 |
| k_col | transposed/columnized k tile | 1-2 |
| state_scaled | resident scaled state tiles | 4-8 |
| outer | one outer-product tile | 1-2 |
| state_next_internal | resident next-state tiles for output | 4-8 |
| state_out | state tiles for writer | 4-8 |
| out | output tile for writer | 1-2 |

This should stay well under the 1.5 MB L1/core budget even with conservative
double-buffering. After the FP32 path is correct, q/k/value can move to BF16 if
the model-level gates tolerate it.

## Component Bring-Up Order

Each component gets its own minimal op or guarded mode before composing the full
kernel:

1. `gdn_decay_state`: `state_scaled = alpha * state`.
2. `gdn_prediction`: `prediction = k @ state_scaled`.
3. `gdn_delta`: `delta = beta * (value - prediction)`.
4. `gdn_outer_update`: `state_next = state_scaled + k_col @ delta`.
5. `gdn_output`: `out = q @ state_next`.
6. `gdn_decode`: full block composition.

For each stage, compare device output against the CPU oracle fixture. Use tight
FP32 tolerances for FP32 bring-up, then define BF16 gates separately.

## Single-Device First

Single-device bring-up should use one raw device only when the host server is
not running on that machine. Since qb1 is currently reserved and qb2 runs the
resident server, do not run raw device tests until explicitly coordinated.

The first C++ build can still be developed locally in source form and synced to
qb2's `~/tenstorrent/tt-metal` only when ready to build.

## Mesh Extension

After single-device correctness:

1. Build a mesh wrapper that accepts local shards as the logical contract:
   `[slots_per_chip, 128, 128]` and `[slots_per_chip, 128]`.
2. Validate synthetic mesh with `ShardTensorToMesh(dim=0)` against CPU oracle.
3. Validate real server tensors.
4. Only then consider in-place state update and production decode integration.

For multi-chip performance work, communication overlap belongs outside the
single-chip recurrence body at first. The recurrence itself is local per chip;
collectives occur after row-parallel projections. We should not complicate the
first GDN kernel with CCL until local correctness and timing are stable.
