# qwen36_conv1d_decode_owned — Owned 4-Tap Depthwise Conv1d Decode Op

Custom single-device TT-Metal op that fuses the 4-tap depthwise conv1d +
state-shift currently implemented as 6 separate ttnn ops in
`experiments/serve/server_tp.py:570-580`. Targets DeltaNet_conv (rank 4
in the post-owned_gdn eager profile — 288 calls / 83 ms /
11.30%).

Bring-up plan: `research/conv1d_custom_kernel_plan_2026_05_18.md`.

## What it does

Implements the per-step math:
```
out[d]            = silu( state0[d]*w0[d] + state1[d]*w1[d] + state2[d]*w2[d] + mixed[d]*w3[d] )
state0[d] (next)  = state1[d]    # shift left by one tap
state1[d] (next)  = state2[d]
state2[d] (next)  = mixed[d]
```

In one kernel launch, with the state shift performed by the **writer
kernel** (sidesteps the in-place L1 update problem the prior
`update_cache_for_token_` ring-buffer and `ttnn.copy(src, ttnn.slice(buf,...))`
attempts hit; see `feedback_conv1d_circular_buffer.md`).

## API

```cpp
namespace ttnn::experimental {
Tensor qwen36_conv1d_decode_owned(
    const Tensor& mixed,    // [D, 1] padded [D, 32], bf16 TILE_LAYOUT — current input
    const Tensor& state0,   // [D, 1] padded [D, 32]                 — oldest tap (mutated in place)
    const Tensor& state1,   // [D, 1]                                  (mutated in place)
    const Tensor& state2,   // [D, 1]                                  (mutated in place)
    const Tensor& weight0,  // [D, 1] padded [D, 32]                 — tap 0 weight (state0 * w0)
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,                                          // — tap 3 weight (mixed * w3)
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);
}
```

Returns: `out` of shape `[D, 1]` padded `[D, 32]`. State buffers are
overwritten in place via the writer kernel: `state0 ← state1`,
`state1 ← state2`, `state2 ← mixed`.

`debug_fill=true` makes the compute kernel write a known constant to `out`
instead of the real compute — used for first-build sanity that the
scaffolding compiles, links, dispatches, and the writer's state-shift
works without any real math interference.

## Layout choice

Each of the 8 input tensors is `[D, 1]` padded to `[D, 32]` (i.e. tile-padded
along the column dim, real data in column 0). The reader broadcasts the
column-0 value to all 32 columns via the host tile-padding (zeros
elsewhere), so the compute kernel only ever multiplies the real data
column with the real weights — no in-kernel column slicing.

This requires the production server to **pre-split** `dn['w_conv']`
(currently `[D, 4]`) and `dn['conv_st']` (currently `[D, 3]`) into 4 + 3
separate `[D, 1]` tensors at upload time. That's a one-time
bootstrap cost; no per-step host work changes. Pre-split happens in the
integration commit when G2 (guarded trace) lands.

## File map

| file | purpose |
|---|---|
| `qwen36_conv1d_decode_owned.{hpp,cpp}` | public C++ entry, forwards to ttnn::prim |
| `qwen36_conv1d_decode_owned_nanobind.{hpp,cpp}` | Python binding |
| `device/qwen36_conv1d_decode_owned_device_operation_types.hpp` | Params + Inputs structs |
| `device/qwen36_conv1d_decode_owned_device_operation.{hpp,cpp}` | TTNN op: validate + specs + create_outputs |
| `device/qwen36_conv1d_decode_owned_program_factory.{hpp,cpp}` | CBs + kernel creation + runtime args |
| `device/kernels/compute/qwen36_conv1d_decode_owned.cpp` | the fused 4-tap mul-add-silu |
| `device/kernels/dataflow/reader_qwen36_conv1d_decode_owned.cpp` | loads mixed/state/weight tiles + shift CBs |
| `device/kernels/dataflow/writer_qwen36_conv1d_decode_owned.cpp` | writes out + writes shifted state in place |
| `sources.cmake` | tt-metal build manifest |
| `integrate_into_ttmetal.py` | installer: copies tree + patches CMake/nanobind in tt-metal |
| `test_qwen36_conv1d_decode_owned.py` | standalone correctness test (BF16 ladder against numpy oracle) |
| `INTEGRATION.md` | full install + build + test recipe |

## Reference

The friend repo vendors a structurally identical op at
`experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_causal_conv_decode/`.
That op is the implementation reference for the kernel pattern (explicit
4-tap unroll instead of a `ttnn.sum` reduce; state shift in the writer
kernel via re-reading state CBs into separate shift CBs). Our
implementation is written from scratch using their design as a guide;
the friend repo is known to have model-level errors elsewhere and is
treated as a pattern catalog, not ground truth.

## Status

Scaffold + skeleton kernels written 2026-05-18; not yet built or tested
on qb2. See `INTEGRATION.md` for the install/build/test recipe and
`research/conv1d_custom_kernel_plan_2026_05_18.md` for the staged
validation gates (G0 standalone → G1 real-tensor probe → G2 guarded
trace → G3 cosine_ladder_tp 500 positions → G4 default flip).
