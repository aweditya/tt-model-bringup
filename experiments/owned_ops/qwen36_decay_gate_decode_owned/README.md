# qwen36_decay_gate_decode_owned — Owned DeltaNet Decay/Gate Decode Op

Custom single-device TT-Metal op that fuses the 10-op
`add → softplus → exp → neg → mul → exp → sigmoid → reshape × 2` chain
currently implemented in `experiments/serve/server_tp.py:681-690`.
Targets the **biggest remaining cluster** in the post-owned-gdn profile:
DeltaNet_decay_gate (#1 at 480 ops / 99.32 ms / 13.52% per
`research/post_owned_gdn_profile_2026_05_18.md`).

Bring-up plan: `research/decay_gate_custom_kernel_plan_2026_05_18.md`.

## What it does

Per-element along NV_PER_CHIP:
```
softplus_a = softplus(a + dt_bias)
g          = -exp(A_log) * softplus_a
decay      = exp(g)
beta       = sigmoid(b)
```

In one kernel launch. Compute fits in a single 32×32 tile (NV_PER_CHIP=12
≤ 32). 10 ttnn ops per layer per token → 1 owned-op call.

## API

```cpp
namespace ttnn::experimental {
std::tuple<Tensor, Tensor> qwen36_decay_gate_decode_owned(
    const Tensor& a,         // [1, NV] logical, [1, 32] padded (bf16 TILE)
    const Tensor& b,         // [1, NV]
    const Tensor& dt_bias,   // [1, NV]
    const Tensor& A_log,     // [1, NV]
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_decay = std::nullopt,
    const std::optional<Tensor>& output_beta = std::nullopt);
}
```

Returns `(decay, beta)` both shaped `[1, NV]` padded `[1, 32]`. The Python
caller can then `ttnn.reshape(decay, [1, NV, 1, 1])` (metadata-only) for
the owned_gdn recurrence input.

`debug_fill=true` makes the compute kernel emit a copy of `a` to `decay`
and `b` to `beta` (no real math) — used for scaffold integration sanity
testing.

## Layout choice

All 4 inputs and both outputs are `[1, NV]` logical / `[1, 32]` padded — 1
tile each. This matches the production tensor shapes after the
slice + reshape preamble; no per-step layout shuffling needed.

## File map

| file | purpose |
|---|---|
| `qwen36_decay_gate_decode_owned.{hpp,cpp}` | public C++ entry, forwards to ttnn::prim |
| `qwen36_decay_gate_decode_owned_nanobind.{hpp,cpp}` | Python binding |
| `device/qwen36_decay_gate_decode_owned_device_operation_types.hpp` | Params + Inputs structs |
| `device/qwen36_decay_gate_decode_owned_device_operation.{hpp,cpp}` | TTNN op: validate + specs + create_outputs |
| `device/qwen36_decay_gate_decode_owned_program_factory.{hpp,cpp}` | CBs + kernel creation + runtime args |
| `device/kernels/compute/qwen36_decay_gate_decode_owned.cpp` | the fused softplus + exp + sigmoid chain |
| `device/kernels/dataflow/reader_qwen36_decay_gate_decode_owned.cpp` | loads 4 input tiles |
| `device/kernels/dataflow/writer_qwen36_decay_gate_decode_owned.cpp` | writes 2 output tiles |
| `sources.cmake` | tt-metal build manifest |
| `integrate_into_ttmetal.py` | installer (patches CMake/nanobind, anchored on GDN/conv1d entries) |
| `test_qwen36_decay_gate_decode_owned.py` | standalone correctness test (BF16 ladder vs numpy oracle) |
| `INTEGRATION.md` | install + build + test recipe |

## Open design questions (resolved at G0)

1. **`softplus_tile` SFPU vs manual `log(exp(x)+1)`**: G0 uses SFPU
   softplus_tile and validates against numpy. If it doesn't match the
   production manual chain at BF16 tolerance, fall back to the manual
   chain inside the kernel.

2. **Output expansion**: kernel emits compact `[1, NV]`. Python caller
   reshapes to `[1, NV, 1, 1]` before passing to owned_gdn — reshape is
   metadata-only, essentially free.

## Status

Scaffold written 2026-05-18 evening; not yet built or tested on qb2. See
`INTEGRATION.md` for the install/build/test recipe.
