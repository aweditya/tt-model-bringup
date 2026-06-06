# qwen36_topk_owned design doc

Owned ttnn op that forks `ttnn::topk` (reduction) and unconditionally enables
the LLK stable-sort flag (PR #31989). Unblocks tracing of the Nemotron-3 Nano
MoE router and the Qwen3.6 MoE router by removing tie-break drift across
23 MoE layers / 40+ decode steps.

Refs
- tt-metal #20625 (SFPSWAP magnitude-then-sign asymmetry, unstable ties)
- tt-metal PR #31989 (LLK stable-sort flag landed)
- tt-metal #33492 (ttnn.sort stable=True still wrong; ttnn.topk has no flag exposed)
- MEMORY: `feedback_ttnn_topk_tie_break_drift.md`
- existing probe: `experiments/cb/isolate/nemotron3_v040hc_topk_tiebreak_probe.py`

## Hypothesis

`ttnn.topk` does NOT thread `stable_sort` through to the LLK call site
(`topk_local_sort`, `topk_merge`, `topk_rebuild` in
`compute_kernel_api.h:513-590`). The default of all three templates is
`stable_sort = false`. Producing an op-binding fork that calls these LLK
intrinsics with the explicit-template `<true>` variant should yield bit-stable
tie-break vs `numpy.argpartition` (with explicit lowest-idx-wins resolution).

This is a plumbing fix only: PR #31989 already shipped the LLK code path;
ttnn.topk just never opted in.

## What gets copied where

Source dirs (this repo): `experiments/owned_ops/qwen36_topk_owned/`
Target dir (tt-metal):    `ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/`

Following the existing pattern in `nemotron3_mamba2_decode_owned/integrate_into_ttmetal.py`
the owned op lives under `experimental/transformer/` regardless of its
original category. Same `experimental_nanobind.cpp` + `experimental/transformer/CMakeLists.txt`
+ root `ttnn/CMakeLists.txt` patch sites already used by the GDN/Mamba2 owned ops.

Files in this op's tree (mirrors topk's reduction layout, renamed):

```
qwen36_topk_owned/
  qwen36_topk_owned.hpp                  # forked from reduction/topk/topk.hpp
  qwen36_topk_owned.cpp                  # forked from reduction/topk/topk.cpp
  qwen36_topk_owned_nanobind.hpp         # new (follows GDN/Mamba2 pattern)
  qwen36_topk_owned_nanobind.cpp         # new (follows GDN/Mamba2 pattern)
  sources.cmake                          # follows nemotron3_mamba2_decode_owned pattern
  integrate_into_ttmetal.py              # forks nemotron3_mamba2/integrate_into_ttmetal.py
  INTEGRATION.md
  README.md
  test_qwen36_topk_owned.py              # smoke unit test (qb1)
  device/
    qwen36_topk_owned_device_operation.hpp / .cpp     # forked from reduction/topk/device/topk_device_operation.{hpp,cpp}
    qwen36_topk_owned_device_operation_types.hpp     # forked from topk_device_operation_types.hpp
    qwen36_topk_owned_single_core_program_factory.hpp / .cpp   # forked from topk_single_core_program_factory.{hpp,cpp}
    qwen36_topk_owned_multi_core_program_factory.hpp / .cpp    # forked from topk_multi_core_program_factory.{hpp,cpp}
    qwen36_topk_owned_constants.hpp                  # forked from topk_constants.hpp
    qwen36_topk_owned_utils.hpp / .cpp                # forked from topk_utils.{hpp,cpp}
    kernels/
      compute/
        qwen36_topk_owned.cpp           # forked from topk.cpp + stable_sort=true on LLK calls
        qwen36_topk_owned_local.cpp     # forked from topk_local.cpp + stable_sort=true
        qwen36_topk_owned_final.cpp     # forked from topk_final.cpp + stable_sort=true
        qwen36_topk_owned_common_funcs.hpp  # forked from topk_common_funcs.hpp + stable_sort=true
      dataflow/
        # forked verbatim from reduction/topk/device/kernels/dataflow/*
```

## Where the `stable=true` flag gets threaded through

LLK API entry points (already template-parameterised, default false):

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_api.h:513-590
template <bool stable_sort = false>
ALWI void topk_local_sort(...);
template <bool idir = false, bool stable_sort = false>
ALWI void topk_merge(uint32_t idst, int m_iter, int k);
template <bool stable_sort = false>
ALWI void topk_rebuild(uint32_t idst, bool idir, int m_iter, int k, int logk, int skip_second);
```

These dispatch through `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_topk.h`
to `tt-llk/{tt_llk_blackhole,tt_llk_wormhole_b0}/common/inc/sfpu/ckernel_sfpu_topk.h`
where `STABLE_SORT` controls the `bitonic_topk_*` rebuild/merge specialisation.

Call sites (in upstream topk that need swapping to `<true>`):

```text
ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp:319
  ckernel::topk_local_sort(0, (int)!largest, end_phase);
  →
  ckernel::topk_local_sort<true>(0, (int)!largest, end_phase);

ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp:41
  ckernel::topk_local_sort(0, (int)ascending, end_phase);
  → ckernel::topk_local_sort<true>(...)

ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp:95
  ckernel::topk_rebuild(0, (uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);
  → ckernel::topk_rebuild<true>(...)

ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp:155
  ckernel::topk_merge<false>(0, m_iter, K);
  → ckernel::topk_merge<false, true>(0, m_iter, K);

ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp:157
  ckernel::topk_merge<true>(0, m_iter, K);
  → ckernel::topk_merge<true, true>(0, m_iter, K);
```

The forked op hardcodes `stable_sort = true` at every LLK call site (no
compile-time toggle, no compute-args flag). This is the simplest and
strictest fix — we always want stable for the router.

If a future caller wants the unstable fast path, they keep using `ttnn.topk`.
`ttnn.experimental.qwen36_topk_owned` is purely "I need byte-stable ties".

## API

Host-side surface mirrors `ttnn.topk` (`ttnn/cpp/ttnn/operations/reduction/topk/topk.hpp`):

```cpp
namespace ttnn::experimental {
std::vector<Tensor> qwen36_topk_owned(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim = -1,
    bool largest = true,
    bool sorted = true,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& indices_tensor = std::nullopt,
    std::optional<std::tuple<Tensor&, Tensor&>> preallocated_output_tensors = std::nullopt);
}
```

There is intentionally NO `stable` argument: this op is always-stable. That
keeps the Python call site one-for-one swappable with `ttnn.topk`.

Python:
```python
values, indices = ttnn.experimental.qwen36_topk_owned(scores, k=6, dim=-1)
```

Input-type contract is inherited from `topk_device_operation.cpp:146-149`:
input must be `BFLOAT16` or `BFLOAT8_B` in `TILE` layout. Indices output is
`UINT16` (or `UINT32` if last-dim > 65535). This is fine — Nemotron-3 MoE
router scores are bf16 and last dim is 128.

## Validation strategy

`experiments/cb/isolate/qwen36_topk_owned_probe.py` — forks
`nemotron3_v040hc_topk_tiebreak_probe.py` and adds a fifth variant `E. owned`:
`ttnn.experimental.qwen36_topk_owned(scores_biased, k=6, dim=-1)`.

Compares against `numpy.argpartition` on the same Nemotron-3 router score
distribution (seed=99, bf16 quantised): the existing tie distribution
documented in `[[feedback-ttnn-topk-tie-break-drift]]` (1-2 tied scores per
row at bf16, ~13% of rows have ties at top-6 boundary).

Pass criteria:
- per-row idx-set match vs numpy: 8/8 rows (vs 0/8 for `baseline` / 7/8 for
  `idx_offset` / 6/8 for `fp32_promote`)
- weights cos >= 0.9999 (relative to numpy ground truth)
- determinism across 10 calls: 10/10 identical bit-for-bit indices

If 8/8 + deterministic: this op is ready for router integration in
Nemotron-3 (`server_nemotron3_ttnn.py` + dev-harness validator).

If less: re-read which LLK call we missed; merge alone might be insufficient
(`bitonic_topk_step_N` is also templated on `STABLE_SORT`; if the kernel
hits that branch we need the `<true>` there too).

## Integration script

`integrate_into_ttmetal.py` forks
`nemotron3_mamba2_decode_owned/integrate_into_ttmetal.py` verbatim with
op-name substitution. The four patch sites are identical:

1. `experimental/transformer/CMakeLists.txt` — kernel glob + api header +
   private sources + unity-build skip
2. `ttnn/CMakeLists.txt` — nanobind source path
3. `experimental/experimental_nanobind.cpp` — include + `bind_*(mod)` call

`--dry-run` prints intended copies and patches without writing.

Build command on qb1:
```bash
ssh qb1 'cmake --build ~/tenstorrent/tt-metal/build_Release --target ttnn -j8'
```

This session: stop after the first cmake-build attempt. If it succeeds, the
plumbing is sound — correctness validation is the next session's work.

## Known unknowns / TODOs

- **Multi-core PF**: the multi-core program-factory uses `topk_local.cpp` +
  `topk_final.cpp` (a two-pass local-then-merge schedule). All three compute
  kernels need the flag flipped; the design above lists all three.
- **bitonic_topk_step_N branch**: in
  `tt-llk/.../ckernel_sfpu_topk.h:531/844` the `bitonic_topk_step_N<STABLE_SORT>`
  branch is called from inside the merge/rebuild specialisations. If our
  outer `<true>` propagates through, this branch is covered automatically.
  Verify by stepping through the templated call once after first build.
- **fp32_dest_acc_en**: `topk_single_core_program_factory.cpp:204` sets
  `.fp32_dest_acc_en = !uint16_output`. The owned op should inherit
  identically. The LLK `STABLE_SORT` branch is independent of
  `is_fp32_dest_acc_en` so this is orthogonal.
- **bfp8 inputs**: PR #31989's stable flag still operates on the SFPSWAP
  schedule; bf16 ties are perfectly recoverable, bfp8 ties may still drift
  because shared exponents quantise the score further. Document at probe
  time if seen.

## Next-session entry points

1. If cmake build PASSED this session: `experiments/cb/isolate/qwen36_topk_owned_probe.py`
   on qb1 to verify 8/8 + determinism against numpy.argpartition.
2. If cmake build FAILED this session: error logged in this doc + next
   session reads the log + iterates on the per-file fork.
3. Router integration is downstream of (1) — `server_nemotron3_ttnn.py`
   moe_router_topk swap, then 7/7 decode chain re-run.
