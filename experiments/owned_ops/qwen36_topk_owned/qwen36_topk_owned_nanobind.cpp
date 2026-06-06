// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_topk_owned_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_topk_owned/qwen36_topk_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_topk_owned::detail {

void bind_qwen36_topk_owned(nb::module_& mod) {
    ttnn::bind_function<"qwen36_topk_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Stable-sort topk for Qwen3.6 / Nemotron-3 MoE router.

            Identical contract to ``ttnn.topk`` but always uses the LLK
            STABLE_SORT specialisation (PR #31989). Ties are broken
            deterministically by lowest source index, so per-row argmax
            does not drift between calls or between runs.

            Use this op whenever a downstream consumer cares about the
            exact set of top-k indices (e.g. MoE expert routing across
            many layers / many decode steps). For raw value extraction
            where exact tie-break does not matter, keep using
            ``ttnn.topk`` (slightly faster, unstable).

            Args mirror ``ttnn.topk`` exactly; see that op's docstring
            for shape / dtype contracts.
        )doc",
        &ttnn::qwen36_topk_owned,
        nb::arg("input_tensor").noconvert(),
        nb::arg("k") = 32,
        nb::arg("dim") = -1,
        nb::arg("largest") = true,
        nb::arg("sorted") = true,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("indices_tensor") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_topk_owned::detail
