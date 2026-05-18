// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_delta_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_delta/qwen36_gdn_delta.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_delta::detail {

void bind_qwen36_gdn_delta(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_delta", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN component op: delta = beta * (value - prediction).

            The output is a tiled [1, slots, 32, value_dim] tensor. During
            bring-up, value, prediction, and beta are represented as full tiles
            with logical vectors/scalars repeated across tile rows.
        )doc",
        &ttnn::experimental::qwen36_gdn_delta,
        nb::arg("value").noconvert(),
        nb::arg("prediction").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_delta::detail
