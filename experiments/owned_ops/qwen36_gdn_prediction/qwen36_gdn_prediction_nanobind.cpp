// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_prediction_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_prediction/qwen36_gdn_prediction.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_prediction::detail {

void bind_qwen36_gdn_prediction(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_prediction", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN component op: prediction = k @ state_scaled.

            The output is a tiled [1, slots, 32, value_dim] tensor.  During
            bring-up, k is represented as [1, slots, 32, key_dim] with the
            logical vector repeated across tile rows.
        )doc",
        &ttnn::experimental::qwen36_gdn_prediction,
        nb::arg("state_scaled").noconvert(),
        nb::arg("k").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("debug_mode").noconvert() = 0,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_prediction::detail
