// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_output_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_output/qwen36_gdn_output.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_output::detail {

void bind_qwen36_gdn_output(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_output", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN component op: output = q @ state_next.

            The output is a tiled [1, slots, 32, value_dim] tensor.  During
            bring-up, q is represented as [1, slots, 32, key_dim] with the
            logical vector repeated across tile rows.
        )doc",
        &ttnn::experimental::qwen36_gdn_output,
        nb::arg("state_next").noconvert(),
        nb::arg("q").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_output::detail
