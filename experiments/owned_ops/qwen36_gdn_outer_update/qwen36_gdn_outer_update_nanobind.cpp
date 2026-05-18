// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_outer_update_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_outer_update/qwen36_gdn_outer_update.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_outer_update::detail {

void bind_qwen36_gdn_outer_update(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_outer_update", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN component op:
                state_next = state_scaled + k_col * delta

            The output is a tiled [1, slots, key_dim, value_dim] tensor. During
            bring-up, k_col repeats each key scalar across a 32-column tile and
            delta repeats the value vector across tile rows.
        )doc",
        &ttnn::experimental::qwen36_gdn_outer_update,
        nb::arg("state_scaled").noconvert(),
        nb::arg("k_col").noconvert(),
        nb::arg("delta").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_outer_update::detail
