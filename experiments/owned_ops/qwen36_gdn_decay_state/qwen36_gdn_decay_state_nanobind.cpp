// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decay_state_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_decay_state/qwen36_gdn_decay_state.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_decay_state::detail {

void bind_qwen36_gdn_decay_state(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_decay_state", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN component op: state_scaled = alpha * state.

            This is the first correctness bring-up kernel for the GDN recurrence.
            It is intentionally separate-output first; in-place state update is a
            later optimization after component correctness passes.
        )doc",
        &ttnn::experimental::qwen36_gdn_decay_state,
        nb::arg("state").noconvert(),
        nb::arg("alpha").noconvert(),
        nb::kw_only(),
        nb::arg("debug_copy").noconvert() = false,
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_decay_state::detail
