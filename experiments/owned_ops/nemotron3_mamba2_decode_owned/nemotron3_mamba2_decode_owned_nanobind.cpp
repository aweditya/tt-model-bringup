// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decode_owned_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/qwen36_gdn_decode_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_gdn_decode_owned::detail {

void bind_qwen36_gdn_decode_owned(nb::module_& mod) {
    ttnn::bind_function<"qwen36_gdn_decode_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 GDN fused decode op.

            Updates state in place and returns (state, out). During bring-up,
            q/k/value are represented as repeated-row tiled vectors unless
            compact_vectors=True. native_io=True consumes compact row vectors
            and scalar tiles without Python-side pad/repeat adapters.
        )doc",
        &ttnn::experimental::qwen36_gdn_decode_owned,
        nb::arg("state").noconvert(),
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("value").noconvert(),
        nb::arg("alpha").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("k_col") = nb::none(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("compact_vectors").noconvert() = false,
        nb::arg("native_io").noconvert() = false,
        nb::arg("debug_mode").noconvert() = 0,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_gdn_decode_owned::detail
