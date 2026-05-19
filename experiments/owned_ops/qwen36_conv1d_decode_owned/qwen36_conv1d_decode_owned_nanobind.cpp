// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_conv1d_decode_owned_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/qwen36_conv1d_decode_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_conv1d_decode_owned::detail {

void bind_qwen36_conv1d_decode_owned(nb::module_& mod) {
    ttnn::bind_function<"qwen36_conv1d_decode_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 4-tap depthwise conv1d decode op.

            Computes out[d] = silu(state0[d]*w0[d] + state1[d]*w1[d] +
            state2[d]*w2[d] + mixed[d]*w3[d]) per-element along D in one
            kernel launch, and writes shifted state in place via the writer
            kernel: state0 <- state1, state1 <- state2, state2 <- mixed.

            All input tensors must be bf16 TILE_LAYOUT with logical shape
            [D, 1] padded to [D, 32] (real data in column 0). Returns
            (state0, state1, state2, out) — state tensors are the same
            handles as the inputs, now mutated.

            debug_fill=True replaces the math with a constant write so the
            scaffold can be sanity-tested without the real arithmetic.
        )doc",
        &ttnn::experimental::qwen36_conv1d_decode_owned,
        nb::arg("mixed").noconvert(),
        nb::arg("state0").noconvert(),
        nb::arg("state1").noconvert(),
        nb::arg("state2").noconvert(),
        nb::arg("weight0").noconvert(),
        nb::arg("weight1").noconvert(),
        nb::arg("weight2").noconvert(),
        nb::arg("weight3").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_conv1d_decode_owned::detail
