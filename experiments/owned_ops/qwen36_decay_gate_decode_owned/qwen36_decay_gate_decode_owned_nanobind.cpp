// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_decay_gate_decode_owned_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned/qwen36_decay_gate_decode_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_decay_gate_decode_owned::detail {

void bind_qwen36_decay_gate_decode_owned(nb::module_& mod) {
    ttnn::bind_function<"qwen36_decay_gate_decode_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 DeltaNet decay/gate fused decode op.

            Computes:
              softplus_a = softplus(a + dt_bias)
              g          = -exp(A_log) * softplus_a
              decay      = exp(g)
              beta       = sigmoid(b)
            All inputs/outputs are bf16 TILE_LAYOUT, logical shape [1, NV]
            padded [1, 32]. Returns (decay, beta).

            debug_fill=True emits a copy of a -> decay and b -> beta (no
            real math) for scaffold integration sanity testing.
        )doc",
        &ttnn::experimental::qwen36_decay_gate_decode_owned,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("A_log").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_decay") = nb::none(),
        nb::arg("output_beta") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_decay_gate_decode_owned::detail
