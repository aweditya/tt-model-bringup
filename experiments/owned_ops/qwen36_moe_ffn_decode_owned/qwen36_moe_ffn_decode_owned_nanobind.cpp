// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_moe_ffn_decode_owned_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_moe_ffn_decode_owned/qwen36_moe_ffn_decode_owned.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::qwen36_moe_ffn_decode_owned::detail {

void bind_qwen36_moe_ffn_decode_owned(nb::module_& mod) {
    ttnn::bind_function<"qwen36_moe_ffn_decode_owned", "ttnn.experimental.">(
        mod,
        R"doc(
            Owned Qwen3.6 MoE Pattern A batched FFN fused decode op.

            Replaces the three-op chain
                gate_up = h @ W1            # batched over experts
                mid     = silu(gate) * up
                eo      = mid @ W2
                routed  = sum_e (rw[e] * eo[e])
            with one kernel launch, keeping all intermediates in L1.

            G0 (current): scaffold only — output is filled with zeros.
            Verifies that the build, the nanobind binding, and the
            program-factory plumbing work end-to-end before any math
            lands. Subsequent G1..G3 stages add real compute incrementally.

            debug_fill=True copies h's first tile into the output (sanity
            check that input access works).
        )doc",
        &ttnn::experimental::qwen36_moe_ffn_decode_owned,
        nb::arg("h").noconvert(),
        nb::arg("W1").noconvert(),
        nb::arg("W2").noconvert(),
        nb::arg("routing_weight").noconvert(),
        nb::kw_only(),
        nb::arg("debug_fill").noconvert() = false,
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::qwen36_moe_ffn_decode_owned::detail
