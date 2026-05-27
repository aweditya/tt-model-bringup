// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Owned fused MoE FFN decode op (Pattern A batched, single chip).
//
// Replaces the three back-to-back matmuls in moe_forward_ttnn_pattern_a_batched:
//   gate_up = matmul(h, W1)
//   mid     = silu(gate_up[:HALF]) * gate_up[HALF:]
//   eo      = matmul(mid, W2)
//   routed  = sum_e (routing_weight[e] * eo[e])
//
// All in one kernel launch with intermediates resident in L1.
//
// Inputs (per chip):
//   h               [1, HIDDEN]                bf16, TILE_LAYOUT
//   W1              [E_LOCAL, HIDDEN, 2*MOE_INTER]  bf16, TILE_LAYOUT
//   W2              [E_LOCAL, MOE_INTER, HIDDEN]     bf16, TILE_LAYOUT
//   routing_weight  [E_LOCAL]                  bf16 (compact, padded to TILE)
//
// Output:
//   routed          [1, HIDDEN]                bf16
//
// G0 (current): all kernels are stubs; output is filled with zeros. Verifies
// build, nanobind binding, and program-factory plumbing only. No math yet.
Tensor qwen36_moe_ffn_decode_owned(
    const Tensor& h,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& routing_weight,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::experimental
