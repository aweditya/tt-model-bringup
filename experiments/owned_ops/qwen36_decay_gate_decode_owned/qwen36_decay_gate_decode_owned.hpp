// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Owned DeltaNet decay/gate fused decode op.
//
// Computes per-element along NV_PER_CHIP:
//   softplus_a = softplus(a + dt_bias)
//   g          = -exp(A_log) * softplus_a
//   decay      = exp(g)
//   beta       = sigmoid(b)
//
// All inputs/outputs are bf16 TILE_LAYOUT, logical shape [1, NV] padded
// [1, 32]. Returns (decay, beta) as a tuple.
std::tuple<Tensor, Tensor> qwen36_decay_gate_decode_owned(
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& A_log,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_decay = std::nullopt,
    const std::optional<Tensor>& output_beta = std::nullopt);

}  // namespace ttnn::experimental
