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

// Owned 4-tap depthwise conv1d decode op. Computes
//   out[d] = silu( state0[d]*w0[d] + state1[d]*w1[d] + state2[d]*w2[d] + mixed[d]*w3[d] )
// per-element along D, and writes shifted state in place:
//   state0 ← state1, state1 ← state2, state2 ← mixed.
// Returns (state0, state1, state2, out) — the three state tensors are the
// same handles as the inputs (now mutated); the user threads them back into
// their persistent state location.
std::tuple<Tensor, Tensor, Tensor, Tensor> qwen36_conv1d_decode_owned(
    const Tensor& mixed,
    const Tensor& state0,
    const Tensor& state1,
    const Tensor& state2,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::experimental
