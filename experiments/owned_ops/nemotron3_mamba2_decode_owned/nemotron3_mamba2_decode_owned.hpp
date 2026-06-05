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

std::tuple<Tensor, Tensor> qwen36_gdn_decode_owned(
    const Tensor& state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const std::optional<Tensor>& k_col = std::nullopt,
    bool debug_fill = false,
    bool compact_vectors = false,
    bool native_io = false,
    uint32_t debug_mode = 0,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::experimental
