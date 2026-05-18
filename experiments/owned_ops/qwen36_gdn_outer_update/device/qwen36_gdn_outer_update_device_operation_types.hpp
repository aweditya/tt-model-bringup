// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnOuterUpdateParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
};

struct Qwen36GdnOuterUpdateInputs {
    const Tensor& state_scaled;
    const Tensor& k_col;
    const Tensor& delta;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
