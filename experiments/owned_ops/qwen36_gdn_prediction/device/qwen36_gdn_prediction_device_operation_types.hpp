// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnPredictionParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
    uint32_t debug_mode;
};

struct Qwen36GdnPredictionInputs {
    const Tensor& state_scaled;
    const Tensor& k;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
