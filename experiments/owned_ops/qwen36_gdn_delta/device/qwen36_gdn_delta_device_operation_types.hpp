// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnDeltaParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
};

struct Qwen36GdnDeltaInputs {
    const Tensor& value;
    const Tensor& prediction;
    const Tensor& beta;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
