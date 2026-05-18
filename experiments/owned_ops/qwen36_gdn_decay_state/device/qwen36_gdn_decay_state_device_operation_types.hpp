// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnDecayStateParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_copy;
    bool debug_fill;
};

struct Qwen36GdnDecayStateInputs {
    const Tensor& state;
    const Tensor& alpha;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
