// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36MoeFfnDecodeOwnedParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
};

struct Qwen36MoeFfnDecodeOwnedInputs {
    const Tensor& h;
    const Tensor& W1;
    const Tensor& W2;
    const Tensor& routing_weight;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
