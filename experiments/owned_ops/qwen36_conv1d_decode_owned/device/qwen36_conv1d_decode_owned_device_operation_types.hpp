// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36Conv1dDecodeOwnedParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
};

struct Qwen36Conv1dDecodeOwnedInputs {
    const Tensor& mixed;
    const Tensor& state0;
    const Tensor& state1;
    const Tensor& state2;
    const Tensor& weight0;
    const Tensor& weight1;
    const Tensor& weight2;
    const Tensor& weight3;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
