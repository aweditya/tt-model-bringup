// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36DecayGateDecodeOwnedParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
};

struct Qwen36DecayGateDecodeOwnedInputs {
    const Tensor& a;
    const Tensor& b;
    const Tensor& dt_bias;
    const Tensor& A_log;
    const std::optional<Tensor>& preallocated_decay;
    const std::optional<Tensor>& preallocated_beta;
};

}  // namespace ttnn::experimental::prim
