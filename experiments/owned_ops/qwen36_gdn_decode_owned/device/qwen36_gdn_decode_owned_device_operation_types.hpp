// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnDecodeOwnedParams {
    std::optional<MemoryConfig> output_memory_config;
    bool debug_fill;
    bool compact_vectors;
    bool native_io;
    uint32_t debug_mode;
};

struct Qwen36GdnDecodeOwnedInputs {
    const Tensor& state;
    const Tensor& q;
    const Tensor& k;
    const std::optional<Tensor>& k_col;
    const Tensor& value;
    const Tensor& alpha;
    const Tensor& beta;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::experimental::prim
