// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

namespace ttnn::prim {
struct Qwen36TopkOwnedParams {
    uint32_t k{};
    int8_t dim{};
    bool largest{};
    bool sorted{};
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::CoreRangeSet sub_core_grids;
};

struct Qwen36TopkOwnedInputs {
    Tensor input;
    std::optional<Tensor> indices;
    std::optional<std::tuple<Tensor, Tensor>> preallocated_outputs;
};
}  // namespace ttnn::prim
