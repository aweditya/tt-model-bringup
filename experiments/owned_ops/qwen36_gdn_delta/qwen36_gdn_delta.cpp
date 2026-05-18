// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_delta.hpp"

#include "device/qwen36_gdn_delta_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_gdn_delta(
    const Tensor& value,
    const Tensor& prediction,
    const Tensor& beta,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_delta(value, prediction, beta, debug_fill, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
