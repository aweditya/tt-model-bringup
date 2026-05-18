// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_prediction.hpp"

#include "device/qwen36_gdn_prediction_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_gdn_prediction(
    const Tensor& state_scaled,
    const Tensor& k,
    bool debug_fill,
    uint32_t debug_mode,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_prediction(state_scaled, k, debug_fill, debug_mode, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
