// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_outer_update.hpp"

#include "device/qwen36_gdn_outer_update_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_gdn_outer_update(
    const Tensor& state_scaled,
    const Tensor& k_col,
    const Tensor& delta,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_outer_update(
        state_scaled, k_col, delta, debug_fill, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
