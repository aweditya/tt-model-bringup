// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decay_state.hpp"

#include "device/qwen36_gdn_decay_state_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_gdn_decay_state(
    const Tensor& state,
    const Tensor& alpha,
    bool debug_copy,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_decay_state(
        state, alpha, debug_copy, debug_fill, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
