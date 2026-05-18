// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_output.hpp"

#include "device/qwen36_gdn_output_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_gdn_output(
    const Tensor& state_next,
    const Tensor& q,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_output(state_next, q, debug_fill, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
