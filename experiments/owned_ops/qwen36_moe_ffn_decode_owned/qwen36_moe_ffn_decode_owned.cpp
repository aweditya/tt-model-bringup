// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_moe_ffn_decode_owned.hpp"

#include "device/qwen36_moe_ffn_decode_owned_device_operation.hpp"

namespace ttnn::experimental {

Tensor qwen36_moe_ffn_decode_owned(
    const Tensor& h,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& routing_weight,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_moe_ffn_decode_owned(
        h, W1, W2, routing_weight, debug_fill, output_memory_config, output_tensor);
}

}  // namespace ttnn::experimental
