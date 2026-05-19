// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_conv1d_decode_owned.hpp"

#include "device/qwen36_conv1d_decode_owned_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor, Tensor> qwen36_conv1d_decode_owned(
    const Tensor& mixed,
    const Tensor& state0,
    const Tensor& state1,
    const Tensor& state2,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_conv1d_decode_owned(
        mixed,
        state0,
        state1,
        state2,
        weight0,
        weight1,
        weight2,
        weight3,
        debug_fill,
        output_memory_config,
        output_tensor);
}

}  // namespace ttnn::experimental
