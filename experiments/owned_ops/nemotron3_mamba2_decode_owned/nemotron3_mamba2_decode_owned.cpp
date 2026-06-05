// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decode_owned.hpp"

#include "device/qwen36_gdn_decode_owned_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> qwen36_gdn_decode_owned(
    const Tensor& state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const std::optional<Tensor>& k_col,
    bool debug_fill,
    bool compact_vectors,
    bool native_io,
    uint32_t debug_mode,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::qwen36_gdn_decode_owned(
        state,
        q,
        k,
        value,
        alpha,
        beta,
        k_col,
        debug_fill,
        compact_vectors,
        native_io,
        debug_mode,
        output_memory_config,
        output_tensor);
}

}  // namespace ttnn::experimental
