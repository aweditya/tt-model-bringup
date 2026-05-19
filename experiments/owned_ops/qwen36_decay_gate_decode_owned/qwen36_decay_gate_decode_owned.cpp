// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_decay_gate_decode_owned.hpp"

#include "device/qwen36_decay_gate_decode_owned_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> qwen36_decay_gate_decode_owned(
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& A_log,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_decay,
    const std::optional<Tensor>& output_beta) {
    return ttnn::prim::qwen36_decay_gate_decode_owned(
        a, b, dt_bias, A_log, debug_fill,
        output_memory_config, output_decay, output_beta);
}

}  // namespace ttnn::experimental
