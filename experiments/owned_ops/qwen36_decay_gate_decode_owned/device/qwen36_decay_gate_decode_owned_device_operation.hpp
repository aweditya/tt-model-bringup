// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "qwen36_decay_gate_decode_owned_device_operation_types.hpp"
#include "qwen36_decay_gate_decode_owned_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Qwen36DecayGateDecodeOwnedDeviceOperation {
    using operation_attributes_t = Qwen36DecayGateDecodeOwnedParams;
    using tensor_args_t = Qwen36DecayGateDecodeOwnedInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<Qwen36DecayGateDecodeOwnedProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> qwen36_decay_gate_decode_owned(
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& A_log,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_decay = std::nullopt,
    const std::optional<Tensor>& output_beta = std::nullopt);

}  // namespace ttnn::prim
