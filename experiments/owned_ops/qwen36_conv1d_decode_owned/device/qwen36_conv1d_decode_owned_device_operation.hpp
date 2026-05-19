// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "qwen36_conv1d_decode_owned_device_operation_types.hpp"
#include "qwen36_conv1d_decode_owned_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Qwen36Conv1dDecodeOwnedDeviceOperation {
    using operation_attributes_t = Qwen36Conv1dDecodeOwnedParams;
    using tensor_args_t = Qwen36Conv1dDecodeOwnedInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec, TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor, Tensor>;
    using program_factory_t = std::variant<Qwen36Conv1dDecodeOwnedProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor, Tensor> qwen36_conv1d_decode_owned(
    const Tensor& mixed,
    const Tensor& state0,
    const Tensor& state1,
    const Tensor& state2,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::prim
