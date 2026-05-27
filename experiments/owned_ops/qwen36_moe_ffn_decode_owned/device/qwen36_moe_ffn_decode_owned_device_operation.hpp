// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "qwen36_moe_ffn_decode_owned_device_operation_types.hpp"
#include "qwen36_moe_ffn_decode_owned_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Qwen36MoeFfnDecodeOwnedDeviceOperation {
    using operation_attributes_t = Qwen36MoeFfnDecodeOwnedParams;
    using tensor_args_t = Qwen36MoeFfnDecodeOwnedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Qwen36MoeFfnDecodeOwnedProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_moe_ffn_decode_owned(
    const Tensor& h,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& routing_weight,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::prim
