// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "qwen36_gdn_prediction_device_operation_types.hpp"
#include "qwen36_gdn_prediction_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnPredictionDeviceOperation {
    using operation_attributes_t = Qwen36GdnPredictionParams;
    using tensor_args_t = Qwen36GdnPredictionInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Qwen36GdnPredictionProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_prediction(
    const Tensor& state_scaled,
    const Tensor& k,
    bool debug_fill = false,
    uint32_t debug_mode = 0,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::prim
