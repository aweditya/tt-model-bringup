// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "qwen36_gdn_outer_update_device_operation_types.hpp"
#include "qwen36_gdn_outer_update_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnOuterUpdateDeviceOperation {
    using operation_attributes_t = Qwen36GdnOuterUpdateParams;
    using tensor_args_t = Qwen36GdnOuterUpdateInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Qwen36GdnOuterUpdateProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_outer_update(
    const Tensor& state_scaled,
    const Tensor& k_col,
    const Tensor& delta,
    bool debug_fill = false,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::prim
