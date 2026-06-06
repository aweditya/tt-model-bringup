// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_device_operation_types.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_single_core_program_factory.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_multi_core_program_factory.hpp"

#include <optional>
#include <variant>

namespace ttnn::prim {
struct Qwen36TopkOwnedDeviceOperation {
    using operation_attributes_t = Qwen36TopkOwnedParams;
    using tensor_args_t = Qwen36TopkOwnedInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<Qwen36TopkOwnedSingleCoreProgramFactory, Qwen36TopkOwnedMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
std::tuple<ttnn::Tensor, ttnn::Tensor> qwen36_topk_owned(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const tt::tt_metal::MemoryConfig& memory_config,
    const tt::tt_metal::CoreRangeSet& sub_core_grids,
    const std::optional<Tensor>& indices_tensor = std::nullopt,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors = std::nullopt);
}  // namespace ttnn::prim
