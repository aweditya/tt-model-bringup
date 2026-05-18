// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_outer_update_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

void validate_tiled_device_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as state_scaled", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

}  // namespace

void Qwen36GdnOuterUpdateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state_scaled = tensor_args.state_scaled;
    validate_tiled_device_tensor(state_scaled, state_scaled, "state_scaled");
    validate_tiled_device_tensor(tensor_args.k_col, state_scaled, "k_col");
    validate_tiled_device_tensor(tensor_args.delta, state_scaled, "delta");
    TT_FATAL(tensor_args.k_col.dtype() == state_scaled.dtype(), "k_col dtype must match state_scaled dtype");
    TT_FATAL(tensor_args.delta.dtype() == state_scaled.dtype(), "delta dtype must match state_scaled dtype");

    const auto& state_logical = state_scaled.logical_shape();
    const auto& state_padded = state_scaled.padded_shape();
    TT_FATAL(state_logical.rank() == 4, "state_scaled must be rank 4");
    TT_FATAL(state_logical[0] == 1, "state_scaled dim 0 must be 1");
    TT_FATAL(state_logical[1] > 0, "state_scaled slots must be non-zero");
    TT_FATAL(state_logical[-2] > 0 && state_logical[-2] <= 128, "key dim must be in [32, 128]");
    TT_FATAL(state_logical[-1] > 0 && state_logical[-1] <= 128, "value dim must be in [32, 128]");
    TT_FATAL(state_logical[-2] % TILE == 0, "key dim must be a whole-tile multiple");
    TT_FATAL(state_logical[-1] % TILE == 0, "value dim must be a whole-tile multiple");
    TT_FATAL(state_padded[0] == 1 && state_padded[1] == state_logical[1], "state padded slots must match logical");
    TT_FATAL(state_padded[-2] == state_logical[-2], "state padded key dim must match logical dim");
    TT_FATAL(state_padded[-1] == state_logical[-1], "state padded value dim must match logical dim");

    const auto& k_logical = tensor_args.k_col.logical_shape();
    const auto& k_padded = tensor_args.k_col.padded_shape();
    TT_FATAL(k_logical.rank() == 4, "k_col must be rank 4");
    TT_FATAL(k_logical[0] == 1 && k_logical[1] == state_logical[1], "k_col leading dims must match state_scaled");
    TT_FATAL(k_logical[-2] == state_logical[-2], "k_col key dim must match state_scaled");
    TT_FATAL(k_logical[-1] == TILE, "k_col must repeat each key scalar across a 32-column tile");
    TT_FATAL(k_padded[0] == 1 && k_padded[1] == state_logical[1], "k_col padded leading dims must match");
    TT_FATAL(k_padded[-2] == state_logical[-2] && k_padded[-1] == TILE, "k_col padded shape must match contract");

    const auto& delta_logical = tensor_args.delta.logical_shape();
    const auto& delta_padded = tensor_args.delta.padded_shape();
    TT_FATAL(delta_logical.rank() == 4, "delta must be rank 4");
    TT_FATAL(delta_logical[0] == 1 && delta_logical[1] == state_logical[1], "delta leading dims must match state_scaled");
    TT_FATAL(delta_logical[-2] == TILE, "delta row dim must be 32 for full-tile vector bring-up");
    TT_FATAL(delta_logical[-1] == state_logical[-1], "delta value dim must match state_scaled");
    TT_FATAL(delta_padded[0] == 1 && delta_padded[1] == state_logical[1], "delta padded leading dims must match");
    TT_FATAL(delta_padded[-2] == TILE && delta_padded[-1] == state_logical[-1], "delta padded shape must match contract");

    if (tensor_args.preallocated_output.has_value()) {
        validate_tiled_device_tensor(tensor_args.preallocated_output.value(), state_scaled, "output_tensor");
        TT_FATAL(
            tensor_args.preallocated_output->dtype() == state_scaled.dtype(),
            "output_tensor dtype must match state_scaled dtype");
        TT_FATAL(
            tensor_args.preallocated_output->logical_shape() == state_logical,
            "output_tensor logical shape must match state_scaled");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36GdnOuterUpdateDeviceOperation::spec_return_value_t Qwen36GdnOuterUpdateDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        auto spec = tensor_args.preallocated_output->tensor_spec();
        if (args.output_memory_config.has_value()) {
            spec = spec.with_memory_config(args.output_memory_config.value());
        }
        return spec;
    }
    return TensorSpec(
        tensor_args.state_scaled.logical_shape(),
        TensorLayout(
            tensor_args.state_scaled.dtype(),
            tensor_args.state_scaled.tensor_spec().page_config(),
            args.output_memory_config.value_or(tensor_args.state_scaled.memory_config())));
}

Qwen36GdnOuterUpdateDeviceOperation::tensor_return_value_t Qwen36GdnOuterUpdateDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.state_scaled.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_outer_update(
    const Tensor& state_scaled,
    const Tensor& k_col,
    const Tensor& delta,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36GdnOuterUpdateDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .state_scaled = state_scaled,
            .k_col = k_col,
            .delta = delta,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
