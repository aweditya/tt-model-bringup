// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decay_state_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

void validate_tiled_device_tensor(const Tensor& tensor, const Tensor& state, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == state.device(), "{} must be on the same device as state", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

}  // namespace

void Qwen36GdnDecayStateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state = tensor_args.state;
    validate_tiled_device_tensor(state, state, "state");
    validate_tiled_device_tensor(tensor_args.alpha, state, "alpha");
    TT_FATAL(tensor_args.alpha.dtype() == state.dtype(), "alpha dtype must match state dtype");

    const auto& state_logical = state.logical_shape();
    const auto& state_padded = state.padded_shape();
    TT_FATAL(state_logical.rank() == 4, "state must be rank 4");
    TT_FATAL(state_logical[0] == 1, "state dim 0 must be 1");
    TT_FATAL(state_padded[0] == 1, "state padded dim 0 must be 1");
    TT_FATAL(state_logical[-2] > 0 && state_logical[-1] > 0, "state trailing dims must be non-zero");
    TT_FATAL(state_logical[-2] <= 128 && state_logical[-1] <= 128, "state trailing dims must be <= [128, 128]");
    TT_FATAL(state_logical[-2] % TILE == 0, "state key dim must be a whole-tile multiple");
    TT_FATAL(state_logical[-1] % TILE == 0, "state value dim must be a whole-tile multiple");
    TT_FATAL(state_padded[-2] == state_logical[-2], "state padded key dim must match logical key dim");
    TT_FATAL(state_padded[-1] == state_logical[-1], "state padded value dim must match logical value dim");
    const uint32_t slots = state_logical[1];
    TT_FATAL(slots > 0, "state slots must be non-zero");

    const auto& alpha_logical = tensor_args.alpha.logical_shape();
    const auto& alpha_padded = tensor_args.alpha.padded_shape();
    TT_FATAL(alpha_logical.rank() == 4, "alpha must be rank 4");
    TT_FATAL(alpha_logical[0] == 1, "alpha dim 0 must be 1");
    TT_FATAL(alpha_logical[1] == slots, "alpha slots must match state slots");
    TT_FATAL(alpha_logical[-2] == TILE, "alpha row dim must be 32 for the full-tile bring-up op");
    TT_FATAL(alpha_logical[-1] == TILE, "alpha col dim must be 32 for the full-tile bring-up op");
    TT_FATAL(alpha_padded[0] == 1 && alpha_padded[1] == slots, "alpha padded slots must match state slots");
    TT_FATAL(alpha_padded[-2] == TILE && alpha_padded[-1] == TILE, "alpha padded shape must end in [32, 32]");

    if (tensor_args.preallocated_output.has_value()) {
        validate_tiled_device_tensor(tensor_args.preallocated_output.value(), state, "output_tensor");
        TT_FATAL(tensor_args.preallocated_output->dtype() == state.dtype(), "output_tensor dtype must match state dtype");
        TT_FATAL(
            tensor_args.preallocated_output->logical_shape() == state.logical_shape(),
            "output_tensor logical shape must match state");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36GdnDecayStateDeviceOperation::spec_return_value_t Qwen36GdnDecayStateDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        auto spec = tensor_args.preallocated_output->tensor_spec();
        if (args.output_memory_config.has_value()) {
            spec = spec.with_memory_config(args.output_memory_config.value());
        }
        return spec;
    }
    return TensorSpec(
        tensor_args.state.logical_shape(),
        TensorLayout(
            tensor_args.state.dtype(),
            tensor_args.state.tensor_spec().page_config(),
            args.output_memory_config.value_or(tensor_args.state.memory_config())));
}

Qwen36GdnDecayStateDeviceOperation::tensor_return_value_t Qwen36GdnDecayStateDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.state.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_decay_state(
    const Tensor& state,
    const Tensor& alpha,
    bool debug_copy,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36GdnDecayStateDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_copy = debug_copy,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .state = state,
            .alpha = alpha,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
