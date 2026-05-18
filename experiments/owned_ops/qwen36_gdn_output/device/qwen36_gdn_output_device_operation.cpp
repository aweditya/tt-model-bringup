// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_output_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

void validate_tiled_device_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as state_next", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

}  // namespace

void Qwen36GdnOutputDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state_next = tensor_args.state_next;
    validate_tiled_device_tensor(state_next, state_next, "state_next");
    validate_tiled_device_tensor(tensor_args.q, state_next, "q");
    TT_FATAL(tensor_args.q.dtype() == state_next.dtype(), "q dtype must match state_next dtype");

    const auto& state_logical = state_next.logical_shape();
    const auto& state_padded = state_next.padded_shape();
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

    const auto& q_logical = tensor_args.q.logical_shape();
    const auto& q_padded = tensor_args.q.padded_shape();
    TT_FATAL(q_logical.rank() == 4, "q must be rank 4");
    TT_FATAL(q_logical[0] == 1, "q dim 0 must be 1");
    TT_FATAL(q_logical[1] == slots, "q slots must match state slots");
    TT_FATAL(q_logical[-2] == TILE, "q row dim must be 32 for full-tile vector bring-up");
    TT_FATAL(q_logical[-1] == state_logical[-2], "q col dim must match state key dim");
    TT_FATAL(q_padded[0] == 1 && q_padded[1] == slots, "q padded slots must match state slots");
    TT_FATAL(q_padded[-2] == TILE, "q padded row dim must be 32");
    TT_FATAL(q_padded[-1] == state_padded[-2], "q padded col dim must match state padded key dim");

    const auto output_shape = Shape({1, slots, TILE, state_logical[-1]});

    if (tensor_args.preallocated_output.has_value()) {
        validate_tiled_device_tensor(tensor_args.preallocated_output.value(), state_next, "output_tensor");
        TT_FATAL(
            tensor_args.preallocated_output->dtype() == state_next.dtype(),
            "output_tensor dtype must match state_next dtype");
        TT_FATAL(
            tensor_args.preallocated_output->logical_shape() == output_shape,
            "output_tensor logical shape must be [1, slots, 32, value_dim]");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36GdnOutputDeviceOperation::spec_return_value_t Qwen36GdnOutputDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        auto spec = tensor_args.preallocated_output->tensor_spec();
        if (args.output_memory_config.has_value()) {
            spec = spec.with_memory_config(args.output_memory_config.value());
        }
        return spec;
    }
    return TensorSpec(
        Shape({1, tensor_args.state_next.logical_shape()[1], TILE, tensor_args.state_next.logical_shape()[-1]}),
        TensorLayout(
            tensor_args.state_next.dtype(),
            tensor_args.state_next.tensor_spec().page_config(),
            args.output_memory_config.value_or(tensor_args.state_next.memory_config())));
}

Qwen36GdnOutputDeviceOperation::tensor_return_value_t Qwen36GdnOutputDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.state_next.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_output(
    const Tensor& state_next,
    const Tensor& q,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36GdnOutputDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .state_next = state_next,
            .q = q,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
