// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_delta_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

void validate_tiled_device_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as value", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

}  // namespace

void Qwen36GdnDeltaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& value = tensor_args.value;
    validate_tiled_device_tensor(value, value, "value");
    validate_tiled_device_tensor(tensor_args.prediction, value, "prediction");
    validate_tiled_device_tensor(tensor_args.beta, value, "beta");
    TT_FATAL(tensor_args.prediction.dtype() == value.dtype(), "prediction dtype must match value dtype");
    TT_FATAL(tensor_args.beta.dtype() == value.dtype(), "beta dtype must match value dtype");

    const auto& value_logical = value.logical_shape();
    const auto& value_padded = value.padded_shape();
    TT_FATAL(value_logical.rank() == 4, "value must be rank 4");
    TT_FATAL(value_logical[0] == 1, "value dim 0 must be 1");
    TT_FATAL(value_logical[1] > 0, "value slots must be non-zero");
    TT_FATAL(value_logical[-2] == TILE, "value row dim must be 32 for full-tile vector bring-up");
    TT_FATAL(value_logical[-1] > 0 && value_logical[-1] <= 128, "value dim must be in [32, 128]");
    TT_FATAL(value_logical[-1] % TILE == 0, "value dim must be a whole-tile multiple");
    TT_FATAL(value_padded[0] == 1 && value_padded[1] == value_logical[1], "value padded slots must match logical");
    TT_FATAL(value_padded[-2] == TILE, "value padded row dim must be 32");
    TT_FATAL(value_padded[-1] == value_logical[-1], "value padded dim must match logical dim");

    TT_FATAL(tensor_args.prediction.logical_shape() == value_logical, "prediction logical shape must match value");
    TT_FATAL(tensor_args.prediction.padded_shape() == value_padded, "prediction padded shape must match value");

    const auto& beta_logical = tensor_args.beta.logical_shape();
    const auto& beta_padded = tensor_args.beta.padded_shape();
    TT_FATAL(beta_logical.rank() == 4, "beta must be rank 4");
    TT_FATAL(beta_logical[0] == 1, "beta dim 0 must be 1");
    TT_FATAL(beta_logical[1] == value_logical[1], "beta slots must match value slots");
    TT_FATAL(beta_logical[-2] == TILE && beta_logical[-1] == TILE, "beta must end in [32, 32]");
    TT_FATAL(beta_padded[0] == 1 && beta_padded[1] == value_logical[1], "beta padded slots must match value");
    TT_FATAL(beta_padded[-2] == TILE && beta_padded[-1] == TILE, "beta padded shape must end in [32, 32]");

    if (tensor_args.preallocated_output.has_value()) {
        validate_tiled_device_tensor(tensor_args.preallocated_output.value(), value, "output_tensor");
        TT_FATAL(tensor_args.preallocated_output->dtype() == value.dtype(), "output_tensor dtype must match value dtype");
        TT_FATAL(
            tensor_args.preallocated_output->logical_shape() == value_logical,
            "output_tensor logical shape must match value");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36GdnDeltaDeviceOperation::spec_return_value_t Qwen36GdnDeltaDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        auto spec = tensor_args.preallocated_output->tensor_spec();
        if (args.output_memory_config.has_value()) {
            spec = spec.with_memory_config(args.output_memory_config.value());
        }
        return spec;
    }
    return TensorSpec(
        tensor_args.value.logical_shape(),
        TensorLayout(
            tensor_args.value.dtype(),
            tensor_args.value.tensor_spec().page_config(),
            args.output_memory_config.value_or(tensor_args.value.memory_config())));
}

Qwen36GdnDeltaDeviceOperation::tensor_return_value_t Qwen36GdnDeltaDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.value.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_gdn_delta(
    const Tensor& value,
    const Tensor& prediction,
    const Tensor& beta,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36GdnDeltaDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .value = value,
            .prediction = prediction,
            .beta = beta,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
