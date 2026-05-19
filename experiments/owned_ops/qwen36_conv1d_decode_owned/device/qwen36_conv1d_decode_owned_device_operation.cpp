// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_conv1d_decode_owned_device_operation.hpp"

#include <string_view>
#include <tuple>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

void validate_tiled_device_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as mixed", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

void validate_tap_tensor(
    const Tensor& tensor, const Tensor& reference, uint32_t d_padded, std::string_view name) {
    validate_tiled_device_tensor(tensor, reference, name);
    TT_FATAL(tensor.dtype() == reference.dtype(), "{} dtype must match mixed dtype", name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 2, "{} must be rank 2; got rank {}", name, logical.rank());
    TT_FATAL(logical[0] > 0, "{} dim 0 must be > 0", name);
    TT_FATAL(logical[1] == 1, "{} dim 1 must be 1 (single-tap column); got {}", name, logical[1]);
    TT_FATAL(padded[0] == d_padded, "{} padded dim 0 must equal mixed's; got {} vs {}", name, padded[0], d_padded);
    TT_FATAL(padded[1] == TILE, "{} padded dim 1 must be 32 (TILE); got {}", name, padded[1]);
}

}  // namespace

void Qwen36Conv1dDecodeOwnedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mixed = tensor_args.mixed;
    validate_tiled_device_tensor(mixed, mixed, "mixed");

    const auto& mixed_logical = mixed.logical_shape();
    const auto& mixed_padded = mixed.padded_shape();
    TT_FATAL(mixed_logical.rank() == 2, "mixed must be rank 2");
    TT_FATAL(mixed_logical[0] > 0, "mixed dim 0 (D) must be > 0");
    TT_FATAL(mixed_logical[1] == 1, "mixed dim 1 must be 1 (single-tap column)");
    TT_FATAL(mixed_padded[0] % TILE == 0, "mixed padded dim 0 must be a multiple of 32; got {}", mixed_padded[0]);
    TT_FATAL(mixed_padded[1] == TILE, "mixed padded dim 1 must be 32 (TILE)");

    const uint32_t d_padded = mixed_padded[0];

    validate_tap_tensor(tensor_args.state0, mixed, d_padded, "state0");
    validate_tap_tensor(tensor_args.state1, mixed, d_padded, "state1");
    validate_tap_tensor(tensor_args.state2, mixed, d_padded, "state2");
    validate_tap_tensor(tensor_args.weight0, mixed, d_padded, "weight0");
    validate_tap_tensor(tensor_args.weight1, mixed, d_padded, "weight1");
    validate_tap_tensor(tensor_args.weight2, mixed, d_padded, "weight2");
    validate_tap_tensor(tensor_args.weight3, mixed, d_padded, "weight3");

    if (tensor_args.preallocated_output.has_value()) {
        validate_tap_tensor(tensor_args.preallocated_output.value(), mixed, d_padded, "output_tensor");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36Conv1dDecodeOwnedDeviceOperation::spec_return_value_t Qwen36Conv1dDecodeOwnedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TensorSpec state0_spec = tensor_args.state0.tensor_spec();
    TensorSpec state1_spec = tensor_args.state1.tensor_spec();
    TensorSpec state2_spec = tensor_args.state2.tensor_spec();
    TensorSpec output_spec =
        tensor_args.preallocated_output.has_value()
            ? tensor_args.preallocated_output->tensor_spec()
            : TensorSpec(
                  tensor_args.mixed.logical_shape(),
                  TensorLayout(
                      tensor_args.mixed.dtype(),
                      tensor_args.mixed.tensor_spec().page_config(),
                      args.output_memory_config.value_or(tensor_args.mixed.memory_config())));
    if (tensor_args.preallocated_output.has_value() && args.output_memory_config.has_value()) {
        output_spec = output_spec.with_memory_config(args.output_memory_config.value());
    }
    return {state0_spec, state1_spec, state2_spec, output_spec};
}

Qwen36Conv1dDecodeOwnedDeviceOperation::tensor_return_value_t
Qwen36Conv1dDecodeOwnedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    Tensor output = tensor_args.preallocated_output.has_value()
                        ? tensor_args.preallocated_output.value()
                        : create_device_tensor(std::get<3>(output_specs), tensor_args.mixed.device());
    return {tensor_args.state0, tensor_args.state1, tensor_args.state2, output};
}

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
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36Conv1dDecodeOwnedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .mixed = mixed,
            .state0 = state0,
            .state1 = state1,
            .state2 = state2,
            .weight0 = weight0,
            .weight1 = weight1,
            .weight2 = weight2,
            .weight3 = weight3,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
