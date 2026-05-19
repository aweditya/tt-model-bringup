// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_decay_gate_decode_owned_device_operation.hpp"

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
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as a", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

void validate_row_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    validate_tiled_device_tensor(tensor, reference, name);
    TT_FATAL(tensor.dtype() == reference.dtype(), "{} dtype must match a dtype", name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 2, "{} must be rank 2; got rank {}", name, logical.rank());
    TT_FATAL(logical[0] == 1, "{} dim 0 must be 1; got {}", name, logical[0]);
    TT_FATAL(logical[1] > 0 && logical[1] <= TILE,
        "{} dim 1 (NV) must be in (0, 32]; got {}", name, logical[1]);
    TT_FATAL(padded[0] == TILE, "{} padded dim 0 must be 32 (TILE); got {}", name, padded[0]);
    TT_FATAL(padded[1] == TILE, "{} padded dim 1 must be 32 (TILE); got {}", name, padded[1]);
}

}  // namespace

void Qwen36DecayGateDecodeOwnedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_row_tensor(tensor_args.a, tensor_args.a, "a");
    validate_row_tensor(tensor_args.b, tensor_args.a, "b");
    validate_row_tensor(tensor_args.dt_bias, tensor_args.a, "dt_bias");
    validate_row_tensor(tensor_args.A_log, tensor_args.a, "A_log");

    // All inputs must agree on the logical NV column count.
    const auto& a_logical = tensor_args.a.logical_shape();
    TT_FATAL(tensor_args.b.logical_shape()[1] == a_logical[1], "b NV mismatch with a");
    TT_FATAL(tensor_args.dt_bias.logical_shape()[1] == a_logical[1], "dt_bias NV mismatch with a");
    TT_FATAL(tensor_args.A_log.logical_shape()[1] == a_logical[1], "A_log NV mismatch with a");

    if (tensor_args.preallocated_decay.has_value()) {
        validate_row_tensor(tensor_args.preallocated_decay.value(), tensor_args.a, "output_decay");
    }
    if (tensor_args.preallocated_beta.has_value()) {
        validate_row_tensor(tensor_args.preallocated_beta.value(), tensor_args.a, "output_beta");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36DecayGateDecodeOwnedDeviceOperation::spec_return_value_t
Qwen36DecayGateDecodeOwnedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TensorSpec decay_spec = tensor_args.preallocated_decay.has_value()
                                ? tensor_args.preallocated_decay->tensor_spec()
                                : TensorSpec(
                                      tensor_args.a.logical_shape(),
                                      TensorLayout(
                                          tensor_args.a.dtype(),
                                          tensor_args.a.tensor_spec().page_config(),
                                          args.output_memory_config.value_or(tensor_args.a.memory_config())));
    TensorSpec beta_spec = tensor_args.preallocated_beta.has_value()
                               ? tensor_args.preallocated_beta->tensor_spec()
                               : TensorSpec(
                                     tensor_args.b.logical_shape(),
                                     TensorLayout(
                                         tensor_args.b.dtype(),
                                         tensor_args.b.tensor_spec().page_config(),
                                         args.output_memory_config.value_or(tensor_args.b.memory_config())));
    return {decay_spec, beta_spec};
}

Qwen36DecayGateDecodeOwnedDeviceOperation::tensor_return_value_t
Qwen36DecayGateDecodeOwnedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    Tensor decay = tensor_args.preallocated_decay.has_value()
                       ? tensor_args.preallocated_decay.value()
                       : create_device_tensor(std::get<0>(output_specs), tensor_args.a.device());
    Tensor beta = tensor_args.preallocated_beta.has_value()
                      ? tensor_args.preallocated_beta.value()
                      : create_device_tensor(std::get<1>(output_specs), tensor_args.b.device());
    return {decay, beta};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> qwen36_decay_gate_decode_owned(
    const Tensor& a,
    const Tensor& b,
    const Tensor& dt_bias,
    const Tensor& A_log,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_decay,
    const std::optional<Tensor>& output_beta) {
    using OperationType = ttnn::experimental::prim::Qwen36DecayGateDecodeOwnedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .a = a,
            .b = b,
            .dt_bias = dt_bias,
            .A_log = A_log,
            .preallocated_decay = output_decay,
            .preallocated_beta = output_beta,
        });
}

}  // namespace ttnn::prim
