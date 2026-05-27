// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_moe_ffn_decode_owned_device_operation.hpp"

#include <string_view>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// G0 scaffolding: validate inputs are on-device, TILE_LAYOUT, bf16. Output
// shape is [1, HIDDEN] matching h's last-dim.
void validate_device_tensor(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as h", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} must be BFLOAT16 for G0", name);
}

}  // namespace

void Qwen36MoeFfnDecodeOwnedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_device_tensor(tensor_args.h, tensor_args.h, "h");
    validate_device_tensor(tensor_args.W1, tensor_args.h, "W1");
    validate_device_tensor(tensor_args.W2, tensor_args.h, "W2");
    validate_device_tensor(tensor_args.routing_weight, tensor_args.h, "routing_weight");

    // h logical shape [1, HIDDEN] with padded last-dim a multiple of TILE.
    const auto& h_logical = tensor_args.h.logical_shape();
    const auto& h_padded = tensor_args.h.padded_shape();
    TT_FATAL(h_logical.rank() == 2, "h must be rank 2; got rank {}", h_logical.rank());
    TT_FATAL(h_logical[0] == 1, "h dim 0 must be 1; got {}", h_logical[0]);
    TT_FATAL(h_logical[1] > 0, "h hidden dim must be > 0");
    TT_FATAL(h_padded[-1] % TILE == 0, "h padded hidden must be a multiple of {}; got {}", TILE, h_padded[-1]);

    if (tensor_args.preallocated_output.has_value()) {
        validate_device_tensor(tensor_args.preallocated_output.value(), tensor_args.h, "output_tensor");
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in G0 bring-up");
    }
}

Qwen36MoeFfnDecodeOwnedDeviceOperation::spec_return_value_t
Qwen36MoeFfnDecodeOwnedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Output has the same shape and layout as h.
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return TensorSpec(
        tensor_args.h.logical_shape(),
        TensorLayout(
            tensor_args.h.dtype(),
            tensor_args.h.tensor_spec().page_config(),
            args.output_memory_config.value_or(tensor_args.h.memory_config())));
}

Qwen36MoeFfnDecodeOwnedDeviceOperation::tensor_return_value_t
Qwen36MoeFfnDecodeOwnedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.h.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor qwen36_moe_ffn_decode_owned(
    const Tensor& h,
    const Tensor& W1,
    const Tensor& W2,
    const Tensor& routing_weight,
    bool debug_fill,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36MoeFfnDecodeOwnedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
        },
        OperationType::tensor_args_t{
            .h = h,
            .W1 = W1,
            .W2 = W2,
            .routing_weight = routing_weight,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
