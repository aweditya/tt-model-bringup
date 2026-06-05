// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decode_owned_device_operation.hpp"

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
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as state", name);
    TT_FATAL(tensor.is_allocated(), "{} must have an allocated device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "{} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(!tensor.is_sharded(), "{} sharded layout is not supported in the single-device bring-up op", name);
}

void validate_same_dtype(const Tensor& tensor, const Tensor& reference, std::string_view name) {
    validate_tiled_device_tensor(tensor, reference, name);
    TT_FATAL(tensor.dtype() == reference.dtype(), "{} dtype must match state dtype", name);
}

void validate_vector_rows(
    const Tensor& tensor,
    const Tensor& state,
    uint32_t slots,
    uint32_t key_dim,
    std::string_view name,
    bool compact_vectors = false) {
    validate_same_dtype(tensor, state, name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 4, "{} must be rank 4", name);
    TT_FATAL(logical[0] == 1 && logical[1] == slots, "{} leading dims must match state", name);
    if (compact_vectors) {
        TT_FATAL(logical[-2] == 1 || logical[-2] == TILE, "{} row dim must be 1 or 32 in compact mode", name);
    } else {
        TT_FATAL(logical[-2] == TILE, "{} row dim must be 32", name);
    }
    TT_FATAL(logical[-1] == key_dim, "{} last dim must match state key dim", name);
    TT_FATAL(padded[0] == 1 && padded[1] == slots, "{} padded leading dims must match state", name);
    TT_FATAL(padded[-2] == TILE && padded[-1] == key_dim, "{} padded shape must match contract", name);
}

void validate_vector_cols(
    const Tensor& tensor, const Tensor& state, uint32_t slots, uint32_t key_dim, std::string_view name) {
    validate_same_dtype(tensor, state, name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 4, "{} must be rank 4", name);
    TT_FATAL(logical[0] == 1 && logical[1] == slots, "{} leading dims must match state", name);
    TT_FATAL(logical[-2] == key_dim, "{} row dim must match state key dim", name);
    TT_FATAL(logical[-1] == TILE, "{} last dim must be 32 for repeated-column bring-up", name);
    TT_FATAL(padded[0] == 1 && padded[1] == slots, "{} padded leading dims must match state", name);
    TT_FATAL(padded[-2] == key_dim && padded[-1] == TILE, "{} padded shape must match contract", name);
}

void validate_value_rows(
    const Tensor& tensor,
    const Tensor& state,
    uint32_t slots,
    uint32_t value_dim,
    std::string_view name,
    bool compact_vectors = false) {
    validate_same_dtype(tensor, state, name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 4, "{} must be rank 4", name);
    TT_FATAL(logical[0] == 1 && logical[1] == slots, "{} leading dims must match state", name);
    if (compact_vectors) {
        TT_FATAL(logical[-2] == 1 || logical[-2] == TILE, "{} row dim must be 1 or 32 in compact mode", name);
    } else {
        TT_FATAL(logical[-2] == TILE, "{} row dim must be 32", name);
    }
    TT_FATAL(logical[-1] == value_dim, "{} last dim must match state value dim", name);
    TT_FATAL(padded[0] == 1 && padded[1] == slots, "{} padded leading dims must match state", name);
    TT_FATAL(padded[-2] == TILE && padded[-1] == value_dim, "{} padded shape must match contract", name);
}

void validate_scalar_tiles(const Tensor& tensor, const Tensor& state, uint32_t slots, std::string_view name) {
    validate_same_dtype(tensor, state, name);
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    TT_FATAL(logical.rank() == 4, "{} must be rank 4", name);
    TT_FATAL(logical[0] == 1 && logical[1] == slots, "{} leading dims must match state", name);
    TT_FATAL(logical[-2] == TILE && logical[-1] == TILE, "{} must end in [32, 32]", name);
    TT_FATAL(padded[0] == 1 && padded[1] == slots, "{} padded leading dims must match state", name);
    TT_FATAL(padded[-2] == TILE && padded[-1] == TILE, "{} padded shape must end in [32, 32]", name);
}

}  // namespace

void Qwen36GdnDecodeOwnedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state = tensor_args.state;
    validate_tiled_device_tensor(state, state, "state");

    const auto& state_logical = state.logical_shape();
    const auto& state_padded = state.padded_shape();
    TT_FATAL(state_logical.rank() == 4, "state must be rank 4");
    TT_FATAL(state_logical[0] == 1, "state dim 0 must be 1");
    TT_FATAL(state_logical[1] > 0, "state slots must be non-zero");
    TT_FATAL(state_logical[-2] > 0 && state_logical[-2] <= 128, "key dim must be in [32, 128]");
    TT_FATAL(state_logical[-1] > 0 && state_logical[-1] <= 128, "value dim must be in [32, 128]");
    TT_FATAL(state_logical[-2] % TILE == 0, "key dim must be a whole-tile multiple");
    TT_FATAL(state_logical[-1] % TILE == 0, "value dim must be a whole-tile multiple");
    TT_FATAL(state_padded[0] == 1 && state_padded[1] == state_logical[1], "state padded leading dims must match");
    TT_FATAL(state_padded[-2] == state_logical[-2], "state padded key dim must match logical dim");
    TT_FATAL(state_padded[-1] == state_logical[-1], "state padded value dim must match logical dim");

    const uint32_t slots = state_logical[1];
    const uint32_t key_dim = state_logical[-2];
    const uint32_t value_dim = state_logical[-1];
    validate_vector_rows(tensor_args.q, state, slots, key_dim, "q", args.compact_vectors || args.native_io);
    validate_vector_rows(tensor_args.k, state, slots, key_dim, "k", args.compact_vectors || args.native_io);
    TT_FATAL(
        !((args.compact_vectors || args.native_io) && tensor_args.k_col.has_value()),
        "compact/native vector modes currently use compute-side K transpose and do not accept k_col");
    if (tensor_args.k_col.has_value()) {
        validate_vector_cols(tensor_args.k_col.value(), state, slots, key_dim, "k_col");
    }
    validate_value_rows(tensor_args.value, state, slots, value_dim, "value", args.compact_vectors || args.native_io);
    if (args.native_io) {
        validate_same_dtype(tensor_args.alpha, state, "alpha");
        validate_same_dtype(tensor_args.beta, state, "beta");
        const auto& alpha_logical = tensor_args.alpha.logical_shape();
        const auto& beta_logical = tensor_args.beta.logical_shape();
        TT_FATAL(alpha_logical.rank() == 4, "alpha must be rank 4 in native_io mode");
        TT_FATAL(beta_logical.rank() == 4, "beta must be rank 4 in native_io mode");
        TT_FATAL(
            alpha_logical[0] == 1 && alpha_logical[1] == slots && alpha_logical[-2] == 1 && alpha_logical[-1] == 1,
            "alpha must be [1, slots, 1, 1] in native_io mode");
        TT_FATAL(
            beta_logical[0] == 1 && beta_logical[1] == slots && beta_logical[-2] == 1 && beta_logical[-1] == 1,
            "beta must be [1, slots, 1, 1] in native_io mode");
    } else {
        validate_scalar_tiles(tensor_args.alpha, state, slots, "alpha");
        validate_scalar_tiles(tensor_args.beta, state, slots, "beta");
    }

    const auto output_shape = args.native_io ? Shape({1, slots * value_dim}) : Shape({1, slots, TILE, value_dim});
    if (tensor_args.preallocated_output.has_value()) {
        validate_same_dtype(tensor_args.preallocated_output.value(), state, "output_tensor");
        if (args.native_io) {
            TT_FATAL(
                tensor_args.preallocated_output->logical_shape() == output_shape,
                "output_tensor logical shape must be [1, slots * value_dim]");
        } else {
            TT_FATAL(
                tensor_args.preallocated_output->logical_shape() == output_shape,
                "output_tensor logical shape must be [1, slots, 32, value_dim]");
        }
    }
    if (args.output_memory_config.has_value()) {
        TT_FATAL(!args.output_memory_config->is_sharded(), "output sharding is not supported in first bring-up op");
    }
}

Qwen36GdnDecodeOwnedDeviceOperation::spec_return_value_t Qwen36GdnDecodeOwnedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TensorSpec state_spec = tensor_args.state.tensor_spec();
    TensorSpec output_spec = tensor_args.preallocated_output.has_value()
                                 ? tensor_args.preallocated_output->tensor_spec()
                                 : TensorSpec(
                                       args.native_io
                                           ? Shape({1,
                                                    tensor_args.state.logical_shape()[1] *
                                                        tensor_args.state.logical_shape()[-1]})
                                           : Shape({
                                                 1,
                                                 tensor_args.state.logical_shape()[1],
                                                 TILE,
                                                 tensor_args.state.logical_shape()[-1],
                                             }),
                                       TensorLayout(
                                           tensor_args.state.dtype(),
                                           tensor_args.state.tensor_spec().page_config(),
                                           args.output_memory_config.value_or(tensor_args.state.memory_config())));
    if (tensor_args.preallocated_output.has_value() && args.output_memory_config.has_value()) {
        output_spec = output_spec.with_memory_config(args.output_memory_config.value());
    }
    return {state_spec, output_spec};
}

Qwen36GdnDecodeOwnedDeviceOperation::tensor_return_value_t Qwen36GdnDecodeOwnedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    Tensor output = tensor_args.preallocated_output.has_value()
                        ? tensor_args.preallocated_output.value()
                        : create_device_tensor(std::get<1>(output_specs), tensor_args.state.device());
    return {tensor_args.state, output};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> qwen36_gdn_decode_owned(
    const Tensor& state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& value,
    const Tensor& alpha,
    const Tensor& beta,
    const std::optional<Tensor>& k_col,
    bool debug_fill,
    bool compact_vectors,
    bool native_io,
    uint32_t debug_mode,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& output_tensor) {
    using OperationType = ttnn::experimental::prim::Qwen36GdnDecodeOwnedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
            .compact_vectors = compact_vectors,
            .native_io = native_io,
            .debug_mode = debug_mode,
        },
        OperationType::tensor_args_t{
            .state = state,
            .q = q,
            .k = k,
            .k_col = k_col,
            .value = value,
            .alpha = alpha,
            .beta = beta,
            .preallocated_output = output_tensor,
        });
}

}  // namespace ttnn::prim
