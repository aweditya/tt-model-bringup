// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_decay_gate_decode_owned_program_factory.hpp"

#include <tuple>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// 4 input CBs + 3 intermediates + 2 output CBs = 9 total.
constexpr uint32_t CB_A = tt::CBIndex::c_0;
constexpr uint32_t CB_B = tt::CBIndex::c_1;
constexpr uint32_t CB_DT_BIAS = tt::CBIndex::c_2;
constexpr uint32_t CB_A_LOG = tt::CBIndex::c_3;
constexpr uint32_t CB_SOFTPLUS = tt::CBIndex::c_4;
constexpr uint32_t CB_NEG_EXP_A = tt::CBIndex::c_5;
constexpr uint32_t CB_G = tt::CBIndex::c_6;
constexpr uint32_t CB_DECAY_OUT = tt::CBIndex::c_7;
constexpr uint32_t CB_BETA_OUT = tt::CBIndex::c_8;

CBHandle create_circular_buffer(
    Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    uint32_t num_tiles,
    uint32_t tile_size,
    const tt::DataFormat& format) {
    const CircularBufferConfig config =
        CircularBufferConfig(num_tiles * tile_size, {{cb_id, format}}).set_page_size(cb_id, tile_size);
    return CreateCircularBuffer(program, cores, config);
}

}  // namespace

Qwen36DecayGateDecodeOwnedProgramFactory::cached_program_t Qwen36DecayGateDecodeOwnedProgramFactory::create(
    const Qwen36DecayGateDecodeOwnedParams& operation_attributes,
    const Qwen36DecayGateDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& a = tensor_args.a;
    auto& decay_out = std::get<0>(output_tensors);
    auto& beta_out = std::get<1>(output_tensors);

    auto* a_buffer = a.buffer();
    auto* b_buffer = tensor_args.b.buffer();
    auto* dt_bias_buffer = tensor_args.dt_bias.buffer();
    auto* A_log_buffer = tensor_args.A_log.buffer();
    auto* decay_buffer = decay_out.buffer();
    auto* beta_buffer = beta_out.buffer();

    // The compute is tiny — 1 work block (1 tile per CB) on 1 core.
    CoreRangeSet single_core = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    // Double-buffer all CBs (2 tiles each).
    create_circular_buffer(program, single_core, CB_A, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_B, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_DT_BIAS, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_A_LOG, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_SOFTPLUS, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_NEG_EXP_A, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_G, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_DECAY_OUT, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_BETA_OUT, 2, tensor_tile_size, tensor_format);

    std::vector<uint32_t> reader_compile_time_args = {CB_A, CB_B, CB_DT_BIAS, CB_A_LOG};
    TensorAccessorArgs(a_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(b_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(dt_bias_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(A_log_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        CB_A, CB_B, CB_DT_BIAS, CB_A_LOG,
        CB_SOFTPLUS, CB_NEG_EXP_A, CB_G,
        CB_DECAY_OUT, CB_BETA_OUT};

    std::vector<uint32_t> writer_compile_time_args = {CB_DECAY_OUT, CB_BETA_OUT};
    TensorAccessorArgs(decay_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(beta_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned/device/kernels/dataflow/"
        "reader_qwen36_decay_gate_decode_owned.cpp",
        single_core,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned/device/kernels/dataflow/"
        "writer_qwen36_decay_gate_decode_owned.cpp",
        single_core,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned/device/kernels/compute/"
        "qwen36_decay_gate_decode_owned.cpp",
        single_core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_compile_time_args});

    std::vector<CoreCoord> cores = {CoreCoord(0, 0)};
    Qwen36DecayGateDecodeOwnedSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .num_cores = 1,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensors);
    return cached_program;
}

void Qwen36DecayGateDecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Qwen36DecayGateDecodeOwnedParams& operation_attributes,
    const Qwen36DecayGateDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    auto& decay_out = std::get<0>(output_tensors);
    auto& beta_out = std::get<1>(output_tensors);
    auto* a_buffer = tensor_args.a.buffer();
    auto* b_buffer = tensor_args.b.buffer();
    auto* dt_bias_buffer = tensor_args.dt_bias.buffer();
    auto* A_log_buffer = tensor_args.A_log.buffer();
    auto* decay_buffer = decay_out.buffer();
    auto* beta_buffer = beta_out.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t debug_fill = operation_attributes.debug_fill ? 1 : 0;

    std::vector<uint32_t> reader_args = {
        a_buffer->address(), b_buffer->address(),
        dt_bias_buffer->address(), A_log_buffer->address()};
    std::vector<uint32_t> compute_args = {debug_fill};
    std::vector<uint32_t> writer_args = {decay_buffer->address(), beta_buffer->address()};

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores[0], reader_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores[0], compute_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores[0], writer_args);
}

}  // namespace ttnn::experimental::prim
