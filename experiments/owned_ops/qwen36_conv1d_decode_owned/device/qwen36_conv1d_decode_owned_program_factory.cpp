// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_conv1d_decode_owned_program_factory.hpp"

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

constexpr uint32_t CB_MIXED = tt::CBIndex::c_0;
constexpr uint32_t CB_STATE0 = tt::CBIndex::c_1;
constexpr uint32_t CB_STATE1 = tt::CBIndex::c_2;
constexpr uint32_t CB_STATE2 = tt::CBIndex::c_3;
constexpr uint32_t CB_WEIGHT0 = tt::CBIndex::c_4;
constexpr uint32_t CB_WEIGHT1 = tt::CBIndex::c_5;
constexpr uint32_t CB_WEIGHT2 = tt::CBIndex::c_6;
constexpr uint32_t CB_WEIGHT3 = tt::CBIndex::c_7;
constexpr uint32_t CB_PRODUCT = tt::CBIndex::c_8;
constexpr uint32_t CB_ACC0 = tt::CBIndex::c_9;
constexpr uint32_t CB_ACC1 = tt::CBIndex::c_10;
constexpr uint32_t CB_CONV_OUT = tt::CBIndex::c_11;
constexpr uint32_t CB_SHIFT_STATE0 = tt::CBIndex::c_12;
constexpr uint32_t CB_SHIFT_STATE1 = tt::CBIndex::c_13;
constexpr uint32_t CB_SHIFT_STATE2 = tt::CBIndex::c_14;

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

Qwen36Conv1dDecodeOwnedProgramFactory::cached_program_t Qwen36Conv1dDecodeOwnedProgramFactory::create(
    const Qwen36Conv1dDecodeOwnedParams& operation_attributes,
    const Qwen36Conv1dDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor, Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& mixed = tensor_args.mixed;
    auto& output = std::get<3>(output_tensors);

    auto* mixed_buffer = mixed.buffer();
    auto* state0_buffer = tensor_args.state0.buffer();
    auto* state1_buffer = tensor_args.state1.buffer();
    auto* state2_buffer = tensor_args.state2.buffer();
    auto* weight0_buffer = tensor_args.weight0.buffer();
    auto* weight1_buffer = tensor_args.weight1.buffer();
    auto* weight2_buffer = tensor_args.weight2.buffer();
    auto* weight3_buffer = tensor_args.weight3.buffer();
    auto* output_buffer = output.buffer();

    const uint32_t d_padded = mixed.padded_shape()[0];
    const uint32_t total_tiles = d_padded / TILE;

    const bool row_major = true;
    auto grid_size = mixed.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            split_work_to_cores(grid_size, total_tiles, row_major);

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(mixed.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    create_circular_buffer(program, all_cores, CB_MIXED, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE0, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE1, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE2, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_WEIGHT0, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_WEIGHT1, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_WEIGHT2, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_WEIGHT3, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_PRODUCT, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_ACC0, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_ACC1, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_CONV_OUT, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_SHIFT_STATE0, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_SHIFT_STATE1, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_SHIFT_STATE2, 2, tensor_tile_size, tensor_format);

    std::vector<uint32_t> reader_compile_time_args = {
        CB_MIXED,
        CB_STATE0,
        CB_STATE1,
        CB_STATE2,
        CB_WEIGHT0,
        CB_WEIGHT1,
        CB_WEIGHT2,
        CB_WEIGHT3,
        CB_SHIFT_STATE0,
        CB_SHIFT_STATE1,
        CB_SHIFT_STATE2};
    TensorAccessorArgs(mixed_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(state0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(state1_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(state2_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight1_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight2_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight3_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        CB_MIXED,
        CB_STATE0,
        CB_STATE1,
        CB_STATE2,
        CB_WEIGHT0,
        CB_WEIGHT1,
        CB_WEIGHT2,
        CB_WEIGHT3,
        CB_PRODUCT,
        CB_ACC0,
        CB_ACC1,
        CB_CONV_OUT};

    std::vector<uint32_t> writer_compile_time_args = {
        CB_CONV_OUT, CB_SHIFT_STATE0, CB_SHIFT_STATE1, CB_SHIFT_STATE2};
    TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(state0_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(state1_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(state2_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/device/kernels/dataflow/"
        "reader_qwen36_conv1d_decode_owned.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/device/kernels/dataflow/"
        "writer_qwen36_conv1d_decode_owned.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/device/kernels/compute/"
        "qwen36_conv1d_decode_owned.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_compile_time_args});

    auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, row_major);
    Qwen36Conv1dDecodeOwnedSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .num_cores = num_cores,
        .g1_numcores = core_group_1.num_cores(),
        .g2_numcores = core_group_2.num_cores(),
        .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
        .num_tiles_per_core_group_2 = num_tiles_per_core_group_2,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensors);
    return cached_program;
}

void Qwen36Conv1dDecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Qwen36Conv1dDecodeOwnedParams& operation_attributes,
    const Qwen36Conv1dDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor, Tensor>& output_tensors) {
    auto& output = std::get<3>(output_tensors);
    auto* mixed_buffer = tensor_args.mixed.buffer();
    auto* state0_buffer = tensor_args.state0.buffer();
    auto* state1_buffer = tensor_args.state1.buffer();
    auto* state2_buffer = tensor_args.state2.buffer();
    auto* weight0_buffer = tensor_args.weight0.buffer();
    auto* weight1_buffer = tensor_args.weight1.buffer();
    auto* weight2_buffer = tensor_args.weight2.buffer();
    auto* weight3_buffer = tensor_args.weight3.buffer();
    auto* output_buffer = output.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t debug_fill = operation_attributes.debug_fill ? 1 : 0;

    std::vector<std::vector<uint32_t>> reader_runtime_args(shared.cores.size(), std::vector<uint32_t>(10, 0));
    std::vector<std::vector<uint32_t>> compute_runtime_args(shared.cores.size(), std::vector<uint32_t>(2, 0));
    std::vector<std::vector<uint32_t>> writer_runtime_args(shared.cores.size(), std::vector<uint32_t>(6, 0));

    uint32_t tiles_written = 0;
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const uint32_t tiles_per_core = i < shared.g1_numcores ? shared.num_tiles_per_core_group_1
                                                                : shared.num_tiles_per_core_group_2;

        reader_runtime_args[i][0] = mixed_buffer->address();
        reader_runtime_args[i][1] = state0_buffer->address();
        reader_runtime_args[i][2] = state1_buffer->address();
        reader_runtime_args[i][3] = state2_buffer->address();
        reader_runtime_args[i][4] = weight0_buffer->address();
        reader_runtime_args[i][5] = weight1_buffer->address();
        reader_runtime_args[i][6] = weight2_buffer->address();
        reader_runtime_args[i][7] = weight3_buffer->address();
        reader_runtime_args[i][8] = tiles_written;
        reader_runtime_args[i][9] = tiles_per_core;

        compute_runtime_args[i][0] = tiles_per_core;
        compute_runtime_args[i][1] = debug_fill;

        writer_runtime_args[i][0] = output_buffer->address();
        writer_runtime_args[i][1] = state0_buffer->address();
        writer_runtime_args[i][2] = state1_buffer->address();
        writer_runtime_args[i][3] = state2_buffer->address();
        writer_runtime_args[i][4] = tiles_written;
        writer_runtime_args[i][5] = tiles_per_core;

        tiles_written += tiles_per_core;
    }

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores, reader_runtime_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores, compute_runtime_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores, writer_runtime_args);
}

}  // namespace ttnn::experimental::prim
