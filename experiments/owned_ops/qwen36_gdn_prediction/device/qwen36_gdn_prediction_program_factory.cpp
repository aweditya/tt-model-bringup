// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_prediction_program_factory.hpp"

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

constexpr uint32_t CB_STATE_IN = tt::CBIndex::c_0;
constexpr uint32_t CB_K = tt::CBIndex::c_1;
constexpr uint32_t CB_K_COL = tt::CBIndex::c_2;
constexpr uint32_t CB_PRODUCT = tt::CBIndex::c_3;
constexpr uint32_t CB_REDUCE_SCALER = tt::CBIndex::c_4;
constexpr uint32_t CB_REDUCED = tt::CBIndex::c_5;
constexpr uint32_t CB_ACCUM_A = tt::CBIndex::c_6;
constexpr uint32_t CB_ACCUM_B = tt::CBIndex::c_7;
constexpr uint32_t CB_PRED_OUT = tt::CBIndex::c_16;

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

Qwen36GdnPredictionProgramFactory::cached_program_t Qwen36GdnPredictionProgramFactory::create(
    const Qwen36GdnPredictionParams& operation_attributes,
    const Qwen36GdnPredictionInputs& tensor_args,
    Tensor& output_tensor) {
    Program program = CreateProgram();

    const auto& state_scaled = tensor_args.state_scaled;
    const auto& k = tensor_args.k;
    auto* state_buffer = state_scaled.buffer();
    auto* k_buffer = k.buffer();
    auto* output_buffer = output_tensor.buffer();

    const uint32_t slots = state_scaled.logical_shape()[1];
    const uint32_t key_tiles = state_scaled.padded_shape()[-2] / TILE;
    const uint32_t value_tiles = state_scaled.padded_shape()[-1] / TILE;
    const uint32_t total_blocks = slots * value_tiles;

    const bool row_major = true;
    auto grid_size = state_scaled.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            split_work_to_cores(grid_size, total_blocks, row_major);

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(state_scaled.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    create_circular_buffer(program, all_cores, CB_STATE_IN, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_K, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_K_COL, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_PRODUCT, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_REDUCE_SCALER, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_REDUCED, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_ACCUM_A, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_ACCUM_B, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_PRED_OUT, 2, tensor_tile_size, tensor_format);

    std::vector<uint32_t> reader_compile_time_args = {CB_STATE_IN, CB_K};
    TensorAccessorArgs(state_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(k_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        CB_STATE_IN,
        CB_K,
        CB_K_COL,
        CB_PRODUCT,
        CB_REDUCE_SCALER,
        CB_REDUCED,
        CB_ACCUM_A,
        CB_ACCUM_B,
        CB_PRED_OUT};

    std::vector<uint32_t> writer_compile_time_args = {CB_PRED_OUT};
    TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prediction/device/kernels/dataflow/"
        "reader_qwen36_gdn_prediction.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prediction/device/kernels/dataflow/"
        "writer_qwen36_gdn_prediction.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prediction/device/kernels/compute/"
        "qwen36_gdn_prediction.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_compile_time_args});

    auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, row_major);
    Qwen36GdnPredictionSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .num_cores = num_cores,
        .g1_numcores = core_group_1.num_cores(),
        .g2_numcores = core_group_2.num_cores(),
        .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
        .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensor);
    return cached_program;
}

void Qwen36GdnPredictionProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Qwen36GdnPredictionParams& operation_attributes,
    const Qwen36GdnPredictionInputs& tensor_args,
    Tensor& output_tensor) {
    auto* state_buffer = tensor_args.state_scaled.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* output_buffer = output_tensor.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t key_tiles = tensor_args.state_scaled.padded_shape()[-2] / TILE;
    const uint32_t value_tiles = tensor_args.state_scaled.padded_shape()[-1] / TILE;
    const uint32_t debug_mode = operation_attributes.debug_mode != 0
                                    ? operation_attributes.debug_mode
                                    : (operation_attributes.debug_fill ? 1 : 0);

    std::vector<std::vector<uint32_t>> reader_runtime_args(shared.cores.size(), std::vector<uint32_t>(6, 0));
    std::vector<std::vector<uint32_t>> compute_runtime_args(shared.cores.size(), std::vector<uint32_t>(3, 0));
    std::vector<std::vector<uint32_t>> writer_runtime_args(shared.cores.size(), std::vector<uint32_t>(5, 0));

    uint32_t blocks_written = 0;
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const uint32_t blocks_per_core =
            i < shared.g1_numcores ? shared.num_blocks_per_core_group_1 : shared.num_blocks_per_core_group_2;

        reader_runtime_args[i][0] = state_buffer->address();
        reader_runtime_args[i][1] = k_buffer->address();
        reader_runtime_args[i][2] = blocks_written;
        reader_runtime_args[i][3] = blocks_per_core;
        reader_runtime_args[i][4] = key_tiles;
        reader_runtime_args[i][5] = value_tiles;

        compute_runtime_args[i][0] = blocks_per_core;
        compute_runtime_args[i][1] = key_tiles;
        compute_runtime_args[i][2] = debug_mode;

        writer_runtime_args[i][0] = output_buffer->address();
        writer_runtime_args[i][1] = blocks_written;
        writer_runtime_args[i][2] = blocks_per_core;
        writer_runtime_args[i][3] = key_tiles;
        writer_runtime_args[i][4] = value_tiles;

        blocks_written += blocks_per_core;
    }

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores, reader_runtime_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores, compute_runtime_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores, writer_runtime_args);
}

}  // namespace ttnn::experimental::prim
