// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_gdn_decode_owned_program_factory.hpp"

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
constexpr uint32_t CB_Q = tt::CBIndex::c_1;
constexpr uint32_t CB_K = tt::CBIndex::c_2;
constexpr uint32_t CB_VALUE = tt::CBIndex::c_3;
constexpr uint32_t CB_ALPHA = tt::CBIndex::c_4;
constexpr uint32_t CB_BETA = tt::CBIndex::c_5;
constexpr uint32_t CB_STATE_SCALED = tt::CBIndex::c_6;
constexpr uint32_t CB_PRED = tt::CBIndex::c_7;
constexpr uint32_t CB_DELTA_TMP = tt::CBIndex::c_8;
constexpr uint32_t CB_DELTA = tt::CBIndex::c_9;
constexpr uint32_t CB_K_COL = tt::CBIndex::c_10;
constexpr uint32_t CB_OUTER = tt::CBIndex::c_11;
constexpr uint32_t CB_STATE_NEXT_INTERNAL = tt::CBIndex::c_12;
constexpr uint32_t CB_Q_PREP = tt::CBIndex::c_13;
constexpr uint32_t CB_K_PREP = tt::CBIndex::c_14;
constexpr uint32_t CB_VALUE_PREP = tt::CBIndex::c_15;
constexpr uint32_t CB_STATE_OUT = tt::CBIndex::c_16;
constexpr uint32_t CB_OUT = tt::CBIndex::c_17;

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

Qwen36GdnDecodeOwnedProgramFactory::cached_program_t Qwen36GdnDecodeOwnedProgramFactory::create(
    const Qwen36GdnDecodeOwnedParams& operation_attributes,
    const Qwen36GdnDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& state = tensor_args.state;
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& k_col = tensor_args.k_col;
    const auto& value = tensor_args.value;
    const auto& alpha = tensor_args.alpha;
    const auto& beta = tensor_args.beta;
    auto& output = std::get<1>(output_tensors);

    auto* state_buffer = state.buffer();
    auto* q_buffer = q.buffer();
    auto* k_buffer = k.buffer();
    auto* k_col_buffer = k_col.has_value() ? k_col->buffer() : k_buffer;
    auto* value_buffer = value.buffer();
    auto* alpha_buffer = alpha.buffer();
    auto* beta_buffer = beta.buffer();
    auto* output_buffer = output.buffer();

    const uint32_t slots = state.logical_shape()[1];
    const uint32_t key_tiles = state.padded_shape()[-2] / TILE;
    const uint32_t value_tiles = state.padded_shape()[-1] / TILE;
    const uint32_t total_blocks = slots * value_tiles;

    const bool row_major = true;
    auto grid_size = state.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            split_work_to_cores(grid_size, total_blocks, row_major);

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(state.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    create_circular_buffer(program, all_cores, CB_STATE_IN, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_Q, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_K, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_VALUE, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_ALPHA, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_BETA, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE_SCALED, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_PRED, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_DELTA_TMP, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_DELTA, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_K_COL, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_OUTER, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE_NEXT_INTERNAL, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_Q_PREP, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_K_PREP, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_VALUE_PREP, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_STATE_OUT, key_tiles * 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, all_cores, CB_OUT, 2, tensor_tile_size, tensor_format);
    std::vector<uint32_t> reader_compile_time_args = {CB_STATE_IN, CB_Q, CB_K, CB_VALUE, CB_ALPHA, CB_BETA, CB_K_COL};
    TensorAccessorArgs(state_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(q_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(k_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(k_col_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(value_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(alpha_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(beta_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        CB_STATE_IN,
        CB_Q,
        CB_K,
        CB_VALUE,
        CB_ALPHA,
        CB_BETA,
        CB_STATE_SCALED,
        CB_PRED,
        CB_DELTA_TMP,
        CB_DELTA,
        CB_K_COL,
        CB_OUTER,
        CB_STATE_NEXT_INTERNAL,
        CB_Q_PREP,
        CB_K_PREP,
        CB_VALUE_PREP,
        CB_STATE_OUT,
        CB_OUT};

    std::vector<uint32_t> writer_compile_time_args = {CB_STATE_OUT, CB_OUT};
    TensorAccessorArgs(state_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/device/kernels/dataflow/"
        "reader_qwen36_gdn_decode_owned.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/device/kernels/dataflow/"
        "writer_qwen36_gdn_decode_owned.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/device/kernels/compute/"
        "qwen36_gdn_decode_owned.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_compile_time_args});

    auto cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, row_major);
    Qwen36GdnDecodeOwnedSharedVariables shared_variables{
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
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensors);
    return cached_program;
}

void Qwen36GdnDecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Qwen36GdnDecodeOwnedParams& operation_attributes,
    const Qwen36GdnDecodeOwnedInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    auto& output = std::get<1>(output_tensors);
    auto* state_buffer = tensor_args.state.buffer();
    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* k_col_buffer = tensor_args.k_col.has_value() ? tensor_args.k_col->buffer() : k_buffer;
    auto* value_buffer = tensor_args.value.buffer();
    auto* alpha_buffer = tensor_args.alpha.buffer();
    auto* beta_buffer = tensor_args.beta.buffer();
    auto* output_buffer = output.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t key_tiles = tensor_args.state.padded_shape()[-2] / TILE;
    const uint32_t value_tiles = tensor_args.state.padded_shape()[-1] / TILE;
    const uint32_t debug_mode = operation_attributes.debug_mode != 0
                                    ? operation_attributes.debug_mode
                                    : (operation_attributes.debug_fill ? 1 : 0);
    const uint32_t use_pretransposed_k_col = tensor_args.k_col.has_value() ? 1 : 0;
    const uint32_t compact_vectors = operation_attributes.compact_vectors ? 1 : 0;
    const uint32_t native_io = operation_attributes.native_io ? 1 : 0;

    std::vector<std::vector<uint32_t>> reader_runtime_args(shared.cores.size(), std::vector<uint32_t>(12, 0));
    std::vector<std::vector<uint32_t>> compute_runtime_args(shared.cores.size(), std::vector<uint32_t>(6, 0));
    std::vector<std::vector<uint32_t>> writer_runtime_args(shared.cores.size(), std::vector<uint32_t>(7, 0));

    uint32_t blocks_written = 0;
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        const uint32_t blocks_per_core =
            i < shared.g1_numcores ? shared.num_blocks_per_core_group_1 : shared.num_blocks_per_core_group_2;

        reader_runtime_args[i][0] = state_buffer->address();
        reader_runtime_args[i][1] = q_buffer->address();
        reader_runtime_args[i][2] = k_buffer->address();
        reader_runtime_args[i][3] = k_col_buffer->address();
        reader_runtime_args[i][4] = value_buffer->address();
        reader_runtime_args[i][5] = alpha_buffer->address();
        reader_runtime_args[i][6] = beta_buffer->address();
        reader_runtime_args[i][7] = blocks_written;
        reader_runtime_args[i][8] = blocks_per_core;
        reader_runtime_args[i][9] = key_tiles;
        reader_runtime_args[i][10] = value_tiles;
        reader_runtime_args[i][11] = use_pretransposed_k_col;

        compute_runtime_args[i][0] = blocks_per_core;
        compute_runtime_args[i][1] = key_tiles;
        compute_runtime_args[i][2] = debug_mode;
        compute_runtime_args[i][3] = use_pretransposed_k_col;
        compute_runtime_args[i][4] = compact_vectors;
        compute_runtime_args[i][5] = native_io;

        writer_runtime_args[i][0] = state_buffer->address();
        writer_runtime_args[i][1] = output_buffer->address();
        writer_runtime_args[i][2] = blocks_written;
        writer_runtime_args[i][3] = blocks_per_core;
        writer_runtime_args[i][4] = value_tiles;
        writer_runtime_args[i][5] = key_tiles;
        writer_runtime_args[i][6] = native_io;

        blocks_written += blocks_per_core;
    }

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores, reader_runtime_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores, compute_runtime_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores, writer_runtime_args);
}

}  // namespace ttnn::experimental::prim
