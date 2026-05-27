// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen36_moe_ffn_decode_owned_program_factory.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// G0 scaffold CBs — minimal set:
//   CB_H:        input row, one tile wide
//   CB_OUT:      output row, one tile wide
// Real CBs for matmul/silu/etc come in G1+.
constexpr uint32_t CB_H = tt::CBIndex::c_0;
constexpr uint32_t CB_OUT = tt::CBIndex::c_1;

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

Qwen36MoeFfnDecodeOwnedProgramFactory::cached_program_t Qwen36MoeFfnDecodeOwnedProgramFactory::create(
    const Qwen36MoeFfnDecodeOwnedParams& operation_attributes,
    const Qwen36MoeFfnDecodeOwnedInputs& tensor_args,
    Tensor& output_tensor) {
    Program program = CreateProgram();

    const auto& h = tensor_args.h;
    auto* h_buffer = h.buffer();
    auto* out_buffer = output_tensor.buffer();

    // G0: single core, work = HIDDEN/TILE output tiles. The kernel writes one
    // zero tile per output column tile. G1+ will split work across cores.
    CoreRangeSet single_core = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(h.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    const uint32_t hidden_tiles = h.padded_shape()[-1] / TILE;

    // Double-buffered: 2 tiles per CB so reader can prefetch while compute runs.
    create_circular_buffer(program, single_core, CB_H, 2, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_OUT, 2, tensor_tile_size, tensor_format);

    std::vector<uint32_t> reader_compile_time_args = {CB_H};
    TensorAccessorArgs(h_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {CB_H, CB_OUT};

    std::vector<uint32_t> writer_compile_time_args = {CB_OUT};
    TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_moe_ffn_decode_owned/device/kernels/dataflow/"
        "reader_qwen36_moe_ffn_decode_owned.cpp",
        single_core,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_moe_ffn_decode_owned/device/kernels/dataflow/"
        "writer_qwen36_moe_ffn_decode_owned.cpp",
        single_core,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_moe_ffn_decode_owned/device/kernels/compute/"
        "qwen36_moe_ffn_decode_owned.cpp",
        single_core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_compile_time_args});

    std::vector<CoreCoord> cores = {CoreCoord(0, 0)};
    Qwen36MoeFfnDecodeOwnedSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .num_cores = 1,
        .hidden_tiles = hidden_tiles,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensor);
    return cached_program;
}

void Qwen36MoeFfnDecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Qwen36MoeFfnDecodeOwnedParams& operation_attributes,
    const Qwen36MoeFfnDecodeOwnedInputs& tensor_args,
    Tensor& output_tensor) {
    auto* h_buffer = tensor_args.h.buffer();
    auto* out_buffer = output_tensor.buffer();

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t debug_fill = operation_attributes.debug_fill ? 1 : 0;

    std::vector<uint32_t> reader_args = {h_buffer->address(), shared.hidden_tiles};
    std::vector<uint32_t> compute_args = {shared.hidden_tiles, debug_fill};
    std::vector<uint32_t> writer_args = {out_buffer->address(), shared.hidden_tiles};

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores[0], reader_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores[0], compute_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores[0], writer_args);
}

}  // namespace ttnn::experimental::prim
