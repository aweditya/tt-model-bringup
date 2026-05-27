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

// CB index assignments (single-core G1b).
constexpr uint32_t CB_H         = tt::CBIndex::c_0;
constexpr uint32_t CB_W1        = tt::CBIndex::c_1;
constexpr uint32_t CB_W2        = tt::CBIndex::c_2;
constexpr uint32_t CB_RW        = tt::CBIndex::c_3;
constexpr uint32_t CB_GATE_UP   = tt::CBIndex::c_4;
constexpr uint32_t CB_SILU_TMP  = tt::CBIndex::c_5;
constexpr uint32_t CB_MID       = tt::CBIndex::c_6;
constexpr uint32_t CB_EO        = tt::CBIndex::c_7;
constexpr uint32_t CB_EO_SCALED = tt::CBIndex::c_8;
constexpr uint32_t CB_PARTIAL   = tt::CBIndex::c_9;
constexpr uint32_t CB_OUT       = tt::CBIndex::c_10;

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
    const auto& W1 = tensor_args.W1;
    const auto& W2 = tensor_args.W2;
    const auto& rw = tensor_args.routing_weight;

    auto* h_buffer  = h.buffer();
    auto* w1_buffer = W1.buffer();
    auto* w2_buffer = W2.buffer();
    auto* rw_buffer = rw.buffer();
    auto* out_buffer = output_tensor.buffer();

    // Single-core G1a (D-G1a-02).
    CoreRangeSet single_core = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    const tt::DataFormat tensor_format = datatype_to_dataformat_converter(h.dtype());
    const uint32_t tensor_tile_size = tt::tile_size(tensor_format);

    // Shape-derived tile counts.
    const auto& h_padded  = h.padded_shape();
    const auto& W1_padded = W1.padded_shape();
    const auto& W2_padded = W2.padded_shape();
    const uint32_t hidden_tiles  = h_padded[-1] / TILE;             // HIDDEN/TILE
    const uint32_t gate_up_tiles = W1_padded[-1] / TILE;            // 2*MOE_INTER/TILE
    const uint32_t mid_tiles     = W2_padded[-2] / TILE;            // MOE_INTER/TILE
    // `experts` is consumed at override_runtime_arguments time (recomputed
    // from tensor shapes there); kept here to centralize the shape derivation.
    [[maybe_unused]] const uint32_t experts =
        static_cast<uint32_t>(W1.logical_shape()[0]);

    // CB sizes. Compute calls cb_wait_front(cb_w1, hidden_tiles) and
    // cb_wait_front(cb_w2, mid_tiles), so CB_W1 must hold ≥ hidden_tiles
    // and CB_W2 must hold ≥ mid_tiles to avoid reader/compute deadlock.
    // 2× for ping-pong headroom (reader prefetches the next column while
    // compute drains the current one).
    //   CB_H:         hidden_tiles tiles resident (read once, used all experts)
    //   CB_W1:        2*hidden_tiles (must fit one full inner-reduce + prefetch)
    //   CB_W2:        2*mid_tiles    (must fit one full inner-reduce + prefetch)
    //   CB_RW:        2 tiles (one rw_broadcast tile per expert, double-buffered)
    //   CB_GATE_UP:   gate_up_tiles resident (one expert's full output row)
    //   CB_SILU_TMP:  2 tiles scratch
    //   CB_MID:       mid_tiles resident
    //   CB_EO:        2 tiles (one expert_out tile at a time)
    //   CB_EO_SCALED: 2 tiles (eo[j] * rw_broadcast[e])
    //   CB_PARTIAL:   2*hidden_tiles depth (ping-pong headroom; D-G1a-04 note)
    //   CB_OUT:       2 tiles drain
    create_circular_buffer(program, single_core, CB_H,         hidden_tiles,     tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_W1,        2 * hidden_tiles, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_W2,        2 * mid_tiles,    tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_RW,        2,                tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_GATE_UP,   gate_up_tiles,    tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_SILU_TMP,  2,                tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_MID,       mid_tiles,        tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_EO,        2,                tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_EO_SCALED, 2,                tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_PARTIAL,   2 * hidden_tiles, tensor_tile_size, tensor_format);
    create_circular_buffer(program, single_core, CB_OUT,       2,                tensor_tile_size, tensor_format);

    std::vector<uint32_t> reader_compile_time_args = {CB_H, CB_W1, CB_W2, CB_RW};
    TensorAccessorArgs(h_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(w1_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(w2_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(rw_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        CB_H, CB_W1, CB_W2, CB_RW,
        CB_GATE_UP, CB_SILU_TMP, CB_MID,
        CB_EO, CB_EO_SCALED, CB_PARTIAL, CB_OUT};

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
    // Stash extra shape info on shared via additional fields below would require
    // header changes; instead we recompute from buffers in override_runtime_arguments.

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, output_tensor);
    return cached_program;
}

void Qwen36MoeFfnDecodeOwnedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const Qwen36MoeFfnDecodeOwnedParams& operation_attributes,
    const Qwen36MoeFfnDecodeOwnedInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    auto* h_buffer  = tensor_args.h.buffer();
    auto* w1_buffer = tensor_args.W1.buffer();
    auto* w2_buffer = tensor_args.W2.buffer();
    auto* rw_buffer = tensor_args.routing_weight.buffer();
    auto* out_buffer = output_tensor.buffer();

    // Recompute the per-call dimensions (in tiles).
    const uint32_t hidden_tiles  = tensor_args.h.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
    const uint32_t gate_up_tiles = tensor_args.W1.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
    const uint32_t mid_tiles     = tensor_args.W2.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t experts       = static_cast<uint32_t>(tensor_args.W1.logical_shape()[0]);

    std::vector<uint32_t> reader_args = {
        h_buffer->address(), w1_buffer->address(), w2_buffer->address(), rw_buffer->address(),
        hidden_tiles, gate_up_tiles, mid_tiles, experts};
    std::vector<uint32_t> compute_args = {hidden_tiles, gate_up_tiles, mid_tiles, experts};
    std::vector<uint32_t> writer_args = {out_buffer->address(), hidden_tiles};

    SetRuntimeArgs(program, shared.reader_kernel_id, shared.cores[0], reader_args);
    SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores[0], compute_args);
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.cores[0], writer_args);
}

}  // namespace ttnn::experimental::prim
