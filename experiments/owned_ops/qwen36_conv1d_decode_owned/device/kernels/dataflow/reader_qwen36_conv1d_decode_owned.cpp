// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for qwen36_conv1d_decode_owned.
//
// Per tile, this loads:
//   - mixed, state0/1/2, weight0/1/2/3 into their compute CBs
//   - state1, state2, mixed into the writer's shift CBs (so the writer
//     can write state0_addr <- state1, state1_addr <- state2,
//     state2_addr <- mixed for the next-step state shift)
//
// The shift values come from re-reading the same input buffers via separate
// CBs; this sidesteps the in-place L1 update problem the prior
// update_cache_for_token_ / slice-write attempts hit
// (feedback_conv1d_circular_buffer.md).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void read_tile_to_cb(const Accessor& accessor, uint32_t tile_id, uint32_t cb_id, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, 1);
    noc_async_read(accessor.get_noc_addr(tile_id), get_write_ptr(cb_id), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_mixed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_state0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_state1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_state2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weight0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_weight1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weight2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_weight3 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_shift_state0 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_shift_state1 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_shift_state2 = get_compile_time_arg_val(10);

    constexpr auto mixed_args = TensorAccessorArgs<11>();
    constexpr auto state0_args = TensorAccessorArgs<mixed_args.next_compile_time_args_offset()>();
    constexpr auto state1_args = TensorAccessorArgs<state0_args.next_compile_time_args_offset()>();
    constexpr auto state2_args = TensorAccessorArgs<state1_args.next_compile_time_args_offset()>();
    constexpr auto weight0_args = TensorAccessorArgs<state2_args.next_compile_time_args_offset()>();
    constexpr auto weight1_args = TensorAccessorArgs<weight0_args.next_compile_time_args_offset()>();
    constexpr auto weight2_args = TensorAccessorArgs<weight1_args.next_compile_time_args_offset()>();
    constexpr auto weight3_args = TensorAccessorArgs<weight2_args.next_compile_time_args_offset()>();

    const uint32_t mixed_addr = get_arg_val<uint32_t>(0);
    const uint32_t state0_addr = get_arg_val<uint32_t>(1);
    const uint32_t state1_addr = get_arg_val<uint32_t>(2);
    const uint32_t state2_addr = get_arg_val<uint32_t>(3);
    const uint32_t weight0_addr = get_arg_val<uint32_t>(4);
    const uint32_t weight1_addr = get_arg_val<uint32_t>(5);
    const uint32_t weight2_addr = get_arg_val<uint32_t>(6);
    const uint32_t weight3_addr = get_arg_val<uint32_t>(7);
    const uint32_t start_tile = get_arg_val<uint32_t>(8);
    const uint32_t tile_count = get_arg_val<uint32_t>(9);

    const auto mixed_accessor = TensorAccessor(mixed_args, mixed_addr);
    const auto state0_accessor = TensorAccessor(state0_args, state0_addr);
    const auto state1_accessor = TensorAccessor(state1_args, state1_addr);
    const auto state2_accessor = TensorAccessor(state2_args, state2_addr);
    const auto weight0_accessor = TensorAccessor(weight0_args, weight0_addr);
    const auto weight1_accessor = TensorAccessor(weight1_args, weight1_addr);
    const auto weight2_accessor = TensorAccessor(weight2_args, weight2_addr);
    const auto weight3_accessor = TensorAccessor(weight3_args, weight3_addr);

    const uint32_t tile_bytes = get_tile_size(cb_mixed);

    for (uint32_t tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
        const uint32_t tile_id = start_tile + tile_offset;

        // Compute-side inputs.
        read_tile_to_cb(mixed_accessor, tile_id, cb_mixed, tile_bytes);
        read_tile_to_cb(state0_accessor, tile_id, cb_state0, tile_bytes);
        read_tile_to_cb(state1_accessor, tile_id, cb_state1, tile_bytes);
        read_tile_to_cb(state2_accessor, tile_id, cb_state2, tile_bytes);
        read_tile_to_cb(weight0_accessor, tile_id, cb_weight0, tile_bytes);
        read_tile_to_cb(weight1_accessor, tile_id, cb_weight1, tile_bytes);
        read_tile_to_cb(weight2_accessor, tile_id, cb_weight2, tile_bytes);
        read_tile_to_cb(weight3_accessor, tile_id, cb_weight3, tile_bytes);

        // Writer-side shift inputs:
        //   shift_state0 <- state1 (new state0 value)
        //   shift_state1 <- state2 (new state1 value)
        //   shift_state2 <- mixed  (new state2 value)
        read_tile_to_cb(state1_accessor, tile_id, cb_shift_state0, tile_bytes);
        read_tile_to_cb(state2_accessor, tile_id, cb_shift_state1, tile_bytes);
        read_tile_to_cb(mixed_accessor, tile_id, cb_shift_state2, tile_bytes);
    }
}
