// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for qwen36_conv1d_decode_owned.
//
// Per tile:
//   1. Write cb_conv_out         to out_addr     (the silu(sum(mul...)) result)
//   2. Write cb_shift_state0     to state0_addr  (= old state1's content)
//   3. Write cb_shift_state1     to state1_addr  (= old state2's content)
//   4. Write cb_shift_state2     to state2_addr  (= old mixed's content)
//
// The shift CBs are populated by the reader from the original state/mixed
// buffers BEFORE compute runs, so by the time we write back to those
// addresses, the reader has already finished reading them and there is no
// read-after-write hazard.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void write_tile_from_cb(const Accessor& accessor, uint32_t tile_id, uint32_t cb_id, uint32_t tile_bytes) {
    noc_async_write(get_read_ptr(cb_id), accessor.get_noc_addr(tile_id), tile_bytes);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_conv_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_shift_state0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_shift_state1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_shift_state2 = get_compile_time_arg_val(3);

    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr auto state0_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto state1_args = TensorAccessorArgs<state0_args.next_compile_time_args_offset()>();
    constexpr auto state2_args = TensorAccessorArgs<state1_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t state0_addr = get_arg_val<uint32_t>(1);
    const uint32_t state1_addr = get_arg_val<uint32_t>(2);
    const uint32_t state2_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);
    const uint32_t tile_count = get_arg_val<uint32_t>(5);

    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const auto state0_accessor = TensorAccessor(state0_args, state0_addr);
    const auto state1_accessor = TensorAccessor(state1_args, state1_addr);
    const auto state2_accessor = TensorAccessor(state2_args, state2_addr);

    const uint32_t tile_bytes = get_tile_size(cb_conv_out);

    for (uint32_t tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
        const uint32_t tile_id = start_tile + tile_offset;

        cb_wait_front(cb_conv_out, 1);
        cb_wait_front(cb_shift_state0, 1);
        cb_wait_front(cb_shift_state1, 1);
        cb_wait_front(cb_shift_state2, 1);

        write_tile_from_cb(out_accessor, tile_id, cb_conv_out, tile_bytes);
        write_tile_from_cb(state0_accessor, tile_id, cb_shift_state0, tile_bytes);
        write_tile_from_cb(state1_accessor, tile_id, cb_shift_state1, tile_bytes);
        write_tile_from_cb(state2_accessor, tile_id, cb_shift_state2, tile_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_conv_out, 1);
        cb_pop_front(cb_shift_state0, 1);
        cb_pop_front(cb_shift_state1, 1);
        cb_pop_front(cb_shift_state2, 1);
    }
}
