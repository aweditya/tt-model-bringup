// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void read_tile_to_cb(
    const Accessor& accessor,
    uint32_t tile_id,
    uint32_t cb_id,
    [[maybe_unused]] uint32_t page_bytes) {
    const uint32_t write_addr = get_write_ptr(cb_id);
    noc_async_read_tile(tile_id, accessor, write_addr);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k_col = get_compile_time_arg_val(1);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(2);

    constexpr auto state_args = TensorAccessorArgs<3>();
    constexpr auto k_col_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto delta_args = TensorAccessorArgs<k_col_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_col_addr = get_arg_val<uint32_t>(1);
    const uint32_t delta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t block_count = get_arg_val<uint32_t>(4);
    const uint32_t key_tiles = get_arg_val<uint32_t>(5);
    const uint32_t value_tiles = get_arg_val<uint32_t>(6);
    const uint32_t blocks_per_slot = key_tiles * value_tiles;

    const auto state_accessor = TensorAccessor(state_args, state_addr);
    const auto k_col_accessor = TensorAccessor(k_col_args, k_col_addr);
    const auto delta_accessor = TensorAccessor(delta_args, delta_addr);

    const uint32_t state_tile_bytes = get_tile_size(cb_state);
    const uint32_t k_col_tile_bytes = get_tile_size(cb_k_col);
    const uint32_t delta_tile_bytes = get_tile_size(cb_delta);

    for (uint32_t block_offset = 0; block_offset < block_count; ++block_offset) {
        const uint32_t block = start_block + block_offset;
        const uint32_t slot = block / blocks_per_slot;
        const uint32_t block_in_slot = block % blocks_per_slot;
        const uint32_t key_tile = block_in_slot / value_tiles;
        const uint32_t value_tile = block_in_slot % value_tiles;

        cb_reserve_back(cb_state, 1);
        cb_reserve_back(cb_k_col, 1);
        cb_reserve_back(cb_delta, 1);

        read_tile_to_cb(state_accessor, block, cb_state, state_tile_bytes);
        read_tile_to_cb(k_col_accessor, slot * key_tiles + key_tile, cb_k_col, k_col_tile_bytes);
        read_tile_to_cb(delta_accessor, slot * value_tiles + value_tile, cb_delta, delta_tile_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_state, 1);
        cb_push_back(cb_k_col, 1);
        cb_push_back(cb_delta, 1);
    }
}
