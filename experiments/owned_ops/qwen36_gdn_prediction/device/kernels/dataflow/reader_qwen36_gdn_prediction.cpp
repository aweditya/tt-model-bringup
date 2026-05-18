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
    uint32_t page_offset,
    uint32_t page_bytes) {
    const uint32_t write_addr = get_write_ptr(cb_id) + page_offset * page_bytes;
    noc_async_read_tile(tile_id, accessor, write_addr);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);

    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto k_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t block_count = get_arg_val<uint32_t>(3);
    const uint32_t key_tiles = get_arg_val<uint32_t>(4);
    const uint32_t value_tiles = get_arg_val<uint32_t>(5);
    const uint32_t state_tiles_per_slot = key_tiles * value_tiles;

    const auto state_accessor = TensorAccessor(state_args, state_addr);
    const auto k_accessor = TensorAccessor(k_args, k_addr);

    const uint32_t state_tile_bytes = get_tile_size(cb_state_in);
    const uint32_t k_tile_bytes = get_tile_size(cb_k);

    for (uint32_t block_offset = 0; block_offset < block_count; ++block_offset) {
        const uint32_t block = start_block + block_offset;
        const uint32_t slot = block / value_tiles;
        const uint32_t value_tile = block % value_tiles;
        const uint32_t state_slot_base = slot * state_tiles_per_slot;

        cb_reserve_back(cb_state_in, key_tiles);
        cb_reserve_back(cb_k, key_tiles);

        for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
            read_tile_to_cb(
                state_accessor,
                state_slot_base + key_tile * value_tiles + value_tile,
                cb_state_in,
                key_tile,
                state_tile_bytes);
            read_tile_to_cb(k_accessor, slot * key_tiles + key_tile, cb_k, key_tile, k_tile_bytes);
        }
        noc_async_read_barrier();

        cb_push_back(cb_state_in, key_tiles);
        cb_push_back(cb_k, key_tiles);
    }
}
