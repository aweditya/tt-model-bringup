// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void write_tile_from_cb(
    const Accessor& accessor,
    uint32_t tile_id,
    uint32_t cb_id,
    uint32_t page_offset,
    uint32_t page_bytes) {
    const uint32_t read_addr = get_read_ptr(cb_id) + page_offset * page_bytes;
    noc_async_write_tile(tile_id, accessor, read_addr);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t block_count = get_arg_val<uint32_t>(3);
    const uint32_t value_tiles = get_arg_val<uint32_t>(4);
    const uint32_t key_tiles = get_arg_val<uint32_t>(5);

    const auto state_accessor = TensorAccessor(state_args, state_addr);
    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_state_out);

    for (uint32_t block_offset = 0; block_offset < block_count; ++block_offset) {
        const uint32_t block = start_block + block_offset;
        const uint32_t slot = block / value_tiles;
        const uint32_t value_tile = block % value_tiles;
        const uint32_t state_slot_base = slot * key_tiles * value_tiles;

        cb_wait_front(cb_state_out, key_tiles);
        cb_wait_front(cb_out, 1);
        for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
            write_tile_from_cb(
                state_accessor,
                state_slot_base + key_tile * value_tiles + value_tile,
                cb_state_out,
                key_tile,
                tile_bytes);
        }
        write_tile_from_cb(out_accessor, slot * value_tiles + value_tile, cb_out, 0, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, key_tiles);
        cb_pop_front(cb_out, 1);
    }
}
