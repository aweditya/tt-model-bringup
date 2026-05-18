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
    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pred = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(2);

    constexpr auto value_args = TensorAccessorArgs<3>();
    constexpr auto pred_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<pred_args.next_compile_time_args_offset()>();

    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t pred_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t block_count = get_arg_val<uint32_t>(4);
    const uint32_t value_tiles = get_arg_val<uint32_t>(5);

    const auto value_accessor = TensorAccessor(value_args, value_addr);
    const auto pred_accessor = TensorAccessor(pred_args, pred_addr);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr);

    const uint32_t value_tile_bytes = get_tile_size(cb_value);
    const uint32_t pred_tile_bytes = get_tile_size(cb_pred);
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);

    for (uint32_t block_offset = 0; block_offset < block_count; ++block_offset) {
        const uint32_t block = start_block + block_offset;
        const uint32_t slot = block / value_tiles;

        cb_reserve_back(cb_value, 1);
        cb_reserve_back(cb_pred, 1);
        cb_reserve_back(cb_beta, 1);

        read_tile_to_cb(value_accessor, block, cb_value, value_tile_bytes);
        read_tile_to_cb(pred_accessor, block, cb_pred, pred_tile_bytes);
        read_tile_to_cb(beta_accessor, slot, cb_beta, beta_tile_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_value, 1);
        cb_push_back(cb_pred, 1);
        cb_push_back(cb_beta, 1);
    }
}
