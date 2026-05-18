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
    constexpr uint32_t cb_q = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k = get_compile_time_arg_val(2);
    constexpr uint32_t cb_value = get_compile_time_arg_val(3);
    constexpr uint32_t cb_alpha = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
    constexpr uint32_t cb_k_col = get_compile_time_arg_val(6);

    constexpr auto state_args = TensorAccessorArgs<7>();
    constexpr auto q_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_col_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<k_col_args.next_compile_time_args_offset()>();
    constexpr auto alpha_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<alpha_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_col_addr = get_arg_val<uint32_t>(3);
    const uint32_t value_addr = get_arg_val<uint32_t>(4);
    const uint32_t alpha_addr = get_arg_val<uint32_t>(5);
    const uint32_t beta_addr = get_arg_val<uint32_t>(6);
    const uint32_t start_block = get_arg_val<uint32_t>(7);
    const uint32_t block_count = get_arg_val<uint32_t>(8);
    const uint32_t key_tiles = get_arg_val<uint32_t>(9);
    const uint32_t value_tiles = get_arg_val<uint32_t>(10);
    const uint32_t use_pretransposed_k_col = get_arg_val<uint32_t>(11);
    const uint32_t state_tiles_per_slot = key_tiles * value_tiles;

    const auto state_accessor = TensorAccessor(state_args, state_addr);
    const auto q_accessor = TensorAccessor(q_args, q_addr);
    const auto k_accessor = TensorAccessor(k_args, k_addr);
    const auto k_col_accessor = TensorAccessor(k_col_args, k_col_addr);
    const auto value_accessor = TensorAccessor(value_args, value_addr);
    const auto alpha_accessor = TensorAccessor(alpha_args, alpha_addr);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr);

    const uint32_t state_tile_bytes = get_tile_size(cb_state_in);
    const uint32_t q_tile_bytes = get_tile_size(cb_q);
    const uint32_t k_tile_bytes = get_tile_size(cb_k);
    const uint32_t k_col_tile_bytes = get_tile_size(cb_k_col);
    const uint32_t value_tile_bytes = get_tile_size(cb_value);
    const uint32_t alpha_tile_bytes = get_tile_size(cb_alpha);
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);

    for (uint32_t block_offset = 0; block_offset < block_count; ++block_offset) {
        const uint32_t block = start_block + block_offset;
        const uint32_t slot = block / value_tiles;
        const uint32_t value_tile = block % value_tiles;
        const uint32_t state_slot_base = slot * state_tiles_per_slot;

        cb_reserve_back(cb_state_in, key_tiles);
        cb_reserve_back(cb_q, key_tiles);
        cb_reserve_back(cb_k, key_tiles);
        if (use_pretransposed_k_col != 0) {
            cb_reserve_back(cb_k_col, key_tiles);
        }
        cb_reserve_back(cb_value, 1);
        cb_reserve_back(cb_alpha, 1);
        cb_reserve_back(cb_beta, 1);

        for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
            read_tile_to_cb(
                state_accessor,
                state_slot_base + key_tile * value_tiles + value_tile,
                cb_state_in,
                key_tile,
                state_tile_bytes);
            read_tile_to_cb(q_accessor, slot * key_tiles + key_tile, cb_q, key_tile, q_tile_bytes);
            read_tile_to_cb(k_accessor, slot * key_tiles + key_tile, cb_k, key_tile, k_tile_bytes);
            if (use_pretransposed_k_col != 0) {
                read_tile_to_cb(k_col_accessor, slot * key_tiles + key_tile, cb_k_col, key_tile, k_col_tile_bytes);
            }
        }
        read_tile_to_cb(value_accessor, slot * value_tiles + value_tile, cb_value, 0, value_tile_bytes);
        read_tile_to_cb(alpha_accessor, slot, cb_alpha, 0, alpha_tile_bytes);
        read_tile_to_cb(beta_accessor, slot, cb_beta, 0, beta_tile_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_state_in, key_tiles);
        cb_push_back(cb_q, key_tiles);
        cb_push_back(cb_k, key_tiles);
        if (use_pretransposed_k_col != 0) {
            cb_push_back(cb_k_col, key_tiles);
        }
        cb_push_back(cb_value, 1);
        cb_push_back(cb_alpha, 1);
        cb_push_back(cb_beta, 1);
    }
}
