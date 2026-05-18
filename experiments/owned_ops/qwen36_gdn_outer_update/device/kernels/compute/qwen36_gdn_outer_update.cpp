// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reconfig_data_format.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void mul_to_tmp(uint32_t cb_k_col, uint32_t cb_delta, uint32_t cb_tmp) {
    mul_tiles_init(cb_k_col, cb_delta);
    cb_wait_front(cb_k_col, ONE_TILE);
    cb_wait_front(cb_delta, ONE_TILE);
    cb_reserve_back(cb_tmp, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_k_col, cb_delta, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tmp);
    tile_regs_release();

    cb_push_back(cb_tmp, ONE_TILE);
}

FORCE_INLINE void add_state(uint32_t cb_state, uint32_t cb_tmp, uint32_t cb_out) {
    reconfig_data_format(cb_state, cb_tmp);
    pack_reconfig_data_format(cb_out);
    add_tiles_init(cb_state, cb_tmp);
    cb_wait_front(cb_state, ONE_TILE);
    cb_wait_front(cb_tmp, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_state, cb_tmp, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void fill_one(uint32_t cb_out) {
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k_col = get_compile_time_arg_val(1);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(3);
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(4);

    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t debug_mode = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_k_col, cb_delta, cb_tmp);

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            cb_wait_front(cb_state, ONE_TILE);
            cb_wait_front(cb_k_col, ONE_TILE);
            cb_wait_front(cb_delta, ONE_TILE);
            fill_one(cb_state_out);
        } else {
            mul_to_tmp(cb_k_col, cb_delta, cb_tmp);
            add_state(cb_state, cb_tmp, cb_state_out);
            cb_pop_front(cb_tmp, ONE_TILE);
        }
        cb_pop_front(cb_state, ONE_TILE);
        cb_pop_front(cb_k_col, ONE_TILE);
        cb_pop_front(cb_delta, ONE_TILE);
    }
}
