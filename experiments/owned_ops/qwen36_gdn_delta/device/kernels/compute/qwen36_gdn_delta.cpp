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

FORCE_INLINE void sub_to_tmp(uint32_t cb_value, uint32_t cb_pred, uint32_t cb_tmp) {
    sub_tiles_init(cb_value, cb_pred);
    cb_wait_front(cb_value, ONE_TILE);
    cb_wait_front(cb_pred, ONE_TILE);
    cb_reserve_back(cb_tmp, ONE_TILE);

    tile_regs_acquire();
    sub_tiles(cb_value, cb_pred, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tmp);
    tile_regs_release();

    cb_push_back(cb_tmp, ONE_TILE);
}

FORCE_INLINE void mul_beta(uint32_t cb_tmp, uint32_t cb_beta, uint32_t cb_out) {
    reconfig_data_format(cb_tmp, cb_beta);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_tmp, cb_beta);
    cb_wait_front(cb_tmp, ONE_TILE);
    cb_wait_front(cb_beta, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_tmp, cb_beta, 0, 0, 0);
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
    constexpr uint32_t cb_value = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pred = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(3);
    constexpr uint32_t cb_delta_out = get_compile_time_arg_val(4);

    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t debug_mode = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_value, cb_pred, cb_tmp);

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            cb_wait_front(cb_value, ONE_TILE);
            cb_wait_front(cb_pred, ONE_TILE);
            cb_wait_front(cb_beta, ONE_TILE);
            fill_one(cb_delta_out);
        } else {
            sub_to_tmp(cb_value, cb_pred, cb_tmp);
            mul_beta(cb_tmp, cb_beta, cb_delta_out);
            cb_pop_front(cb_tmp, ONE_TILE);
        }
        cb_pop_front(cb_value, ONE_TILE);
        cb_pop_front(cb_pred, ONE_TILE);
        cb_pop_front(cb_beta, ONE_TILE);
    }
}
