// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void matmul_reduce(uint32_t cb_q, uint32_t cb_state, uint32_t key_tiles, uint32_t cb_out) {
    mm_init(cb_q, cb_state, cb_out);
    cb_wait_front(cb_q, key_tiles);
    cb_wait_front(cb_state, key_tiles);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        matmul_tiles(cb_q, cb_state, key_tile, key_tile, 0);
    }
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
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);

    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t key_tiles = get_arg_val<uint32_t>(1);
    const uint32_t debug_mode = get_arg_val<uint32_t>(2);

    binary_op_init_common(cb_q, cb_state_in, cb_output);

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            cb_wait_front(cb_state_in, key_tiles);
            cb_wait_front(cb_q, key_tiles);
            fill_one(cb_output);
        } else {
            matmul_reduce(cb_q, cb_state_in, key_tiles, cb_output);
        }
        cb_pop_front(cb_state_in, key_tiles);
        cb_pop_front(cb_q, key_tiles);
    }
}
