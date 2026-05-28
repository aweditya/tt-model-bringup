// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose_wh.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

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

FORCE_INLINE void mul_alpha_tile_indexed(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_init(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
}

FORCE_INLINE void mul_alpha_scalar_tile_indexed(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_bcast_scalar_init_short(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
}

FORCE_INLINE void mul_alpha_tile_indexed_auto(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled,
    uint32_t native_io) {
    if (native_io != 0) {
        mul_alpha_scalar_tile_indexed(cb_state, cb_alpha, tile_index, cb_state_scaled);
    } else {
        mul_alpha_tile_indexed(cb_state, cb_alpha, tile_index, cb_state_scaled);
    }
}

FORCE_INLINE void mul_alpha_tile_indexed_to_two(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_init(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
    cb_push_back(cb_state_out, ONE_TILE);
}

FORCE_INLINE void mul_alpha_scalar_tile_indexed_to_two(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_bcast_scalar_init_short(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
    cb_push_back(cb_state_out, ONE_TILE);
}

FORCE_INLINE void mul_alpha_tile_indexed_to_two_auto(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_scaled,
    uint32_t cb_state_out,
    uint32_t native_io) {
    if (native_io != 0) {
        mul_alpha_scalar_tile_indexed_to_two(cb_state, cb_alpha, tile_index, cb_state_scaled, cb_state_out);
    } else {
        mul_alpha_tile_indexed_to_two(cb_state, cb_alpha, tile_index, cb_state_scaled, cb_state_out);
    }
}

FORCE_INLINE void mul_alpha_tile_indexed_to_out(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_out);
    mul_tiles_init(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_out, ONE_TILE);
}

FORCE_INLINE void mul_alpha_scalar_tile_indexed_to_out(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state, cb_alpha);
    pack_reconfig_data_format(cb_state_out);
    mul_tiles_bcast_scalar_init_short(cb_state, cb_alpha);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_out, ONE_TILE);
}

FORCE_INLINE void mul_alpha_tile_indexed_to_out_auto(
    uint32_t cb_state,
    uint32_t cb_alpha,
    uint32_t tile_index,
    uint32_t cb_state_out,
    uint32_t native_io) {
    if (native_io != 0) {
        mul_alpha_scalar_tile_indexed_to_out(cb_state, cb_alpha, tile_index, cb_state_out);
    } else {
        mul_alpha_tile_indexed_to_out(cb_state, cb_alpha, tile_index, cb_state_out);
    }
}

FORCE_INLINE void matmul_reduce(uint32_t cb_lhs, uint32_t cb_rhs, uint32_t key_tiles, uint32_t cb_out) {
    mm_init(cb_lhs, cb_rhs, cb_out);
    cb_wait_front(cb_lhs, key_tiles);
    cb_wait_front(cb_rhs, key_tiles);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        matmul_tiles(cb_lhs, cb_rhs, key_tile, key_tile, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void sub_to_tmp(uint32_t cb_value, uint32_t cb_pred, uint32_t cb_tmp) {
    reconfig_data_format(cb_value, cb_pred);
    pack_reconfig_data_format(cb_tmp);
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

FORCE_INLINE void mul_beta(uint32_t cb_tmp, uint32_t cb_beta, uint32_t cb_delta) {
    reconfig_data_format(cb_tmp, cb_beta);
    pack_reconfig_data_format(cb_delta);
    mul_tiles_init(cb_tmp, cb_beta);
    cb_wait_front(cb_tmp, ONE_TILE);
    cb_wait_front(cb_beta, ONE_TILE);
    cb_reserve_back(cb_delta, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_tmp, cb_beta, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_delta);
    tile_regs_release();

    cb_push_back(cb_delta, ONE_TILE);
}

FORCE_INLINE void mul_beta_scalar(uint32_t cb_tmp, uint32_t cb_beta, uint32_t cb_delta) {
    reconfig_data_format(cb_tmp, cb_beta);
    pack_reconfig_data_format(cb_delta);
    mul_tiles_bcast_scalar_init_short(cb_tmp, cb_beta);
    cb_wait_front(cb_tmp, ONE_TILE);
    cb_wait_front(cb_beta, ONE_TILE);
    cb_reserve_back(cb_delta, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_tmp, cb_beta, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_delta);
    tile_regs_release();

    cb_push_back(cb_delta, ONE_TILE);
}

FORCE_INLINE void mul_beta_auto(uint32_t cb_tmp, uint32_t cb_beta, uint32_t cb_delta, uint32_t native_io) {
    if (native_io != 0) {
        mul_beta_scalar(cb_tmp, cb_beta, cb_delta);
    } else {
        mul_beta(cb_tmp, cb_beta, cb_delta);
    }
}

FORCE_INLINE void broadcast_row_tile_indexed(uint32_t cb_in, uint32_t tile_index, uint32_t cb_out) {
    unary_bcast_init<BroadcastType::ROW>(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_in, tile_index + 1);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    unary_bcast<BroadcastType::ROW>(cb_in, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void transpose_k_indexed(uint32_t cb_k, uint32_t tile_index, uint32_t cb_k_col) {
    unary_op_init_common(cb_k, cb_k_col);
    transpose_wh_init_short(cb_k);
    pack_reconfig_data_format(cb_k_col);
    cb_wait_front(cb_k, tile_index + 1);
    cb_reserve_back(cb_k_col, ONE_TILE);

    tile_regs_acquire();
    transpose_wh_tile(cb_k, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_k_col);
    tile_regs_release();

    cb_push_back(cb_k_col, ONE_TILE);
}

FORCE_INLINE void mul_outer(uint32_t cb_k_col, uint32_t cb_delta, uint32_t cb_outer) {
    reconfig_data_format(cb_k_col, cb_delta);
    pack_reconfig_data_format(cb_outer);
    mul_tiles_init(cb_k_col, cb_delta);
    cb_wait_front(cb_k_col, ONE_TILE);
    cb_wait_front(cb_delta, ONE_TILE);
    cb_reserve_back(cb_outer, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_k_col, cb_delta, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_outer);
    tile_regs_release();

    cb_push_back(cb_outer, ONE_TILE);
}

FORCE_INLINE void matmul_outer(uint32_t cb_k_col, uint32_t cb_delta, uint32_t cb_outer) {
    mm_init(cb_k_col, cb_delta, cb_outer);
    cb_wait_front(cb_k_col, ONE_TILE);
    cb_wait_front(cb_delta, ONE_TILE);
    cb_reserve_back(cb_outer, ONE_TILE);

    tile_regs_acquire();
    matmul_tiles(cb_k_col, cb_delta, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_outer);
    tile_regs_release();

    cb_push_back(cb_outer, ONE_TILE);
}

FORCE_INLINE void add_state_to_two(
    uint32_t cb_state_scaled,
    uint32_t cb_outer,
    uint32_t tile_index,
    uint32_t cb_state_next_internal,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state_scaled, cb_outer);
    pack_reconfig_data_format(cb_state_next_internal);
    add_tiles_init(cb_state_scaled, cb_outer);
    cb_wait_front(cb_state_scaled, tile_index + 1);
    cb_wait_front(cb_outer, ONE_TILE);
    cb_reserve_back(cb_state_next_internal, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_state_scaled, cb_outer, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_next_internal);
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_next_internal, ONE_TILE);
    cb_push_back(cb_state_out, ONE_TILE);
}

FORCE_INLINE void add_state_to_out(
    uint32_t cb_state_scaled,
    uint32_t cb_outer,
    uint32_t tile_index,
    uint32_t cb_state_out) {
    reconfig_data_format(cb_state_scaled, cb_outer);
    pack_reconfig_data_format(cb_state_out);
    add_tiles_init(cb_state_scaled, cb_outer);
    cb_wait_front(cb_state_scaled, tile_index + 1);
    cb_wait_front(cb_outer, ONE_TILE);
    cb_reserve_back(cb_state_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_state_scaled, cb_outer, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_out);
    tile_regs_release();

    cb_push_back(cb_state_out, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k = get_compile_time_arg_val(2);
    constexpr uint32_t cb_value = get_compile_time_arg_val(3);
    constexpr uint32_t cb_alpha = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
    constexpr uint32_t cb_state_scaled = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pred = get_compile_time_arg_val(7);
    constexpr uint32_t cb_delta_tmp = get_compile_time_arg_val(8);
    constexpr uint32_t cb_delta = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k_col = get_compile_time_arg_val(10);
    constexpr uint32_t cb_outer = get_compile_time_arg_val(11);
    constexpr uint32_t cb_state_next_internal = get_compile_time_arg_val(12);
    constexpr uint32_t cb_q_prep = get_compile_time_arg_val(13);
    constexpr uint32_t cb_k_prep = get_compile_time_arg_val(14);
    constexpr uint32_t cb_value_prep = get_compile_time_arg_val(15);
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);
    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t key_tiles = get_arg_val<uint32_t>(1);
    const uint32_t debug_mode = get_arg_val<uint32_t>(2);
    const uint32_t use_pretransposed_k_col = get_arg_val<uint32_t>(3);
    const uint32_t compact_vectors = get_arg_val<uint32_t>(4);
    const uint32_t native_io = get_arg_val<uint32_t>(5);

    binary_op_init_common(cb_state_in, cb_alpha, cb_state_scaled);

    for (uint32_t block = 0; block < block_count; ++block) {
        bool consumed_pretransposed_k_col = false;
        const uint32_t q_cb = native_io != 0 ? cb_q_prep : cb_q;
        const uint32_t k_cb = native_io != 0 ? cb_k_prep : cb_k;
        const uint32_t value_cb = native_io != 0 ? cb_value_prep : cb_value;
        const uint32_t vector_mode = native_io != 0 ? 0 : compact_vectors;
        if (native_io != 0) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                broadcast_row_tile_indexed(cb_q, key_tile, cb_q_prep);
                broadcast_row_tile_indexed(cb_k, key_tile, cb_k_prep);
            }
            broadcast_row_tile_indexed(cb_value, 0, cb_value_prep);
        }
        if (debug_mode == 1) {
            cb_wait_front(cb_state_in, key_tiles);
            cb_wait_front(cb_q, key_tiles);
            cb_wait_front(cb_k, key_tiles);
            if (use_pretransposed_k_col != 0) {
                cb_wait_front(cb_k_col, key_tiles);
            }
            cb_wait_front(cb_value, ONE_TILE);
            cb_wait_front(cb_alpha, ONE_TILE);
            cb_wait_front(cb_beta, ONE_TILE);
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                fill_one(cb_state_out);
            }
            fill_one(cb_out);
        } else if (debug_mode == 2) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_to_out_auto(cb_state_in, cb_alpha, key_tile, cb_state_out, native_io);
            }
            fill_one(cb_out);
        } else if (debug_mode == 3) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_to_two_auto(
                    cb_state_in, cb_alpha, key_tile, cb_state_scaled, cb_state_out, native_io);
            }
            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_out);
            cb_pop_front(cb_state_scaled, key_tiles);
        } else if (debug_mode == 4) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_to_two_auto(
                    cb_state_in, cb_alpha, key_tile, cb_state_scaled, cb_state_out, native_io);
            }
            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_out, native_io);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
        } else if (debug_mode == 5) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_auto(cb_state_in, cb_alpha, key_tile, cb_state_scaled, native_io);
            }

            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_delta, native_io);

            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (use_pretransposed_k_col == 0) {
                    transpose_k_indexed(k_cb, key_tile, cb_k_col);
                } else {
                    consumed_pretransposed_k_col = true;
                }
                if (vector_mode == 0) {
                    mul_outer(cb_k_col, cb_delta, cb_outer);
                } else {
                    matmul_outer(cb_k_col, cb_delta, cb_outer);
                }
                add_state_to_two(cb_state_scaled, cb_outer, key_tile, cb_state_next_internal, cb_state_out);
                cb_pop_front(cb_k_col, ONE_TILE);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            fill_one(cb_out);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_delta, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
            cb_pop_front(cb_state_next_internal, key_tiles);
        } else if (debug_mode == 6) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                fill_one(cb_state_out);
            }
            matmul_reduce(q_cb, cb_state_in, key_tiles, cb_out);
        } else if (debug_mode == 7) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_to_two_auto(
                    cb_state_in, cb_alpha, key_tile, cb_state_scaled, cb_state_out, native_io);
            }
            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_delta, native_io);

            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (use_pretransposed_k_col == 0) {
                    transpose_k_indexed(k_cb, key_tile, cb_k_col);
                } else {
                    consumed_pretransposed_k_col = true;
                }
                cb_pop_front(cb_k_col, ONE_TILE);
            }

            fill_one(cb_out);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_delta, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
        } else if (debug_mode == 8) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_to_two_auto(
                    cb_state_in, cb_alpha, key_tile, cb_state_scaled, cb_state_out, native_io);
            }
            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_delta, native_io);

            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (use_pretransposed_k_col == 0) {
                    transpose_k_indexed(k_cb, key_tile, cb_k_col);
                } else {
                    consumed_pretransposed_k_col = true;
                }
                if (vector_mode == 0) {
                    mul_outer(cb_k_col, cb_delta, cb_outer);
                } else {
                    matmul_outer(cb_k_col, cb_delta, cb_outer);
                }
                cb_pop_front(cb_k_col, ONE_TILE);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            fill_one(cb_out);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_delta, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
        } else if (debug_mode == 9) {
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_auto(cb_state_in, cb_alpha, key_tile, cb_state_scaled, native_io);
            }

            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_delta, native_io);

            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (use_pretransposed_k_col == 0) {
                    transpose_k_indexed(k_cb, key_tile, cb_k_col);
                } else {
                    consumed_pretransposed_k_col = true;
                }
                if (vector_mode == 0) {
                    mul_outer(cb_k_col, cb_delta, cb_outer);
                } else {
                    matmul_outer(cb_k_col, cb_delta, cb_outer);
                }
                add_state_to_out(cb_state_scaled, cb_outer, key_tile, cb_state_out);
                cb_pop_front(cb_k_col, ONE_TILE);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            fill_one(cb_out);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_delta, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
        } else {
            // Production path (mode 0) + batched-safe variant (debug_mode == 10).
            // safe_out routes the output matmul through cb_state_next_internal
            // (produced AND consumed by compute) instead of cb_state_out. mode 0
            // reads cb_state_out for the output AND lets the writer pop it — a
            // dual-consumer race: once a core processes >1 block (slots > ~24),
            // the writer pops a block's state_out tiles before this output matmul
            // reads them, corrupting early slots. The two-CB form (cf. README
            // "duplicate internal state_next CB", removed for B=1 perf) decouples
            // them: state_next_internal → output (compute-owned), state_out →
            // writer only. Done as an in-loop conditional (not a duplicate
            // branch) to stay under the TENSIX kernel-config size limit.
            const bool safe_out = (debug_mode == 10);
            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                mul_alpha_tile_indexed_auto(cb_state_in, cb_alpha, key_tile, cb_state_scaled, native_io);
            }

            matmul_reduce(k_cb, cb_state_scaled, key_tiles, cb_pred);
            sub_to_tmp(value_cb, cb_pred, cb_delta_tmp);
            mul_beta_auto(cb_delta_tmp, cb_beta, cb_delta, native_io);

            for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
                if (use_pretransposed_k_col == 0) {
                    transpose_k_indexed(k_cb, key_tile, cb_k_col);
                } else {
                    consumed_pretransposed_k_col = true;
                }
                if (vector_mode == 0) {
                    mul_outer(cb_k_col, cb_delta, cb_outer);
                } else {
                    matmul_outer(cb_k_col, cb_delta, cb_outer);
                }
                if (safe_out) {
                    add_state_to_two(cb_state_scaled, cb_outer, key_tile, cb_state_next_internal, cb_state_out);
                } else {
                    add_state_to_out(cb_state_scaled, cb_outer, key_tile, cb_state_out);
                }
                cb_pop_front(cb_k_col, ONE_TILE);
                cb_pop_front(cb_outer, ONE_TILE);
            }

            matmul_reduce(q_cb, safe_out ? cb_state_next_internal : cb_state_out, key_tiles, cb_out);
            cb_pop_front(cb_pred, ONE_TILE);
            cb_pop_front(cb_delta_tmp, ONE_TILE);
            cb_pop_front(cb_delta, ONE_TILE);
            cb_pop_front(cb_state_scaled, key_tiles);
            if (safe_out) {
                cb_pop_front(cb_state_next_internal, key_tiles);
            }
        }

        if (use_pretransposed_k_col != 0 && !consumed_pretransposed_k_col) {
            cb_pop_front(cb_k_col, key_tiles);
        }
        cb_pop_front(cb_state_in, key_tiles);
        cb_pop_front(cb_q, key_tiles);
        cb_pop_front(cb_k, key_tiles);
        cb_pop_front(cb_value, ONE_TILE);
        if (native_io != 0) {
            cb_pop_front(cb_q_prep, key_tiles);
            cb_pop_front(cb_k_prep, key_tiles);
            cb_pop_front(cb_value_prep, ONE_TILE);
        }
        cb_pop_front(cb_alpha, ONE_TILE);
        cb_pop_front(cb_beta, ONE_TILE);
    }
}
