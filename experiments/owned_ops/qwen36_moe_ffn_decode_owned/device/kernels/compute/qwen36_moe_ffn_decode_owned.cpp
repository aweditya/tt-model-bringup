// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for qwen36_moe_ffn_decode_owned — G1a.
// Single-core fused MoE FFN, no routing-weight scaling (D-G1a-01).
//
// Per expert e in [0, E):
//   gate_up = h @ W1[e]                       (gate_up_tiles output tiles)
//   mid     = silu(gate_up[:MOE_INTER]) * gate_up[MOE_INTER:]  (mid_tiles tiles)
//   eo      = mid @ W2[e]                     (hidden_tiles output tiles)
//   For j in [0, hidden_tiles):
//     CB_PARTIAL[j] = (e == 0 ? eo[j] : prev_partial[j] + eo[j])
// After all experts: emit CB_PARTIAL[j] -> CB_OUT[j].
//
// CB_PARTIAL pattern: depth=2*hidden_tiles. For e>0 we ping-pong
// (wait_front 1 prior tile, push sum to back, pop front consumed tile).
// At any moment CB_PARTIAL holds between hidden_tiles and hidden_tiles+1 entries.
//
// Patterns lifted from experiments/owned_ops/qwen36_gdn_decode_owned/
// device/kernels/compute/ (matmul_one_tile pattern).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void matmul_reduce_one_tile(
    uint32_t cb_lhs, uint32_t cb_rhs, uint32_t key_tiles, uint32_t cb_out) {
    mm_init(cb_lhs, cb_rhs, cb_out);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    for (uint32_t k = 0; k < key_tiles; ++k) {
        matmul_tiles(cb_lhs, cb_rhs, k, k, /*transpose=*/0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void silu_one_tile(uint32_t cb_in, uint32_t in_idx, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
    silu_tile_init();
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, in_idx, 0);
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void mul_one_tile(
    uint32_t cb_a, uint32_t a_idx, uint32_t cb_b, uint32_t b_idx, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_a, cb_b);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, a_idx, b_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void add_one_tile(
    uint32_t cb_a, uint32_t a_idx, uint32_t cb_b, uint32_t b_idx, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    add_tiles_init(cb_a, cb_b);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, a_idx, b_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void copy_one_tile(uint32_t cb_in, uint32_t in_idx, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, in_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_h         = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w1        = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w2        = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rw        = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gate_up   = get_compile_time_arg_val(4);
    constexpr uint32_t cb_silu_tmp  = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mid       = get_compile_time_arg_val(6);
    constexpr uint32_t cb_eo        = get_compile_time_arg_val(7);
    constexpr uint32_t cb_eo_scaled = get_compile_time_arg_val(8);
    constexpr uint32_t cb_partial   = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out       = get_compile_time_arg_val(10);

    const uint32_t hidden_tiles   = get_arg_val<uint32_t>(0);
    const uint32_t gate_up_tiles  = get_arg_val<uint32_t>(1);
    const uint32_t mid_tiles      = get_arg_val<uint32_t>(2);
    const uint32_t experts        = get_arg_val<uint32_t>(3);

    binary_op_init_common(cb_h, cb_w1, cb_gate_up);

    // h stays resident across all expert iterations.
    cb_wait_front(cb_h, hidden_tiles);

    for (uint32_t e = 0; e < experts; ++e) {
        // -------- gate_up = h @ W1[e] (gate_up_tiles output tiles) --------
        for (uint32_t j = 0; j < gate_up_tiles; ++j) {
            cb_wait_front(cb_w1, hidden_tiles);
            matmul_reduce_one_tile(cb_h, cb_w1, hidden_tiles, cb_gate_up);
            cb_pop_front(cb_w1, hidden_tiles);
        }

        // -------- mid = silu(gate) * up (mid_tiles tiles) --------
        cb_wait_front(cb_gate_up, gate_up_tiles);
        for (uint32_t j = 0; j < mid_tiles; ++j) {
            silu_one_tile(cb_gate_up, j, cb_silu_tmp);
            cb_wait_front(cb_silu_tmp, ONE_TILE);
            mul_one_tile(cb_silu_tmp, 0, cb_gate_up, mid_tiles + j, cb_mid);
            cb_pop_front(cb_silu_tmp, ONE_TILE);
        }
        cb_pop_front(cb_gate_up, gate_up_tiles);

        // -------- eo = mid @ W2[e], scale by rw[e], accumulate into CB_PARTIAL --------
        // rw_broadcast tile for this expert is already waiting in cb_rw.
        cb_wait_front(cb_rw, ONE_TILE);
        cb_wait_front(cb_mid, mid_tiles);
        for (uint32_t j = 0; j < hidden_tiles; ++j) {
            cb_wait_front(cb_w2, mid_tiles);
            matmul_reduce_one_tile(cb_mid, cb_w2, mid_tiles, cb_eo);
            cb_pop_front(cb_w2, mid_tiles);

            // Scale eo[j] by rw_broadcast[e] (one tile of rw[e] broadcast).
            cb_wait_front(cb_eo, ONE_TILE);
            mul_one_tile(cb_eo, 0, cb_rw, 0, cb_eo_scaled);
            cb_pop_front(cb_eo, ONE_TILE);

            cb_wait_front(cb_eo_scaled, ONE_TILE);
            if (e == 0) {
                copy_one_tile(cb_eo_scaled, 0, cb_partial);
            } else {
                cb_wait_front(cb_partial, ONE_TILE);
                add_one_tile(cb_partial, 0, cb_eo_scaled, 0, cb_partial);
                cb_pop_front(cb_partial, ONE_TILE);
            }
            cb_pop_front(cb_eo_scaled, ONE_TILE);
        }
        cb_pop_front(cb_mid, mid_tiles);
        cb_pop_front(cb_rw, ONE_TILE);
    }

    cb_pop_front(cb_h, hidden_tiles);

    // -------- emit CB_PARTIAL -> CB_OUT --------
    for (uint32_t j = 0; j < hidden_tiles; ++j) {
        cb_wait_front(cb_partial, ONE_TILE);
        copy_one_tile(cb_partial, 0, cb_out);
        cb_pop_front(cb_partial, ONE_TILE);
    }
}
