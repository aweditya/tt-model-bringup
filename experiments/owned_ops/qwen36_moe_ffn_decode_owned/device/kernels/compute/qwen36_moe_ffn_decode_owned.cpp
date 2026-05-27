// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for qwen36_moe_ffn_decode_owned — G0 scaffold.
//
// G0 emits IDENTITY: copy each h tile from CB_H to CB_OUT. This is a stronger
// plumbing check than zero-emission: it proves the full
// reader → compute → writer pipeline carries data correctly, and we can
// gate G0 by asserting output == h. Real fused FFN math lands in G1.
//
// Init pattern + include set borrowed from
// experiments/owned_ops/qwen36_decay_gate_decode_owned/device/kernels/compute/
// qwen36_decay_gate_decode_owned.cpp (the 27B compute kernel that builds
// on this same qb1/qb2 tt-metal tree).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"  // init_sfpu
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"               // copy_tile

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void copy_one_tile(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    const uint32_t hidden_tiles = get_arg_val<uint32_t>(0);
    // debug_fill: in G0 every call is "fill from h"; the flag becomes
    // meaningful in G1 when the default path runs real compute.

    for (uint32_t t = 0; t < hidden_tiles; ++t) {
        copy_one_tile(cb_h, cb_out);
    }
}
