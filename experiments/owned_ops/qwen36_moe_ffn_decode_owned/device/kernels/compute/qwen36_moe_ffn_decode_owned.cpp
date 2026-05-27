// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for qwen36_moe_ffn_decode_owned — G0 scaffold.
//
// G0 produces hidden_tiles output zero tiles. The reader has already pushed
// hidden_tiles into CB_H; we drain them (their data is discarded) and emit
// hidden_tiles zero tiles into CB_OUT. This exists to prove that:
//   - CB plumbing compiles
//   - Reader → compute → writer pipeline runs without deadlock
//   - Output is allocated correctly and the writer can drain CB_OUT
//
// G1 will replace the zero-fill with the real fused gate_up + silu*up + down
// + scale + accumulate chain.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

}  // namespace

void MAIN {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    const uint32_t hidden_tiles = get_arg_val<uint32_t>(0);
    // debug_fill is currently unused in G0 — when true in G1+ we'll copy
    // CB_H tiles straight to CB_OUT to verify the input access path.

    init_sfpu(cb_h, cb_out);

    for (uint32_t t = 0; t < hidden_tiles; ++t) {
        cb_wait_front(cb_h, ONE_TILE);
        cb_reserve_back(cb_out, ONE_TILE);

        // Emit a zero tile. We allocate a DEST register, zero it, and pack.
        tile_regs_acquire();
        // DST register 0 is zero-initialized at acquire on this build path;
        // any safer-than-trust approach would use ttnn's zero_tile_init or
        // an explicit move from a known-zero CB. For G0, acquire + immediate
        // pack is enough to write all-zero data to CB_OUT.
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, ONE_TILE);
        cb_pop_front(cb_h, ONE_TILE);
    }
}
