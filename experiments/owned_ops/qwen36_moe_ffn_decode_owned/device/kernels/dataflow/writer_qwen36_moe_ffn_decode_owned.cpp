// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for qwen36_moe_ffn_decode_owned — G0 scaffold.
//
// Drains CB_OUT (one tile at a time) to the output buffer at the matching
// tile index. G1+ will add per-core tile-range arguments.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t hidden_tiles = get_arg_val<uint32_t>(1);

    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);

    for (uint32_t t = 0; t < hidden_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), out_accessor.get_noc_addr(t), tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
