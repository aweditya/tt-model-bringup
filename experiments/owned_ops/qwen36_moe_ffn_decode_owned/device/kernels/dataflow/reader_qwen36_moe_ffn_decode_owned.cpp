// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for qwen36_moe_ffn_decode_owned — G0 scaffold.
//
// Reads h's tiles into CB_H. The G0 compute kernel discards the data and
// emits zero tiles, so this reader exists only to exercise the input
// access path. G1 will extend this to stream W1[e] and W2[e] per expert.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr auto h_args = TensorAccessorArgs<1>();

    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t hidden_tiles = get_arg_val<uint32_t>(1);

    const auto h_accessor = TensorAccessor(h_args, h_addr);
    const uint32_t tile_bytes = get_tile_size(cb_h);

    // Stream all h tiles into CB_H, one at a time (double-buffered).
    for (uint32_t t = 0; t < hidden_tiles; ++t) {
        cb_reserve_back(cb_h, 1);
        noc_async_read(h_accessor.get_noc_addr(t), get_write_ptr(cb_h), tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_h, 1);
    }
}
