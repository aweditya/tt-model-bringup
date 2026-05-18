// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <vector>

#include "qwen36_gdn_delta_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Qwen36GdnDeltaSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
    uint32_t g1_numcores{};
    uint32_t g2_numcores{};
    uint32_t num_blocks_per_core_group_1{};
    uint32_t num_blocks_per_core_group_2{};
};

struct Qwen36GdnDeltaProgramFactory {
    using shared_variables_t = Qwen36GdnDeltaSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Qwen36GdnDeltaParams& operation_attributes,
        const Qwen36GdnDeltaInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Qwen36GdnDeltaParams& operation_attributes,
        const Qwen36GdnDeltaInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
