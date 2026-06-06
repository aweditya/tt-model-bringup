// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_device_operation_types.hpp"

namespace ttnn::prim {

struct Qwen36TopkOwnedSingleCoreSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle binary_writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct Qwen36TopkOwnedSingleCoreProgramFactory {
    using shared_variables_t = Qwen36TopkOwnedSingleCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Qwen36TopkOwnedParams& args, const Qwen36TopkOwnedInputs& tensor_args, std::tuple<Tensor, Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Qwen36TopkOwnedParams& args,
        const Qwen36TopkOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);
};

}  // namespace ttnn::prim
