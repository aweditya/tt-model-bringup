// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::qwen36_gdn_outer_update::detail {

void bind_qwen36_gdn_outer_update(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::qwen36_gdn_outer_update::detail
