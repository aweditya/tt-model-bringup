// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::qwen36_decay_gate_decode_owned::detail {

void bind_qwen36_decay_gate_decode_owned(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::qwen36_decay_gate_decode_owned::detail
