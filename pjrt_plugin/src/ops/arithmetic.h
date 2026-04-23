// Arithmetic op handlers: add, subtract, multiply, divide.
// Phase 1: Header only. Implementation in Phase 3.

#ifndef TT_PJRT_OPS_ARITHMETIC_H_
#define TT_PJRT_OPS_ARITHMETIC_H_

// Op handlers will be registered here in Phase 3.
// Each handler takes a HandlerContext (op, value_map, device) and returns
// the result ttnn tensor(s).
//
// Example (Phase 3):
//   ttnn::Tensor HandleAdd(const HandlerContext& ctx) {
//     return ttnn::add(ctx.GetInput(0), ctx.GetInput(1));
//   }

#endif  // TT_PJRT_OPS_ARITHMETIC_H_
