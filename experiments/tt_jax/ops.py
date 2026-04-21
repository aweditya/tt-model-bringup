"""
Op registry: maps Jaxpr primitive names to TT-NN implementations.

Each op handler takes (interpreter, invars, params, eqn) and returns
a TT-NN tensor (or tuple of tensors for multi-output ops).

The registry is a plain dict — easy to extend, easy to test.
"""

import numpy as np
import ttnn
from . import tensors


def _get_shapes(eqn):
    """Extract input/output shapes from a Jaxpr equation."""
    in_shapes = []
    for v in eqn.invars:
        if tensors.is_literal(v):
            in_shapes.append(())
        elif hasattr(v, 'aval'):
            in_shapes.append(v.aval.shape)
        else:
            in_shapes.append(())
    out_shape = eqn.outvars[0].aval.shape if eqn.outvars else ()
    return in_shapes, out_shape


def _binary_with_broadcast(interp, invars, eqn, tt_fn, scalar_fn=None):
    """Handle a binary op with scalar detection and broadcast."""
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        if scalar_fn:
            return scalar_fn(b, tensors.literal_val(invars[0]))
        return tt_fn(interp.to_device(np.array(tensors.literal_val(invars[0]))), b)
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        if scalar_fn:
            return scalar_fn(a, tensors.literal_val(invars[1]))
        return tt_fn(a, interp.to_device(np.array(tensors.literal_val(invars[1]))))

    a = interp.eval_var(invars[0])
    b = interp.eval_var(invars[1])

    in_shapes, out_shape = _get_shapes(eqn)
    a, b = tensors.broadcast_to_match(a, b, in_shapes[0], in_shapes[1],
                                       out_shape, interp.device)
    return tt_fn(a, b)


# ============================================================
# Elementwise ops
# ============================================================

def op_add(interp, invars, params, eqn):
    return _binary_with_broadcast(interp, invars, eqn, ttnn.add,
                                   scalar_fn=lambda t, s: ttnn.add(t, s))

def op_sub(interp, invars, params, eqn):
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        return ttnn.add(ttnn.neg(b), tensors.literal_val(invars[0]))
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        return ttnn.add(a, -tensors.literal_val(invars[1]))

    a = interp.eval_var(invars[0])
    b = interp.eval_var(invars[1])
    in_shapes, out_shape = _get_shapes(eqn)
    a, b = tensors.broadcast_to_match(a, b, in_shapes[0], in_shapes[1],
                                       out_shape, interp.device)
    return ttnn.sub(a, b)

def op_mul(interp, invars, params, eqn):
    return _binary_with_broadcast(interp, invars, eqn, ttnn.mul,
                                   scalar_fn=lambda t, s: ttnn.multiply(t, s))

def op_div(interp, invars, params, eqn):
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        return ttnn.multiply(a, 1.0 / tensors.literal_val(invars[1]))
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        return ttnn.multiply(ttnn.reciprocal(b), tensors.literal_val(invars[0]))

    a = interp.eval_var(invars[0])
    b = interp.eval_var(invars[1])
    in_shapes, out_shape = _get_shapes(eqn)
    a, b = tensors.broadcast_to_match(a, b, in_shapes[0], in_shapes[1],
                                       out_shape, interp.device)
    return ttnn.mul(a, ttnn.reciprocal(b))

def op_neg(interp, invars, params, eqn):
    return ttnn.neg(interp.eval_var(invars[0]))

def op_exp(interp, invars, params, eqn):
    return ttnn.exp(interp.eval_var(invars[0]))

def op_log(interp, invars, params, eqn):
    return ttnn.log(interp.eval_var(invars[0]))

def op_sqrt(interp, invars, params, eqn):
    return ttnn.sqrt(interp.eval_var(invars[0]))

def op_rsqrt(interp, invars, params, eqn):
    return ttnn.rsqrt(interp.eval_var(invars[0]))

def op_reciprocal(interp, invars, params, eqn):
    return ttnn.reciprocal(interp.eval_var(invars[0]))

def op_max(interp, invars, params, eqn):
    """Element-wise max. Detects max(x, 0) → relu."""
    if tensors.is_literal(invars[1]) and tensors.literal_val(invars[1]) == 0.0:
        return ttnn.relu(interp.eval_var(invars[0]))
    if tensors.is_literal(invars[0]) and tensors.literal_val(invars[0]) == 0.0:
        return ttnn.relu(interp.eval_var(invars[1]))
    return _binary_with_broadcast(interp, invars, eqn, ttnn.maximum)

def op_integer_pow(interp, invars, params, eqn):
    """x^n for small integer n. Uses repeated multiplication."""
    a = interp.eval_var(invars[0])
    n = params['y']
    if n == 0:
        return interp.to_device(np.ones((1, 1), dtype=np.float32))
    if n == 1:
        return a
    if n == 2:
        return ttnn.mul(a, a)
    if n == -1:
        return ttnn.reciprocal(a)
    if n < 0:
        a = ttnn.reciprocal(a)
        n = -n
    result = a
    for _ in range(n - 1):
        result = ttnn.mul(result, a)
    return result


# ============================================================
# Matmul
# ============================================================

def op_dot_general(interp, invars, params, eqn):
    return ttnn.matmul(interp.eval_var(invars[0]), interp.eval_var(invars[1]))


# ============================================================
# Reductions
# ============================================================

def op_reduce_max(interp, invars, params, eqn):
    a = interp.eval_var(invars[0])
    for ax in sorted(params['axes'], reverse=True):
        a = ttnn.max(a, dim=ax, keepdim=True)
    return a

def op_reduce_sum(interp, invars, params, eqn):
    a = interp.eval_var(invars[0])
    for ax in sorted(params['axes'], reverse=True):
        a = ttnn.sum(a, dim=ax, keepdim=True)
    return a


# ============================================================
# Shape manipulation
# ============================================================

def op_broadcast_in_dim(interp, invars, params, eqn):
    """Handle broadcast_in_dim via CPU round-trip when shapes differ."""
    a = interp.eval_var(invars[0])
    in_shape = eqn.invars[0].aval.shape if not tensors.is_literal(eqn.invars[0]) else ()
    out_shape = eqn.outvars[0].aval.shape

    if in_shape == out_shape:
        return a

    broadcast_dims = params.get('broadcast_dimensions', tuple(range(len(in_shape))))

    inter_shape = [1] * len(out_shape)
    for i, bd in enumerate(broadcast_dims):
        if i < len(in_shape):
            inter_shape[bd] = in_shape[i]

    t = ttnn.to_torch(a).float()
    while t.dim() < 2:
        t = t.unsqueeze(0)
    while t.dim() > len(inter_shape):
        t = t.squeeze(0)

    t_np = t.numpy()
    slices = tuple(slice(0, s) for s in inter_shape)
    try:
        t_np = t_np[slices].copy()
    except IndexError:
        n = 1
        for d in in_shape:
            n *= d
        t_np = t.numpy().reshape(-1)[:n].reshape(inter_shape)

    return interp.to_device(np.broadcast_to(t_np, out_shape).copy())

def op_reshape(interp, invars, params, eqn):
    a = interp.eval_var(invars[0])
    new_shape = params.get('new_sizes', params.get('dimensions', None))
    return ttnn.reshape(a, new_shape) if new_shape else a

def op_transpose(interp, invars, params, eqn):
    return ttnn.permute(interp.eval_var(invars[0]), params['permutation'])

def op_squeeze(interp, invars, params, eqn):
    return interp.eval_var(invars[0])


# ============================================================
# Pass-through / identity ops
# ============================================================

def op_convert_element_type(interp, invars, params, eqn):
    return interp.eval_var(invars[0])

def op_stop_gradient(interp, invars, params, eqn):
    return interp.eval_var(invars[0])


# ============================================================
# Registry
# ============================================================

REGISTRY = {
    # Elementwise
    'add': op_add,
    'sub': op_sub,
    'mul': op_mul,
    'div': op_div,
    'neg': op_neg,
    'exp': op_exp,
    'log': op_log,
    'sqrt': op_sqrt,
    'rsqrt': op_rsqrt,
    'reciprocal': op_reciprocal,
    'max': op_max,
    'integer_pow': op_integer_pow,
    # Matmul
    'dot_general': op_dot_general,
    # Reductions
    'reduce_max': op_reduce_max,
    'reduce_sum': op_reduce_sum,
    # Shape
    'broadcast_in_dim': op_broadcast_in_dim,
    'reshape': op_reshape,
    'transpose': op_transpose,
    'squeeze': op_squeeze,
    # Pass-through
    'convert_element_type': op_convert_element_type,
    'stop_gradient': op_stop_gradient,
}
