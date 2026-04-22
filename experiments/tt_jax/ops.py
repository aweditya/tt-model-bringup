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


def _binary_with_broadcast(interp, invars, eqn, tt_fn, np_fn, scalar_fn=None):
    """Handle a binary op with scalar detection, broadcast, and CPU fallback."""
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        if scalar_fn:
            return scalar_fn(b, tensors.literal_val(invars[0]))
        return tt_fn(interp.eval_var(invars[0]), b)
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        if scalar_fn:
            return scalar_fn(a, tensors.literal_val(invars[1]))
        return tt_fn(a, interp.eval_var(invars[1]))

    a = interp.eval_var(invars[0])
    b = interp.eval_var(invars[1])

    in_shapes, out_shape = _get_shapes(eqn)
    a, b = tensors.broadcast_to_match(a, b, in_shapes[0], in_shapes[1],
                                       out_shape, interp.device)
    try:
        return tt_fn(a, b)
    except RuntimeError:
        # CPU fallback for cases TT-NN can't handle (e.g., 4D broadcast)
        a_np = tensors.from_device(a, out_shape)
        b_np = tensors.from_device(b, out_shape)
        return interp.to_device(np_fn(a_np, b_np))


# ============================================================
# Elementwise ops
# ============================================================

def op_add(interp, invars, params, eqn):
    return _binary_with_broadcast(interp, invars, eqn, ttnn.add, np.add,
                                   scalar_fn=lambda t, s: ttnn.add(t, s))

def op_sub(interp, invars, params, eqn):
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        return ttnn.add(ttnn.neg(b), tensors.literal_val(invars[0]))
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        return ttnn.add(a, -tensors.literal_val(invars[1]))
    return _binary_with_broadcast(interp, invars, eqn, ttnn.sub, np.subtract)

def op_mul(interp, invars, params, eqn):
    return _binary_with_broadcast(interp, invars, eqn, ttnn.mul, np.multiply,
                                   scalar_fn=lambda t, s: ttnn.multiply(t, s))

def op_div(interp, invars, params, eqn):
    if tensors.is_literal(invars[1]):
        a = interp.eval_var(invars[0])
        return ttnn.multiply(a, 1.0 / tensors.literal_val(invars[1]))
    if tensors.is_literal(invars[0]):
        b = interp.eval_var(invars[1])
        return ttnn.multiply(ttnn.reciprocal(b), tensors.literal_val(invars[0]))
    return _binary_with_broadcast(interp, invars, eqn,
                                   lambda a, b: ttnn.mul(a, ttnn.reciprocal(b)),
                                   lambda a, b: a / b)

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
    return _binary_with_broadcast(interp, invars, eqn, ttnn.maximum, np.maximum)

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

def op_tanh(interp, invars, params, eqn):
    """Hyperbolic tangent — used in GELU activation."""
    return ttnn.tanh(interp.eval_var(invars[0]))


# ============================================================
# Comparison and selection ops (needed for causal masking)
# ============================================================

def op_iota(interp, invars, params, eqn):
    """Generate an index array. Used for causal mask construction.

    iota(dimension=0, shape=(T,T)) produces [[0,0,...],[1,1,...],...]
    iota(dimension=1, shape=(T,T)) produces [[0,1,...],[0,1,...],...]
    """
    shape = params['shape']
    dim = params['dimension']
    # Build on CPU, send to device
    arr = np.zeros(shape, dtype=np.float32)
    idx = [slice(None)] * len(shape)
    for i in range(shape[dim]):
        idx[dim] = i
        arr[tuple(idx)] = float(i)
    return interp.to_device(arr)

def op_ge(interp, invars, params, eqn):
    """Greater-than-or-equal comparison. Returns 1.0 where a >= b, else 0.0."""
    return _binary_with_broadcast(interp, invars, eqn, ttnn.ge,
                                   lambda a, b: (a >= b).astype(np.float32))

def op_select_n(interp, invars, params, eqn):
    """Select between values based on condition.

    select_n(cond, on_false, on_true): where cond is true, pick on_true.
    JAX convention: select_n(pred, false_val, true_val)
    """
    cond = interp.eval_var(invars[0])
    on_false = interp.eval_var(invars[1])
    on_true = interp.eval_var(invars[2])
    return ttnn.where(cond, on_true, on_false)


# ============================================================
# Split op (needed for Q/K/V separation)
# ============================================================

def op_split(interp, invars, params, eqn):
    """Split a tensor along an axis into multiple parts.

    Used for splitting QKV projections: (B, T, 3*C) -> 3x (B, T, C).
    Falls back to CPU slice + re-upload since ttnn lacks native split.
    """
    a = interp.eval_var(invars[0])
    sizes = params['sizes']
    axis = params['axis']

    in_shape = eqn.invars[0].aval.shape

    # Read back to CPU, split, re-upload
    a_np = tensors.from_device(a, in_shape)

    results = []
    offset = 0
    for size in sizes:
        slices = [slice(None)] * len(in_shape)
        slices[axis] = slice(offset, offset + size)
        chunk = a_np[tuple(slices)].copy()
        results.append(interp.to_device(chunk))
        offset += size

    return results


def op_slice(interp, invars, params, eqn):
    """Slice a tensor along one or more axes.

    JAX's slice primitive takes start_indices, limit_indices, strides.
    Used for per-head Q/K/V extraction in flat attention:
      x[:, :, h*d:(h+1)*d] -> slice with start=(0,0,h*d), limit=(B,T,(h+1)*d)
    Falls back to CPU since ttnn lacks general slicing.
    """
    a = interp.eval_var(invars[0])
    in_shape = eqn.invars[0].aval.shape
    start = params['start_indices']
    limit = params['limit_indices']
    strides = params.get('strides', None)

    a_np = tensors.from_device(a, in_shape)
    slices = []
    for i in range(len(start)):
        s = strides[i] if strides else 1
        slices.append(slice(start[i], limit[i], s))
    result = a_np[tuple(slices)].copy()
    return interp.to_device(result)


def op_dynamic_slice(interp, invars, params, eqn):
    """Dynamic slice: extract a window from a tensor at runtime indices.

    invars[0] = tensor, invars[1:] = start indices for each dim.
    params['slice_sizes'] = size of the window in each dim.
    """
    a = interp.eval_var(invars[0])
    in_shape = eqn.invars[0].aval.shape
    a_np = tensors.from_device(a, in_shape)

    sizes = params['slice_sizes']
    starts = []
    for iv in invars[1:]:
        if tensors.is_literal(iv):
            starts.append(int(tensors.literal_val(iv)))
        else:
            s = tensors.from_device(interp.eval_var(iv), ())
            starts.append(int(s))

    slices = [slice(s, s + sz) for s, sz in zip(starts, sizes)]
    result = a_np[tuple(slices)].copy()
    return interp.to_device(result)


def op_concatenate(interp, invars, params, eqn):
    """Concatenate multiple tensors along an axis.

    Used to reassemble per-head attention outputs:
      concat([head_0, head_1, ..., head_11], axis=-1)
    Falls back to CPU since ttnn.concat has shape constraints.
    """
    dim = params['dimension']
    arrays = []
    for i, iv in enumerate(invars):
        shape = eqn.invars[i].aval.shape
        t = interp.eval_var(iv)
        arrays.append(tensors.from_device(t, shape))
    result = np.concatenate(arrays, axis=dim)
    return interp.to_device(result)


# ============================================================
# Matmul
# ============================================================

def op_dot_general(interp, invars, params, eqn):
    """General dot product with dimension_numbers support.

    Simple matmuls go through ttnn.matmul on device. Complex contractions
    (mismatched batch dims, non-standard contraction axes) fall back to CPU.
    """
    a = interp.eval_var(invars[0])
    b = interp.eval_var(invars[1])

    dim_nums = params.get('dimension_numbers', None)
    a_shape = eqn.invars[0].aval.shape
    b_shape = eqn.invars[1].aval.shape

    is_simple = True
    if dim_nums is not None:
        (ca, cb), (ba, bb) = dim_nums
        is_simple = (
            len(ca) == 1 and len(cb) == 1 and
            ca[0] == len(a_shape) - 1 and
            cb[0] == len(b_shape) - 2 and
            tuple(ba) == tuple(bb) and
            list(ba) == list(range(len(ba)))
        )

    if is_simple:
        try:
            return ttnn.matmul(a, b)
        except RuntimeError:
            pass

    # CPU fallback for complex dot_general
    import jax.lax
    a_np = tensors.from_device(a, a_shape)
    b_np = tensors.from_device(b, b_shape)
    result_np = np.array(jax.lax.dot_general(
        jax.numpy.array(a_np), jax.numpy.array(b_np),
        dimension_numbers=dim_nums
    ))
    return interp.to_device(result_np)


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
    """Handle broadcast_in_dim, using on-device repeat when possible."""
    a = interp.eval_var(invars[0])
    in_shape = eqn.invars[0].aval.shape if not tensors.is_literal(eqn.invars[0]) else ()
    out_shape = eqn.outvars[0].aval.shape

    if in_shape == out_shape:
        return a

    broadcast_dims = params.get('broadcast_dimensions', tuple(range(len(in_shape))))

    # Compute intermediate shape (input dims placed at broadcast positions)
    inter_shape = [1] * len(out_shape)
    for i, bd in enumerate(broadcast_dims):
        if i < len(in_shape):
            inter_shape[bd] = in_shape[i]

    # Try on-device broadcast via ttnn.repeat
    try:
        return tensors._repeat_broadcast(a, tuple(inter_shape), out_shape)
    except Exception:
        pass

    # Fallback: CPU round-trip (needed for complex reshapes)
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
    'tanh': op_tanh,
    # Comparison / selection
    'iota': op_iota,
    'ge': op_ge,
    'select_n': op_select_n,
    # Split / slice / concat
    'split': op_split,
    'slice': op_slice,
    'dynamic_slice': op_dynamic_slice,
    'concatenate': op_concatenate,
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
