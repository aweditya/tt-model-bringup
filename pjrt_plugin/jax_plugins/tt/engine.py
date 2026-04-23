"""StableHLO interpreter engine for TT PJRT plugin.

Parses MLIR bytecode from JAX and executes StableHLO ops on numpy arrays.
This is the "Python engine" in our "thin C++ shell + Python engine" design.

Called from C++ via CPython API during PJRT_LoadedExecutable_Execute.
"""

import numpy as np
import re
import sys


def bytecode_to_text(bytecode: bytes) -> str:
    """Convert MLIR bytecode to text using jaxlib's MLIR bindings.

    JAX sends StableHLO programs as VHLO portable artifacts (not plain MLIR
    bytecode). We first try deserializing as a portable artifact, then fall
    back to plain MLIR bytecode parsing.
    """
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo as stablehlo_dialect

    # Check if this is a StableHLO portable artifact (starts with ML\xefR...StableHLO)
    if b'StableHLO' in bytecode[:30]:
        # VHLO portable artifact — deserialize to MLIR native bytecode first
        from jaxlib.mlir._mlir_libs._stablehlo import deserialize_portable_artifact_str
        native_bytecode = deserialize_portable_artifact_str(bytecode)
        # native_bytecode is MLIR native bytecode (bytes), parse it
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            stablehlo_dialect.register_dialect(ctx)
            module = ir.Module.parse(native_bytecode, ctx)
            return str(module)
    else:
        # Plain MLIR bytecode (from module_to_bytecode in tests)
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            stablehlo_dialect.register_dialect(ctx)
            module = ir.Module.parse(bytecode, ctx)
            return str(module)


def parse_stablehlo(text: str):
    """Parse StableHLO text IR into a list of op descriptors.

    Returns:
        args: list of (name, type_str) for function arguments
        ops: list of dicts with keys: name, op, operands, attrs, result_type
        returns: list of SSA value names to return
    """
    # Extract function body (between ^bb0 and closing brace)
    # Handle both "func.func" and "\"func.func\"" syntax
    lines = text.strip().split('\n')
    in_func = False
    func_args = []
    ops = []
    returns = []

    for line in lines:
        line = line.strip()

        # Detect function entry block
        if line.startswith('^bb0('):
            in_func = True
            # Parse arguments: ^bb0(%arg0: tensor<4xf32>, ...):
            arg_str = line[len('^bb0('):]
            arg_str = arg_str.rstrip('):')
            for arg in arg_str.split(','):
                arg = arg.strip()
                if ':' in arg:
                    name, type_str = arg.split(':', 1)
                    func_args.append((name.strip().lstrip('%'), type_str.strip()))
            continue

        if not in_func:
            continue

        # Detect return
        if 'func.return' in line or line.startswith('return '):
            # "func.return"(%1) : ...  or  return %1 : ...
            m = re.findall(r'%[a-zA-Z0-9_]+', line.split(':')[0])
            returns = [v.lstrip('%') for v in m]
            continue

        # Parse SSA assignment: %name = op operands attrs : type
        m = re.match(r'(%[a-zA-Z0-9_]+)\s*=\s*(.+)', line)
        if not m:
            continue

        result_name = m.group(1).lstrip('%')
        rest = m.group(2)

        op_desc = parse_op(result_name, rest)
        if op_desc:
            ops.append(op_desc)

    return func_args, ops, returns


def parse_op(result_name: str, text: str) -> dict:
    """Parse a single StableHLO op from text."""

    # stablehlo.constant dense<1.0> : tensor<f32>
    if text.startswith('stablehlo.constant'):
        return parse_constant(result_name, text)

    # stablehlo.add %a, %b : tensor<4xf32>
    if text.startswith('stablehlo.add '):
        return parse_binary_op(result_name, 'add', text)

    if text.startswith('stablehlo.subtract '):
        return parse_binary_op(result_name, 'subtract', text)

    if text.startswith('stablehlo.multiply '):
        return parse_binary_op(result_name, 'multiply', text)

    if text.startswith('stablehlo.divide '):
        return parse_binary_op(result_name, 'divide', text)

    if text.startswith('stablehlo.maximum '):
        return parse_binary_op(result_name, 'maximum', text)

    if text.startswith('stablehlo.minimum '):
        return parse_binary_op(result_name, 'minimum', text)

    # stablehlo.negate %a : tensor<4xf32>
    if text.startswith('stablehlo.negate '):
        return parse_unary_op(result_name, 'negate', text)

    if text.startswith('stablehlo.abs '):
        return parse_unary_op(result_name, 'abs', text)

    if text.startswith('stablehlo.exponential '):
        return parse_unary_op(result_name, 'exp', text)

    if text.startswith('stablehlo.log '):
        return parse_unary_op(result_name, 'log', text)

    if text.startswith('stablehlo.tanh '):
        return parse_unary_op(result_name, 'tanh', text)

    if text.startswith('stablehlo.rsqrt '):
        return parse_unary_op(result_name, 'rsqrt', text)

    if text.startswith('stablehlo.sqrt '):
        return parse_unary_op(result_name, 'sqrt', text)

    # stablehlo.broadcast_in_dim %a, dims = [...] : (...) -> result_type
    if text.startswith('stablehlo.broadcast_in_dim '):
        return parse_broadcast(result_name, text)

    # stablehlo.reshape %a : (...) -> result_type
    if text.startswith('stablehlo.reshape '):
        return parse_reshape(result_name, text)

    # stablehlo.transpose %a, permutation = [...] : (...) -> result_type
    if text.startswith('stablehlo.transpose '):
        return parse_transpose(result_name, text)

    # stablehlo.convert %a : (...) -> result_type
    if text.startswith('stablehlo.convert '):
        return parse_unary_op(result_name, 'convert', text)

    # stablehlo.dot_general
    if text.startswith('stablehlo.dot_general '):
        return parse_dot_general(result_name, text)

    # Unknown op — store for error reporting
    return {
        'name': result_name,
        'op': 'unknown',
        'text': text[:100],
    }


def parse_constant(name: str, text: str) -> dict:
    """Parse: stablehlo.constant dense<1.0> : tensor<f32>"""
    # Extract the value between dense< and >
    m = re.search(r'dense<(.+?)>\s*:', text)
    if not m:
        return {'name': name, 'op': 'constant', 'value': 0.0, 'result_type': ''}

    val_str = m.group(1)
    result_type = extract_result_type(text)

    return {
        'name': name,
        'op': 'constant',
        'value_str': val_str,
        'result_type': result_type,
    }


def parse_binary_op(name: str, op: str, text: str) -> dict:
    """Parse: stablehlo.add %a, %b : tensor<4xf32>"""
    # Extract the part after "stablehlo.xxx " and before ":"
    after_op = re.sub(r'^stablehlo\.\w+\s+', '', text)
    before_type = after_op.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', before_type)
    return {
        'name': name,
        'op': op,
        'operands': operands,
        'result_type': extract_result_type(text),
    }


def parse_unary_op(name: str, op: str, text: str) -> dict:
    """Parse: stablehlo.negate %a : tensor<4xf32>"""
    after_op = re.sub(r'^stablehlo\.\w+\s+', '', text)
    before_type = after_op.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', before_type)
    return {
        'name': name,
        'op': op,
        'operands': operands,
        'result_type': extract_result_type(text),
    }


def parse_broadcast(name: str, text: str) -> dict:
    """Parse: stablehlo.broadcast_in_dim %a, dims = [0] : (...) -> tensor<4xf32>"""
    # Extract operand: first %name after the op name
    after_op = re.sub(r'^stablehlo\.broadcast_in_dim\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(',')[0])
    dims_m = re.search(r'dims\s*=\s*\[([^\]]*)\]', text)
    dims = []
    if dims_m and dims_m.group(1).strip():
        dims = [int(d.strip()) for d in dims_m.group(1).split(',')]

    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'broadcast_in_dim',
        'operands': operands,
        'dims': dims,
        'result_type': result_type,
    }


def parse_reshape(name: str, text: str) -> dict:
    """Parse: stablehlo.reshape %a : (...) -> tensor<2x3xf32>"""
    after_op = re.sub(r'^stablehlo\.reshape\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(':')[0])
    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'reshape',
        'operands': operands,
        'result_type': result_type,
    }


def parse_transpose(name: str, text: str) -> dict:
    """Parse: stablehlo.transpose %a, permutation = [1, 0] : (...) -> tensor<3x2xf32>"""
    after_op = re.sub(r'^stablehlo\.transpose\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(',')[0])
    perm_m = re.search(r'permutation\s*=\s*\[([^\]]+)\]', text)
    perm = []
    if perm_m:
        perm = [int(d.strip()) for d in perm_m.group(1).split(',')]
    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'transpose',
        'operands': operands,
        'permutation': perm,
        'result_type': result_type,
    }


def parse_dot_general(name: str, text: str) -> dict:
    """Parse stablehlo.dot_general with dimension numbers.

    Format: stablehlo.dot_general %a, %b,
            contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
            : (tensor<...>, tensor<...>) -> tensor<...>
    """
    after_op = re.sub(r'^stablehlo\.dot_general\s+', '', text)

    # Extract the two operands (before contracting_dims)
    operand_part = after_op.split('contracting_dims')[0] if 'contracting_dims' in after_op else after_op.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', operand_part)

    # Parse contracting_dims = [1] x [0]
    lhs_contract = []
    rhs_contract = []
    lhs_batch = []
    rhs_batch = []

    cd = re.search(r'contracting_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]', text)
    if cd:
        if cd.group(1).strip():
            lhs_contract = [int(d.strip()) for d in cd.group(1).split(',')]
        if cd.group(2).strip():
            rhs_contract = [int(d.strip()) for d in cd.group(2).split(',')]

    bd = re.search(r'batching_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]', text)
    if bd:
        if bd.group(1).strip():
            lhs_batch = [int(d.strip()) for d in bd.group(1).split(',')]
        if bd.group(2).strip():
            rhs_batch = [int(d.strip()) for d in bd.group(2).split(',')]

    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'dot_general',
        'operands': operands,
        'lhs_contracting': lhs_contract,
        'rhs_contracting': rhs_contract,
        'lhs_batching': lhs_batch,
        'rhs_batching': rhs_batch,
        'result_type': result_type,
    }


def extract_result_type(text: str) -> str:
    """Extract the result type from an op's text.
    For ops with ->, returns the type after ->. Otherwise the type after last :.
    """
    if '->' in text:
        return text.split('->')[-1].strip().rstrip(')')
    parts = text.rsplit(':', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return ''


def parse_tensor_type(type_str: str):
    """Parse 'tensor<4x3xf32>' into (shape, dtype).
    Returns (shape_tuple, numpy_dtype).
    """
    type_str = type_str.strip()
    m = re.match(r'tensor<(.+)>', type_str)
    if not m:
        return (), np.float32

    inner = m.group(1)
    # Split on 'x' but the last part is the dtype
    parts = inner.split('x')
    if len(parts) == 1:
        # Scalar tensor like tensor<f32>
        return (), dtype_from_str(parts[0])

    dtype = dtype_from_str(parts[-1])
    shape = tuple(int(p) for p in parts[:-1])
    return shape, dtype


def dtype_from_str(s: str):
    """Convert MLIR dtype string to numpy dtype."""
    s = s.strip()
    mapping = {
        'f32': np.float32,
        'f64': np.float64,
        'f16': np.float16,
        'bf16': np.float32,  # numpy doesn't have bf16, use f32
        'i1': np.bool_,
        'i8': np.int8,
        'i16': np.int16,
        'i32': np.int32,
        'i64': np.int64,
        'ui8': np.uint8,
        'ui16': np.uint16,
        'ui32': np.uint32,
        'ui64': np.uint64,
    }
    return mapping.get(s, np.float32)


def execute_stablehlo(bytecode: bytes, inputs: list) -> list:
    """Execute a StableHLO program on numpy array inputs.

    Args:
        bytecode: MLIR bytecode from PJRT_Client_Compile
        inputs: list of numpy arrays (one per function argument)

    Returns:
        list of numpy arrays (one per return value)
    """
    # Parse bytecode → text → op list
    text = bytecode_to_text(bytecode)
    func_args, ops, returns = parse_stablehlo(text)

    # Build value map: SSA name → numpy array
    values = {}
    for i, (arg_name, type_str) in enumerate(func_args):
        values[arg_name] = inputs[i]

    # Execute ops
    for op in ops:
        result = execute_op(op, values)
        values[op['name']] = result

    # Gather return values
    return [values[r] for r in returns]


def execute_op(op: dict, values: dict) -> np.ndarray:
    """Execute a single StableHLO op."""
    op_type = op['op']

    if op_type == 'constant':
        return execute_constant(op)

    if op_type == 'add':
        a, b = get_operands(op, values, 2)
        return np.add(a, b)

    if op_type == 'subtract':
        a, b = get_operands(op, values, 2)
        return np.subtract(a, b)

    if op_type == 'multiply':
        a, b = get_operands(op, values, 2)
        return np.multiply(a, b)

    if op_type == 'divide':
        a, b = get_operands(op, values, 2)
        return np.divide(a, b)

    if op_type == 'maximum':
        a, b = get_operands(op, values, 2)
        return np.maximum(a, b)

    if op_type == 'minimum':
        a, b = get_operands(op, values, 2)
        return np.minimum(a, b)

    if op_type == 'negate':
        a = get_operands(op, values, 1)[0]
        return np.negative(a)

    if op_type == 'abs':
        a = get_operands(op, values, 1)[0]
        return np.abs(a)

    if op_type == 'exp':
        a = get_operands(op, values, 1)[0]
        return np.exp(a)

    if op_type == 'log':
        a = get_operands(op, values, 1)[0]
        return np.log(a)

    if op_type == 'tanh':
        a = get_operands(op, values, 1)[0]
        return np.tanh(a)

    if op_type == 'rsqrt':
        a = get_operands(op, values, 1)[0]
        return 1.0 / np.sqrt(a)

    if op_type == 'sqrt':
        a = get_operands(op, values, 1)[0]
        return np.sqrt(a)

    if op_type == 'convert':
        a = get_operands(op, values, 1)[0]
        _, target_dtype = parse_tensor_type(op['result_type'])
        return a.astype(target_dtype)

    if op_type == 'broadcast_in_dim':
        return execute_broadcast(op, values)

    if op_type == 'reshape':
        a = get_operands(op, values, 1)[0]
        target_shape, _ = parse_tensor_type(op['result_type'])
        if not target_shape:
            return a.reshape(())
        return a.reshape(target_shape)

    if op_type == 'transpose':
        a = get_operands(op, values, 1)[0]
        return np.transpose(a, op['permutation'])

    if op_type == 'dot_general':
        return execute_dot_general(op, values)

    raise ValueError(f"Unsupported op: {op_type} (text: {op.get('text', '')})")


def get_operands(op: dict, values: dict, n: int) -> list:
    """Get n operand arrays from the value map."""
    operands = op.get('operands', [])
    result = []
    for name in operands[:n]:
        if name not in values:
            raise ValueError(f"Unknown operand %{name} in op {op['op']}")
        result.append(values[name])
    return result


def execute_constant(op: dict) -> np.ndarray:
    """Execute a stablehlo.constant op."""
    val_str = op.get('value_str', '0')
    result_type = op.get('result_type', 'tensor<f32>')
    shape, dtype = parse_tensor_type(result_type)

    # Parse the value string
    # Formats: "1.0", "1.000000e+00", "[[1, 2], [3, 4]]", "true", "false"
    val_str = val_str.strip()

    if val_str in ('true', 'false'):
        val = val_str == 'true'
        arr = np.full(shape if shape else (), val, dtype=dtype)
        return arr

    if val_str.startswith('['):
        # Array constant — parse nested list
        # Replace stablehlo syntax with Python syntax
        arr = np.array(eval(val_str), dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr

    # Scalar constant
    val = float(val_str)
    arr = np.full(shape if shape else (), val, dtype=dtype)
    return arr


def execute_broadcast(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.broadcast_in_dim."""
    a = get_operands(op, values, 1)[0]
    dims = op.get('dims', [])
    result_type = op['result_type']
    target_shape, target_dtype = parse_tensor_type(result_type)

    if not target_shape:
        return a

    # broadcast_in_dim: dims maps source dimensions to target dimensions
    # First, reshape source to have size 1 in all non-mapped dimensions
    result = a
    if a.ndim == 0:
        # Scalar broadcast
        result = np.broadcast_to(a, target_shape).copy()
    else:
        # Insert size-1 dimensions for broadcasting
        new_shape = [1] * len(target_shape)
        for src_dim, tgt_dim in enumerate(dims):
            new_shape[tgt_dim] = a.shape[src_dim]
        result = a.reshape(new_shape)
        result = np.broadcast_to(result, target_shape).copy()

    return result.astype(target_dtype)


def execute_dot_general(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.dot_general (generalized matmul)."""
    a, b = get_operands(op, values, 2)
    lhs_contract = op['lhs_contracting']
    rhs_contract = op['rhs_contracting']
    lhs_batch = op.get('lhs_batching', [])
    rhs_batch = op.get('rhs_batching', [])

    # Simple case: standard matmul (no batch dims, single contraction)
    if not lhs_batch and len(lhs_contract) == 1 and len(rhs_contract) == 1:
        return np.tensordot(a, b, axes=(lhs_contract, rhs_contract))

    # General case: use np.einsum
    # Build einsum string
    # This handles batched matmul and multi-contraction
    return np.einsum_dot_general(a, b, lhs_contract, rhs_contract,
                                 lhs_batch, rhs_batch)


def np_einsum_dot_general(a, b, lhs_contract, rhs_contract, lhs_batch, rhs_batch):
    """Implement dot_general using numpy operations."""
    # For now, handle the common cases
    if not lhs_batch:
        return np.tensordot(a, b, axes=(lhs_contract, rhs_contract))

    # Batched case — move batch dims to front, do batched matmul
    # This is a simplification; full implementation would use einsum
    raise ValueError("Batched dot_general not yet implemented")
