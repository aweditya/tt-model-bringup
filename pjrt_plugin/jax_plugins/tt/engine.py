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

    Handles modules with multiple functions. Returns the main (first)
    function's data plus a list of private (subsequent) functions.

    Returns:
        args: list of (name, type_str) for main function arguments
        ops: list of dicts for main function ops
        returns: list of SSA value names to return from main
        private_fns: list of (args, ops, returns) for private functions
    """
    lines = text.strip().split('\n')

    # Parse all functions in the module
    all_functions = []
    current_func = None
    pending = None  # For multi-line ops (scatter body regions, etc.)

    for line in lines:
        line = line.strip()

        # Multi-line op accumulation MUST be checked first — absorbs inner
        # ^bb0 lines from body regions (e.g. scatter) before they can be
        # mistaken for new function entry blocks.
        if pending is not None:
            pending['lines'].append(line)
            # Count body region nesting: ({ opens, }) closes
            if '({' in line:
                pending['depth'] += line.count('({')
            if '})' in line:
                pending['depth'] -= line.count('})')
            if pending['depth'] <= 0:
                # End of multi-line op — join and parse
                full_text = ' '.join(pending['lines'])
                op_desc = parse_op(pending['name'], full_text)
                if op_desc:
                    current_func['ops'].append(op_desc)
                pending = None
            continue

        # Detect function entry block (with arguments)
        if line.startswith('^bb0('):
            func_args = []
            arg_str = line[len('^bb0('):]
            arg_str = arg_str.rstrip('):')
            for arg in arg_str.split(','):
                arg = arg.strip()
                if ':' in arg:
                    name, type_str = arg.split(':', 1)
                    func_args.append((name.strip().lstrip('%'), type_str.strip()))
            current_func = {'args': func_args, 'ops': [], 'returns': []}
            continue

        # Detect no-argument function: "func.func"() ... ({
        if '"func.func"' in line and '({' in line and current_func is None:
            current_func = {'args': [], 'ops': [], 'returns': []}
            continue

        if current_func is None:
            continue

        # Detect return — ends the current function
        if 'func.return' in line or (line.startswith('return ') and '%' in line):
            m = re.findall(r'%[a-zA-Z0-9_]+', line.split(':')[0])
            current_func['returns'] = [v.lstrip('%') for v in m]
            all_functions.append(current_func)
            current_func = None
            continue

        # Parse SSA assignment: %name = op operands attrs : type
        m = re.match(r'(%[a-zA-Z0-9_]+)\s*=\s*(.+)', line)
        if not m:
            continue

        result_name = m.group(1).lstrip('%')
        rest = m.group(2)

        # Check for multi-line op with body region (scatter, etc.)
        if rest.startswith('"stablehlo.') and '({' in rest:
            depth = rest.count('({') - rest.count('})')
            if depth > 0:
                pending = {
                    'name': result_name, 'lines': [rest], 'depth': depth,
                }
                continue

        op_desc = parse_op(result_name, rest)
        if op_desc:
            current_func['ops'].append(op_desc)

    # First function is main, rest are private
    if not all_functions:
        return [], [], [], []

    main = all_functions[0]
    private_fns = [
        (f['args'], f['ops'], f['returns']) for f in all_functions[1:]
    ]

    return main['args'], main['ops'], main['returns'], private_fns


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

    # stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1]
    if text.startswith('stablehlo.reduce'):
        return parse_reduce(result_name, text)

    # stablehlo.slice %arg0 [0:1, 0:4, 0:8, 0:16] : (...) -> (...)
    if text.startswith('stablehlo.slice '):
        return parse_slice(result_name, text)

    # stablehlo.compare GT, %a, %b, FLOAT : (...) -> tensor<...xi1>
    if text.startswith('stablehlo.compare '):
        return parse_compare(result_name, text)

    # stablehlo.select %pred, %true, %false : tensor<...xi1>, tensor<...>
    if text.startswith('stablehlo.select '):
        return parse_select(result_name, text)

    # stablehlo.iota dim = 0 : tensor<4x4xi32>
    if text.startswith('stablehlo.iota '):
        return parse_iota(result_name, text)

    # stablehlo.concatenate %a, %b, dim = 1 : (...) -> (...)
    if text.startswith('stablehlo.concatenate '):
        return parse_concatenate(result_name, text)

    # stablehlo.and / stablehlo.or (boolean logic)
    if text.startswith('stablehlo.and '):
        return parse_binary_op(result_name, 'and', text)

    if text.startswith('stablehlo.or '):
        return parse_binary_op(result_name, 'or', text)

    # "stablehlo.scatter"(%operand, %indices, %updates) <{...}> ({...}) : (...) -> result
    if text.startswith('"stablehlo.scatter"'):
        return parse_scatter(result_name, text)

    # "stablehlo.gather"(%operand, %indices) <{...}> : (...) -> result
    if text.startswith('"stablehlo.gather"'):
        return parse_gather(result_name, text)

    # "func.call"(%arg) <...> : (...) -> result_type
    if text.startswith('"func.call"'):
        return parse_func_call(result_name, text)

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
    """Parse: stablehlo.transpose %a, dims = [1, 0] : (...)  or  permutation = [1, 0]"""
    after_op = re.sub(r'^stablehlo\.transpose\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(',')[0])
    # Try both "permutation = [...]" and "dims = [...]" (bytecode format uses dims)
    perm_m = re.search(r'(?:permutation|dims)\s*=\s*\[([^\]]+)\]', text)
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


def parse_reduce(name: str, text: str) -> dict:
    """Parse stablehlo.reduce with `applies` shorthand.

    Format: stablehlo.reduce(%x init: %cst) applies stablehlo.add
            across dimensions = [1] : (...) -> tensor<2xf32>
    """
    # Extract input operand: first %name after reduce(
    input_m = re.search(r'reduce\((%[a-zA-Z0-9_]+)', text)
    input_operand = input_m.group(1).lstrip('%') if input_m else None

    # Extract init value: after "init:"
    init_m = re.search(r'init:\s*(%[a-zA-Z0-9_]+)', text)
    init_operand = init_m.group(1).lstrip('%') if init_m else None

    # Extract reduction function: "applies stablehlo.XXX"
    applies_m = re.search(r'applies\s+stablehlo\.(\w+)', text)
    reduce_fn = applies_m.group(1) if applies_m else 'add'

    # Extract dimensions: "across dimensions = [1]" or "dimensions = [1]"
    dims_m = re.search(r'dimensions\s*=\s*\[([^\]]*)\]', text)
    dims = []
    if dims_m and dims_m.group(1).strip():
        dims = [int(d.strip()) for d in dims_m.group(1).split(',')]

    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'reduce',
        'operands': [input_operand] if input_operand else [],
        'init_operand': init_operand,
        'reduce_fn': reduce_fn,
        'dimensions': dims,
        'result_type': result_type,
    }


def parse_func_call(name: str, text: str) -> dict:
    """Parse "func.call"(%arg) <...> : (...) -> result_type

    In the bytecode→text format, func.call doesn't include the callee name
    directly. We track call index to map to the Nth private function.
    """
    # Extract operands
    operand_part = text.split('<')[0] if '<' in text else text.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', operand_part)

    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'func_call',
        'operands': operands,
        'result_type': result_type,
    }


def parse_slice(name: str, text: str) -> dict:
    """Parse: stablehlo.slice %arg0 [0:1, 0:4, 0:8, 0:16] : (...) -> (...)"""
    after_op = re.sub(r'^stablehlo\.slice\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split('[')[0])
    # Extract [start:end, start:end, ...] ranges
    bracket_m = re.search(r'\[([^\]]+)\]', text)
    starts, limits, strides = [], [], []
    if bracket_m:
        for part in bracket_m.group(1).split(','):
            part = part.strip()
            pieces = part.split(':')
            starts.append(int(pieces[0]))
            limits.append(int(pieces[1]))
            strides.append(int(pieces[2]) if len(pieces) > 2 else 1)
    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'slice', 'operands': operands,
        'starts': starts, 'limits': limits, 'strides': strides,
        'result_type': result_type,
    }


def parse_compare(name: str, text: str) -> dict:
    """Parse: stablehlo.compare GT, %a, %b, FLOAT : (...) -> tensor<...xi1>"""
    after_op = re.sub(r'^stablehlo\.compare\s+', '', text)
    # Direction is the first word: GT, LT, GE, LE, EQ, NE
    parts = after_op.split(',')
    direction = parts[0].strip()
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(':')[0])
    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'compare', 'operands': operands,
        'direction': direction, 'result_type': result_type,
    }


def parse_select(name: str, text: str) -> dict:
    """Parse: stablehlo.select %pred, %true, %false : tensor<...xi1>, tensor<...>"""
    after_op = re.sub(r'^stablehlo\.select\s+', '', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', after_op.split(':')[0])
    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'select', 'operands': operands,
        'result_type': result_type,
    }


def parse_iota(name: str, text: str) -> dict:
    """Parse: stablehlo.iota dim = 0 : tensor<4x4xi32>"""
    dim_m = re.search(r'dim\s*=\s*(\d+)', text)
    dim = int(dim_m.group(1)) if dim_m else 0
    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'iota', 'operands': [],
        'dim': dim, 'result_type': result_type,
    }


def parse_concatenate(name: str, text: str) -> dict:
    """Parse: stablehlo.concatenate %a, %b, dim = 1 : (...) -> (...)"""
    after_op = re.sub(r'^stablehlo\.concatenate\s+', '', text)
    # Operands are before "dim ="
    operand_part = after_op.split('dim')[0] if 'dim' in after_op else after_op.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', operand_part)
    dim_m = re.search(r'dim\s*=\s*(\d+)', text)
    dim = int(dim_m.group(1)) if dim_m else 0
    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'concatenate', 'operands': operands,
        'dim': dim, 'result_type': result_type,
    }


def parse_scatter(name: str, text: str) -> dict:
    """Parse "stablehlo.scatter"(%operand, %indices, %updates) ...

    For KV cache updates, the body region is always "return new value" (overwrite).
    We extract: operand, indices, updates, and scatter_dims_to_operand_dims.
    """
    # Extract operands from the first part: "stablehlo.scatter"(%a, %b, %c)
    paren_m = re.search(r'"stablehlo\.scatter"\(([^)]+)\)', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', paren_m.group(1)) if paren_m else []

    # Extract scatter_dims_to_operand_dims from the attribute dict
    sdto = re.search(r'scatter_dims_to_operand_dims\s*=\s*\[([^\]]*)\]', text)
    scatter_dims = []
    if sdto and sdto.group(1).strip():
        scatter_dims = [int(d.strip()) for d in sdto.group(1).split(',')]

    # Extract update_window_dims
    uwd = re.search(r'update_window_dims\s*=\s*\[([^\]]*)\]', text)
    update_window_dims = []
    if uwd and uwd.group(1).strip():
        update_window_dims = [int(d.strip()) for d in uwd.group(1).split(',')]

    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'scatter', 'operands': operands,
        'scatter_dims_to_operand_dims': scatter_dims,
        'update_window_dims': update_window_dims,
        'result_type': result_type,
    }


def parse_gather(name: str, text: str) -> dict:
    """Parse "stablehlo.gather"(%operand, %indices) <{dimension_numbers = ..., slice_sizes = ...}>

    For embedding lookup: operand[indices] with specified dimension mapping.
    """
    # Extract operands
    paren_m = re.search(r'"stablehlo\.gather"\(([^)]+)\)', text)
    operands = re.findall(r'%([a-zA-Z0-9_]+)', paren_m.group(1)) if paren_m else []

    # Extract slice_sizes = array<i64: 1, 64>
    ss_m = re.search(r'slice_sizes\s*=\s*array<i64:\s*([^>]+)>', text)
    slice_sizes = []
    if ss_m:
        slice_sizes = [int(d.strip()) for d in ss_m.group(1).split(',')]

    # Extract dimension_numbers
    od_m = re.search(r'offset_dims\s*=\s*\[([^\]]*)\]', text)
    offset_dims = [int(d) for d in od_m.group(1).split(',')] if od_m and od_m.group(1).strip() else []

    cd_m = re.search(r'collapsed_slice_dims\s*=\s*\[([^\]]*)\]', text)
    collapsed_dims = [int(d) for d in cd_m.group(1).split(',')] if cd_m and cd_m.group(1).strip() else []

    sim_m = re.search(r'start_index_map\s*=\s*\[([^\]]*)\]', text)
    start_index_map = [int(d) for d in sim_m.group(1).split(',')] if sim_m and sim_m.group(1).strip() else []

    ivd_m = re.search(r'index_vector_dim\s*=\s*(\d+)', text)
    index_vector_dim = int(ivd_m.group(1)) if ivd_m else 1

    result_type = extract_result_type(text)
    return {
        'name': name, 'op': 'gather', 'operands': operands,
        'slice_sizes': slice_sizes,
        'offset_dims': offset_dims,
        'collapsed_slice_dims': collapsed_dims,
        'start_index_map': start_index_map,
        'index_vector_dim': index_vector_dim,
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
    global _private_functions, _call_counter

    # Parse bytecode → text → op list
    text = bytecode_to_text(bytecode)
    func_args, ops, returns, private_fns = parse_stablehlo(text)

    # Set up private functions for func.call dispatch
    _private_functions = private_fns
    _call_counter = 0

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

    if op_type == 'reduce':
        return execute_reduce(op, values)

    if op_type == 'slice':
        return execute_slice(op, values)

    if op_type == 'compare':
        return execute_compare(op, values)

    if op_type == 'select':
        a, b, c = get_operands(op, values, 3)
        return np.where(a, b, c)

    if op_type == 'iota':
        return execute_iota(op)

    if op_type == 'concatenate':
        return execute_concatenate(op, values)

    if op_type == 'and':
        a, b = get_operands(op, values, 2)
        return np.logical_and(a, b)

    if op_type == 'or':
        a, b = get_operands(op, values, 2)
        return np.logical_or(a, b)

    if op_type == 'scatter':
        return execute_scatter(op, values)

    if op_type == 'gather':
        return execute_gather(op, values)

    if op_type == 'func_call':
        return execute_func_call(op, values)

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

    # Scalar constant — may be decimal, scientific notation, or hex float
    if val_str.startswith('0x') or val_str.startswith('0X'):
        # IEEE 754 hex representation (e.g., 0xFF800000 = -inf for f32)
        import struct
        hex_val = int(val_str, 16)
        if dtype == np.float32 or len(val_str) <= 10:  # 0x + 8 hex digits
            val = struct.unpack('f', struct.pack('I', hex_val))[0]
        else:
            val = struct.unpack('d', struct.pack('Q', hex_val))[0]
    else:
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
    """Execute stablehlo.dot_general (generalized matmul).

    Uses np.einsum for the general case (batched + multi-contraction).
    """
    a, b = get_operands(op, values, 2)
    lhs_contract = op['lhs_contracting']
    rhs_contract = op['rhs_contracting']
    lhs_batch = op.get('lhs_batching', [])
    rhs_batch = op.get('rhs_batching', [])

    # Simple case: no batch dims, single contraction → tensordot
    if not lhs_batch and len(lhs_contract) == 1 and len(rhs_contract) == 1:
        return np.tensordot(a, b, axes=(lhs_contract, rhs_contract))

    # General case: build einsum string
    # Assign letters: a-z for dimensions
    letters = 'abcdefghijklmnopqrstuvwxyz'
    idx = [0]
    def next_letter():
        c = letters[idx[0]]
        idx[0] += 1
        return c

    lhs_labels = [''] * a.ndim
    rhs_labels = [''] * b.ndim

    # Batch dims share labels
    for ld, rd in zip(lhs_batch, rhs_batch):
        c = next_letter()
        lhs_labels[ld] = c
        rhs_labels[rd] = c

    # Contracting dims share labels
    for ld, rd in zip(lhs_contract, rhs_contract):
        c = next_letter()
        lhs_labels[ld] = c
        rhs_labels[rd] = c

    # Free dims get unique labels
    for i in range(a.ndim):
        if not lhs_labels[i]:
            lhs_labels[i] = next_letter()
    for i in range(b.ndim):
        if not rhs_labels[i]:
            rhs_labels[i] = next_letter()

    # Output: batch dims + lhs free dims + rhs free dims
    lhs_free = [i for i in range(a.ndim) if i not in lhs_batch and i not in lhs_contract]
    rhs_free = [i for i in range(b.ndim) if i not in rhs_batch and i not in rhs_contract]

    out_labels = (
        [lhs_labels[d] for d in lhs_batch] +
        [lhs_labels[d] for d in lhs_free] +
        [rhs_labels[d] for d in rhs_free]
    )

    subscripts = f"{''.join(lhs_labels)},{''.join(rhs_labels)}->{''.join(out_labels)}"
    return np.einsum(subscripts, a, b)


def execute_reduce(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.reduce.

    Supported reduction functions: add (sum), maximum, minimum.
    Reduces along specified dimensions.
    """
    a = get_operands(op, values, 1)[0]
    reduce_fn = op['reduce_fn']
    dims = op['dimensions']

    # Map reduction function name to numpy operation
    if reduce_fn == 'add':
        return np.sum(a, axis=tuple(dims))
    elif reduce_fn == 'maximum':
        return np.max(a, axis=tuple(dims))
    elif reduce_fn == 'minimum':
        return np.min(a, axis=tuple(dims))
    elif reduce_fn == 'multiply':
        return np.prod(a, axis=tuple(dims))
    else:
        raise ValueError(f"Unsupported reduce function: stablehlo.{reduce_fn}")


def execute_slice(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.slice with static start/limit/strides."""
    a = get_operands(op, values, 1)[0]
    slices = tuple(
        slice(s, l, st)
        for s, l, st in zip(op['starts'], op['limits'], op['strides'])
    )
    return a[slices]


def execute_compare(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.compare with direction (GT, LT, GE, LE, EQ, NE)."""
    a, b = get_operands(op, values, 2)
    direction = op['direction']
    if direction == 'GT':
        return np.greater(a, b)
    elif direction == 'LT':
        return np.less(a, b)
    elif direction == 'GE':
        return np.greater_equal(a, b)
    elif direction == 'LE':
        return np.less_equal(a, b)
    elif direction == 'EQ':
        return np.equal(a, b)
    elif direction == 'NE':
        return np.not_equal(a, b)
    else:
        raise ValueError(f"Unsupported compare direction: {direction}")


def execute_iota(op: dict) -> np.ndarray:
    """Execute stablehlo.iota — generate indices along a dimension."""
    result_type = op['result_type']
    shape, dtype = parse_tensor_type(result_type)
    dim = op['dim']
    # Create an array where values along `dim` are 0, 1, 2, ...
    result = np.zeros(shape, dtype=dtype)
    idx = [np.newaxis] * len(shape)
    idx[dim] = slice(None)
    result = np.arange(shape[dim], dtype=dtype)[tuple(idx)] * np.ones(shape, dtype=dtype)
    return result.astype(dtype)


def execute_concatenate(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.concatenate along a dimension."""
    arrays = [values[name] for name in op['operands']]
    return np.concatenate(arrays, axis=op['dim'])


def execute_scatter(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.scatter (overwrite mode for KV cache updates).

    Handles the common case: scatter along one operand dimension,
    body region = "return new value" (overwrite, not accumulate).
    """
    operand = values[op['operands'][0]]
    indices = values[op['operands'][1]]
    updates = values[op['operands'][2]]
    scatter_dims = op['scatter_dims_to_operand_dims']

    result = operand.copy()
    # Simple case: scatter along a single dimension with scalar indices
    if len(scatter_dims) == 1:
        dim = scatter_dims[0]
        # indices is a 1D array of indices (e.g., [5] for pos=5)
        idx = int(indices.flat[0])
        # Build a slice tuple to index into the result
        slices = [slice(None)] * result.ndim
        slices[dim] = idx
        # Updates might have the scattered dim as size 1; squeeze if needed
        upd = updates
        if upd.shape[dim] == 1:
            # The update covers exactly 1 position along the scatter dim
            slices[dim] = slice(idx, idx + 1)
            result[tuple(slices)] = upd
        else:
            result[tuple(slices)] = upd
    else:
        raise ValueError(f"Multi-dim scatter not supported: {scatter_dims}")

    return result


def execute_gather(op: dict, values: dict) -> np.ndarray:
    """Execute stablehlo.gather (embedding lookup pattern).

    Handles the common case: gather rows from a 2D table using 1D indices.
    dimension_numbers = {offset_dims=[1], collapsed_slice_dims=[0],
                         start_index_map=[0], index_vector_dim=1}
    """
    operand = values[op['operands'][0]]
    indices = values[op['operands'][1]]
    slice_sizes = op['slice_sizes']
    start_index_map = op['start_index_map']
    collapsed_dims = op['collapsed_slice_dims']
    offset_dims = op['offset_dims']
    index_vector_dim = op['index_vector_dim']
    result_shape, result_dtype = parse_tensor_type(op['result_type'])

    # Common embedding lookup: operand[indices[:, 0], :]
    # indices shape: [N, 1] or [N], start_index_map=[0], collapsed=[0]
    if (len(start_index_map) == 1 and start_index_map[0] == 0
            and len(collapsed_dims) == 1 and collapsed_dims[0] == 0):
        # Extract the actual index values
        if indices.ndim > 1 and index_vector_dim == indices.ndim - 1:
            idx = indices[..., 0]
        else:
            idx = indices
        # Gather rows
        result = operand[idx]
        # Slice to the specified sizes for non-collapsed dims
        if len(slice_sizes) > 1:
            trailing_slices = tuple(
                slice(0, s) for i, s in enumerate(slice_sizes) if i not in collapsed_dims
            )
            result = result[(..., *trailing_slices)]
        return result.astype(result_dtype)

    raise ValueError(f"Unsupported gather pattern: start_index_map={start_index_map}, "
                     f"collapsed={collapsed_dims}")


# ============================================================
# Multi-function support for func.call
# ============================================================

# Module-level storage for private functions parsed from the module.
# Set by execute_stablehlo before executing the main function.
_private_functions = []
_call_counter = 0


def execute_func_call(op: dict, values: dict) -> np.ndarray:
    """Execute a func.call by running the corresponding private function."""
    global _call_counter

    if _call_counter >= len(_private_functions):
        raise ValueError(
            f"func.call #{_call_counter} but only {len(_private_functions)} "
            f"private functions found in module"
        )

    func_args, func_ops, func_returns = _private_functions[_call_counter]
    _call_counter += 1

    # Build local value map: function args ← call operands
    local_values = dict(values)  # inherit outer scope for init values etc.
    call_operands = [values[name] for name in op['operands']]
    for i, (arg_name, _type_str) in enumerate(func_args):
        local_values[arg_name] = call_operands[i]

    # Execute function body
    for func_op in func_ops:
        result = execute_op(func_op, local_values)
        local_values[func_op['name']] = result

    # Return the function's return value(s)
    if len(func_returns) == 1:
        return local_values[func_returns[0]]
    return [local_values[r] for r in func_returns]
