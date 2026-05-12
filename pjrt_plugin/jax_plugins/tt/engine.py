"""StableHLO interpreter engine for TT PJRT plugin.

Parses MLIR bytecode from JAX and executes StableHLO ops.
Dual-mode: numpy (CPU) by default, ttnn (Blackhole device) when
TT_PJRT_USE_DEVICE=1 is set.

This is the "Python engine" in our "thin C++ shell + Python engine" design.
Called from C++ via CPython API during PJRT_LoadedExecutable_Execute.
"""

import atexit
import numpy as np
import os
import re
import struct
import sys

# ============================================================
# Device mode: ttnn on Blackhole when TT_PJRT_USE_DEVICE=1
# ============================================================

_USE_DEVICE = os.environ.get('TT_PJRT_USE_DEVICE', '0') == '1'
_device = None
_logical_shapes = {}  # Per-execution SSA name → StableHLO logical shape

if _USE_DEVICE:
    try:
        import ttnn
        import torch
        # Redirect ALL ttnn paths off /tmp and into the project cache.
        # Project rule: nothing under /tmp; everything under ~/tt-xla/.cache/.
        # We override the in-memory ttnn.CONFIG since non-interactive ssh
        # doesn't source ~/.bashrc, so env vars set there don't apply.
        _cache_root = os.environ.get(
            'TT_PJRT_CACHE_ROOT',
            os.path.expanduser('~/tt-xla/.cache'))
        for sub in ('ttnn', 'ttnn/models', 'ttnn-tmp'):
            os.makedirs(os.path.join(_cache_root, sub), exist_ok=True)
        ttnn.CONFIG.cache_path = os.path.join(_cache_root, 'ttnn')
        ttnn.CONFIG.model_cache_path = os.path.join(_cache_root, 'ttnn', 'models')
        ttnn.CONFIG.tmp_dir = os.path.join(_cache_root, 'ttnn-tmp')
    except ImportError:
        _USE_DEVICE = False

def _get_device():
    """Lazily open Blackhole device 0 on first use."""
    global _device
    if _device is None:
        _device = ttnn.open_device(device_id=0)
        # Release any captured traces BEFORE closing the device.
        # ttnn keeps trace handles alive on device; closing without
        # release can corrupt subsequent runs in the same process.
        atexit.register(_release_trace_cache)
        atexit.register(lambda: ttnn.close_device(_device))
    return _device


def _release_trace_cache():
    """Free every captured trace and clear the cache. Safe to call twice."""
    global _trace_cache, _parse_cache
    for key, entry in list(_trace_cache.items()):
        tid = entry.get('trace_id') if isinstance(entry, dict) else None
        if tid is not None:
            try:
                ttnn.release_trace(_device, tid)
            except Exception:
                pass
    _trace_cache.clear()
    _parse_cache.clear()


def _to_device(arr):
    """Convert numpy array to ttnn tensor on device (bf16, TILE_LAYOUT)."""
    if isinstance(arr, (int, float, np.integer, np.floating)):
        arr = np.array([[float(arr)]], dtype=np.float32)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)
    t = torch.from_numpy(arr.copy()).float()
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=_get_device(),
                           layout=ttnn.TILE_LAYOUT)


def _from_device(tensor, shape):
    """Convert ttnn tensor back to numpy array with correct shape."""
    t = ttnn.to_torch(tensor).float()
    if len(shape) == 0:
        return t.numpy().flatten()[0]
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        pass
    try:
        return t.squeeze().numpy().reshape(shape)
    except (RuntimeError, ValueError):
        pass
    # Handle tile padding: flatten and take first N elements
    t_np = t.numpy()
    target_size = 1
    for d in shape:
        target_size *= d
    if t_np.size >= target_size:
        return t_np.flatten()[:target_size].reshape(shape).copy()
    raise ValueError(f"Cannot reshape tensor of size {t_np.size} to {shape}")


def bytecode_to_text(bytecode: bytes) -> str:
    """Convert MLIR bytecode to text using jaxlib's MLIR bindings.

    JAX sends StableHLO programs as VHLO portable artifacts (not plain MLIR
    bytecode). We first try deserializing as a portable artifact, then fall
    back to plain MLIR bytecode parsing.

    Func.call ops carry their callee name in the assembly form when emitted
    via the operation tree walk in `_module_to_text_with_callees`. The
    default `str(module)` printer drops them under `allow_unregistered_dialects`,
    so for multi-function modules we use the walker.
    """
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo as stablehlo_dialect

    if b'StableHLO' in bytecode[:30]:
        from jaxlib.mlir._mlir_libs._stablehlo import deserialize_portable_artifact_str
        native_bytecode = deserialize_portable_artifact_str(bytecode)
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            stablehlo_dialect.register_dialect(ctx)
            module = ir.Module.parse(native_bytecode, ctx)
            return _module_to_text_with_callees(module)
    else:
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            stablehlo_dialect.register_dialect(ctx)
            module = ir.Module.parse(bytecode, ctx)
            return _module_to_text_with_callees(module)


def _module_to_text_with_callees(module) -> str:
    """Render an MLIR module to text, annotating each func.call with the
    callee name in a form our parser can recognize.

    The standard `str(module)` printer drops callee names when the func
    dialect is unregistered (we use allow_unregistered_dialects=True). We
    walk the operation tree to extract every (op, callee, sym_name) and
    splice the callee into the text via a deterministic pattern match.
    """
    text = str(module)
    # Try to harvest (callee, sym_name) pairs by walking the module
    # operation tree. If the walker can read 'callee' from attributes, we
    # rewrite the call sites; otherwise the original text is returned and
    # the engine falls back to single-private-function dispatch.
    try:
        calls = []     # list of callee symbol names, in walk order
        sym_names = [] # list of function symbol names, in walk order

        def _walk(op):
            if op.name == "func.call":
                callee = None
                try:
                    for attr in op.attributes:
                        if attr.name == "callee":
                            s = str(attr.attr)
                            m = re.match(r'@(\S+)', s)
                            if m:
                                callee = m.group(1)
                            else:
                                callee = s
                            break
                except Exception:
                    pass
                calls.append(callee)
            elif op.name == "func.func":
                sym = None
                try:
                    for attr in op.attributes:
                        if attr.name == "sym_name":
                            s = str(attr.attr)
                            m = re.match(r'"([^"]+)"', s)
                            sym = m.group(1) if m else s.strip('"')
                            break
                except Exception:
                    pass
                sym_names.append(sym)
            for r in op.regions:
                for blk in r:
                    for child in blk:
                        _walk(child)

        _walk(module.operation)

        # Splice callee names into the text by walking call-site lines in
        # order. Lines containing `"func.call"(...)` and lacking
        # `callee = ` get rewritten with the corresponding callee.
        if calls:
            out_lines = []
            i = 0  # index into `calls` list
            for line in text.splitlines():
                if '"func.call"' in line and 'callee' not in line and i < len(calls):
                    callee = calls[i]
                    if callee:
                        # Insert `<{callee = @CalleeName}>` after the operands
                        # paren. Pattern: %X = "func.call"(%Y) ...
                        # Replace the first ' <' or ' :' after the paren with
                        # ` <{callee = @<name>}>` followed by original tail.
                        line = re.sub(
                            r'("func\.call"\([^)]*\))\s*(<|:)',
                            r'\1 <{callee = @' + callee + r'}> \2',
                            line,
                            count=1,
                        )
                    i += 1
                out_lines.append(line)

            # Splice sym_name into func.func definitions too. Same walk order.
            j = 0
            new_lines = []
            for line in out_lines:
                if '"func.func"' in line and j < len(sym_names) and sym_names[j]:
                    sym = sym_names[j]
                    # Make sure the existing func.func line doesn't already include sym_name=
                    if 'sym_name' not in line:
                        line = re.sub(
                            r'("func\.func"\(\))',
                            r'\1 <{sym_name = "' + sym + r'"}>',
                            line,
                            count=1,
                        )
                    j += 1
                elif '"func.func"' in line:
                    j += 1
                new_lines.append(line)
            text = '\n'.join(new_lines)
    except Exception:
        pass
    return text


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
    _pending_sym_name = None  # sym_name harvested before ^bb0 line

    for line in lines:
        line = line.strip()

        # Multi-line op accumulation MUST be checked first — absorbs inner
        # ^bb0 lines from body regions (e.g. scatter, reduce with reducer)
        # before they can be mistaken for new function entry blocks.
        if pending is not None:
            pending['lines'].append(line)
            # Track brace nesting for both ({ }) and plain { }
            for ch in line:
                if ch == '{':
                    pending['depth'] += 1
                    pending['seen_open'] = True
                elif ch == '}':
                    pending['depth'] -= 1
            if pending['seen_open'] and pending['depth'] <= 0:
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
            current_func = {'args': func_args, 'ops': [], 'returns': [],
                            'sym_name': _pending_sym_name}
            _pending_sym_name = None
            continue

        # Detect no-argument function: "func.func"() ... ({
        if '"func.func"' in line and '({' in line and current_func is None:
            # Capture sym_name if it's in the func.func line
            sym_match = re.search(r'sym_name\s*=\s*"([^"]+)"', line)
            _pending_sym_name = sym_match.group(1) if sym_match else None
            current_func = {'args': [], 'ops': [], 'returns': [],
                            'sym_name': _pending_sym_name}
            _pending_sym_name = None
            continue

        # Also catch func.func that appears alone (will set sym_name for the
        # next ^bb0 block).
        if '"func.func"' in line and current_func is None:
            sym_match = re.search(r'sym_name\s*=\s*"([^"]+)"', line)
            _pending_sym_name = sym_match.group(1) if sym_match else None
            continue

        if current_func is None:
            continue

        # Detect return — ends the current function
        if 'func.return' in line or (line.startswith('return ') and '%' in line):
            # Match %name and %name#N (multi-output element access)
            m = re.findall(r'%([a-zA-Z0-9_]+(?:#\d+)?)', line.split(':')[0])
            current_func['returns'] = m
            all_functions.append(current_func)
            current_func = None
            continue

        # Parse SSA assignment: %name = op  (also handles %name:N for multi-output)
        m = re.match(r'(%[a-zA-Z0-9_]+(?::\d+)?)\s*=\s*(.+)', line)
        if not m:
            continue

        result_name = m.group(1).lstrip('%')
        rest = m.group(2)

        # Check for multi-line op with body region (scatter, reduce with reducer)
        # Scatter uses "stablehlo.scatter"(...) ({ ... })
        # Multi-output reduce uses stablehlo.reduce(...) ... reducer(...) { ... }
        has_body = '({' in rest
        is_multi_reduce = rest.startswith('stablehlo.reduce') and ':' in result_name
        if has_body or is_multi_reduce:
            depth = sum(1 for c in rest if c == '{') - sum(1 for c in rest if c == '}')
            if depth > 0 or is_multi_reduce:
                pending = {
                    'name': result_name, 'lines': [rest], 'depth': depth,
                    # Multi-output reduce body starts on next line — track if
                    # we've seen any { yet so we don't terminate at depth=0
                    # before the body opens.
                    'seen_open': depth > 0,
                }
                continue

        op_desc = parse_op(result_name, rest)
        if op_desc:
            current_func['ops'].append(op_desc)

    # First function is main, rest are private.
    # Private function tuples are (args, ops, returns, sym_name) when
    # the sym_name was harvested; sym_name may be None.
    if not all_functions:
        return [], [], [], []

    main = all_functions[0]
    private_fns = [
        (f['args'], f['ops'], f['returns'], f.get('sym_name'))
        for f in all_functions[1:]
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
    """Parse stablehlo.reduce — both simple (applies) and complex (reducer body).

    Simple format: stablehlo.reduce(%x init: %cst) applies stablehlo.add
                   across dimensions = [1] : (...) -> tensor<2xf32>

    Argmax format: stablehlo.reduce(%x init: %cst), (%idx init: %c)
                   across dimensions = [1] : (...) -> (tensor<f32>, tensor<i32>)
                   reducer(...) { ...compare/select body... }
    """
    # Check for multi-output argmax pattern (has 'reducer' body, not 'applies')
    if 'reducer' in text and ':' in name:
        return parse_reduce_argmax(name, text)

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


def parse_reduce_argmax(name: str, text: str) -> dict:
    """Parse multi-output reduce used by jnp.argmax.

    JAX compiles argmax as a dual reduce: track both max value and its index.
    Format: stablehlo.reduce(%values init: %neg_inf), (%indices init: %zero)
            across dimensions = [N] : (...) -> (tensor<...xf32>, tensor<...xi32>)
            reducer(...) { compare+select body }

    We recognize this pattern and map it to np.argmax.
    """
    # Extract input: first (operand init: value) pair
    input_m = re.search(r'reduce\((%[a-zA-Z0-9_]+)', text)
    input_operand = input_m.group(1).lstrip('%') if input_m else None

    # Extract dimensions
    dims_m = re.search(r'dimensions\s*=\s*\[([^\]]*)\]', text)
    dims = []
    if dims_m and dims_m.group(1).strip():
        dims = [int(d.strip()) for d in dims_m.group(1).split(',')]

    # Extract result types from -> (type1, type2)
    arrow_m = re.search(r'->\s*\(([^)]+)\)', text)
    result_types = []
    if arrow_m:
        result_types = [t.strip() for t in arrow_m.group(1).split(',')]

    # Strip the :N from name (e.g., "1:2" -> "1")
    base_name = name.split(':')[0]

    return {
        'name': base_name,
        'op': 'reduce_argmax',
        'operands': [input_operand] if input_operand else [],
        'dimensions': dims,
        'result_types': result_types,
        'num_outputs': int(name.split(':')[1]) if ':' in name else 2,
    }


def parse_func_call(name: str, text: str) -> dict:
    """Parse "func.call"(%arg) <{callee = @sym_name}> : (...) -> result_type

    In the bytecode→text format, func.call carries the callee inside `<...>`:
        "func.call"(%67) <{callee = @SomeFn}> : (tensor<...>) -> tensor<...>
    OR the printer may drop the callee (only emitting `<loc(...)>`). We
    extract it when present; otherwise downstream dispatch falls back to
    'use the single private function' / positional rules.
    """
    # Extract operands
    operand_part = text.split('<')[0] if '<' in text else text.split(':')[0]
    operands = re.findall(r'%([a-zA-Z0-9_]+)', operand_part)

    # Extract callee from <{callee = @Name}> if present
    callee = None
    m = re.search(r'callee\s*=\s*@([A-Za-z_][A-Za-z0-9_$.]*)', text)
    if m:
        callee = m.group(1)

    result_type = extract_result_type(text)
    return {
        'name': name,
        'op': 'func_call',
        'operands': operands,
        'callee': callee,
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

    For ops with ->, returns the type after ->. Otherwise the type(s) after
    the last `:`. Some ops (e.g. stablehlo.select) put multiple comma-
    separated types after the colon — the result type is the LAST one
    (the value type, not the predicate type).
    """
    if '->' in text:
        return text.split('->')[-1].strip().rstrip(')')
    parts = text.rsplit(':', 1)
    if len(parts) > 1:
        tail = parts[1].strip()
        # Multi-type case: take the last "tensor<...>" occurrence.
        matches = re.findall(r'tensor<[^>]+>', tail)
        if len(matches) > 1:
            return matches[-1]
        return tail
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


def count_outputs(bytecode: bytes) -> int:
    """Count the number of outputs from a StableHLO program.

    Called from C++ during PJRT_Client_Compile to set num_outputs correctly.
    """
    text = bytecode_to_text(bytecode)
    _, _, returns, _ = parse_stablehlo(text)
    return len(returns)


# ============================================================
# Trace cache (Phase 5 Step 6)
# ============================================================
#
# `execute_stablehlo` is called once per (compiled program, input batch).
# In eager mode we re-parse the bytecode and re-dispatch every op every
# call — Step 5 showed parse alone is 1.4-1.7ms (50-70% of wall time on
# small programs). Trace cache attacks this:
#
#   First call:  parse + warm-up execute + ttnn.begin/end_trace_capture
#   Replay:      copy_host_to_device_tensor(inputs) → execute_trace
#
# Cache key: hash(bytecode). Python guarantees `bytes` hashes are stable
# within a process. Across processes JAX rebuilds bytecode so the cache
# is per-process — fine for v0.
#
# Constraints on traceable programs:
#   - All ops must run on-device with NO host roundtrip during the trace.
#   - Data-independent host ops (constant, iota) are evaluated during
#     warm-up and the resulting device tensor is reused — safe.
#   - Any other host roundtrip op (broadcast_in_dim, slice, gather, ...)
#     means we cannot trace this program; we fall back to eager + parse cache.

_TRACE_CACHE_MAX = 64
_trace_cache = {}            # bytecode_hash -> dict (see _capture_trace)
_parse_cache = {}            # bytecode_hash -> (func_args, ops, returns, private_fns)
_NO_TRACE = os.environ.get('TT_PJRT_NO_TRACE', '0') == '1'

# Ops we can safely materialize once during warmup (their value doesn't
# depend on subsequent input data).
_DATA_INDEPENDENT_OPS = {'constant', 'iota'}

# Ops whose device implementation currently does a host roundtrip. If any
# op in this set is data-dependent in the program, we cannot capture a
# trace (the cached result would be stale on replay).
#
# Step 7a (2026-05-11): `broadcast_in_dim` removed from this set.
# `_execute_broadcast_device` has an on-device `ttnn.repeat` path that
# handles the five patterns JAX actually emits (see Step 7 plan). The
# CPU fallback still exists (raises caught upstream), so any pattern
# that doesn't work on-device just drops the trace and falls back to
# parse-cached eager — same behavior as before. Net: softmax / LN /
# RMSNorm become traceable.
_HOST_TRANSFER_DEVICE_OPS = {
    'slice', 'gather', 'scatter',
    'and', 'or', 'reduce_argmax', 'compare',
}


def _is_traceable(ops):
    """Decide if a parsed op list can run end-to-end inside a ttnn trace.

    The simple rule: every host-transfer device op must be data-
    independent (i.e., its inputs are all upstream-data-independent).
    For v0 we approximate this strictly: NO host-transfer op may appear
    unless it's `concatenate` of constants (rare). This is conservative
    but correct.
    """
    data_indep = set()  # SSA names whose value is data-independent
    for op in ops:
        if op['op'] in _DATA_INDEPENDENT_OPS:
            data_indep.add(op['name'])
            continue
        if op['op'] in _HOST_TRANSFER_DEVICE_OPS:
            # Allow only if every operand is data-independent
            for src in op.get('operands', []):
                if src not in data_indep:
                    return False
            data_indep.add(op['name'])
            continue
        # Pure on-device op — its result may or may not be data-
        # independent depending on operands; we don't need to track.
    return True


def _evict_lru():
    """Cap the trace cache size by dropping the oldest entry."""
    if len(_trace_cache) <= _TRACE_CACHE_MAX:
        return
    # Python dicts preserve insertion order — drop the first key.
    first_key = next(iter(_trace_cache))
    entry = _trace_cache.pop(first_key)
    try:
        if entry.get('trace_id') is not None:
            ttnn.release_trace(_get_device(), entry['trace_id'])
    except Exception:
        pass


def _capture_trace(key, bytecode, inputs, parsed):
    """Run the program once eagerly, then capture an executable trace.

    Returns the eager result (so the caller has a correct first-call
    answer) and populates _trace_cache[key].
    """
    func_args, ops, returns, private_fns = parsed

    # Eager warm-up — this is identical to the old execute_stablehlo path.
    global _private_functions, _call_counter, _logical_shapes
    _private_functions = private_fns
    _call_counter = 0
    _logical_shapes = {}

    values = {}
    input_placeholders = []   # ttnn tensors we'll reuse across calls
    for i, (arg_name, type_str) in enumerate(func_args):
        _logical_shapes[arg_name], _ = parse_tensor_type(type_str)
        t = _to_device(inputs[i])
        input_placeholders.append(t)
        values[arg_name] = t

    for op in ops:
        result = execute_op(op, values)
        if isinstance(result, dict):
            values.update(result)
            for k in result:
                rts = op.get('result_types', [])
                idx_str = k.split('#')[1] if '#' in k else '0'
                idx = int(idx_str)
                if idx < len(rts):
                    _logical_shapes[k], _ = parse_tensor_type(rts[idx])
        else:
            values[op['name']] = result
            rt = op.get('result_type', '')
            if rt:
                try:
                    _logical_shapes[op['name']], _ = parse_tensor_type(rt)
                except (ValueError, TypeError):
                    pass

    # Snapshot the warmup result for eager return AND for traced replay
    # (the device tensors of data-independent ops are pinned here).
    warmup_values = dict(values)

    eager_results = []
    for r in returns:
        val = values[r]
        if not isinstance(val, np.ndarray):
            shape = _logical_shapes.get(r) or _infer_result_shape(r, ops, func_args, returns)
            eager_results.append(_from_device(val, shape))
        else:
            eager_results.append(val)

    # Build the trace. During capture we MUST NOT do host transfers, so
    # we skip any data-independent op (its device tensor is already in
    # `values`) and re-execute only the pure-device ops.
    try:
        device = _get_device()
        ttnn.synchronize_device(device)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)

        # Reset state for trace re-execution
        _private_functions = private_fns
        _call_counter = 0
        _logical_shapes = {}
        trace_values = {}
        for i, (arg_name, type_str) in enumerate(func_args):
            _logical_shapes[arg_name], _ = parse_tensor_type(type_str)
            trace_values[arg_name] = input_placeholders[i]

        for op in ops:
            if op['op'] in _DATA_INDEPENDENT_OPS:
                # Pin warm-up value (a ttnn tensor); don't re-execute.
                trace_values[op['name']] = warmup_values[op['name']]
                rt = op.get('result_type', '')
                if rt:
                    try:
                        _logical_shapes[op['name']], _ = parse_tensor_type(rt)
                    except (ValueError, TypeError):
                        pass
                continue
            if op['op'] in _HOST_TRANSFER_DEVICE_OPS:
                # Same — these were already classified as data-indep by
                # _is_traceable. Pin warm-up value.
                trace_values[op['name']] = warmup_values[op['name']]
                rt = op.get('result_type', '')
                if rt:
                    try:
                        _logical_shapes[op['name']], _ = parse_tensor_type(rt)
                    except (ValueError, TypeError):
                        pass
                continue

            result = execute_op(op, trace_values)
            if isinstance(result, dict):
                trace_values.update(result)
                for k in result:
                    rts = op.get('result_types', [])
                    idx_str = k.split('#')[1] if '#' in k else '0'
                    idx = int(idx_str)
                    if idx < len(rts):
                        _logical_shapes[k], _ = parse_tensor_type(rts[idx])
            else:
                trace_values[op['name']] = result
                rt = op.get('result_type', '')
                if rt:
                    try:
                        _logical_shapes[op['name']], _ = parse_tensor_type(rt)
                    except (ValueError, TypeError):
                        pass

        output_tensors = []
        output_shapes = []
        for r in returns:
            val = trace_values[r]
            shape = _logical_shapes.get(r) or _infer_result_shape(r, ops, func_args, returns)
            output_tensors.append(val)
            output_shapes.append(shape)

        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    except Exception as e:
        # Capture failed — drop the partial trace, keep eager-only path.
        try:
            ttnn.end_trace_capture(_get_device(), trace_id, cq_id=0)
            ttnn.release_trace(_get_device(), trace_id)
        except Exception:
            pass
        _trace_cache[key] = {'failed': True, 'error': str(e)}
        return eager_results

    _trace_cache[key] = {
        'failed': False,
        'trace_id': trace_id,
        'input_placeholders': input_placeholders,
        'output_tensors': output_tensors,
        'output_shapes': output_shapes,
    }
    _evict_lru()
    return eager_results


def _replay_trace(key, inputs):
    """Execute a previously-captured trace with new inputs."""
    entry = _trace_cache[key]
    device = _get_device()

    # Copy host inputs into the placeholder device tensors.
    for i, arr in enumerate(inputs):
        placeholder = entry['input_placeholders'][i]
        # Build a torch tensor with the SAME padded shape as the placeholder
        # so copy_host_to_device_tensor doesn't shape-mismatch.
        if isinstance(arr, (int, float, np.integer, np.floating)):
            arr = np.array([[float(arr)]], dtype=np.float32)
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float32)
        t = torch.from_numpy(arr.copy()).float()
        while t.dim() < 2:
            t = t.unsqueeze(0)
        # Pad to placeholder's logical shape, accounting for tile padding
        ph_shape = tuple(placeholder.shape)
        if tuple(t.shape) != ph_shape:
            # Right-pad with zeros to match placeholder shape
            pads = []
            for src, dst in zip(t.shape[::-1], ph_shape[::-1]):
                pads.extend([0, max(0, dst - src)])
            if any(p > 0 for p in pads):
                t = torch.nn.functional.pad(t, pads)
        new_t = ttnn.from_torch(t, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(new_t, placeholder)

    ttnn.execute_trace(device, entry['trace_id'], cq_id=0, blocking=True)

    results = []
    for tensor, shape in zip(entry['output_tensors'], entry['output_shapes']):
        results.append(_from_device(tensor, shape))
    return results


def execute_stablehlo(bytecode: bytes, inputs: list) -> list:
    """Execute a StableHLO program on numpy array inputs.

    Fast path: if the bytecode has been compiled to a ttnn trace before,
    replay the trace with new inputs (skipping parse and per-op dispatch).
    Slow path: parse + interpret eagerly, then attempt to capture a trace
    for next time.

    Args:
        bytecode: MLIR bytecode from PJRT_Client_Compile
        inputs: list of numpy arrays (one per function argument)

    Returns:
        list of numpy arrays (one per return value)
    """
    global _private_functions, _call_counter, _logical_shapes

    # --- Trace fast path ---
    if _USE_DEVICE and not _NO_TRACE:
        key = hash(bytecode)
        entry = _trace_cache.get(key)
        if entry is not None and not entry.get('failed'):
            return _replay_trace(key, inputs)
        if entry is None:
            # First time we see this bytecode — try to capture a trace.
            # Parse once and stash for potential reuse on failure.
            parsed = _parse_cache.get(key)
            if parsed is None:
                text = bytecode_to_text(bytecode)
                parsed = parse_stablehlo(text)
                _parse_cache[key] = parsed
            _, ops, _, _ = parsed
            if _is_traceable(ops):
                return _capture_trace(key, bytecode, inputs, parsed)
            # Not traceable — mark and fall through to eager path.
            _trace_cache[key] = {'failed': True, 'error': 'not traceable'}

    # --- Eager fallback (also numpy mode) ---
    # Parse (with cache hit if we already parsed for trace attempt)
    if _USE_DEVICE and not _NO_TRACE:
        key = hash(bytecode)
        parsed = _parse_cache.get(key)
        if parsed is None:
            text = bytecode_to_text(bytecode)
            parsed = parse_stablehlo(text)
            _parse_cache[key] = parsed
    else:
        text = bytecode_to_text(bytecode)
        parsed = parse_stablehlo(text)
    func_args, ops, returns, private_fns = parsed

    # Set up private functions for func.call dispatch
    _private_functions = private_fns
    _call_counter = 0

    # Track LOGICAL shapes (from StableHLO IR) for each SSA value.
    # ttnn pads tensors to 2D-min and tile-aligned, so the device tensor's
    # .shape doesn't match the StableHLO type. Ops like broadcast_in_dim
    # need the logical input shape to be correct.
    _logical_shapes = {}

    # Build value map: SSA name → value (numpy array or ttnn tensor)
    values = {}
    for i, (arg_name, type_str) in enumerate(func_args):
        _logical_shapes[arg_name], _ = parse_tensor_type(type_str)
        if _USE_DEVICE:
            values[arg_name] = _to_device(inputs[i])
        else:
            values[arg_name] = inputs[i]

    # Execute ops, populating logical shapes as we go.
    for op in ops:
        result = execute_op(op, values)
        if isinstance(result, dict):
            # Multi-output op (e.g., reduce_argmax): store as name#0, name#1, ...
            values.update(result)
            for k in result:
                # Multi-output result types are in op['result_types'] list
                rts = op.get('result_types', [])
                idx_str = k.split('#')[1] if '#' in k else '0'
                idx = int(idx_str)
                if idx < len(rts):
                    _logical_shapes[k], _ = parse_tensor_type(rts[idx])
        else:
            values[op['name']] = result
            rt = op.get('result_type', '')
            if rt:
                try:
                    _logical_shapes[op['name']], _ = parse_tensor_type(rt)
                except (ValueError, TypeError):
                    # Multi-output result_type like "(tensor<a>, tensor<b>)" —
                    # the multi-output branch above handles result_types list.
                    pass

    # Gather return values, converting back to numpy if on device
    results = []
    for r in returns:
        val = values[r]
        if _USE_DEVICE and not isinstance(val, np.ndarray):
            shape = _logical_shapes.get(r) or _infer_result_shape(r, ops, func_args, returns)
            val = _from_device(val, shape)
        results.append(val)
    return results


def _infer_result_shape(name, ops, func_args, returns):
    """Infer the numpy shape for a return value from its op's result_type."""
    # Check if it's a function argument
    for arg_name, type_str in func_args:
        if arg_name == name:
            shape, _ = parse_tensor_type(type_str)
            return shape

    # Check ops for this name (handles multi-output name#N too)
    base_name = name.split('#')[0] if '#' in name else name
    for op in ops:
        if op['name'] == name or op['name'] == base_name:
            rt = op.get('result_type', '')
            if rt:
                shape, _ = parse_tensor_type(rt)
                return shape
            # Multi-output: check result_types list
            if '#' in name:
                idx = int(name.split('#')[1])
                rts = op.get('result_types', [])
                if idx < len(rts):
                    shape, _ = parse_tensor_type(rts[idx])
                    return shape
    return ()


def execute_op(op: dict, values: dict):
    """Execute a single StableHLO op.

    Dispatches to ttnn (device) or numpy (CPU) depending on _USE_DEVICE.
    """
    if _USE_DEVICE:
        return _execute_op_device(op, values)
    return _execute_op_numpy(op, values)


def _execute_op_numpy(op: dict, values: dict) -> np.ndarray:
    """Execute a single StableHLO op on numpy (CPU path)."""
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

    if op_type == 'reduce_argmax':
        return execute_reduce_argmax(op, values)

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


# ============================================================
# Device execution path (ttnn on Blackhole)
# ============================================================

def _device_to_numpy(tensor, op):
    """Helper: convert ttnn tensor to numpy for CPU-roundtrip ops."""
    rt = op.get('result_type', '')
    if rt:
        shape, _ = parse_tensor_type(rt)
    else:
        shape = ()
    return _from_device(tensor, shape)


def _operand_to_numpy(val, shape=None):
    """Convert a ttnn tensor or numpy value to numpy for CPU fallback."""
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, (int, float, np.integer, np.floating)):
        return np.array(val)
    # ttnn tensor — need shape hint
    if shape is not None:
        return _from_device(val, shape)
    # No shape hint: try scalar, then flat
    t = ttnn.to_torch(val).float()
    return t.numpy()


def _execute_op_device(op: dict, values: dict):
    """Execute a single StableHLO op on ttnn device."""
    op_type = op['op']

    # --- Constants: generate on CPU, send to device ---
    if op_type == 'constant':
        np_val = execute_constant(op)
        return _to_device(np_val)

    # --- Tier 1: Elementwise (direct ttnn equivalents) ---
    if op_type == 'add':
        a, b = get_operands(op, values, 2)
        return ttnn.add(a, b)

    if op_type == 'subtract':
        a, b = get_operands(op, values, 2)
        return ttnn.sub(a, b)

    if op_type == 'multiply':
        a, b = get_operands(op, values, 2)
        return ttnn.mul(a, b)

    if op_type == 'divide':
        a, b = get_operands(op, values, 2)
        return ttnn.mul(a, ttnn.reciprocal(b))

    if op_type == 'maximum':
        a, b = get_operands(op, values, 2)
        return ttnn.maximum(a, b)

    if op_type == 'minimum':
        a, b = get_operands(op, values, 2)
        return ttnn.minimum(a, b)

    if op_type == 'negate':
        a = get_operands(op, values, 1)[0]
        return ttnn.neg(a)

    if op_type == 'abs':
        a = get_operands(op, values, 1)[0]
        return ttnn.abs(a)

    if op_type == 'exp':
        a = get_operands(op, values, 1)[0]
        return ttnn.exp(a)

    if op_type == 'log':
        a = get_operands(op, values, 1)[0]
        return ttnn.log(a)

    if op_type == 'tanh':
        a = get_operands(op, values, 1)[0]
        return ttnn.tanh(a)

    if op_type == 'rsqrt':
        a = get_operands(op, values, 1)[0]
        return ttnn.rsqrt(a)

    if op_type == 'sqrt':
        a = get_operands(op, values, 1)[0]
        return ttnn.sqrt(a)

    # --- Tier 2: Shape ops ---
    if op_type == 'convert':
        # ttnn handles types internally; bf16 throughout
        return get_operands(op, values, 1)[0]

    if op_type == 'broadcast_in_dim':
        return _execute_broadcast_device(op, values)

    if op_type == 'reshape':
        a = get_operands(op, values, 1)[0]
        target_shape, _ = parse_tensor_type(op['result_type'])
        if not target_shape:
            # Scalar output — keep on device as 1x1
            return a
        return ttnn.reshape(a, list(target_shape))

    if op_type == 'transpose':
        a = get_operands(op, values, 1)[0]
        return ttnn.permute(a, op['permutation'])

    # --- Tier 3: Matmul ---
    if op_type == 'dot_general':
        return _execute_dot_general_device(op, values)

    # --- Tier 2/4: Reductions ---
    if op_type == 'reduce':
        return _execute_reduce_device(op, values)

    if op_type == 'reduce_argmax':
        return _execute_reduce_argmax_device(op, values)

    # --- Tier 4: CPU-roundtrip ops ---
    if op_type == 'slice':
        return _execute_slice_device(op, values)

    if op_type == 'compare':
        return _execute_compare_device(op, values)

    if op_type == 'select':
        a, b, c = get_operands(op, values, 3)
        return ttnn.where(a, b, c)

    if op_type == 'iota':
        np_val = execute_iota(op)
        return _to_device(np_val)

    if op_type == 'concatenate':
        return _execute_concatenate_device(op, values)

    if op_type == 'and':
        return _execute_logical_device(op, values, 'and')

    if op_type == 'or':
        return _execute_logical_device(op, values, 'or')

    if op_type == 'scatter':
        return _execute_scatter_device(op, values)

    if op_type == 'gather':
        return _execute_gather_device(op, values)

    if op_type == 'func_call':
        return execute_func_call(op, values)

    raise ValueError(f"Unsupported op: {op_type} (text: {op.get('text', '')})")


def _execute_broadcast_device(op, values):
    """Device-mode broadcast_in_dim.

    Tries on-device `ttnn.repeat` first (no host roundtrip — needed for
    trace capture). Falls back to CPU broadcast + `_to_device` if shapes
    or layout prevent on-device repeat.

    Critical: use the StableHLO LOGICAL input shape (from _logical_shapes),
    NOT the ttnn tensor's .shape. ttnn pads tensors to 2D-min and tile-
    aligned, so the device tensor's shape is misleading. For example, an
    operand declared as tensor<4xf32> in StableHLO has logical shape (4,)
    but ttnn .shape reports (1, 4) after our _to_device unsqueeze.
    """
    a = get_operands(op, values, 1)[0]
    dims = op.get('dims', [])
    result_type = op['result_type']
    target_shape, _ = parse_tensor_type(result_type)

    if not target_shape:
        return a

    # Get the LOGICAL source shape from the StableHLO IR, not from ttnn.
    operand_name = op['operands'][0] if op.get('operands') else None
    a_logical = _logical_shapes.get(operand_name, ())

    # Build intermediate shape: source dim i → target dim dims[i]
    inter_shape = [1] * len(target_shape)
    for src_dim, tgt_dim in enumerate(dims):
        if src_dim < len(a_logical):
            inter_shape[tgt_dim] = a_logical[src_dim]

    # On-device path: ttnn.repeat with repeat counts computed from
    # inter_shape → target_shape. Trace-capture safe (no host transfer).
    if isinstance(a, np.ndarray):
        # Already numpy (constant warmup path) — fall through to CPU.
        pass
    else:
        try:
            repeat_counts = []
            for src, dst in zip(inter_shape, target_shape):
                if src == dst:
                    repeat_counts.append(1)
                elif src == 1:
                    repeat_counts.append(dst)
                else:
                    raise ValueError(
                        f"non-broadcastable dim {src}->{dst}")
            # Reshape source onto inter_shape so ttnn.repeat sees aligned dims.
            # Pad to match device tensor rank so ttnn.reshape doesn't strip.
            dev_rank = len(a.shape)
            inter_padded = list(inter_shape)
            while len(inter_padded) < dev_rank:
                inter_padded.insert(0, 1)
            t = a
            if list(t.shape) != inter_padded:
                t = ttnn.reshape(t, inter_padded)
            repeat_padded = list(repeat_counts)
            while len(repeat_padded) < dev_rank:
                repeat_padded.insert(0, 1)
            if any(r > 1 for r in repeat_padded):
                t = ttnn.repeat(t, ttnn.Shape(repeat_padded))
            # Final reshape to exactly target_shape (may strip padding dims)
            if list(t.shape) != list(target_shape):
                try:
                    t = ttnn.reshape(t, list(target_shape))
                except Exception:
                    pass
            return t
        except Exception:
            pass  # Fall through to CPU fallback

    # CPU fallback path (always correct, used when on-device repeat fails)
    a_np = _operand_to_numpy(a, a_logical or None)
    if a_np.ndim == 0:
        result = np.broadcast_to(a_np, target_shape).copy()
    else:
        # Reshape to logical source shape first, then to intermediate
        try:
            a_np = a_np.reshape(a_logical) if a_logical else a_np
        except ValueError:
            # ttnn padding may have left extra elements; truncate to logical size
            n = 1
            for d in a_logical:
                n *= d
            a_np = a_np.flatten()[:n].reshape(a_logical)
        result = np.broadcast_to(a_np.reshape(inter_shape), target_shape).copy()
    return _to_device(result)


def _execute_dot_general_device(op, values):
    """Device-mode dot_general: ttnn.matmul for simple cases, CPU fallback."""
    a, b = get_operands(op, values, 2)
    lhs_contract = op['lhs_contracting']
    rhs_contract = op['rhs_contracting']
    lhs_batch = op.get('lhs_batching', [])
    rhs_batch = op.get('rhs_batching', [])

    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    # Simple case: standard matmul (contract last of A with second-to-last of B)
    is_simple = (
        not lhs_batch and
        len(lhs_contract) == 1 and len(rhs_contract) == 1 and
        lhs_contract[0] == len(a_shape) - 1 and
        rhs_contract[0] == len(b_shape) - 2
    )

    # Batched matmul: batch dims are leading and aligned
    is_batched_simple = (
        len(lhs_contract) == 1 and len(rhs_contract) == 1 and
        lhs_contract[0] == len(a_shape) - 1 and
        rhs_contract[0] == len(b_shape) - 2 and
        tuple(lhs_batch) == tuple(rhs_batch) and
        list(lhs_batch) == list(range(len(lhs_batch)))
    )

    if is_simple or is_batched_simple:
        try:
            return ttnn.matmul(a, b)
        except RuntimeError:
            pass

    # CPU fallback for complex dot_general
    a_np = _operand_to_numpy(a, a_shape)
    b_np = _operand_to_numpy(b, b_shape)
    result_np = execute_dot_general(op, {'__a': a_np, '__b': b_np,
                                          op['operands'][0]: a_np,
                                          op['operands'][1]: b_np})
    return _to_device(result_np)


def _execute_reduce_device(op, values):
    """Device-mode reduce: ttnn.sum/ttnn.max for supported cases."""
    a = get_operands(op, values, 1)[0]
    reduce_fn = op['reduce_fn']
    dims = op['dimensions']

    try:
        if reduce_fn == 'add':
            result = a
            for ax in sorted(dims, reverse=True):
                result = ttnn.sum(result, dim=ax, keepdim=True)
            # Squeeze reduced dims to match StableHLO semantics
            target_shape, _ = parse_tensor_type(op.get('result_type', ''))
            if target_shape:
                result = ttnn.reshape(result, list(target_shape))
            return result
        elif reduce_fn == 'maximum':
            result = a
            for ax in sorted(dims, reverse=True):
                result = ttnn.max(result, dim=ax, keepdim=True)
            target_shape, _ = parse_tensor_type(op.get('result_type', ''))
            if target_shape:
                result = ttnn.reshape(result, list(target_shape))
            return result
    except Exception:
        pass

    # CPU fallback for min, prod, or on ttnn failure
    a_np = _operand_to_numpy(a)
    np_result = execute_reduce(op, {op['operands'][0]: a_np})
    return _to_device(np_result)


def _execute_reduce_argmax_device(op, values):
    """Device-mode argmax: CPU roundtrip (ttnn lacks native argmax)."""
    a = get_operands(op, values, 1)[0]
    a_shape = tuple(a.shape)
    a_np = _operand_to_numpy(a, a_shape)

    # Reuse numpy path
    np_result = execute_reduce_argmax(op, {op['operands'][0]: a_np})
    # np_result is a dict of name#0 → numpy, name#1 → numpy
    device_result = {}
    for key, val in np_result.items():
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
            # Keep integer results as numpy (argmax indices)
            device_result[key] = val
        else:
            device_result[key] = _to_device(val)
    return device_result


def _execute_slice_device(op, values):
    """Device-mode slice: CPU roundtrip (ttnn lacks general slicing)."""
    a = get_operands(op, values, 1)[0]
    a_shape = tuple(a.shape)
    a_np = _operand_to_numpy(a, a_shape)
    slices = tuple(
        slice(s, l, st)
        for s, l, st in zip(op['starts'], op['limits'], op['strides'])
    )
    result = a_np[slices]
    return _to_device(result)


def _execute_compare_device(op, values):
    """Device-mode compare: try ttnn, fall back to CPU."""
    a, b = get_operands(op, values, 2)
    direction = op['direction']

    try:
        if direction == 'GE':
            return ttnn.ge(a, b)
        elif direction == 'GT':
            return ttnn.gt(a, b)
        elif direction == 'LE':
            return ttnn.le(a, b)
        elif direction == 'LT':
            return ttnn.lt(a, b)
        elif direction == 'EQ':
            return ttnn.eq(a, b)
        elif direction == 'NE':
            return ttnn.ne(a, b)
    except Exception:
        pass

    # CPU fallback
    a_np = _operand_to_numpy(a)
    b_np = _operand_to_numpy(b)
    np_result = execute_compare(op, {op['operands'][0]: a_np,
                                      op['operands'][1]: b_np})
    return _to_device(np_result.astype(np.float32))


def _execute_concatenate_device(op, values):
    """Device-mode concatenate: try ttnn.concat, fall back to CPU."""
    tensors = [values[name] for name in op['operands']]
    try:
        return ttnn.concat(tensors, dim=op['dim'])
    except Exception:
        pass

    # CPU fallback
    arrays = []
    for name in op['operands']:
        val = values[name]
        arrays.append(_operand_to_numpy(val))
    result = np.concatenate(arrays, axis=op['dim'])
    return _to_device(result)


def _execute_logical_device(op, values, logic_op):
    """Device-mode and/or: CPU roundtrip."""
    a, b = get_operands(op, values, 2)
    a_np = _operand_to_numpy(a)
    b_np = _operand_to_numpy(b)
    if logic_op == 'and':
        result = np.logical_and(a_np, b_np)
    else:
        result = np.logical_or(a_np, b_np)
    return _to_device(result.astype(np.float32))


def _execute_scatter_device(op, values):
    """Device-mode scatter: CPU roundtrip with shape-aware reads."""
    operand_names = op['operands']
    operand = _operand_to_numpy(values[operand_names[0]],
                                 _logical_shapes.get(operand_names[0]))
    indices = _operand_to_numpy(values[operand_names[1]],
                                 _logical_shapes.get(operand_names[1]))
    updates = _operand_to_numpy(values[operand_names[2]],
                                 _logical_shapes.get(operand_names[2]))
    if not np.issubdtype(indices.dtype, np.integer):
        indices = indices.astype(np.int64)

    np_values = {operand_names[0]: operand, operand_names[1]: indices,
                 operand_names[2]: updates}
    result = execute_scatter(op, np_values)
    return _to_device(result)


def _execute_gather_device(op, values):
    """Device-mode gather: CPU roundtrip.

    Indices come back as floats from ttnn (we always upload as bf16), so
    we must cast them to int before using them as numpy indices.
    """
    operand_names = op['operands']
    operand_shape = _logical_shapes.get(operand_names[0])
    indices_shape = _logical_shapes.get(operand_names[1])
    operand = _operand_to_numpy(values[operand_names[0]], operand_shape)
    indices = _operand_to_numpy(values[operand_names[1]], indices_shape)
    if not np.issubdtype(indices.dtype, np.integer):
        indices = indices.astype(np.int64)

    np_values = {operand_names[0]: operand, operand_names[1]: indices}
    result = execute_gather(op, np_values)
    return _to_device(result)


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


def execute_reduce_argmax(op: dict, values: dict) -> dict:
    """Execute multi-output reduce for argmax pattern.

    JAX compiles argmax as a dual reduce returning (max_value, argmax_index).
    Returns a dict mapping name#0 -> max values, name#1 -> argmax indices.
    """
    a = get_operands(op, values, 1)[0]
    dims = op['dimensions']
    base_name = op['name']
    axis = tuple(dims)

    max_vals = np.max(a, axis=axis)
    argmax_idx = np.argmax(a, axis=axis[0] if len(axis) == 1 else axis)

    # Parse result dtypes
    result_types = op.get('result_types', [])
    if len(result_types) >= 2:
        _, max_dtype = parse_tensor_type(result_types[0])
        _, idx_dtype = parse_tensor_type(result_types[1])
        max_vals = max_vals.astype(max_dtype)
        argmax_idx = argmax_idx.astype(idx_dtype)

    return {
        f'{base_name}#0': max_vals,
        f'{base_name}#1': argmax_idx,
    }


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
# _private_functions can be either:
#   - list of (args, ops, returns) tuples (positional dispatch by _call_counter)
#   - dict {callee_name: (args, ops, returns)} (name-based dispatch)
# When a func.call op carries a 'callee' field we use the dict form;
# otherwise we fall back to positional. When there's exactly ONE private
# function and the call carries no callee, we always dispatch to it
# (JAX commonly deduplicates a single helper like silu and emits multiple
# func.calls to the same private function — see research/pjrt_real_model_plan.md).
_private_functions = []
_call_counter = 0


def execute_func_call(op: dict, values: dict) -> np.ndarray:
    """Execute a func.call by running the corresponding private function."""
    global _call_counter

    callee = op.get('callee', None)
    func_entry = None

    # private_functions is a list of 4-tuples (args, ops, returns, sym_name)
    # (legacy 3-tuples are also tolerated)
    n_private = len(_private_functions)

    def _entry_at(idx):
        e = _private_functions[idx]
        return e[:3]  # (args, ops, returns)

    # Try callee-name dispatch first
    if callee is not None:
        for entry in _private_functions:
            if len(entry) >= 4 and entry[3] == callee:
                func_entry = entry[:3]
                break

    if func_entry is None:
        # Common case: bytecode_to_text drops callee names; JAX deduplicates
        # helpers (e.g. silu) into ONE private function with N callsites.
        # When there's exactly one private function, dispatch all calls to it.
        if n_private == 1:
            func_entry = _entry_at(0)
        elif _call_counter < n_private:
            func_entry = _entry_at(_call_counter)
            _call_counter += 1
        else:
            raise ValueError(
                f"func.call cannot resolve callee={callee!r}; "
                f"{n_private} private function(s), _call_counter={_call_counter}"
            )

    func_args, func_ops, func_returns = func_entry

    # Build local value map: function args ← call operands
    local_values = dict(values)  # inherit outer scope for init values etc.
    call_operands = [values[name] for name in op['operands']]
    for i, (arg_name, type_str) in enumerate(func_args):
        local_values[arg_name] = call_operands[i]
        # Mirror the caller's logical shape into the private function's
        # arg name so ops inside the function can resolve it.
        caller_operand = op['operands'][i]
        if caller_operand in _logical_shapes:
            _logical_shapes[arg_name] = _logical_shapes[caller_operand]
        else:
            shape, _ = parse_tensor_type(type_str)
            _logical_shapes[arg_name] = shape

    # Execute function body, populating logical shapes for each op
    for func_op in func_ops:
        result = execute_op(func_op, local_values)
        if isinstance(result, dict):
            local_values.update(result)
        else:
            local_values[func_op['name']] = result
            rt = func_op.get('result_type', '')
            if rt:
                try:
                    _logical_shapes[func_op['name']], _ = parse_tensor_type(rt)
                except (ValueError, TypeError):
                    pass  # Multi-output or malformed; handled by result_types

    # Return the function's return value(s)
    if len(func_returns) == 1:
        return local_values[func_returns[0]]
    return [local_values[r] for r in func_returns]
