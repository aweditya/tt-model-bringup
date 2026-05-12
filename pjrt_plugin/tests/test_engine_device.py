"""Test StableHLO engine execution on ttnn device (Blackhole).

Mirrors test_engine.py but runs with TT_PJRT_USE_DEVICE=1 so all ops
execute on the Tenstorrent device via ttnn. Tolerances are wider (bf16).

Run on remote host:
    TT_PJRT_USE_DEVICE=1 python3 -m pytest test_engine_device.py -v
"""

import os
import sys
import numpy as np
import pytest

# Force device mode ON before importing engine
os.environ['TT_PJRT_USE_DEVICE'] = '1'

# Add pjrt_plugin to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use the SAME engine module instance that the C++ PJRT plugin invokes
# (`jax_plugins.tt.engine`). If we imported a second copy via
# importlib.spec_from_file_location, we'd get two `_device` globals →
# the second one tries to ttnn.open_device(0) again and crashes with
# `context_id.get() >= 0`.
from jax_plugins.tt import engine


def _check_device_mode():
    """Skip all tests if ttnn is not available."""
    if not engine._USE_DEVICE:
        pytest.skip("ttnn not available — device tests require TT_PJRT_USE_DEVICE=1 and ttnn")


# bf16 tolerances
ATOL = 1e-2
RTOL = 1e-2


def _run_op(op_desc, values_np):
    """Run a single op in device mode and return numpy result."""
    _check_device_mode()
    # Convert inputs to device tensors
    values = {}
    for k, v in values_np.items():
        values[k] = engine._to_device(v)
    result = engine._execute_op_device(op_desc, values)
    if isinstance(result, dict):
        return {k: engine._from_device(v, ()) if not isinstance(v, np.ndarray) else v
                for k, v in result.items()}
    if isinstance(result, np.ndarray):
        return result
    # ttnn tensor — infer shape from result_type
    rt = op_desc.get('result_type', '')
    if rt:
        shape, _ = engine.parse_tensor_type(rt)
    else:
        shape = ()
    return engine._from_device(result, shape)


# ============================================================
# Tier 1: Elementwise ops
# ============================================================

class TestElementwise:
    def test_add(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        b = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'add', 'operands': ['a', 'b'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, a + b, atol=ATOL, rtol=RTOL)

    def test_subtract(self):
        a = np.array([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32)
        b = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'subtract', 'operands': ['a', 'b'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, a - b, atol=ATOL, rtol=RTOL)

    def test_multiply(self):
        a = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        b = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        op = {'name': '0', 'op': 'multiply', 'operands': ['a', 'b'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, a * b, atol=ATOL, rtol=RTOL)

    def test_divide(self):
        a = np.array([[6.0, 8.0, 9.0, 12.0]], dtype=np.float32)
        b = np.array([[2.0, 4.0, 3.0, 6.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'divide', 'operands': ['a', 'b'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, a / b, atol=ATOL, rtol=RTOL)

    def test_negate(self):
        a = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'negate', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, -a, atol=ATOL, rtol=RTOL)

    def test_exp(self):
        a = np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float32)
        op = {'name': '0', 'op': 'exp', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, np.exp(a), atol=ATOL, rtol=RTOL)

    def test_log(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'log', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, np.log(a), atol=ATOL, rtol=RTOL)

    def test_tanh(self):
        a = np.array([[0.0, 1.0, -1.0, 3.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'tanh', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, np.tanh(a), atol=ATOL, rtol=RTOL)

    def test_rsqrt(self):
        a = np.array([[1.0, 4.0, 9.0, 16.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'rsqrt', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, 1.0 / np.sqrt(a), atol=ATOL, rtol=RTOL)

    def test_sqrt(self):
        a = np.array([[1.0, 4.0, 9.0, 16.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'sqrt', 'operands': ['a'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, np.sqrt(a), atol=ATOL, rtol=RTOL)

    def test_maximum(self):
        a = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'maximum', 'operands': ['a', 'b'],
              'result_type': 'tensor<1x4xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, np.maximum(a, b), atol=ATOL, rtol=RTOL)


# ============================================================
# Tier 2: Shape ops
# ============================================================

class TestShape:
    def test_reshape(self):
        _check_device_mode()
        a = np.random.randn(1, 2, 4).astype(np.float32) * 0.5
        op = {'name': '0', 'op': 'reshape', 'operands': ['a'],
              'result_type': 'tensor<2x4xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, a.reshape(2, 4), atol=ATOL, rtol=RTOL)

    def test_transpose(self):
        _check_device_mode()
        a = np.random.randn(2, 4).astype(np.float32) * 0.5
        op = {'name': '0', 'op': 'transpose', 'operands': ['a'],
              'permutation': [1, 0], 'result_type': 'tensor<4x2xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, a.T, atol=ATOL, rtol=RTOL)

    def test_broadcast_scalar(self):
        _check_device_mode()
        a = np.array([[2.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'broadcast_in_dim', 'operands': ['a'],
              'dims': [0, 1], 'result_type': 'tensor<1x64xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, np.full((1, 64), 2.0), atol=ATOL, rtol=RTOL)


# ============================================================
# Tier 3: Matmul
# ============================================================

class TestMatmul:
    def test_simple_matmul(self):
        _check_device_mode()
        np.random.seed(42)
        a = np.random.randn(2, 64).astype(np.float32) * 0.1
        b = np.random.randn(64, 32).astype(np.float32) * 0.1
        op = {'name': '0', 'op': 'dot_general', 'operands': ['a', 'b'],
              'lhs_contracting': [1], 'rhs_contracting': [0],
              'lhs_batching': [], 'rhs_batching': [],
              'result_type': 'tensor<2x32xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        np.testing.assert_allclose(result, a @ b, atol=0.05, rtol=0.05)

    def test_batched_matmul(self):
        _check_device_mode()
        np.random.seed(42)
        a = np.random.randn(4, 8, 32).astype(np.float32) * 0.1
        b = np.random.randn(4, 32, 16).astype(np.float32) * 0.1
        op = {'name': '0', 'op': 'dot_general', 'operands': ['a', 'b'],
              'lhs_contracting': [2], 'rhs_contracting': [1],
              'lhs_batching': [0], 'rhs_batching': [0],
              'result_type': 'tensor<4x8x16xf32>'}
        result = _run_op(op, {'a': a, 'b': b})
        ref = np.einsum('bij,bjk->bik', a, b)
        np.testing.assert_allclose(result, ref, atol=0.05, rtol=0.05)


# ============================================================
# Tier 4: Reduce, Slice, Compare, etc.
# ============================================================

class TestReduce:
    def test_reduce_sum(self):
        _check_device_mode()
        a = np.random.randn(2, 64).astype(np.float32)
        op = {'name': '0', 'op': 'reduce', 'operands': ['a'],
              'init_operand': None, 'reduce_fn': 'add',
              'dimensions': [1], 'result_type': 'tensor<2xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result.flatten()[:2], np.sum(a, axis=1), atol=0.5, rtol=0.1)

    def test_reduce_max(self):
        _check_device_mode()
        a = np.random.randn(2, 64).astype(np.float32)
        op = {'name': '0', 'op': 'reduce', 'operands': ['a'],
              'init_operand': None, 'reduce_fn': 'maximum',
              'dimensions': [1], 'result_type': 'tensor<2xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result.flatten()[:2], np.max(a, axis=1), atol=ATOL, rtol=RTOL)


class TestSlice:
    def test_static_slice(self):
        _check_device_mode()
        a = np.random.randn(1, 64).astype(np.float32)
        op = {'name': '0', 'op': 'slice', 'operands': ['a'],
              'starts': [0, 0], 'limits': [1, 32], 'strides': [1, 1],
              'result_type': 'tensor<1x32xf32>'}
        result = _run_op(op, {'a': a})
        np.testing.assert_allclose(result, a[:, :32], atol=ATOL, rtol=RTOL)


class TestCompare:
    def test_ge(self):
        _check_device_mode()
        a = np.array([[1.0, -1.0, 0.0, 2.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        op = {'name': '0', 'op': 'compare', 'operands': ['a', 'b'],
              'direction': 'GE', 'result_type': 'tensor<1x4xi1>'}
        result = _run_op(op, {'a': a, 'b': b})
        expected = (a >= b).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=ATOL)


class TestConcatenate:
    def test_concat(self):
        _check_device_mode()
        a = np.random.randn(1, 32).astype(np.float32)
        b = np.random.randn(1, 32).astype(np.float32)
        op = {'name': '0', 'op': 'concatenate', 'operands': ['a', 'b'],
              'dim': 1, 'result_type': 'tensor<1x64xf32>'}
        values_np = {'a': a, 'b': b}
        _check_device_mode()
        values = {k: engine._to_device(v) for k, v in values_np.items()}
        result = engine._execute_op_device(op, values)
        result_np = engine._from_device(result, (1, 64))
        np.testing.assert_allclose(result_np, np.concatenate([a, b], axis=1),
                                   atol=ATOL, rtol=RTOL)


# ============================================================
# Composite: softmax, RMS norm, attention
# ============================================================

class TestComposite:
    def test_softmax(self):
        """Softmax = max → sub → exp → sum → div (5 ops on device)."""
        _check_device_mode()
        np.random.seed(42)
        x = np.random.randn(1, 64).astype(np.float32)

        # Build the op sequence manually
        ops = [
            {'name': 'max', 'op': 'reduce', 'operands': ['x'],
             'init_operand': None, 'reduce_fn': 'maximum',
             'dimensions': [1], 'result_type': 'tensor<1xf32>'},
            {'name': 'max_bc', 'op': 'broadcast_in_dim', 'operands': ['max'],
             'dims': [0], 'result_type': 'tensor<1x64xf32>'},
            {'name': 'sub', 'op': 'subtract', 'operands': ['x', 'max_bc'],
             'result_type': 'tensor<1x64xf32>'},
            {'name': 'exp', 'op': 'exp', 'operands': ['sub'],
             'result_type': 'tensor<1x64xf32>'},
            {'name': 'sum', 'op': 'reduce', 'operands': ['exp'],
             'init_operand': None, 'reduce_fn': 'add',
             'dimensions': [1], 'result_type': 'tensor<1xf32>'},
            {'name': 'sum_bc', 'op': 'broadcast_in_dim', 'operands': ['sum'],
             'dims': [0], 'result_type': 'tensor<1x64xf32>'},
            {'name': 'out', 'op': 'divide', 'operands': ['exp', 'sum_bc'],
             'result_type': 'tensor<1x64xf32>'},
        ]

        values = {'x': engine._to_device(x)}
        for op in ops:
            result = engine._execute_op_device(op, values)
            values[op['name']] = result

        out_np = engine._from_device(values['out'], (1, 64))
        ref = np.exp(x - np.max(x, axis=1, keepdims=True))
        ref = ref / np.sum(ref, axis=1, keepdims=True)
        np.testing.assert_allclose(out_np, ref, atol=0.02, rtol=0.02)

    def test_linear_layer(self):
        """x @ w + b through device matmul + add."""
        _check_device_mode()
        np.random.seed(42)
        x = np.random.randn(2, 64).astype(np.float32) * 0.1
        w = np.random.randn(64, 32).astype(np.float32) * 0.1
        b = np.random.randn(1, 32).astype(np.float32) * 0.1

        values = {
            'x': engine._to_device(x),
            'w': engine._to_device(w),
            'b': engine._to_device(b),
        }

        # matmul
        mm_op = {'name': 'mm', 'op': 'dot_general', 'operands': ['x', 'w'],
                 'lhs_contracting': [1], 'rhs_contracting': [0],
                 'lhs_batching': [], 'rhs_batching': [],
                 'result_type': 'tensor<2x32xf32>'}
        values['mm'] = engine._execute_op_device(mm_op, values)

        # add bias
        add_op = {'name': 'out', 'op': 'add', 'operands': ['mm', 'b'],
                  'result_type': 'tensor<2x32xf32>'}
        values['out'] = engine._execute_op_device(add_op, values)

        out_np = engine._from_device(values['out'], (2, 32))
        ref = x @ w + b
        np.testing.assert_allclose(out_np, ref, atol=0.1, rtol=0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
