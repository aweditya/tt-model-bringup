"""Smoke test: verify engine.py loads in device mode on Blackhole.

Run on remote host:
    cd ~/tt-xla && TT_PJRT_USE_DEVICE=1 python3 pjrt_plugin/scripts/smoke_test_device.py
"""

import os
import sys

os.environ['TT_PJRT_USE_DEVICE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
engine_path = os.path.join(os.path.dirname(__file__), '..', 'jax_plugins', 'tt', 'engine.py')
spec = importlib.util.spec_from_file_location('engine', engine_path)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

import numpy as np

print(f"_USE_DEVICE = {engine._USE_DEVICE}")
print(f"_HAS_TTNN (via import) = {hasattr(engine, 'ttnn')}")

if not engine._USE_DEVICE:
    print("FAIL: ttnn not available, cannot test device mode")
    sys.exit(1)

# Test 1: Device opens successfully
device = engine._get_device()
print(f"Device opened: {device}")

# Test 2: Round-trip a tensor
a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
t = engine._to_device(a)
print(f"Tensor on device: shape={t.shape}")
back = engine._from_device(t, (1, 4))
print(f"Round-trip result: {back}")
np.testing.assert_allclose(back, a, atol=0.01)
print("Round-trip OK!")

# Test 3: Simple add on device
b = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
t_a = engine._to_device(a)
t_b = engine._to_device(b)
import ttnn
t_c = ttnn.add(t_a, t_b)
c = engine._from_device(t_c, (1, 4))
np.testing.assert_allclose(c, a + b, atol=0.01)
print(f"Device add: {a} + {b} = {c}  OK!")

# Test 4: Execute op through engine dispatch
op = {'name': '0', 'op': 'add', 'operands': ['a', 'b'],
      'result_type': 'tensor<1x4xf32>'}
values = {'a': t_a, 'b': t_b}
result = engine._execute_op_device(op, values)
result_np = engine._from_device(result, (1, 4))
np.testing.assert_allclose(result_np, a + b, atol=0.01)
print(f"Engine dispatch add: OK!")

print("\n=== ALL SMOKE TESTS PASSED ===")
