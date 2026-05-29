"""
jit_tt: A jax.jit-like API that dispatches to Tenstorrent Blackhole.

Usage:
    import tt_jax

    device = ttnn.open_device(device_id=0)
    executor = tt_jax.Executor(device)

    @executor.jit
    def mlp(x, w, b):
        return jax.nn.relu(jnp.dot(x, w) + b)

    result = mlp(x_np, w_np, b_np)  # Runs on Blackhole!

First call: traces Jaxpr, warmup, trace capture.
Subsequent calls: ttnn.copy + execute_trace (fast path).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import ttnn

from . import tensors
from .interpret import Interpreter


class CompiledFunction:
    """A function compiled for traced execution on Blackhole."""

    def __init__(self, fn, device):
        self.fn = fn
        self.device = device
        self._compiled = False
        self._trace_id = None
        self._interp = None
        self._jaxpr = None
        self._input_buffers = None
        self._literal_cache = None

    def _compile(self, args):
        """First-call compilation: trace Jaxpr, warmup, capture trace."""
        # Step 1: Trace to Jaxpr
        jax_args = [jnp.array(a) for a in args]
        self._jaxpr = make_jaxpr(self.fn)(*jax_args)

        # Step 2: Pre-materialize literals
        self._literal_cache = {}
        for eqn in self._jaxpr.jaxpr.eqns:
            for v in eqn.invars:
                if tensors.is_literal(v):
                    fval = float(v.val)
                    if fval not in self._literal_cache:
                        self._literal_cache[fval] = tensors.to_device(v.val, self.device)

        # Step 3: Load inputs
        self._input_buffers = [tensors.to_device(a, self.device) for a in args]

        # Step 4: Warmup (allocates all intermediate buffers)
        self._interp = Interpreter(self.device, literal_cache=self._literal_cache)
        for var, t in zip(self._jaxpr.jaxpr.invars, self._input_buffers):
            self._interp.env[var] = t
        for var, const in zip(self._jaxpr.jaxpr.constvars, self._jaxpr.consts):
            self._interp.env[var] = tensors.to_device(const, self.device)
        for eqn in self._jaxpr.jaxpr.eqns:
            self._interp._exec(eqn)

        # Step 5: Trace capture
        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        for eqn in self._jaxpr.jaxpr.eqns:
            self._interp._exec(eqn)
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)

        self._compiled = True

    def __call__(self, *args):
        if not self._compiled:
            self._compile(args)
        else:
            # Fast path: copy new inputs into existing buffers
            for buf, arg in zip(self._input_buffers, args):
                new_tt = tensors.to_device(arg, self.device)
                ttnn.copy(new_tt, buf)

        # Execute trace
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)

        # Read output(s)
        outvars = self._jaxpr.jaxpr.outvars
        if len(outvars) == 1:
            return tensors.from_device(
                self._interp.env[outvars[0]], outvars[0].aval.shape)
        return [tensors.from_device(self._interp.env[v], v.aval.shape)
                for v in outvars]

    def release(self):
        """Release device resources."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
            self._compiled = False


class Executor:
    """Device executor that provides jit-like API for Blackhole."""

    def __init__(self, device):
        self.device = device
        self._compiled_fns = []

    def jit(self, fn):
        """Decorator: compile a function for traced execution on Blackhole."""
        compiled = CompiledFunction(fn, self.device)
        self._compiled_fns.append(compiled)
        return compiled

    def release_all(self):
        """Release all compiled function traces."""
        for cf in self._compiled_fns:
            cf.release()
        self._compiled_fns.clear()
