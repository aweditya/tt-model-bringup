"""
Jaxpr interpreter: walks a Jaxpr and executes each equation on Blackhole.

Uses the op registry (ops.py) for dispatch and tensor utilities (tensors.py)
for host ↔ device transfer.
"""

from jax._src import core as jax_core
from . import tensors
from .ops import REGISTRY


class Interpreter:
    """Execute a Jaxpr on Tenstorrent Blackhole via TT-NN."""

    def __init__(self, device, literal_cache=None):
        self.device = device
        self.env = {}
        self.ops_seen = set()
        self.literal_cache = literal_cache or {}

    def to_device(self, val):
        return tensors.to_device(val, self.device)

    def eval_var(self, var):
        if tensors.is_literal(var):
            fval = float(var.val)
            if fval in self.literal_cache:
                return self.literal_cache[fval]
            return self.to_device(var.val)
        return self.env[var]

    def run(self, jaxpr, args):
        """Execute a closed Jaxpr with given arguments. Returns numpy array(s)."""
        self.env.clear()
        self.ops_seen.clear()

        # Bind inputs
        for var, arg in zip(jaxpr.jaxpr.invars, args):
            self.env[var] = self.to_device(arg)

        # Bind constants
        for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            self.env[var] = self.to_device(const)

        # Execute
        for eqn in jaxpr.jaxpr.eqns:
            self._exec(eqn)

        # Collect outputs
        results = []
        for var in jaxpr.jaxpr.outvars:
            shape = var.aval.shape
            results.append(tensors.from_device(self.eval_var(var), shape))

        return results[0] if len(results) == 1 else results

    def _exec(self, eqn):
        """Execute one Jaxpr equation."""
        name = eqn.primitive.name
        self.ops_seen.add(name)

        # Look up in registry
        handler = REGISTRY.get(name)

        # Handle sub-jaxpr ops (custom_jvp_call, pjit)
        if handler is None and name in ('custom_jvp_call', 'pjit'):
            result = self._exec_sub_jaxpr(eqn)
        elif handler is None:
            raise NotImplementedError(
                f"Unsupported Jaxpr primitive: '{name}'. "
                f"Supported: {sorted(REGISTRY.keys())}"
            )
        else:
            result = handler(self, eqn.invars, eqn.params, eqn)

        # Bind outputs
        if isinstance(result, (list, tuple)):
            for var, val in zip(eqn.outvars, result):
                self.env[var] = val
        else:
            self.env[eqn.outvars[0]] = result

    def _exec_sub_jaxpr(self, eqn):
        """Execute a sub-jaxpr (custom_jvp_call wraps relu, pjit wraps some ops)."""
        name = eqn.primitive.name
        params = eqn.params
        invars = eqn.invars

        if name == 'custom_jvp_call':
            sub = params['call_jaxpr']
        elif name == 'pjit':
            sub = params['jaxpr']
        else:
            raise NotImplementedError(f"Unknown sub-jaxpr type: {name}")

        # Unwrap ClosedJaxpr
        if hasattr(sub, 'jaxpr'):
            raw = sub.jaxpr
            consts = getattr(sub, 'consts', [])
        else:
            raw = sub
            consts = []

        # Bind constants and inputs
        for sv, c in zip(raw.constvars, consts):
            self.env[sv] = self.to_device(c)
        for sv, iv in zip(raw.invars, invars):
            self.env[sv] = self.eval_var(iv)

        # Execute sub-equations
        for sub_eqn in raw.eqns:
            self._exec(sub_eqn)

        # Return outputs
        if len(raw.outvars) == 1:
            return self.eval_var(raw.outvars[0])
        return [self.eval_var(v) for v in raw.outvars]
