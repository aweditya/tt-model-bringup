"""
Trace capture wrapper: compiles a Jaxpr into a replayable TT-NN trace.

Usage:
    executor = TracedExecutor(device)
    executor.compile(jaxpr, sample_args)  # captures trace
    result = executor.run(new_args)       # replays trace with new data
    executor.release()                    # frees device resources

Key constraint: TT-NN forbids ALL host-device transfers during trace capture
(no reads, no writes). Ops that do CPU round-trips (broadcast_in_dim, binary
ops with shape mismatch) are handled by caching their warm-up results and
injecting them during the trace run.
"""

import ttnn
import numpy as np
from jax._src import core as jax_core
from . import tensors
from .interpret import Interpreter


# Ops that always do CPU round-trips (to_device/from_device calls)
_HOST_TRANSFER_OPS = {'broadcast_in_dim'}


def _needs_broadcast(eqn):
    """Check if a binary op equation will trigger broadcast_to_match."""
    if len(eqn.invars) != 2:
        return False
    shapes = []
    for v in eqn.invars:
        if isinstance(v, jax_core.Literal):
            return False  # scalar literals are handled without broadcast
        if hasattr(v, 'aval'):
            shapes.append(v.aval.shape)
        else:
            shapes.append(())
    if len(shapes) == 2 and shapes[0] != shapes[1]:
        return True
    return False


class TracedExecutor:
    """Compile a Jaxpr into a TT-NN trace for fast replay."""

    def __init__(self, device):
        self.device = device
        self.trace_id = None
        self.input_tensors = []
        self.output_tensor = None
        self.output_shape = None
        self.jaxpr = None

    def compile(self, jaxpr, sample_args):
        """Compile a Jaxpr: run it once, then capture a TT-NN trace."""
        self.jaxpr = jaxpr
        self.output_shape = jaxpr.jaxpr.outvars[0].aval.shape

        # Create input tensors on device
        self.input_tensors = []
        for var, arg in zip(jaxpr.jaxpr.invars, sample_args):
            self.input_tensors.append(tensors.to_device(arg, self.device))

        # Warm-up run
        interp = Interpreter(self.device)
        interp.env.clear()
        for var, t in zip(jaxpr.jaxpr.invars, self.input_tensors):
            interp.env[var] = t
        for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            interp.env[var] = tensors.to_device(const, self.device)
        for eqn in jaxpr.jaxpr.eqns:
            interp._exec(eqn)
        ttnn.synchronize_device(self.device)

        # Build caches from warm-up
        warmup_env = dict(interp.env)
        literal_cache = {}
        self._collect_literals(jaxpr.jaxpr, interp, literal_cache)

        # Identify equations that need host transfers (can't run during trace)
        skip_eqns = self._find_host_transfer_eqns(jaxpr.jaxpr)

        # Capture trace
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        interp2 = Interpreter(self.device, literal_cache=literal_cache)
        interp2.env.clear()
        for var, t in zip(jaxpr.jaxpr.invars, self.input_tensors):
            interp2.env[var] = t
        for var in jaxpr.jaxpr.constvars:
            interp2.env[var] = warmup_env[var]

        self._exec_traced(interp2, jaxpr.jaxpr, warmup_env, skip_eqns)

        self.output_tensor = interp2.eval_var(jaxpr.jaxpr.outvars[0])
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

    def _find_host_transfer_eqns(self, jaxpr_inner):
        """Find equation indices that require host-device transfers."""
        skip = set()
        for i, eqn in enumerate(jaxpr_inner.eqns):
            name = eqn.primitive.name
            if name in _HOST_TRANSFER_OPS:
                skip.add(i)
            elif name in ('add', 'sub', 'mul', 'div', 'max') and _needs_broadcast(eqn):
                skip.add(i)
        return skip

    def _exec_traced(self, interp, jaxpr_inner, warmup_env, skip_eqns):
        """Execute Jaxpr during trace, skipping host-transfer ops."""
        for i, eqn in enumerate(jaxpr_inner.eqns):
            name = eqn.primitive.name

            if i in skip_eqns:
                # Inject warm-up results for ops that need host transfers
                for var in eqn.outvars:
                    if var in warmup_env:
                        interp.env[var] = warmup_env[var]
            elif name in ('custom_jvp_call', 'pjit'):
                self._exec_sub_traced(interp, eqn, warmup_env)
            else:
                interp._exec(eqn)

    def _exec_sub_traced(self, interp, eqn, warmup_env):
        """Execute a sub-jaxpr during trace."""
        name = eqn.primitive.name
        params = eqn.params

        if name == 'custom_jvp_call':
            sub = params['call_jaxpr']
        elif name == 'pjit':
            sub = params['jaxpr']
        else:
            raise NotImplementedError(f"Unknown sub-jaxpr: {name}")

        if hasattr(sub, 'jaxpr'):
            raw = sub.jaxpr
            consts = getattr(sub, 'consts', [])
        else:
            raw = sub
            consts = []

        for sv, c in zip(raw.constvars, consts):
            interp.env[sv] = warmup_env.get(sv, interp.to_device(c))
        for sv, iv in zip(raw.invars, eqn.invars):
            interp.env[sv] = interp.eval_var(iv)

        sub_skip = self._find_host_transfer_eqns(raw)
        self._exec_traced(interp, raw, warmup_env, sub_skip)

        if len(raw.outvars) == 1:
            result = interp.eval_var(raw.outvars[0])
        else:
            result = [interp.eval_var(v) for v in raw.outvars]

        if isinstance(result, (list, tuple)):
            for var, val in zip(eqn.outvars, result):
                interp.env[var] = val
        else:
            interp.env[eqn.outvars[0]] = result

    def _collect_literals(self, jaxpr_inner, interp, cache):
        """Pre-materialize all literal values from warm-up."""
        for eqn in jaxpr_inner.eqns:
            for v in eqn.invars:
                if isinstance(v, jax_core.Literal):
                    fval = float(v.val)
                    if fval not in cache:
                        cache[fval] = interp.to_device(v.val)
            for p in eqn.params.values():
                if hasattr(p, 'jaxpr'):
                    sub = p.jaxpr if hasattr(p.jaxpr, 'eqns') else p
                    if hasattr(sub, 'eqns'):
                        self._collect_literals(sub, interp, cache)
                elif hasattr(p, 'eqns'):
                    self._collect_literals(p, interp, cache)

    def run(self, args):
        """Execute the compiled trace with new input data."""
        if self.trace_id is None:
            raise RuntimeError("Must call compile() before run()")

        for i, arg in enumerate(args):
            new_t = ttnn.from_torch(
                _prepare_torch(arg),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(new_t, self.input_tensors[i])

        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)
        return tensors.from_device(self.output_tensor, self.output_shape)

    def release(self):
        """Free trace resources on device."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None


def _prepare_torch(val):
    """Convert value to a padded torch tensor for host-to-device copy."""
    import torch
    if isinstance(val, np.ndarray):
        val = torch.from_numpy(val.copy()).float()
    while val.dim() < 2:
        val = val.unsqueeze(0)
    h, w = val.shape[-2], val.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        val = torch.nn.functional.pad(val, (0, pad_w, 0, pad_h))
    return val
