"""
Trace capture wrapper: compiles a Jaxpr into a replayable TT-NN trace.

Usage:
    executor = TracedExecutor(device)
    executor.compile(jaxpr, sample_args)  # captures trace
    result = executor.run(new_args)       # replays trace with new data
    executor.release()                    # frees device resources
"""

import ttnn
import numpy as np
from . import tensors
from .interpret import Interpreter


class TracedExecutor:
    """Compile a Jaxpr into a TT-NN trace for fast replay."""

    def __init__(self, device):
        self.device = device
        self.trace_id = None
        self.input_tensors = []  # device tensors to overwrite before replay
        self.output_tensor = None
        self.output_shape = None
        self.jaxpr = None

    def compile(self, jaxpr, sample_args):
        """Compile a Jaxpr: run it once to capture a TT-NN trace.

        sample_args: list of numpy arrays with the shapes that will be used.
        """
        self.jaxpr = jaxpr
        self.output_shape = jaxpr.jaxpr.outvars[0].aval.shape

        # Create input tensors on device (these will be overwritten before replay)
        self.input_tensors = []
        for var, arg in zip(jaxpr.jaxpr.invars, sample_args):
            t = tensors.to_device(arg, self.device)
            self.input_tensors.append(t)

        # Dry run to warm up (ensures kernels are compiled/cached)
        interp = Interpreter(self.device)
        interp.env.clear()
        for var, t in zip(jaxpr.jaxpr.invars, self.input_tensors):
            interp.env[var] = t
        for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            interp.env[var] = tensors.to_device(const, self.device)
        for eqn in jaxpr.jaxpr.eqns:
            interp._exec(eqn)
        ttnn.synchronize_device(self.device)

        # Capture trace
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        interp2 = Interpreter(self.device)
        interp2.env.clear()
        for var, t in zip(jaxpr.jaxpr.invars, self.input_tensors):
            interp2.env[var] = t
        for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            interp2.env[var] = tensors.to_device(const, self.device)
        for eqn in jaxpr.jaxpr.eqns:
            interp2._exec(eqn)

        # Save output tensor reference
        self.output_tensor = interp2.eval_var(jaxpr.jaxpr.outvars[0])

        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

    def run(self, args):
        """Execute the compiled trace with new input data."""
        if self.trace_id is None:
            raise RuntimeError("Must call compile() before run()")

        # Overwrite input buffers with new data
        for i, arg in enumerate(args):
            new_t = ttnn.from_torch(
                _prepare_torch(arg),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(new_t, self.input_tensors[i])

        # Replay trace
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)

        # Read output
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
