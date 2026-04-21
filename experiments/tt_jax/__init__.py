"""
tt_jax: A trace-based JAX backend for Tenstorrent Blackhole.

This module implements a "Level 1" JAX backend that:
  1. Takes a Jaxpr (JAX's intermediate representation)
  2. Maps each primitive to a TT-NN operation
  3. Optionally wraps execution in TT-NN trace capture for 3x+ speedup

Architecture:
  ops.py       — Op registry: maps Jaxpr primitive names → TT-NN callables
  tensors.py   — Tensor utilities: padding, dtype, host↔device transfer
  interpret.py — Jaxpr walker: evaluates a Jaxpr using the op registry
  trace.py     — Trace capture: wraps interpreter for compiled execution
"""

from .interpret import Interpreter
from .trace import TracedExecutor
