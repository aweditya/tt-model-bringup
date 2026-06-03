"""Drift ladder — fp32 H_t + owned_decay_gate also disabled.

Tests whether decay quantization (via the owned_decay_gate kernel
write-back) is an additive contributor on top of H1.
"""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "fp32"
    os.environ["CB35_OWNED_GDN"] = "off"
    os.environ["CB35_OWNED_DECAY_GATE"] = "off"
    return _main(state)
