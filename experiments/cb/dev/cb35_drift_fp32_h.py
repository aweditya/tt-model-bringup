"""Drift ladder — fp32 H_t (H1 from research). owned_gdn auto-disabled."""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "fp32"
    os.environ["CB35_OWNED_GDN"] = "off"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"   # ignored at fp32; left for symmetry
    return _main(state)
