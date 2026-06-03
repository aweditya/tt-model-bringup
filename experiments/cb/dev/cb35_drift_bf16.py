"""Drift ladder — bf16 baseline (default config). H0 = "no drift fix"."""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "bf16"
    os.environ["CB35_OWNED_GDN"] = "on"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"
    return _main(state)
