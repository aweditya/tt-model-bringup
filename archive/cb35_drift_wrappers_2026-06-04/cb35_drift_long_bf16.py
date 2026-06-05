"""Long-context drift ladder — bf16 baseline (85-position prompt).

Tests whether drift compounds across positions in the real
user-facing regime (drift becomes visible at pos 25+ per memory).
"""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "bf16"
    os.environ["CB35_OWNED_GDN"] = "on"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"
    os.environ["CB35_ORACLE_DIR"] = ".cache/hf_oracle_35b_long"
    os.environ["CB35_LADDER_POSITIONS"] = "0,1,5,10,25,40,60,80"
    return _main(state)
