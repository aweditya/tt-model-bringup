"""Long-context drift ladder — H1: fp32 H_t. owned_gdn auto-disabled."""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "fp32"
    os.environ["CB35_OWNED_GDN"] = "off"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"
    os.environ["CB35_ORACLE_DIR"] = ".cache/hf_oracle_35b_long"
    os.environ["CB35_LADDER_POSITIONS"] = "0,1,5,10,25,40,60,80"
    return _main(state)
