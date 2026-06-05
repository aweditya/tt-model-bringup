"""Drift ladder — bf16 state but force manual recurrence (owned_gdn off).

Discriminates between "manual recurrence path is broken" vs
"fp32 typecast chain is broken" — fp32 H_t needs both manual + fp32,
so seeing the regression at pos 0 with bf16 + manual proves the
manual path is the bug.
"""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "bf16"
    os.environ["CB35_OWNED_GDN"] = "off"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"
    os.environ["CB35_ORACLE_DIR"] = ".cache/hf_oracle_35b_long"
    os.environ["CB35_LADDER_POSITIONS"] = "0,1,5,10,25,40,60,80"
    return _main(state)
