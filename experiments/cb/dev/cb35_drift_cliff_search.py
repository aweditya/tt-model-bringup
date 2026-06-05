"""Linear-search probe to localize the pos1→pos5 drift cliff.

Step 1 of `research/35b_drift_next_session_plan.md` §"What to do next".

Baseline data (`archive/cb35_drift_wrappers_2026-06-04/cb35_drift_long_bf16.py`
with sparse positions 0,1,5,10,25,40,60,80) showed cos_L32 = 0.99 @ pos 1
→ 0.32 @ pos 5. This probe linearly scans 0..7 to pinpoint the position
where the cliff lands (the first pos where cos_L32 drops below 0.95).

Result lands as JSON in `.cache/cb35_runtime/` and `last.log`.
Headline metric: `P_cliff` = smallest pos with cos_L32 < 0.95.

Configuration is identical to the archived `cb35_drift_long_bf16.py`
(owned_gdn=ON, owned_decay_gate=ON, bf16 DN state, long ladder oracle);
only `CB35_LADDER_POSITIONS` changes.

NOTE 2026-06-04: 35B drift cliff RESOLVED per
`feedback_35b_drift_resolved_2026-06-04`; this probe is kept as a
regression-watch tool in case the cliff returns.
"""
import os
from cb35_drift_ladder import main as _main

def main(state):
    os.environ["CB35_DN_DTYPE"] = "bf16"
    os.environ["CB35_OWNED_GDN"] = "on"
    os.environ["CB35_OWNED_DECAY_GATE"] = "on"
    os.environ["CB35_ORACLE_DIR"] = ".cache/hf_oracle_35b_long"
    os.environ["CB35_LADDER_POSITIONS"] = "0,1,2,3,4,5,6,7"
    return _main(state)
