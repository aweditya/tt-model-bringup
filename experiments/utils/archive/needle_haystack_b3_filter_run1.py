#!/usr/bin/env python3
"""One-shot helper: filter results.json down to only successful trials so the
B3 needle probe can be resumed without redoing the 4 already-Y cells.

Reads ~/tt-xla/.cache/needle_haystack_b3/results.json, drops any trial whose
score is "ERR", writes the result back. Original is preserved as
results_run1.json (separate copy step).
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/tt-xla"))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PATH = os.path.expanduser("~/tt-xla/.cache/needle_haystack_b3/results.json")

with open(PATH) as f:
    state = json.load(f)

kept = []
dropped = 0
for c in state["results"]:
    if c.get("skipped"):
        continue
    if c.get("score") in ("Y", "P", "N"):
        kept.append(c)
    else:
        dropped += 1

state["results"] = kept
state.pop("end_unix", None)
state.pop("wall_sec", None)
state["aborted_reason"] = None
with open(PATH, "w") as f:
    json.dump(state, f, indent=2)

print(f"kept {len(kept)} trials, dropped {dropped}")
for c in kept:
    print(f"  L={c['L_target']} frac={c['frac']} trial={c['trial']} "
          f"score={c['score']} needle={c['needle']}")
