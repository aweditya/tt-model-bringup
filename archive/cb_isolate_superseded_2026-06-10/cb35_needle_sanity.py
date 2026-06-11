"""Sanity sweep of 35B needle haystack — multi-L, multi-trial.

Forks `cb35_needle_haystack.py` to set the env so the underlying test
runs a finer sweep:
  - Lengths: 100, 200, 300, 460
  - Trials: 2 per (L, frac) to detect single-seed flukes
  - Frac: 0.5 (mid-haystack)

After the 1-trial run showed L=100 N / L=460 Y / L=1024 N, we want to
know whether L=100 N is deterministic across seeds or an outlier.
"""
from __future__ import annotations

import os

from cb35_needle_haystack import main as _main


def main(state):
    os.environ["CB35_NEEDLE_LENGTHS"] = "100,200,300,460"
    os.environ["CB35_NEEDLE_FRAC"] = "0.5"
    os.environ["CB35_NEEDLE_TRIALS"] = "2"
    os.environ["CB35_NEEDLE_MAX_GEN"] = "24"
    return _main(state)
