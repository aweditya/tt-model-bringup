#!/usr/bin/env python3
"""MM7 v0.4.1.g — Nemotron-3 needle test with MAX_NEW=120.

v0.4.1.f at MAX_NEW=24 produced 0/4 retrieval but outputs were
coherent ("The user wants to..." / "The question is:..."), suggesting
the model is reasoning through the prompt rather than directly
echoing the password. Test the hypothesis by letting the reasoning
finish at MAX_NEW=120 tokens (~24s per trial at 0.2s/step).

Single L=128 trial — quick signal before deciding next step.

If retrieved within 120 tokens: prompt-shape issue (analogous to
Gemma 4 Round 9). The model works, our test format triggers
reasoning that runs past the 24-token cap.

If NOT retrieved: do a teacher-forced sanity check (compare per-
position logits against HF) to definitively isolate prompt vs
model-correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "cb" / "isolate"))


def main(state=None) -> int:
    import os
    # Override the constant before importing the baseline module.
    os.environ["NM3_NEEDLE_MAX_NEW"] = "120"
    import nemotron3_v041f_needle_baseline as base
    base.MAX_NEW = 120  # hard override in case env was already read
    return base.main(state=state)


if __name__ == "__main__":
    sys.exit(main())
