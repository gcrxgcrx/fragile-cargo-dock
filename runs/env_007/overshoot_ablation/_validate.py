"""Validate the overshoot-ablation variants: parse + expected edits present."""

import ast
import sys
from pathlib import Path

BASE = Path("runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py")
OUT = Path("runs/env_007/overshoot_ablation")

EXPECT = {
    "o00_base": {"COPY"},
    "o01_overshoot": {"ov"},
    "o02_gentle": {"gen"},
    "o03_both": {"ov", "gen"},
    "o04_overshoot_x10": {"ov10"},
}

base_text = BASE.read_text(encoding="utf-8")
ok = True
for name, expected in EXPECT.items():
    text = (OUT / name / "reward_v1.py").read_text(encoding="utf-8")
    ast.parse(text)
    body = text.split("def compute_reward", 1)[1]
    flags = set()
    if text.endswith(base_text):
        flags.add("COPY")
    if 'components["OVERSHOOT_penalty"] = float(-20.0' in body:
        flags.add("ov")
    if 'components["OVERSHOOT_penalty"] = float(-200.0' in body:
        flags.add("ov10")
    if 'components["REPAIR_gentleness"] = float(' in body:
        flags.add("gen")
    good = flags == expected
    ok = ok and good
    print(f"{name:<18} parses  flags={sorted(flags)} expected={sorted(expected)} "
          f"{'OK' if good else 'MISMATCH'}")

sys.exit(0 if ok else 1)
