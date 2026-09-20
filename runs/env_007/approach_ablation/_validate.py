"""Validate the approach-ablation variants: parse + expected edits present."""

import ast
import sys
from pathlib import Path

BASE = Path("runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py")
OUT = Path("runs/env_007/approach_ablation")

EXPECT = {
    "p01_proximity": {"prox"},
    "p02_progress_x200": {"x200"},
    "p03_x200_gentle": {"x200", "gen"},
    "p04_proximity_gentle": {"prox", "gen"},
    "p05_funnel": {"funnel"},
}

base_text = BASE.read_text(encoding="utf-8")
ok = True
for name, expected in EXPECT.items():
    text = (OUT / name / "reward_v1.py").read_text(encoding="utf-8")
    ast.parse(text)
    body = text.split("def compute_reward", 1)[1]
    flags = set()
    if "2400.0 * progress" in body:
        flags.add("x200")
    if 'components["APPROACH_proximity"] = float(' in body:
        flags.add("prox")
    if 'components["APPROACH_funnel"] = float(' in body:
        flags.add("funnel")
    if 'components["REPAIR_gentleness"] = float(' in body:
        flags.add("gen")
    good = flags == expected
    ok = ok and good
    print(f"{name:<22} parses  flags={sorted(flags)} expected={sorted(expected)} "
          f"{'OK' if good else 'MISMATCH'}")

print(f"\nbase (r09) untouched: {BASE.read_text(encoding='utf-8') == base_text}")
sys.exit(0 if ok else 1)
