"""Mechanical validation of the repair variants: they must parse, and each flag must show up.

Run:  python runs/env_007/repair_test/_validate_variants.py
"""

import ast
import sys
from pathlib import Path

BASE = Path("runs/env_007/prompt_ladder/L2/cand_00/reward_v1.py")
OUT = Path("runs/env_007/repair_test")

EXPECT = {
    "r00_copy": {"BODY==BASE"},
    "r01_guidance": {"guidance"},
    "r02_guidance_gentle": {"guidance", "gentleness"},
    "r03_guidance_x3": {"guidance"},
    "r04_own_progress_x50": {"progress_x50"},
    "r05_guidance_no_events": {"guidance", "gentleness", "events_zeroed"},
    "r06_settled_stream": {"settled_stream"},
    "r07_stream_and_guidance": {"guidance", "gentleness", "settled_stream"},
    "r08_progress_x50_gentle": {"progress_x50", "gentleness"},
    "r09_progress_x50_stream": {"progress_x50", "settled_stream"},
    "r10_x50_stream_gentle": {"progress_x50", "settled_stream", "gentleness"},
    "r11_x50_own_speedpen_x100": {"progress_x50", "speed_pen_x100"},
}

base_text = BASE.read_text(encoding="utf-8")
ok = True
for name, expected in EXPECT.items():
    text = (OUT / name / "reward_v1.py").read_text(encoding="utf-8")
    ast.parse(text)
    body = text.split("def compute_reward", 1)[1]
    flags = set()
    if text.endswith(base_text):
        flags.add("BODY==BASE")
    if 'components["REPAIR_crate_progress"] = float(' in body:
        flags.add("guidance")
    if 'components["REPAIR_gentleness"] = float(' in body:
        flags.add("gentleness")
    if 'components["success_event"] = 0.0' in body:
        flags.add("events_zeroed")
    if "600.0 * progress" in body:
        flags.add("progress_x50")
    if 'components["REPAIR_settled_stream"] = float(' in body:
        flags.add("settled_stream")
    if "-300.0 * near_dock * over_speed" in body:
        flags.add("speed_pen_x100")
    good = flags == expected
    ok = ok and good
    print(f"{name:<26} parses  flags={sorted(flags)}  expected={sorted(expected)}  "
          f"{'OK' if good else 'MISMATCH'}")

r01 = (OUT / "r01_guidance" / "reward_v1.py").read_text(encoding="utf-8")
r03 = (OUT / "r03_guidance_x3" / "reward_v1.py").read_text(encoding="utf-8")
w_ok = "float(1.0 *" in r01 and "float(3.0 *" in r03
print(f"\nr01 weight 1.0 / r03 weight 3.0 present: {w_ok}")
sys.exit(0 if (ok and w_ok) else 1)
