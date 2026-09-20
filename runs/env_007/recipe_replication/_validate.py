"""Validate the recipe-replication variants: parse + expected edits present."""

import ast
import sys
from pathlib import Path

BASE = Path("runs/env_007/prompt_ladder/L2/cand_14/reward_v1.py")
OUT = Path("runs/env_007/recipe_replication")

EXPECT = {
    "g00_copy": set(),
    "g01_stream": {"stream"},
    "g02_recipe": {"x50", "stream"},
}

base_text = BASE.read_text(encoding="utf-8")
ok = True
for name, expected in EXPECT.items():
    text = (OUT / name / "reward_v1.py").read_text(encoding="utf-8")
    ast.parse(text)
    body = text.split("def compute_reward", 1)[1]
    flags = set()
    if "300.0 * progress" in body:
        flags.add("x50")
    if 'components["REPAIR_settled_stream"] = float(20.0)' in body:
        flags.add("stream")
    body_identical = text.endswith(base_text)
    good = flags == expected and (body_identical == (name == "g00_copy"))
    ok = ok and good
    print(f"{name:<12} parses  flags={sorted(flags)} expected={sorted(expected)} "
          f"body==base={body_identical}  {'OK' if good else 'MISMATCH'}")

sys.exit(0 if ok else 1)
