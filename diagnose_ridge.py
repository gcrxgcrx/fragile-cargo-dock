"""Artifact checks for the jagged ridge result.

Four reward files differ only in one coefficient, yet score 16 / 0 / 31 / 0 out of 60. Before
believing in a jagged surface, rule out the boring explanations:

 1. did my generator change anything besides that one number? (diff each file against the
    byte-identical baseline copy)
 2. did any run silently take an error/fallback path, or train with a different clip, env count,
    budget or eval configuration?
 3. did any run produce reward errors during training (reward_error_count_max), or fail the final
    component evaluation?
 4. do the runs' own training curves look like different regimes, or like the same run with noise?
 5. are there stderr contents for any of them?

Run: python diagnose_ridge.py
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

ARMS = {
    "x50 (16/60)": "runs/env_007/overshoot_ablation/o00_base",
    "x75 (0/60)": "runs/env_007/ridge_width/q01_progress_x75",
    "x100 (31/60)": "runs/env_007/ridge_width/q02_progress_x100",
    "x200 (0/60)": "runs/env_007/approach_ablation/p02_progress_x200",
}
BASE = Path(ARMS["x50 (16/60)"])

# ---------------------------------------------------------------- 1. file diffs
print("=" * 100)
print("1. REWARD-FILE DIFF vs the baseline arm (only the coefficient line should differ)")
print("=" * 100)
base_text = (BASE / "reward_v1.py").read_text(encoding="utf-8")
for label, d in ARMS.items():
    p = Path(d) / "reward_v1.py"
    text = p.read_text(encoding="utf-8")
    diff = [ln for ln in difflib.unified_diff(base_text.splitlines(), text.splitlines(),
                                              lineterm="", n=0)
            if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    print(f"\n  {label:<14} {p}")
    if not diff:
        print("      (identical to baseline)")
    for ln in diff:
        print(f"      {ln}")

# ---------------------------------------------------------------- 2/3. summary fields
print()
print("=" * 100)
print("2/3. TRAINING-SUMMARY FIELDS (silent fallbacks, clip, envs, budget, eval config, errors)")
print("=" * 100)
for label, d in ARMS.items():
    sp = Path(d) / "training_summary.json"
    if not sp.exists():
        print(f"\n  {label}: no training_summary.json")
        continue
    s = json.loads(sp.read_text(encoding="utf-8"))
    ext = s.get("external_eval", {}) or {}
    cs = s.get("component_summary", {}) or {}
    print(f"\n  {label}")
    print(f"      n_envs={s.get('n_envs')}  reward_clip={s.get('reward_clip')}  "
          f"total_timesteps={s.get('total_timesteps')}  error_fallback={s.get('error_fallback')}")
    print(f"      reward_source={s.get('reward_source')}")
    print(f"      reward_error_count_max={cs.get('reward_error_count_max')}  "
          f"final_component_errors={ext.get('final_policy_component_error_count')}")
    print(f"      eval episodes={ext.get('eval_episodes')} seed_offset={ext.get('eval_seed_offset')}  "
          f"termination={ext.get('termination_breakdown')}")
    print(f"      mean_eval_reward={ext.get('mean_eval_reward')}")

# ---------------------------------------------------------------- 4. training curves
print()
print("=" * 100)
print("4. TRAINING CURVES (monitor_snapshots: steps, episodes, mean generated reward)")
print("=" * 100)
for label, d in ARMS.items():
    sp = Path(d) / "training_summary.json"
    if not sp.exists():
        continue
    s = json.loads(sp.read_text(encoding="utf-8"))
    snaps = s.get("monitor_snapshots") or []
    pts = [(sn.get("steps"), sn.get("episodes"), sn.get("mean_generated_reward")) for sn in snaps]
    print(f"\n  {label}  ({len(snaps)} checkpoints)")
    for steps, eps, mr in pts[::max(1, len(pts) // 6)] if pts else []:
        print(f"      step {steps:>9}  eps {eps:>4}  mean_generated_reward {mr}")

# ---------------------------------------------------------------- 5. stderr
print()
print("=" * 100)
print("5. STDERR / MONITOR FILES")
print("=" * 100)
logs = Path("runs/env_007")
for label, d in ARMS.items():
    name = Path(d).name
    errs = list(logs.glob(f"train_{name}*.err.log"))
    for e in errs:
        print(f"  {label:<14} {e.name}: {e.stat().st_size} bytes")
    mons = list((Path(d) / "monitor").glob("*")) if (Path(d) / "monitor").exists() else []
    sizes = [m.stat().st_size for m in mons if m.is_file()]
    print(f"  {label:<14} monitor files: {len(sizes)}  total {sum(sizes)} bytes")
