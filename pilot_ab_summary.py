"""Summarise the clip A/B probe: same reward files, two per-step clips.

Usage:  python pilot_ab_summary.py
"""

from __future__ import annotations

import glob
import json
import os

ARMS = [
    ("clip=20   (harness default)", "runs/env_007/terminal_rule_pilot"),
    ("clip=600  (non-binding)", "runs/env_007/terminal_rule_pilot_clip600"),
]


def main() -> None:
    for label, root in ARMS:
        print("=" * 78)
        print(label)
        print("=" * 78)
        pattern = os.path.join(root, "seed_0", "gen_00", "cand_*", "training",
                               "training_summary.json")
        rows = sorted(glob.glob(pattern))
        if not rows:
            print("  (still training / no summaries yet)")
            print()
            continue
        print(f"  {'cand':<8} {'mean':>10} {'max':>10} {'succ':>7} {'len':>6}   breakdown")
        for path in rows:
            data = json.load(open(path, encoding="utf-8"))
            ev = data.get("external_eval", {})
            rewards = ev.get("episode_rewards", [])
            terminated = ev.get("episode_terminated", [])
            succ = sum(1 for r, t in zip(rewards, terminated) if t and r > 0)
            crack = ev.get("termination_breakdown", {})
            name = path.split(os.sep)[-3]
            print(f"  {name:<8} {ev.get('mean_eval_reward', float('nan')):>10.3f} "
                  f"{ev.get('max_eval_reward', float('nan')):>10.3f} "
                  f"{succ:>3}/{len(rewards):<3} "
                  f"{ev.get('mean_episode_length', 0):>6.0f}   {crack}")
        print()


if __name__ == "__main__":
    main()
