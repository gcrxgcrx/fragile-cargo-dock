"""Score trained policies on FRESH evaluation seeds.

The 20 evaluation seeds used inside training (10000..10019) are the ones the
reward search itself selects on, so a score measured there is optimistic. This
script re-scores a trained (model, vecnormalize) pair on seeds the search never
saw, which is the only honest way to compare two candidates at success rates of
a few percent.

Usage
-----
    python eval_fresh_seeds.py --episodes 60 --seed-offset 30000 \
        runs/env_007/terminal_rule_pilot/seed_0/gen_00/cand_03/training \
        runs/env_007/terminal_rule_pilot_clip600/seed_0/gen_00/cand_03/training
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
from run_fragilecargo_baseline import ENV_ID, evaluate, load_trained_policy, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="dirs containing model.zip / vecnormalize.pkl")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=30000)
    args = ap.parse_args()

    print(f"fresh seeds: {args.seed_offset}..{args.seed_offset + args.episodes - 1}"
          f"  ({args.episodes} episodes)")
    for run_dir in args.run_dirs:
        d = Path(run_dir)
        model_path = d / "model.zip"
        vec_path = d / "vecnormalize.pkl"
        if not model_path.exists():
            print(f"\n{d}: no model.zip, skipped")
            continue
        policy, obs_rms = load_trained_policy(str(model_path),
                                              str(vec_path) if vec_path.exists() else None)
        records = evaluate(ENV_ID, policy, args.episodes, args.seed_offset, obs_rms=obs_rms)
        s = summarize("fresh", records)
        succ = sum(1 for r in records if r["success"])
        reasons = Counter(r["reason"] for r in records)
        returns = [r["return"] for r in records]
        print(f"\n=== {d} ===")
        print(f"  mean native return : {s['mean_return']:.3f}")
        print(f"  success            : {succ}/{args.episodes}  ({100.0 * succ / args.episodes:.1f}%)")
        print(f"  best / worst return: {max(returns):.1f} / {min(returns):.1f}")
        print(f"  reasons            : {dict(reasons)}")
        print(f"  dock_entered       : {sum(1 for r in records if r['dock_entered'])}/{args.episodes}")
        fd = [r["final_goal_distance"] for r in records]
        print(f"  final goal dist    : mean {sum(fd) / len(fd):.3f} m, best {min(fd):.3f} m")


if __name__ == "__main__":
    main()
