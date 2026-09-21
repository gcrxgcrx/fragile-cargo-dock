"""Mechanism readout for a trained policy: how close to the 10-step hold does it get?

`eval_pool.py` reports success and dock rate. This adds the quantity that explains the gap:
the **maximum consecutive stable-step count** reached in each episode (success needs 10), plus
the docking-entry speed and the final crate-to-dock distance. It is what separates "reaches
the dock but cannot settle" from "never gets there", and it is the readout that the
close-speed isolation experiment needs (`runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md`).

Usage:
    python diagnose_settling.py --episodes 60 --seed-offset 37000 RUN_DIR [RUN_DIR ...]
    python diagnose_settling.py --episodes 60 --seed-offset 37000 --out runs/env_007/x.json DIR...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
import gymnasium as gym

from run_fragilecargo_baseline import ENV_ID, load_trained_policy

NORM_EPS = 1e-8
OBS_CLIP = 10.0


def run_episode(env, policy, seed, obs_rms):
    obs, info = env.reset(seed=seed)
    steps = 0
    max_stable = 0
    entry_speed = None
    entered = False
    while True:
        policy_obs = obs
        if obs_rms is not None:
            policy_obs = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + NORM_EPS), -OBS_CLIP, OBS_CLIP
            ).astype(np.float32)
        action = policy(policy_obs, info, env)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        s = int(info.get("stable_steps", 0))
        if s > max_stable:
            max_stable = s
        if not entered and info.get("cargo_inside_dock"):
            entered = True
            entry_speed = float(info.get("cargo_speed", float("nan")))
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "reason": str(info.get("termination_reason")),
        "steps": steps,
        "max_stable": max_stable,
        "entered": entered,
        "entry_speed": entry_speed,
        "final_dist": float(info.get("cargo_goal_distance", float("nan"))),
        "final_cargo_speed": float(info.get("cargo_speed", float("nan"))),
        "hard_collisions": int(info.get("hard_collision_count", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=37000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = gym.make(ENV_ID)
    rows = []
    print(f"{'run':<34} {'succ':>7} {'dock':>6} {'max_stable (mean/median/p90)':>30} "
          f"{'entry_v':>8} {'final_d':>8} {'col':>4}")
    print("-" * 106)
    for d in args.run_dirs:
        model = Path(d) / "model.zip"
        vn = Path(d) / "vecnormalize.pkl"
        policy, obs_rms = load_trained_policy(str(model), str(vn) if vn.exists() else None)
        recs = [run_episode(env, policy, args.seed_offset + i, obs_rms)
                for i in range(args.episodes)]
        st = [r["max_stable"] for r in recs]
        succ = sum(r["success"] for r in recs)
        dock = sum(r["entered"] for r in recs) / len(recs)
        speeds = [r["entry_speed"] for r in recs if r["entry_speed"] is not None]
        row = {
            "run_dir": d,
            "episodes": len(recs),
            "success": succ,
            "dock_rate": dock,
            "max_stable_mean": float(statistics.mean(st)),
            "max_stable_median": float(statistics.median(st)),
            "max_stable_p90": float(np.percentile(st, 90)),
            "max_stable_hist": dict(sorted(Counter(st).items())),
            "entry_speed_median": float(np.median(speeds)) if speeds else None,
            "final_distance_median": float(np.median([r["final_dist"] for r in recs])),
            "mean_hard_collisions": float(np.mean([r["hard_collisions"] for r in recs])),
            "reasons": dict(Counter(r["reason"] for r in recs)),
            "records": recs,
        }
        rows.append(row)
        print(f"{Path(d).name:<34} {succ:>4}/{len(recs):<2} {dock:>6.2f} "
              f"{row['max_stable_mean']:>10.2f} / {row['max_stable_median']:>5.1f} / "
              f"{row['max_stable_p90']:>5.1f}      "
              f"{(row['entry_speed_median'] if row['entry_speed_median'] is not None else float('nan')):>8.3f} "
              f"{row['final_distance_median']:>8.3f} {row['mean_hard_collisions']:>4.2f}")
    env.close()

    print()
    print("=== how many of the docked-and-failed episodes got within 3 steps of success? ===")
    for row in rows:
        close = sum(1 for r in row["records"] if r["entered"] and 7 <= r["max_stable"] <= 9)
        docked_failed = sum(1 for r in row["records"] if r["entered"] and not r["success"])
        print(f"  {Path(row['run_dir']).name:<34} docked-but-failed {docked_failed:>3}"
              f"   of which reached 7-9 consecutive stable steps: {close:>3}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
