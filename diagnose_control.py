"""Why does the hand-written control plateau at ~7 % ?

Runs the trained control policy and records, per episode, how far into the
ten-step settling requirement it actually got (`info["stable_steps"]` is
exposed by the raw environment), plus the crate speed at its closest approach
to the dock. That separates "never gets the crate inside the dock" from
"gets inside but never gets it slow".

Usage: python diagnose_control.py RUN_DIR [--episodes 60] [--seed-offset 30000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import custom_envs.registration  # noqa: F401
from run_fragilecargo_baseline import ENV_ID, load_trained_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=30000)
    args = ap.parse_args()

    d = Path(args.run_dir)
    policy, obs_rms = load_trained_policy(str(d / "model.zip"), str(d / "vecnormalize.pkl"))

    import gymnasium as gym
    env = gym.make(ENV_ID)

    rows = []
    for i in range(args.episodes):
        seed = args.seed_offset + i
        obs, info = env.reset(seed=seed)
        best_stable = 0
        docked_steps = 0
        settled_steps = 0
        speed_at_entry = None
        min_dist = 1e9
        entered = False
        while True:
            p_obs = obs
            if obs_rms is not None:
                p_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype(np.float32)
            action = policy(p_obs, info, env)
            obs, _r, terminated, truncated, info = env.step(action)
            if info["cargo_inside_dock"]:
                docked_steps += 1
                entered = True
                if speed_at_entry is None:
                    speed_at_entry = float(info["cargo_speed"])
            if info["cargo_inside_dock"] and info["cargo_speed"] < 0.05:
                settled_steps += 1
            best_stable = max(best_stable, int(info["stable_steps"]))
            min_dist = min(min_dist, float(info["cargo_goal_distance"]))
            if terminated or truncated:
                break
        rows.append(dict(seed=seed, success=bool(info["is_success"]),
                         best_stable=best_stable, docked_steps=docked_steps,
                         settled_steps=settled_steps, entered=entered,
                         speed_at_entry=speed_at_entry, min_dist=min_dist,
                         reason=info["termination_reason"]))
    env.close()

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    entered = sum(r["entered"] for r in rows)
    print(f"episodes                 : {n}")
    print(f"success                  : {succ}/{n} = {100*succ/n:.1f} %")
    print(f"entered the dock at all  : {entered}/{n} = {100*entered/n:.1f} %")
    print(f"reasons                  : {dict(Counter(r['reason'] for r in rows))}")
    print()
    print("best consecutive settled steps reached, over episodes:")
    buckets = Counter()
    for r in rows:
        b = r["best_stable"]
        buckets["0 (never settled)" if b == 0 else
                "1-4" if b < 5 else "5-9" if b < 10 else "10 (success)"] += 1
    for k in ("0 (never settled)", "1-4", "5-9", "10 (success)"):
        print(f"  {k:<20} {buckets.get(k,0):>3} / {n}")
    mean_bs = sum(r["best_stable"] for r in rows) / n
    print(f"  mean best_stable       {mean_bs:.2f}   (needs 10)")
    sp = [r["speed_at_entry"] for r in rows if r["speed_at_entry"] is not None]
    if sp:
        print(f"\ncrate speed when first fully inside the dock (needs < 0.05):")
        print(f"  mean {sum(sp)/len(sp):.3f}  min {min(sp):.3f}  max {max(sp):.3f} m/s")
    md = [r["min_dist"] for r in rows]
    print(f"\nclosest crate-to-dock distance: mean {sum(md)/n:.3f} m, best {min(md):.3f} m")
    ds = [r["docked_steps"] for r in rows]
    ss = [r["settled_steps"] for r in rows]
    print(f"steps spent fully inside the dock   : mean {sum(ds)/n:.1f}  (max {max(ds)})")
    print(f"steps spent inside AND slower than 0.05 m/s: mean {sum(ss)/n:.1f}  (max {max(ss)})")
    worst = sum(1 for r in rows if r["settled_steps"] >= 5 and r["best_stable"] < 10)
    print(f"episodes with >=5 settled steps but never 10 consecutive: {worst}/{n}")


if __name__ == "__main__":
    main()
