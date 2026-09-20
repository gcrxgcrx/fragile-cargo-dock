"""Decompose the *native* reward on scripted trajectories, to see why it learns.

The native reward reaches 96.8 % / 90 %-through-the-wrapper while every LLM reward is
0-5 % and the obs-only hand-written probes are 40-77 %. This script reconstructs the
native reward's own term decomposition from the logged observations/actions of the
scripted-trajectory library, and answers four questions with numbers:

  1. how large is the *shaping* (non-terminal) return, and is it bounded by geometry
     (a telescoping potential) or by the horizon (a farmable integral)?
  2. how large is the terminal return, before and after the harness clip of 20?
  3. what is the ratio of shaping to terminal?
  4. what per-step magnitudes does the policy actually see?

Reconstruction note: `roughness` needs the contact impulse (present only in `info`,
which generated rewards cannot read), so it is reconstructed as 0 and the residual
against the true native return is reported; the residuals are the roughness + hard_hit
terms and are reported explicitly.

Usage:
    python analyze_native_reward.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

W = {"approach_cargo": 1.0, "progress": 1.0, "dock_enter": 5.0,
     "action_cost": 0.0005, "time_cost": 0.002}


def local_xy(obs):
    """(x, y) in metres from a normalised observation pair (dx, dy, half_extent)."""
    return obs[..., 12] * 5.0, obs[..., 13] * 4.0


def decompose(traj):
    obs, nxt, act = traj["obs"], traj["nxt"], traj["act"]
    ox, oy = local_xy(obs)
    nx, ny = local_xy(nxt)
    d_prev = np.hypot(ox, oy)
    d_now = np.hypot(nx, ny)
    progress = d_prev - d_now                       # metres closed this step

    rx, ry = obs[:, 6] * 3.0, obs[:, 7] * 3.0
    nrx, nry = nxt[:, 6] * 3.0, nxt[:, 7] * 3.0
    approach = np.hypot(rx, ry) - np.hypot(nrx, nry)

    inside = (np.abs(nxt[:, 12]) <= 0.024) & (np.abs(nxt[:, 13]) <= 0.030)
    seen = False
    enter = np.zeros(len(inside))
    for i, ins in enumerate(inside):
        if ins and not seen:
            seen = True
            enter[i] = 1.0

    terms = {
        "approach_cargo": W["approach_cargo"] * approach,
        "progress": W["progress"] * progress,
        "dock_enter": W["dock_enter"] * enter,
        "action_cost": -W["action_cost"] * np.sum(act ** 2, axis=1),
        "time_cost": -W["time_cost"] * np.ones(len(act)),
    }
    terminal = traj["native"].copy()
    for v in terms.values():
        terminal = terminal - v
    terms["terminal+roughness+hard_hit (residual)"] = terminal
    return terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--clip", type=float, default=20.0)
    args = ap.parse_args()

    library = trc.build_library(args.seeds)
    print(f"library: {len(library)} trajectories, "
          f"{sum(t['success'] for t in library)} successful\n")

    names = ["approach_cargo", "progress", "dock_enter", "action_cost", "time_cost",
             "terminal+roughness+hard_hit (residual)"]
    hdr = f"{'controller':<15} {'succ':>5} {'steps':>6} " + " ".join(f"{n[:11]:>12}" for n in names)
    print(hdr)
    print("-" * len(hdr))

    agg = []
    for traj in library:
        t = decompose(traj)
        row = [float(np.sum(t[n])) for n in names]
        agg.append((traj, row))
        print(f"{traj['controller']:<15} {str(traj['success']):>5} {len(traj['obs']):>6} "
              + " ".join(f"{v:>12.3f}" for v in row))

    print()
    print("=== per controller (mean over seeds) ===")
    for ctrl in trc.CONTROLLERS:
        rows = [(traj, row) for traj, row in agg if traj["controller"] == ctrl]
        if not rows:
            continue
        m = np.mean([r for _t, r in rows], axis=0)
        succ = np.mean([t["success"] for t, _r in rows])
        steps = np.mean([len(t["obs"]) for t, _r in rows])
        shaping = m[0] + m[1] + m[2] + m[3] + m[4]
        print(f"  {ctrl:<15} success={succ:>4.0%} steps={steps:>6.1f} "
              f"shaping={shaping:>8.3f}  ({', '.join(f'{n.split(chr(43))[0]}={v:+.3f}' for n, v in zip(names[:5], m[:5]))})"
              f"  residual={m[5]:>+8.3f}")

    print()
    print("=== the two questions that matter ===")
    shape_tot = []
    for traj, row in agg:
        shape_tot.append(sum(row[:5]))
    print(f"  shaping total over ALL {len(agg)} trajectories: "
          f"min={min(shape_tot):+.3f} max={max(shape_tot):+.3f}")
    succ = [(traj, row) for traj, row in agg if traj["success"]]
    if succ:
        for traj, row in succ[:3]:
            st = sum(row[:5])
            print(f"  successful traj ({traj['controller']}, {len(traj['obs'])} steps): "
                  f"shaping={st:+.3f}  terminal_residual={row[5]:+.3f}  "
                  f"native_total={traj['oracle_return']:+.3f}")
    print()
    print("  under the harness per-step clip of "
          f"{args.clip}, the largest single-step native reward is "
          f"{max(float(np.max(traj['native'])) for traj, _r in agg):.3f}")


if __name__ == "__main__":
    main()
