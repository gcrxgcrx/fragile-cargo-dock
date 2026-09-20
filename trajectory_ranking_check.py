"""Trajectory-ranking check: does a reward order *physically reachable* behaviours correctly?

Motivation
----------
`analyze_terminal_dominance.py` probes a reward on hand-built synthetic states.
Those states are not guaranteed to be reachable under the environment's dynamics,
so a reward can look right there and behave pathologically on the states a policy
actually visits. This script replaces point probes with **whole trajectories
produced by scripted controllers**, i.e. reachable observation sequences, and asks
a behavioural question instead of a structural one:

    does the candidate reward rank the trajectories the same way the task does?

The task's own ordering is the environment's native return over the trajectory
(the same quantity the paper's evaluator reports), so no new oracle is invented.

It also directly tests the prediction that the missing ingredient on
`FragileCargoDock-v0` is *closing speed at dock entry*: a reward that lacks that
signal should order "gentle entry" and "fast entry" trajectories inconsistently
with the task, and should show a non-positive margin on that specific contrast.

Usage
-----
    python trajectory_ranking_check.py --clip 20 REWARD.py [REWARD.py ...]
    python trajectory_ranking_check.py --clip 20 --library-cache lib.npz ...
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
import gymnasium as gym

import run_fragilecargo_baseline as baseline
from run_fragilecargo_baseline import heuristic_action, wrap_pi, _world_from_obs

ENV_ID = "FragileCargoDock-v0"

# controller name -> RELEASE_DIST (None = the idle controller, "shove" = full throttle)
CONTROLLERS = {
    "idle": None,
    "push_forever": 0.00,
    "release_0.05": 0.05,
    "release_0.15": 0.15,
    "release_0.25": 0.25,
    "release_0.35": 0.35,
    "release_0.50": 0.50,
    "release_0.80": 0.80,
    "shove": "shove",
}

SHOVE_TRIGGER_X = 2.00


def shove_action(obs, info, env):
    """Push at full throttle until the crate is nearly at the dock, then back away.

    This produces the high-dock-entry-speed failure the diagnosis points at; the
    release-based heuristic never does, because it always eases off in time.
    """
    robot_x, robot_y, heading, cargo_x, cargo_y = _world_from_obs(obs)
    if cargo_x < SHOVE_TRIGGER_X:
        want = np.arctan2(cargo_y - robot_y, cargo_x - robot_x)
        err = wrap_pi(want - heading)
        steer = float(np.clip(2.2 * err, -1.0, 1.0))
        throttle = 1.0 if abs(err) < 1.25 else 0.0
    else:
        throttle, steer = -1.0, 0.0
    return np.array([throttle, steer], dtype=np.float32)


def load_reward(path: str):
    spec = importlib.util.spec_from_file_location(f"rk_{abs(hash(path))}", str(Path(path)))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compute_reward


def build_library(seeds_per_controller: int = 6, seed_offset: int = 50000):
    """Roll out each scripted controller; keep the observation sequence and the
    environment's own return, which is the task's ordering oracle."""
    env = gym.make(ENV_ID)
    library = []
    for name, release in CONTROLLERS.items():
        for k in range(seeds_per_controller):
            seed = seed_offset + k
            obs, info = env.reset(seed=seed)
            obs_seq, nxt_seq, act_seq, rew_seq = [], [], [], []
            entry_speed = None
            max_stable = 0
            while True:
                if release is None:
                    action = np.zeros(2, dtype=np.float32)
                elif release == "shove":
                    action = shove_action(obs, info, env)
                else:
                    baseline.RELEASE_DIST = release
                    action = heuristic_action(obs, info, env)
                nxt, native_r, terminated, truncated, info = env.step(action)
                obs_seq.append(np.asarray(obs, dtype=np.float32))
                nxt_seq.append(np.asarray(nxt, dtype=np.float32))
                act_seq.append(np.asarray(action, dtype=np.float32))
                rew_seq.append(float(native_r))
                if entry_speed is None and info["cargo_inside_dock"]:
                    entry_speed = float(info["cargo_speed"])
                max_stable = max(max_stable, int(info["stable_steps"]))
                obs = nxt
                if terminated or truncated:
                    break
            library.append({
                "controller": name,
                "seed": seed,
                "obs": np.stack(obs_seq),
                "nxt": np.stack(nxt_seq),
                "act": np.stack(act_seq),
                "native": np.asarray(rew_seq, dtype=np.float64),
                "oracle_return": float(np.sum(rew_seq)),
                "success": bool(info["is_success"]),
                "entry_speed": entry_speed,
                "max_stable": max_stable,
                "reason": info["termination_reason"],
            })
    env.close()
    return library


def replay(reward_fn, traj, clip: float | None):
    """Return the generated (clipped) return of one trajectory."""
    total = 0.0
    for i in range(len(traj["obs"])):
        try:
            out = reward_fn(traj["obs"][i], traj["act"][i], traj["nxt"][i],
                            float(traj["native"][i]), {}, 0.0)
            val = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
        except Exception:  # noqa: BLE001
            val = 0.0
        if clip is not None:
            val = max(-clip, min(clip, val))
        total += val
    return total


def score(reward_path: str, library, clip: float | None):
    # a fresh module per trajectory, so reward-internal episode state starts clean
    returns = []
    for traj in library:
        returns.append(replay(load_reward(reward_path), traj, clip))
    returns = np.asarray(returns)
    oracle = np.asarray([t["oracle_return"] for t in library])
    succ = np.asarray([t["success"] for t in library])
    entered = np.asarray([t["entry_speed"] is not None for t in library])

    # The outcome-critical contrast: does the reward put every success above
    # every failure? "Closer to the dock" pairs are easy for any progress term
    # and are reported separately so they cannot drown this out.
    n = len(library)
    agree = total = 0.0
    for i in range(n):
        for j in range(n):
            if succ[i] and not succ[j]:
                total += 1
                d = returns[i] - returns[j]
                agree += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    succ_vs_fail = agree / total if total else float("nan")

    all_agree = all_tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy = oracle[i] - oracle[j]
            if dy == 0:
                continue
            dx = returns[i] - returns[j]
            all_tot += 1
            all_agree += 1.0 if (dx != 0 and (dy > 0) == (dx > 0)) else (0.5 if dx == 0 else 0.0)
    acc_all = all_agree / all_tot if all_tot else float("nan")

    r_succ = float(np.mean(returns[succ])) if succ.any() else float("nan")
    r_ent = float(np.mean(returns[entered & ~succ])) if (entered & ~succ).any() else float("nan")
    r_none = float(np.mean(returns[~entered])) if (~entered).any() else float("nan")
    worst_fail = np.nanmax([r_ent, r_none]) if (entered & ~succ).any() or (~entered).any() else float("nan")
    separation = (r_succ - worst_fail) if not np.isnan(r_succ) else float("nan")

    return {"succ_vs_fail": succ_vs_fail, "acc_all": acc_all, "separation": separation,
            "r_succ": r_succ, "r_ent_fail": r_ent, "r_no_entry": r_none,
            "n_succ": int(succ.sum()), "n_ent_fail": int((entered & ~succ).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewards", nargs="+")
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--seeds-per-controller", type=int, default=6)
    ap.add_argument("--cache", default=None, help="npz path to reuse a built library's labels")
    ap.add_argument("--describe", action="store_true",
                    help="print the library's composition (per controller) and exit")
    args = ap.parse_args()

    clip = None if args.clip < 0 else args.clip
    print(f"building trajectory library ({args.seeds_per_controller} seeds x "
          f"{len(CONTROLLERS)} controllers)...", flush=True)
    library = build_library(args.seeds_per_controller)

    if args.describe:
        print(f"\n{'controller':<16} {'n':>3} {'succ':>5} {'entered':>8} "
              f"{'entry_v':>8} {'oracle':>10} {'len':>6}  reasons")
        for name in CONTROLLERS:
            rows = [t for t in library if t["controller"] == name]
            ent = [t for t in rows if t["entry_speed"] is not None]
            ev = np.mean([t["entry_speed"] for t in ent]) if ent else float("nan")
            print(f"{name:<16} {len(rows):>3} {sum(t['success'] for t in rows):>5} "
                  f"{len(ent):>8} {ev:>8.3f} "
                  f"{np.mean([t['oracle_return'] for t in rows]):>10.2f} "
                  f"{np.mean([len(t['obs']) for t in rows]):>6.0f}  "
                  f"{sorted(set(t['reason'] for t in rows))}")
        succ = [t for t in library if t["success"]]
        print(f"\nsuccessful trajectories: {len(succ)}")
        for t in succ:
            print(f"   {t['controller']:<16} seed={t['seed']} entry_v={t['entry_speed']} "
                  f"oracle={t['oracle_return']:.1f} len={len(t['obs'])}")
        fastfail = [t for t in library
                    if not t["success"] and t["entry_speed"] is not None
                    and t["entry_speed"] > 0.4]
        print(f"fast-entry failures (entry_v > 0.4, not success): {len(fastfail)}")
        return

    print(f"library: {len(library)} reachable trajectories, "
          f"{sum(t['success'] for t in library)} successful, "
          f"{sum(1 for t in library if t['entry_speed'] is not None)} entered the dock")
    print()
    print(f"{'reward':<50} {'succ>fail':>9} {'sep':>9} {'all-pairs':>9} "
          f"{'r_succ':>9} {'r_entfl':>9} {'r_none':>9}")
    for path in args.rewards:
        try:
            s = score(path, library, clip)
        except Exception as exc:  # noqa: BLE001
            print(f"{path:<50} ERROR {type(exc).__name__}: {exc}")
            continue
        name = path if len(path) < 50 else "..." + path[-47:]
        print(f"{name:<50} {s['succ_vs_fail']:>9.3f} {s['separation']:>9.2f} "
              f"{s['acc_all']:>9.3f} {s['r_succ']:>9.2f} {s['r_ent_fail']:>9.2f} "
              f"{s['r_no_entry']:>9.2f}")


if __name__ == "__main__":
    main()
