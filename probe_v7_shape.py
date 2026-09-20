"""Measure the *realised* payoff shape of v7 candidates on a genuine success trajectory.

Why this exists
---------------
v7 mandates two things at once: a one-off completion event AND a per-step settled
payoff, and it also keeps v5's "when the completion predicate holds, every component
except the one-off event must contribute exactly 0" rule. v7 candidates resolved that
contradiction by gating the per-step payoff on `not _PAID` (or by zeroing it once the
event fires). This script measures what that costs, on a real trajectory, under the
harness's per-step clip of 20.

For every reward it replays:
  A) the final steps of a scripted SUCCESS trajectory, and reports the per-step
     generated reward with and without the harness clip;
  B) the same trajectory truncated one step before the success step, followed by
     "farm" steps that keep the settled predicate true but never complete it.

Usage:
    python probe_v7_shape.py REWARD.py [REWARD.py ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

CLIP = 20.0


def load_fn(path):
    return trc.load_reward(path)


def call(fn, obs, act, nxt, native):
    out = fn(obs, act, nxt, native, {}, 0.0)
    total = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
    comps = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
    return total, comps


def clipped(v):
    return max(-CLIP, min(CLIP, v))


def success_traj(library):
    for t in library:
        if t["success"] and len(t["obs"]) >= 12:
            return t
    raise SystemExit("no successful trajectory in the library")


def run(path, traj):
    fn = load_fn(path)
    n = len(traj["obs"])
    # --- pass 1: the real trajectory, watched from the start so that any streak
    #     counter is in the state it would be in during training -----------------
    fn(traj["obs"][0], traj["act"][0], traj["nxt"][0], float(traj["native"][0]), {}, 0.0)
    tail = []
    for i in range(1, n):
        v, c = call(fn, traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]))
        tail.append((i, n, v, clipped(v), c))
    return tail


def farm(path, traj, extra=40):
    """Truncate before completion, then repeat the last settled transition."""
    fn = load_fn(path)
    n = len(traj["obs"])
    fn(traj["obs"][0], traj["act"][0], traj["nxt"][0], float(traj["native"][0]), {}, 0.0)
    stop = n - 2                      # don't take the step that completes the streak
    total, steps = 0.0, 0
    for i in range(1, stop + 1):
        v, _ = call(fn, traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]))
        total += clipped(v)
        steps += 1
    i = stop - 1                      # a settled transition, repeated
    for _ in range(extra):
        v, _ = call(fn, traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]))
        total += clipped(v)
        steps += 1
    return total, steps


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: probe_v7_shape.py REWARD.py ...")

    library = trc.build_library(6)
    traj = success_traj(library)
    n = len(traj["obs"])
    print(f"success trajectory: controller={traj['controller']} steps={n} "
          f"entry_speed={traj['entry_speed']}")
    print(f"harness per-step clip = {CLIP}\n")

    for path in paths:
        name = path if len(path) < 44 else "..." + path[-41:]
        tail = run(path, traj)
        print(f"--- {name}")
        print(f"    last 12 steps of the success trajectory "
              f"(step, raw, clipped, top components)")
        for i, _n, v, cv, c in tail[-12:]:
            top = sorted(((k, x) for k, x in c.items()), key=lambda kv: -abs(kv[1]))[:3]
            top_s = " ".join(f"{k}={x:+.2f}" for k, x in top if abs(x) > 1e-9)
            print(f"      t={i:>3}/{n}  raw={v:>9.3f}  clip={cv:>8.3f}   {top_s}")
        complete = sum(cv for _i, _n, _v, cv, _c in tail)
        farmed, fsteps = farm(path, traj)
        print(f"    cumulative to completion (from t=1, clipped): {complete:>9.2f} "
              f"over {n - 1} steps")
        print(f"    truncated + {fsteps - (n - 2)} farm steps    (clipped): {farmed:>9.2f} "
              f"over {fsteps} steps")
        print()


if __name__ == "__main__":
    main()
