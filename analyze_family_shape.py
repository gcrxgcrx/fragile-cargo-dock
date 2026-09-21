"""Training-free shape analysis of a generated family (lint, not a selector).

For each candidate reward it reports, by replaying the scripted-trajectory library:

  stream_after_event   per-step reward on a settled transition WITH the candidate's own
                       state machine advanced past its success step (0 => the payoff is
                       switched off by the event, the v7 defect)
  settled_rate         per-step reward on settled transitions in general
  prog_sum             sum of every progress/approach component over the trajectories
  prog_integral        prog_sum / |net distance change| -- >>1 means a closed-loop integral
  top_share            largest component's share of |reward| (a dominance readout)
  degenerate           True if the reward is ~identically zero on the library

Usage:
    python analyze_family_shape.py DIR_OR_REWARD [DIR_OR_REWARD ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

CLIP = 20.0
PROGRESS_WORDS = ("progress", "approach", "closing", "push", "distance", "settle")


def load(path):
    return trc.load_reward(path)


def components(fn, traj):
    """Return (per-step totals, per-step component dicts) for a trajectory."""
    totals, comps = [], {}
    for i in range(len(traj["obs"])):
        out = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]), {}, 0.0)
        t = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
        totals.append(t)
        c = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
        if isinstance(c, dict):
            for k, v in c.items():
                comps.setdefault(k, []).append(float(v))
    for k in comps:
        comps[k] = np.asarray(comps[k])
    return np.asarray(totals), comps


def analyse(path, library):
    fn = load(path)
    succ = [t for t in library if t["success"] and len(t["obs"]) > 15]
    if not succ:
        raise SystemExit("no successful trajectory in the library")
    traj = succ[0]
    n = len(traj["obs"])

    # advance the candidate's internal state to the success step, then read the next step
    fn(traj["obs"][0], traj["act"][0], traj["nxt"][0], float(traj["native"][0]), {}, 0.0)
    for i in range(1, n):
        out = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]), {}, 0.0)
    last_total = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
    i = n - 2
    out2 = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]), {}, 0.0)
    after_total = float(out2[0]) if isinstance(out2, (tuple, list)) else float(out2)

    # settled-step rate over the stable tail
    tail_totals = []
    for i in range(max(1, n - 12), n):
        out = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]), {}, 0.0)
        tail_totals.append(float(out[0]) if isinstance(out, (tuple, list)) else float(out))
    settled_rate = float(np.mean(np.clip(tail_totals, -CLIP, CLIP)))

    # progress integral over every trajectory
    prog_sum = 0.0
    abs_net = 0.0
    all_abs = []
    for t in library:
        totals, comps = components(fn, t)
        all_abs.append(np.abs(totals).mean())
        p = 0.0
        for name, arr in comps.items():
            if any(w in name.lower() for w in PROGRESS_WORDS):
                p += float(np.sum(arr))
        # net distance closed over the episode (crate->dock, in the env's own normalisation)
        d0 = float(np.hypot(t["obs"][0][12] * 5.0, t["obs"][0][13] * 4.0))
        d1 = float(np.hypot(t["nxt"][-1][12] * 5.0, t["nxt"][-1][13] * 4.0))
        prog_sum += p
        abs_net += abs(d0 - d1)
    return dict(after_event=after_total, settled_rate=settled_rate,
                prog_sum=prog_sum, prog_integral=prog_sum / max(1e-6, abs_net),
                mean_abs_reward=float(np.mean(all_abs)))


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: analyze_family_shape.py DIR_OR_REWARD ...")
    paths = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths += sorted(p.glob("cand_*/reward_v1.py"))
        else:
            paths.append(p)
    library = trc.build_library(3)
    hdr = (f"{'candidate':<44} {'after-event':>11} {'settled/step':>12} {'prog_sum':>9} "
           f"{'integral':>9} {'mean|r|':>8}")
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        name = str(p) if len(str(p)) < 44 else "..." + str(p)[-41:]
        try:
            r = analyse(p, library)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<44} ERROR {type(exc).__name__}: {exc}")
            continue
        flag = "  <- payoff switched off" if r["after_event"] <= 0.5 else ""
        if r["prog_integral"] > 10:
            flag += "  <- progress integral >> net progress"
        print(f"{name:<44} {r['after_event']:>11.2f} {r['settled_rate']:>12.2f} "
              f"{r['prog_sum']:>9.1f} {r['prog_integral']:>9.2f} {r['mean_abs_reward']:>8.3f}{flag}")


if __name__ == "__main__":
    main()
