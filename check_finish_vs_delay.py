"""Finish-versus-delay: the sharpest objective-level check for this task.

Motivation (from an external review, adapted to what this task actually allows):
v7 requires both a one-off completion event AND a per-step payoff on the same state, and
the harness clips every per-step reward to [-20, 20]. Two consequences follow, and both can
be checked in seconds on a *real* success trajectory, without any training:

  (1) FINISH vs DELAY.  Same prefix; completing now vs keeping the settled state alive and
      completing K steps later.  If DELAY wins, the reward's argmax is not the task.
  (2) EVENT vs STREAM.  The one-off event and the settled stream are paid on the same step.
      If their sum exceeds the clip, the event is worth literally nothing (its marginal
      contribution is zero), which is the arithmetic behind "a one-off is worth at most one
      stream step" under clip 20.

Both are reported with and without the harness clip, and with gamma = 0.999 discounting.
This is a LINT on the reward's objective, not a success selector: see
`probe_metric_correlation.py` for why no training-free statistic can select here.

Usage:
    python check_finish_vs_delay.py REWARD.py [REWARD.py ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

CLIP = 20.0
GAMMA = 0.999
K_LIST = (1, 5, 10, 40)


def clip(v):
    return max(-CLIP, min(CLIP, v))


def call(fn, obs, act, nxt, native):
    out = fn(obs, act, nxt, native, {}, 0.0)
    total = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
    comps = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
    return total, (comps if isinstance(comps, dict) else {})


def find_success(library):
    for t in library:
        if t["success"] and len(t["obs"]) >= 30:
            return t
    raise SystemExit("no successful trajectory in the library")


def discounted(seq, gamma=GAMMA, clip_it=True):
    """Discounted return of a reward sequence, optionally clipped per step."""
    total = 0.0
    for i, r in enumerate(seq):
        total += (gamma ** i) * (clip(r) if clip_it else r)
    return total


def analyse(path, traj):
    fn = trc.load_reward(path)
    n = len(traj["obs"])
    # find the last stable (settled) transition = the success step
    last = n - 1
    # replay from the start so any streak counter is in its training state
    fn(traj["obs"][0], traj["act"][0], traj["nxt"][0], float(traj["native"][0]), {}, 0.0)
    prefix = []
    for i in range(1, last):
        r, _ = call(fn, traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]))
        prefix.append(r)
    r_finish, comps_finish = call(fn, traj["obs"][last], traj["act"][last],
                                  traj["nxt"][last], float(traj["native"][last]))

    out = {"finish_raw": r_finish, "finish_clipped": clip(r_finish),
           "finish_components": {k: float(v) for k, v in comps_finish.items()
                                 if abs(float(v)) > 1e-9},
           "delay": {}}
    # delayed completion: keep replaying a settled transition (index last-1) K more times,
    # then complete on the same final transition.
    for K in K_LIST:
        streak = []
        for _ in range(K):
            i = last - 1
            r, _ = call(fn, traj["obs"][i], traj["act"][i], traj["nxt"][i],
                        float(traj["native"][i]))
            streak.append(r)
        r_end, _ = call(fn, traj["obs"][last], traj["act"][last], traj["nxt"][last],
                        float(traj["native"][last]))
        seq_finish = prefix + [r_finish]
        seq_delay = prefix + streak + [r_end]
        d_f = discounted(seq_finish) - discounted(prefix)
        d_d = discounted(seq_delay) - discounted(prefix)
        out["delay"][K] = {
            "finish_disc": d_f, "delay_disc": d_d, "advantage_of_delay": d_d - d_f,
            "finish_disc_unclipped": discounted(seq_finish, clip_it=False)
                                     - discounted(prefix, clip_it=False),
            "delay_disc_unclipped": discounted(seq_delay, clip_it=False)
                                    - discounted(prefix, clip_it=False),
        }
    return out


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: check_finish_vs_delay.py REWARD.py ...")
    library = trc.build_library(6)
    traj = find_success(library)
    print(f"success trajectory: {traj['controller']} seed={traj['seed']} steps={len(traj['obs'])}")
    print(f"harness clip = {CLIP}, gamma = {GAMMA}\n")

    for path in paths:
        name = path if len(path) < 46 else "..." + path[-43:]
        a = analyse(path, traj)
        print(f"--- {name}")
        print(f"    terminal step: raw={a['finish_raw']:+.2f} clipped={a['finish_clipped']:+.2f}"
              f"   components: " +
              (" ".join(f"{k}={v:+.1f}" for k, v in a["finish_components"].items()) or "(none)"))
        print(f"    {'delay K':>8} {'finish(disc)':>14} {'delay(disc)':>13} "
              f"{'delay - finish':>15} {'unclipped adv':>14}")
        for K, d in a["delay"].items():
            adv_un = d["delay_disc_unclipped"] - d["finish_disc_unclipped"]
            print(f"    {K:>8} {d['finish_disc']:>14.2f} {d['delay_disc']:>13.2f} "
                  f"{d['advantage_of_delay']:>15.2f} {adv_un:>14.2f}")
        worst = max(d["advantage_of_delay"] for d in a["delay"].values())
        verdict = ("OK: finishing beats every delay" if worst < -1e-9 else
                   "LINT FAIL: delaying completion is worth more than finishing")
        print(f"    -> {verdict}  (max advantage of delay = {worst:+.2f})\n")


if __name__ == "__main__":
    main()
