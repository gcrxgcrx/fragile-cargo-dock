"""Mechanical test of the rule that v5 forbade: does the reward pay every step while settled?

v5's self-check ④ required the opposite ("call it 12 times on the same settled state; the total
must stop growing"), so a v5-family reward should be flat on repeat calls. The measured repair
requires the growing form. This calls every reward on the *same* settled transition 12 times
and reports the per-call values.

A "settled" transition is taken from the scripted-trajectory library: the last step before
termination of a successful trajectory (termination requires 10 consecutive steps of
inside AND aligned AND slow, so that step satisfies the predicate by construction).

Usage:
    python check_settled_stream.py REWARD.py [REWARD.py ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc


def settled_transition(library):
    for traj in library:
        if traj["success"] and len(traj["obs"]) >= 3:
            i = len(traj["obs"]) - 1          # last step before the success termination
            return traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i])
    raise SystemExit("no successful trajectory in the library")


def main() -> None:
    rewards = sys.argv[1:]
    if not rewards:
        raise SystemExit("usage: check_settled_stream.py REWARD.py ...")

    library = trc.build_library(6)
    obs, act, nxt, native = settled_transition(library)

    print("classification: 'stream' iff the MEDIAN of calls 2..12 is positive.")
    print("(A one-off form pays on the trigger step only; a streak-gated one-off can still fire")
    print(" once inside calls 2..12, which the median ignores. Repeating the same transition also")
    print(" advances any internal streak counter, so per-call values are the honest readout.)")
    print()
    print(f"{'reward':<50} {'call1':>8} {'median(2-12)':>13} {'nz(2-12)':>9} {'stream?':>8}")
    for path in rewards:
        fn = trc.load_reward(path)
        vals = []
        for _ in range(12):
            out = fn(obs, act, nxt, native, {}, 0.0)
            v = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
            vals.append(v)
        tail = sorted(vals[1:])
        median = tail[len(tail) // 2]
        nz = sum(1 for v in vals[1:] if abs(v) > 1e-9)
        name = path if len(path) < 50 else "..." + path[-47:]
        print(f"{name:<50} {vals[0]:>8.3f} {median:>13.3f} {nz:>9} {str(median > 0.5):>8}")


if __name__ == "__main__":
    main()
