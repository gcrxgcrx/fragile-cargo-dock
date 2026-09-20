"""Same-units comparison: native reward vs LLM candidates on identical trajectories.

`analyze_native_reward.py` showed the native reward's cumulative *shaping* is bounded by
geometry (-3.2 .. +8.6 over 27 scripted trajectories) while its terminal term is +300.
This script puts the LLM rewards (and the oracle-repaired ones) into the same units:

    shaping return  = sum of every step's reward EXCEPT the final (terminal) step of a
                      successful trajectory
    terminal return = the reward on that final step
    ratio           = shaping / terminal

and reports how much of the shaping is *recoverable without progressing* (the "farm"
test):
    the same reward, replayed on the same trajectory with progress-dependent terms
    zeroed after the prefix, is not measurable for arbitrary code -- instead we report
    the reward's own late-trajectory shaping rate (mean per-step reward over the last N
    steps before the terminal step), which is what a hovering policy collects.

Usage:
    python compare_shaping_scale.py REWARD.py [REWARD.py ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_native_reward import decompose  # noqa: E402

TAIL = 40


def success_trajectories(library):
    return [t for t in library if t["success"] and len(t["obs"]) > TAIL + 12]


def replay_steps(reward_fn, traj):
    """Per-step generated reward on a trajectory (no clip, and with clip 20)."""
    fn = trc.load_reward(reward_fn) if isinstance(reward_fn, str) else reward_fn
    raw = []
    for i in range(len(traj["obs"])):
        out = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i],
                 float(traj["native"][i]), {}, 0.0)
        raw.append(float(out[0]) if isinstance(out, (tuple, list)) else float(out))
    return np.asarray(raw)


def report_native(traj):
    terms = decompose(traj)
    tail_shaping = sum(float(np.sum(v[:-1])) for v in terms.values())
    return tail_shaping, float(terms["terminal+roughness+hard_hit (residual)"][-1])


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: compare_shaping_scale.py REWARD.py ...")

    library = trc.build_library(3)
    succ = success_trajectories(library)
    if not succ:
        raise SystemExit("no successful scripted trajectory in the library")
    traj = succ[0]
    n = len(traj["obs"])
    print(f"trajectory: {traj['controller']} seed={traj['seed']} steps={n} "
          f"native_total={traj['oracle_return']:+.2f}\n")

    sh, tm = report_native(traj)
    print(f"{'reward':<46} {'shaping':>10} {'terminal':>10} {'ratio':>9} "
          f"{'last40/step':>12}")
    print("-" * 90)
    print(f"{'NATIVE (info-derived, reference)':<46} {sh:>10.2f} {tm:>10.2f} "
          f"{sh / max(1e-9, abs(tm)):>9.3f} {sh / (n - 1):>12.4f}")

    for path in paths:
        try:
            raw = replay_steps(path, traj)
        except Exception as exc:  # noqa: BLE001
            print(f"{path:<46} ERROR {type(exc).__name__}: {exc}")
            continue
        clip = np.clip(raw, -20.0, 20.0)
        shaping = float(np.sum(clip[:-1]))
        terminal = float(clip[-1])
        last40 = float(np.mean(clip[-TAIL - 1:-1]))
        name = path if len(path) < 46 else "..." + path[-43:]
        print(f"{name:<46} {shaping:>10.2f} {terminal:>10.2f} "
              f"{shaping / max(1e-9, abs(terminal)):>9.3f} {last40:>12.4f}")

    print()
    print("reading: 'ratio' is how many times larger the pre-terminal shaping is than the")
    print("terminal payoff. The native reward is ~0.03; anything >> 1 buries the completion")
    print("signal under shaping that a policy can collect without finishing.")


if __name__ == "__main__":
    main()
