"""Join the structural checks with real training outcomes, and ask which check, if any,
predicts the outcome.

Why this script exists
----------------------
DeepSeek's critique of the obvious analysis is correct: correlating a check that all
candidates pass (zero variance) with an outcome that all candidates share (zero variance)
is mathematically undefined. The only way out is a candidate set with variance in BOTH
the checks and the outcome -- that is what the prompt ladder
(`runs/env_007/prompt_ladder/{L0,L1,L2}`) plus a spread of trained candidates provides.

Reported per check:
  * Spearman rho against fresh-seed success (n is small; treat as indicative only)
  * the check value for each trained candidate, with the outcome beside it
and the same for the trajectory-ranking score (ChatGPT's / DeepSeek's preferred probe).

Usage
-----
    python ladder_analysis.py --episodes 60 --seed-offset 30000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
from run_fragilecargo_baseline import ENV_ID, evaluate, load_trained_policy

import analyze_terminal_dominance as atd
import trajectory_ranking_check as trc

CLIP = 20.0
TRAINED = [
    ("L2", "cand_00"), ("L2", "cand_05"), ("L2", "cand_04"), ("L2", "cand_14"),
    ("L0", "cand_02"), ("L0", "cand_11"), ("L0", "cand_13"), ("L0", "cand_00"),
]


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=30000)
    ap.add_argument("--train-root", default="runs/env_007/ladder_train")
    ap.add_argument("--ladder-root", default="runs/env_007/prompt_ladder")
    ap.add_argument("--library-seeds", type=int, default=8)
    ap.add_argument("--skip-trajectories", action="store_true")
    args = ap.parse_args()

    print("building trajectory library once, shared by all candidates...", flush=True)
    library = trc.build_library(args.library_seeds) if not args.skip_trajectories else None

    rows = []
    for arm, cand in TRAINED:
        reward_path = f"{args.ladder_root}/{arm}/{cand}/reward_v1.py"
        train_dir = Path(args.train_root) / f"{arm}_{cand}"
        if not (train_dir / "model.zip").exists():
            print(f"  [skip] {arm}/{cand}: no trained model yet")
            continue

        # ---- structural checks ------------------------------------------------
        form = atd.terminal_form_test(atd.load_reward(reward_path), CLIP)
        fast, slow, gap, _e = atd.gentleness_test(atd.load_reward(reward_path), CLIP)
        probed = atd.probe_states(atd.load_reward(reward_path), CLIP)
        _term, _hover, ratio, _worst = atd.episode_dominance(probed, CLIP)

        # ---- trajectory ranking ----------------------------------------------
        if library is not None:
            tr = trc.score(reward_path, library, CLIP)
            traj = tr["succ_vs_fail"]
        else:
            tr, traj = {}, float("nan")

        # ---- real outcome -----------------------------------------------------
        policy, obs_rms = load_trained_policy(str(train_dir / "model.zip"),
                                              str(train_dir / "vecnormalize.pkl"))
        recs = evaluate(ENV_ID, policy, args.episodes, args.seed_offset, obs_rms=obs_rms)
        succ = sum(1 for r in recs if r["success"])
        returns = [r["return"] for r in recs]

        rows.append({
            "arm": arm, "cand": cand,
            "event": 1.0 if form["has_event"] else 0.0,
            "second": 1.0 if form["sufficient"] else 0.0,
            "term_ratio": form["ratio"] if np.isfinite(form["ratio"]) else 1e6,
            "gentle_gap": gap,
            "hover_ratio": ratio if np.isfinite(ratio) else 1e6,
            "traj_succfail": traj,
            "success": succ,
            "success_rate": succ / args.episodes,
            "mean_return": float(np.mean(returns)),
            "n": args.episodes,
        })
        print(f"  [done] {arm}/{cand}: fresh {succ}/{args.episodes}  "
              f"event={int(form['has_event'])} gap={gap:.3f} ratio={ratio:.2f} "
              f"traj={traj:.3f}", flush=True)

    if not rows:
        print("nothing trained yet")
        return

    print("\n" + "=" * 108)
    print("JOINED TABLE  (fresh seeds "
          f"{args.seed_offset}..{args.seed_offset + args.episodes - 1})")
    print("=" * 108)
    hdr = (f"{'cand':<12} {'event':>5} {'2nd':>4} {'termRatio':>10} {'gentleGap':>10} "
           f"{'hoverRatio':>11} {'trajSucc>Fail':>14} {'fresh':>7} {'meanRet':>9}")
    print(hdr)
    for r in rows:
        print(f"{r['arm']+'/'+r['cand']:<12} {int(r['event']):>5} {int(r['second']):>4} "
              f"{r['term_ratio']:>10.2f} {r['gentle_gap']:>10.3f} {r['hover_ratio']:>11.2f} "
              f"{r['traj_succfail']:>14.3f} {str(r['success'])+'/'+str(r['n']):>7} "
              f"{r['mean_return']:>9.2f}")

    print("\n" + "=" * 108)
    print("PER-CHECK PREDICTIVE POWER  (Spearman rho vs fresh success; n = "
          f"{len(rows)}, indicative only)")
    print("=" * 108)
    y = [r["success_rate"] for r in rows]
    out = [r["mean_return"] for r in rows]
    for key in ("event", "second", "gentle_gap", "term_ratio", "hover_ratio", "traj_succfail"):
        xs = [r[key] for r in rows]
        print(f"  {key:<15} rho(success)={spearman(xs, y):>6.3f}   "
              f"rho(mean_return)={spearman(xs, out):>6.3f}   "
              f"var={np.var(xs):.4g}   distinct={len(set(np.round(xs, 4)))}")

    print(f"\n  outcome variance: successes = {sorted(r['success'] for r in rows)}")
    print(f"  -> {'OUTCOME HAS VARIANCE, correlations are defined' if len({r['success'] for r in rows}) > 1 else 'OUTCOME HAS NO VARIANCE, correlations undefined'}")

    Path("runs/env_007/ladder_analysis.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote runs/env_007/ladder_analysis.json")


if __name__ == "__main__":
    main()
