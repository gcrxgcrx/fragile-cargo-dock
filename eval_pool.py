"""Evaluate a set of trained runs on a fresh-seed block and write machine-readable JSON.

Why this exists
---------------
`eval_fresh_seeds.py` prints a human table but writes nothing, so it cannot feed the
precision/recall/AUPRC report (Branch A) or the seed-dispersion analysis (Branch B) in
`runs/env_007/LADDER_DECISION_PREREGISTRATION.md`. This reuses exactly the same
`load_trained_policy` / `evaluate` code path, so the numbers are comparable with every
fresh-seed number already in SESSION_STATE.md, and dumps them as JSON.

Usage
-----
    # explicit dirs
    python eval_pool.py --episodes 60 --seed-offset 32000 \
        --out runs/env_007/pool_eval_32000.json \
        runs/env_007/ladder_train/L2_cand_00 runs/env_007/ladder_train/L0_cand_02

    # discover every trained run under a root (pattern is relative to the root)
    python eval_pool.py --episodes 60 --seed-offset 32000 \
        --out runs/env_007/pool_eval_32000.json \
        --root runs/env_007/ladder_train --pattern "*/model.zip"

The JSON is a list of records:
    {"run_dir", "name", "success", "episodes", "success_rate", "dock_rate",
     "mean_return", "reasons", "best_return", "worst_return", "mean_final_goal_distance"}

Deterministic: the only inputs are the policy files and the seed block.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
from run_fragilecargo_baseline import ENV_ID, evaluate, load_trained_policy


def discover(root: str, pattern: str) -> list[Path]:
    base = Path(root)
    return sorted(p.parent for p in base.glob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="dirs containing model.zip / vecnormalize.pkl")
    ap.add_argument("--root", default=None, help="root to discover run dirs under")
    ap.add_argument("--pattern", default="*/model.zip",
                    help="glob relative to --root whose parent is a run dir (default */model.zip)")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=32000)
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args()

    dirs: list[Path] = [Path(d) for d in args.run_dirs]
    if args.root:
        dirs += discover(args.root, args.pattern)

    # de-duplicate, keep order
    seen = set()
    unique = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)

    if not unique:
        raise SystemExit("no run dirs given or discovered")

    print(f"fresh seeds {args.seed_offset}..{args.seed_offset + args.episodes - 1} "
          f"({args.episodes} episodes), {len(unique)} runs")

    rows = []
    for d in unique:
        model_path = d / "model.zip"
        vec_path = d / "vecnormalize.pkl"
        if not model_path.exists():
            print(f"  [skip] {d}: no model.zip")
            continue
        policy, obs_rms = load_trained_policy(
            str(model_path), str(vec_path) if vec_path.exists() else None)
        records = evaluate(ENV_ID, policy, args.episodes, args.seed_offset, obs_rms=obs_rms)
        returns = [r["return"] for r in records]
        succ = sum(1 for r in records if r["success"])
        row = {
            "run_dir": str(d),
            "name": d.name,
            "success": succ,
            "episodes": args.episodes,
            "success_rate": succ / args.episodes,
            "dock_rate": sum(1 for r in records if r["dock_entered"]) / args.episodes,
            "mean_return": sum(returns) / len(returns),
            "reasons": dict(Counter(r["reason"] for r in records)),
            "best_return": max(returns),
            "worst_return": min(returns),
            "mean_final_goal_distance": sum(r["final_goal_distance"] for r in records) / len(records),
            "seed_offset": args.seed_offset,
        }
        rows.append(row)
        print(f"  {d.name:<14} success {succ:>3}/{args.episodes}  "
              f"dock {row['dock_rate']:.2f}  return {row['mean_return']:>8.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
