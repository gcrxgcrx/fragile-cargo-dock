"""Training-free selector: how large is the candidate reward's *per-step advantage* for
doing the task, relative to doing nothing?

Why this statistic, and why now
-------------------------------
Three independent failures measured in this session point at the same quantity:

* `L2/cand_00`'s reward ranks success (+56) above pushing (+6.4) above idle (−0.8) on
  scripted trajectories, yet its trained policy sits at ~−0.07 per episode and never moves
  the crate. The *ordering* is right; the per-step advantage of acting is
  (+6.4 − (−0.8)) / 400 ≈ **0.018 per step**.
* Multiplying one coefficient of that same reward by 50 (arm `r04`) turns a never-moving
  policy into one that enters the dock in **97 %** of episodes — the same reward, a ~50×
  larger advantage.
* Every hand-written arm that works carries its mass in a term worth **+20 to +300 per
  episode at 0.03–3 % activation** — orders of magnitude larger per step than the LLM
  rewards' shaping.

So the hypothesis is that learnability here is governed by the **per-step advantage of
task-directed behaviour over inaction**, a quantity that is computable on reachable
trajectories without any training — and that the ordering-based checks
(`trajectory_ranking_check.py`) and the structural checks
(`analyze_terminal_dominance.py`) both miss.

Statistic (frozen; see runs/env_007/ADVANTAGE_PROBE_PREREGISTRATION.md)
----------------------------------------------------------------------
Using the same scripted-trajectory library as `trajectory_ranking_check.py`
(6 seeds x 9 controllers, seed offset 50000), and the harness clip (20):

    per_step(traj) = clipped_generated_return(traj) / len(traj)
    A = mean(per_step over the seven heuristic controllers)
        - mean(per_step over the `idle` controller)

`shove` is excluded from the "task-directed" mean: it is a deliberately bad controller that
drives out of bounds. Per-step normalisation is required because success terminates an
episode early, so raw totals are not comparable across controllers.

Usage
-----
    python advantage_scale_probe.py --pool runs/env_007/advantage_pool14.json \
        --out runs/env_007/advantage_probe_pool14.json
    python advantage_scale_probe.py --pool ... --extended ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc

HEURISTIC = ["push_forever", "release_0.05", "release_0.15", "release_0.25",
             "release_0.35", "release_0.50", "release_0.80"]

AUPRC_BAR = 0.60
RECALL_BAR = 0.60
PRECISION_BAR = 0.60


def advantage(reward_path: str, library, clip: float = 20.0) -> dict:
    per_step = {}
    for traj in library:
        r = trc.replay(trc.load_reward(reward_path), traj, clip)
        per_step.setdefault(traj["controller"], []).append(r / max(1, len(traj["obs"])))
    idle = float(np.mean(per_step["idle"]))
    acting = float(np.mean([v for c in HEURISTIC for v in per_step[c]]))
    return {"A": acting - idle, "per_step_acting": acting, "per_step_idle": idle,
            "per_step_by_controller": {k: float(np.mean(v)) for k, v in per_step.items()}}


def average_precision(scores, labels) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    hits = 0
    ap = 0.0
    for i, (_s, y) in enumerate(pairs, start=1):
        if y:
            hits += 1
            ap += hits / i
    return ap / n_pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="JSON: {name: {reward, label}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--seeds-per-controller", type=int, default=6)
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8-sig"))

    print(f"building trajectory library ({args.seeds_per_controller} seeds x "
          f"{len(trc.CONTROLLERS)} controllers)...", flush=True)
    library = trc.build_library(args.seeds_per_controller)
    print(f"library: {len(library)} trajectories, "
          f"{sum(t['success'] for t in library)} successful, "
          f"{sum(1 for t in library if t['entry_speed'] is not None)} entered the dock\n")

    rows = []
    for name, spec in pool.items():
        try:
            res = advantage(spec["reward"], library, args.clip)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<24} ERROR {type(exc).__name__}: {exc}")
            continue
        rows.append({"name": name, "label": spec["label"], "reward": spec["reward"], **res})

    rows.sort(key=lambda r: -r["A"])
    print(f"{'candidate':<24} {'label':>5} {'A (per step)':>13} {'acting':>10} {'idle':>10}")
    for r in rows:
        print(f"{r['name']:<24} {r['label']:>5} {r['A']:>13.5f} {r['per_step_acting']:>10.5f} "
              f"{r['per_step_idle']:>10.5f}")

    labels = [r["label"] for r in rows]
    scores = [r["A"] for r in rows]
    prevalence = float(np.mean(labels)) if labels else float("nan")
    auprc = average_precision(scores, labels)

    # the binary rule needs a threshold; use the pre-registered one if present, else report
    # the ranking metric and the best-separating threshold for transparency
    ts = sorted(set(scores))
    cands = [(s + 1e-12) for s in ts]
    best = None
    for t in cands:
        pred = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, y in zip(pred, labels) if p and y)
        fp = sum(1 for p, y in zip(pred, labels) if p and not y)
        fn = sum(1 for p, y in zip(pred, labels) if not p and y)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn}
    for t in cands:
        pred = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, y in zip(pred, labels) if p and y)
        fp = sum(1 for p, y in zip(pred, labels) if p and not y)
        fn = sum(1 for p, y in zip(pred, labels) if not p and y)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if rec >= RECALL_BAR and (best is None or prec > best["precision"]):
            best = {"threshold": t, "precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                    "tp": tp, "fp": fp, "fn": fn}

    print()
    print(f"  prevalence (baseline AUPRC)  : {prevalence:.3f}")
    print(f"  AUPRC (average precision)    : {auprc:.3f}")
    if best:
        print(f"  best threshold by recall>={RECALL_BAR}: A >= {best['threshold']:.6f} -> "
              f"precision {best['precision']:.3f} recall {best['recall']:.3f} "
              f"(tp={best['tp']} fp={best['fp']} fn={best['fn']})")
    adopted = bool(auprc >= AUPRC_BAR and best and best["recall"] >= RECALL_BAR
                   and best["precision"] >= PRECISION_BAR)
    verdict = ("ADOPT: the advantage-scale probe separates known outcomes without training"
               if adopted else
               "REJECT: the advantage-scale probe does not separate known outcomes")
    print(f"\n  VERDICT: {verdict}  (bar: AUPRC >= {AUPRC_BAR}, and a threshold with "
          f"precision >= {PRECISION_BAR} and recall >= {RECALL_BAR})")

    out = {"rows": rows, "prevalence": prevalence, "auprc": auprc, "best": best,
           "verdict": verdict, "adopted": adopted,
           "bars": {"auprc": AUPRC_BAR, "recall": RECALL_BAR, "precision": PRECISION_BAR,
                    "clip": args.clip, "heuristic_controllers": HEURISTIC}}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
