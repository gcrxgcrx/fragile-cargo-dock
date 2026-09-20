"""Evaluate the pre-registered mechanistic readout (LADDER_DECISION_PREREGISTRATION.md §3c).

The readout, frozen before any 0.6M summary was read:

  * `sparse` := components with 0 < active_rate < 0.05;
  * `c*` := the element of `sparse` with the largest `magnitude_share` among those with
    `episode_sum_mean > 0` (none -> predict NEGATIVE);
  * predict POSITIVE iff `magnitude_share(c*) >= 0.50` and `episode_sum_mean(c*) >= +20.0`;
  * per candidate, aggregate runs by majority vote;
  * ranking score (for AUPRC) = magnitude_share(c*) * episode_sum_mean(c*), mean over runs;
  * adopted iff AUPRC >= 0.60 and recall >= 0.60; baseline (predict-all) is
    precision = prevalence, recall = 1, AUPRC = prevalence.

Usage
-----
    python mechanistic_readout.py --labels runs/env_007/rung_06m/labels_1m2.json \
        --out runs/env_007/rung_06m/readout_06m.json \
        runs/env_007/rung_06m/*/training_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SHARE_BAR = 0.50
ESM_BAR = 20.0
AR_BAR = 0.05
AUPRC_BAR = 0.60
RECALL_BAR = 0.60

# long-run directory names -> the candidate key used by the short runs / the label file
ALIASES = {"train_1m2": "control_v1", "train_3m": "control_v1"}


def readout(run: dict) -> dict:
    final = run.get("external_eval", {}).get("final_policy_component_evaluation", {}) or {}
    best = None
    for name, v in final.items():
        if not isinstance(v, dict):
            continue
        ar = v.get("active_rate") or 0.0
        esm = v.get("episode_sum_mean") or 0.0
        share = v.get("magnitude_share") or 0.0
        if 0.0 < ar < AR_BAR and esm > 0.0:
            if best is None or share > best[1]:
                best = (name, share, esm, ar)
    if best is None:
        return {"component": None, "share": 0.0, "esm": 0.0, "active_rate": 0.0,
                "score": 0.0, "predict_positive": False}
    name, share, esm, ar = best
    return {"component": name, "share": share, "esm": esm, "active_rate": ar,
            "score": share * esm,
            "predict_positive": bool(share >= SHARE_BAR and esm >= ESM_BAR)}


def average_precision(scores, labels) -> float:
    """Standard AP (area under the precision-recall curve, step interpolation)."""
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
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-negative", nargs="*", default=[],
                    help="candidates that MUST NOT be certified positive for the readout to "
                         "be adopted at this rung (e.g. control_v1); adds a precision condition "
                         "to the verdict, as pre-registered for the 1.0M rung in section 3d")
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))

    per_cand: dict[str, list[dict]] = defaultdict(list)
    for p in args.summaries:
        path = Path(p)
        run = json.loads(path.read_text(encoding="utf-8"))
        key = re.sub(r"_s\d+$", "", path.parent.name)
        key = ALIASES.get(key, key)
        r = readout(run)
        r["run"] = path.parent.name
        r["steps"] = run.get("total_timesteps")
        per_cand[key].append(r)

    rows = []
    for key in sorted(labels):
        runs = per_cand.get(key, [])
        if not runs:
            rows.append({"candidate": key, "label": labels[key], "n_runs": 0,
                         "prediction": None, "score": None, "note": "no runs found"})
            continue
        votes = sum(1 for r in runs if r["predict_positive"])
        pred = votes * 2 >= len(runs)          # majority (ties -> positive)
        score = sum(r["score"] for r in runs) / len(runs)
        rows.append({
            "candidate": key, "label": labels[key], "n_runs": len(runs),
            "prediction": int(pred), "score": score,
            "component": runs[0]["component"],
            "share": runs[0]["share"], "esm": runs[0]["esm"],
            "active_rate": runs[0]["active_rate"],
            "votes_positive": votes,
            "per_run": runs,
        })

    scored = [r for r in rows if r["prediction"] is not None]
    tp = sum(1 for r in scored if r["prediction"] == 1 and r["label"] == 1)
    fp = sum(1 for r in scored if r["prediction"] == 1 and r["label"] == 0)
    fn = sum(1 for r in scored if r["prediction"] == 0 and r["label"] == 1)
    tn = sum(1 for r in scored if r["prediction"] == 0 and r["label"] == 0)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    prevalence = sum(r["label"] for r in scored) / len(scored) if scored else float("nan")
    auprc = average_precision([r["score"] for r in scored], [r["label"] for r in scored])

    print(f"{'candidate':<14} {'label':>5} {'runs':>4} {'c*':<24} {'share':>6} {'esm':>9} "
          f"{'score':>9} {'pred':>5}")
    for r in rows:
        if r["prediction"] is None:
            print(f"{r['candidate']:<14} {r['label']:>5} {r['n_runs']:>4} {r['note']}")
            continue
        print(f"{r['candidate']:<14} {r['label']:>5} {r['n_runs']:>4} "
              f"{str(r['component']):<24} {r['share']:>6.3f} {r['esm']:>9.2f} "
              f"{r['score']:>9.2f} {r['prediction']:>5}")

    print()
    print(f"  confusion           : tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"  precision           : {precision:.3f}")
    print(f"  recall              : {recall:.3f}")
    print(f"  AUPRC (avg prec)    : {auprc:.3f}")
    print(f"  prevalence baseline : precision={prevalence:.3f} recall=1.000 AUPRC={prevalence:.3f}")

    adopted = bool(auprc >= AUPRC_BAR and recall >= RECALL_BAR)
    verdict = ("ADOPT: the mechanistic readout separates at this rung"
               if adopted else
               "REJECT: the readout does not separate at this rung; use a longer rung")

    violated = []
    for key in args.require_negative:
        row = next((r for r in scored if r["candidate"] == key), None)
        if row is None:
            violated.append(f"{key} (no runs)")
        elif row["prediction"] == 1:
            violated.append(f"{key} (certified positive, {row['votes_positive']}/{row['n_runs']} runs)")
    if violated:
        adopted = False
        verdict = ("REJECT: required-negative arm certified positive -> "
                   + "; ".join(violated))
        print(f"\n  REQUIRED-NEGATIVE VIOLATION: {'; '.join(violated)}")

    print(f"\n  VERDICT: {verdict}  (bar: AUPRC >= {AUPRC_BAR} and recall >= {RECALL_BAR}"
          + (f"; must not certify {', '.join(args.require_negative)}" if args.require_negative else "")
          + ")")

    out = {
        "rows": rows, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "auprc": auprc,
        "prevalence": prevalence, "verdict": verdict, "adopted": adopted,
        "require_negative": list(args.require_negative),
        "require_negative_violations": violated,
        "bars": {"share": SHARE_BAR, "esm": ESM_BAR, "active_rate": AR_BAR,
                 "auprc": AUPRC_BAR, "recall": RECALL_BAR},
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
