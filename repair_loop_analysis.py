"""Analyse the evidence-channel repair experiment against its pre-registered predictions.

Reads the fresh-seed evaluation of the 16 repairs plus the training-free advantage readout,
and reports E1-E4 from `runs/env_007/REPAIR_LOOP_PREREGISTRATION.md`.

Note: every repair lives in `.../<arm>/<target>/cand_00`, so the candidate identity must be
parsed from `run_dir`, not from the directory name.

Usage:
    python repair_loop_analysis.py \
        --eval runs/env_007/repair_loop/eval_block33000.json \
        --probe runs/env_007/advantage_probe_repairloop.json \
        --out runs/env_007/repair_loop/analysis.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

STAGE2_LOW, STAGE2_HIGH = 0.05, 0.20


def parse_identity(run_dir: str) -> tuple[str, str]:
    parts = Path(run_dir).parts
    i = parts.index("repair_loop")
    return parts[i + 1], parts[i + 2]          # arm, target


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a,b],[c,d]] (one-sided, greater)."""
    n = a + b + c + d
    total = 0.0
    for k in range(a, min(a + b, a + c) + 1):
        total += comb(a + b, k) * comb(c + d, a + c - k) / comb(n, a + c)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    probe = {}
    if args.probe and Path(args.probe).exists():
        for r in json.loads(Path(args.probe).read_text(encoding="utf-8"))["rows"]:
            arm, target = r["name"].split("/", 1)
            probe[(arm, target)] = r["A"]

    recs = []
    for r in rows:
        arm, target = parse_identity(r["run_dir"])
        recs.append({"arm": arm, "target": target, "success": r["success"],
                     "episodes": r["episodes"], "success_rate": r["success_rate"],
                     "dock_rate": r["dock_rate"], "mean_return": r["mean_return"],
                     "A": probe.get((arm, target))})

    print(f"{'arm':<6} {'target':<12} {'success':>9} {'rate':>7} {'dock':>6} {'return':>9} {'A':>9}")
    for r in sorted(recs, key=lambda r: (r["arm"], r["target"])):
        a = "n/a" if r["A"] is None else f"{r['A']:+.3f}"
        print(f"{r['arm']:<6} {r['target']:<12} {r['success']:>4}/{r['episodes']:<4} "
              f"{r['success_rate']*100:>6.1f}% {r['dock_rate']:>6.2f} {r['mean_return']:>9.2f} {a:>9}")

    hits = {arm: sum(1 for r in recs if r["arm"] == arm and r["success"] > 0) for arm in ("real", "sham")}
    n = {arm: sum(1 for r in recs if r["arm"] == arm) for arm in ("real", "sham")}
    print(f"\n  hits: real {hits['real']}/{n['real']}   sham {hits['sham']}/{n['sham']}")
    p = fisher_one_sided(hits["real"], n["real"] - hits["real"],
                         hits["sham"], n["sham"] - hits["sham"]) if n["real"] and n["sham"] else float("nan")
    print(f"  one-sided Fisher (real > sham) = {p:.4f}")

    with_a = [r for r in recs if r["A"] is not None]
    succ_a = [r for r in with_a if r["success"] > 0]
    print(f"\n  E4 cross-tab (repairs with A>0 vs success):")
    for arm in ("real", "sham"):
        sub = [r for r in with_a if r["arm"] == arm]
        pos = [r for r in sub if r["A"] > 0]
        print(f"    {arm:<5} A>0: {len(pos)}/{len(sub)}   of which successful: "
              f"{sum(1 for r in pos if r['success'] > 0)}")
    if succ_a:
        print(f"    successes: " + ", ".join(
            f"{r['arm']}/{r['target']}={r['success']}/{r['episodes']} (A={r['A']:+.3f})" for r in succ_a))
    else:
        print("    no successes at all")

    e1 = hits["real"] >= 1
    e2 = p < 0.05
    e3 = hits["sham"] == 0
    print(f"\n  E1 (real arm has >=1 success)      : {'HOLDS' if e1 else 'FAILS'}")
    print(f"  E2 (real > sham, p < 0.05)         : {'HOLDS' if e2 else 'FAILS'}  (p={p:.4f})")
    print(f"  E3 (sham arm hit rate ~ 0)         : {'HOLDS' if e3 else 'FAILS'}")
    if not e2 and STAGE2_LOW <= p < STAGE2_HIGH and n["real"] == 8:
        print(f"  -> in the pre-declared stage-2 band [{STAGE2_LOW}, {STAGE2_HIGH}): "
              "run 8 further repairs per arm and pool")
    elif n["real"] == 8:
        print("  -> outside the stage-2 band; the experiment stops at one stage")

    out = {"records": recs, "hits": hits, "n": n, "fisher_p_one_sided": p,
           "E1": e1, "E2": e2, "E3": e3}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
