"""Apply the pre-registered v7 verdicts (runs/env_007/V7_PROMPT_PREREGISTRATION.md).

V7-1: at least one v7 candidate scores > 0/60 on the fresh block (baseline 0/16 for the
      v5 family at 1.2M with 60-episode fresh scoring).
V7-2: the generated code pays every step while settled (already checked, training-free, by
      check_settled_stream.py -> 8/8 vs 0/2 for v5).
V7-3: A > 0 for at least half the candidates (already checked -> 4/8 vs 1/8).
V7-4: any successful candidate has dock_entered > 0 (all eight channel repairs were 0.00).

Usage:
    python v7_analysis.py --eval runs/env_007/prompt_ladder_v7/eval_block34000.json \
        --probe runs/env_007/advantage_probe_v7.json \
        --baseline-hits 0 --baseline-n 16 \
        --out runs/env_007/prompt_ladder_v7/analysis.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for [[a,b],[c,d]]."""
    n = a + b + c + d
    tot = 0.0
    for k in range(a, min(a + b, a + c) + 1):
        tot += comb(a + b, k) * comb(c + d, a + c - k) / comb(n, a + c)
    return tot


def cand_name(run_dir: str) -> str:
    parts = Path(run_dir).parts
    i = parts.index("prompt_ladder_v7")
    return f"{parts[i + 1]}/training"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--baseline-hits", type=int, default=0)
    ap.add_argument("--baseline-n", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    probe = {}
    if args.probe and Path(args.probe).exists():
        for r in json.loads(Path(args.probe).read_text(encoding="utf-8"))["rows"]:
            probe[r["name"].replace("_", "_", 1)] = r["A"]

    recs = []
    for r in rows:
        name = cand_name(r["run_dir"])
        cand = name.split("/")[0]
        recs.append({"candidate": cand, "success": r["success"], "episodes": r["episodes"],
                     "success_rate": r["success_rate"], "dock_rate": r["dock_rate"],
                     "mean_return": r["mean_return"],
                     "A": probe.get("v7_" + cand) or probe.get(cand)})

    print(f"{'candidate':<10} {'success':>9} {'rate':>7} {'dock':>6} {'return':>9} {'A':>9}")
    for r in sorted(recs, key=lambda r: r["candidate"]):
        a = "n/a" if r["A"] is None else f"{r['A']:+.3f}"
        print(f"{r['candidate']:<10} {r['success']:>4}/{r['episodes']:<4} "
              f"{r['success_rate']*100:>6.1f}% {r['dock_rate']:>6.2f} "
              f"{r['mean_return']:>9.2f} {a:>9}")

    hits = [r for r in recs if r["success"] > 0]
    n = len(recs)
    print(f"\n  hits: {len(hits)}/{n}   (baseline v5 family: "
          f"{args.baseline_hits}/{args.baseline_n})")
    p = fisher_one_sided(len(hits), n - len(hits), args.baseline_hits,
                         args.baseline_n - args.baseline_hits)
    print(f"  one-sided Fisher vs baseline = {p:.4f}")

    v7_1 = len(hits) >= 1
    v7_4 = all(r["dock_rate"] > 0 for r in hits) if hits else None
    print(f"\n  V7-1 (>=1 success)                    : {'HOLDS' if v7_1 else 'FAILS'}")
    print(f"  V7-4 (successes have dock_entered > 0): "
          f"{'HOLDS' if v7_4 else ('FAILS' if v7_4 is False else 'n/a (no successes)')}")
    if v7_1:
        print("  successes: " + ", ".join(
            f"{r['candidate']}={r['success']}/{r['episodes']} dock={r['dock_rate']:.2f} "
            f"A={r['A'] if r['A'] is None else round(r['A'], 3)}" for r in hits))
    if 0.05 <= p < 0.20 and n == 8:
        print("  -> in the pre-declared stage-2 band: run 8 further v7 candidates and pool")
    elif n == 8:
        print("  -> outside the stage-2 band; the experiment stops at one stage")

    out = {"records": recs, "hits": len(hits), "n": n, "fisher_p_one_sided": p,
           "V7_1": v7_1, "V7_4": v7_4,
           "baseline": {"hits": args.baseline_hits, "n": args.baseline_n}}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
