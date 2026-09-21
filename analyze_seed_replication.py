"""Seed-level paired analysis of the close-speed isolation (n = 4 training seeds per arm).

Pre-registered in `runs/env_007/SEED_REPLICATION_PREREGISTRATION.md`:
the unit of analysis is the **training seed**, the block is 38000-38059 for every run, and the
decision rule is a 95 % paired t-interval on the per-seed success difference.

Usage:
    python analyze_seed_replication.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EVAL = Path("runs/env_007/close_speed_test/eval_block38000_seeds.json")
REFS = Path("runs/env_007/prompt_ladder_v8/eval_block38000.json")   # v8_cand_01, same block

# training seed -> run dir suffix (both arms)
ARMS = {
    "C0 (control_v1)": {
        0: "close_speed_test/C0_control/seed0",
        1: "close_speed_test/C0_control/seed1",
        2: "close_speed_test/C0_control/seed2",
        3: "close_speed_test/C0_control/seed3",
    },
    "C1 (+closing speed)": {
        0: "ablation_probe/probeD",
        1: "close_speed_test/C1_closing_speed/seed1",
        2: "close_speed_test/C1_closing_speed/seed2",
        3: "close_speed_test/C1_closing_speed/seed3",
    },
}
T95_N4 = 3.1824          # two-sided 95 % t quantile, df = 3


def load():
    out = {}
    for rec in json.loads(EVAL.read_text(encoding="utf-8-sig")):
        out[rec["run_dir"].replace("\\", "/")] = rec
    return out


def find(records, suffix):
    for rd, rec in records.items():
        if rd.endswith(suffix):
            return rec
    return None


def main() -> None:
    recs = load()
    print(f"block 38000-38059, {len(recs)} runs scored\n")
    hdr = f"{'arm':<22} {'seed':>4} {'success':>9} {'dock':>7} {'return':>9}"
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for arm, seeds in ARMS.items():
        table[arm] = {}
        for s, suffix in sorted(seeds.items()):
            r = find(recs, suffix)
            if r is None:
                print(f"{arm:<22} {s:>4}   (missing)")
                continue
            table[arm][s] = r
            print(f"{arm:<22} {s:>4} {r['success']:>4}/{r['episodes']:<4} "
                  f"{r['dock_rate']:>7.2f} {r['mean_return']:>9.2f}")

    c0, c1 = table["C0 (control_v1)"], table["C1 (+closing speed)"]
    shared = sorted(set(c0) & set(c1))
    print()
    print("=== paired per-seed differences (unit = training seed) ===")
    diffs = []
    for s in shared:
        d = c1[s]["success"] - c0[s]["success"]
        diffs.append(d)
        print(f"  seed {s}: C1 {c1[s]['success']:>3} - C0 {c0[s]['success']:>3} = {d:>+4}")
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else float("nan")
    half = T95_N4 * sd / math.sqrt(n) if n > 1 else float("nan")
    lo, hi = mean - half, mean + half
    print(f"\n  n = {n} seeds")
    print(f"  mean difference = {mean:+.1f} successes/60   sd = {sd:.1f}")
    print(f"  95 % paired t-interval = [{lo:+.1f}, {hi:+.1f}]")
    for arm in ARMS:
        vals = [r["success"] for r in table[arm].values()]
        m = sum(vals) / len(vals)
        sdv = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        docks = [r["dock_rate"] for r in table[arm].values()]
        print(f"  {arm:<22} success mean {m:5.1f}/60 sd {sdv:4.1f} range "
              f"[{min(vals)}, {max(vals)}]   dock mean {sum(docks)/len(docks):.2f} "
              f"range [{min(docks):.2f}, {max(docks):.2f}]")

    verdict = ("P (pass): the interval excludes 0 and is positive -> the term is established "
               "at the seed level" if lo > 0 else
               "N (null): the interval contains 0 -> NOT established at the seed level")
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")

    if REFS.exists():
        for r in json.loads(REFS.read_text(encoding="utf-8-sig")):
            if "cand_01" in r["run_dir"]:
                print(f"\n  reference (same block): v8_cand_01  {r['success']}/60, "
                      f"dock {r['dock_rate']:.2f}, return {r['mean_return']:.1f}")


if __name__ == "__main__":
    main()
