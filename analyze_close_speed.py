"""Apply the pre-registered verdict rule for the close-speed isolation experiment.

Reads the `eval_pool.py` JSONs for block 37000-37059 and reports, per arm and seed:
success, dock rate, P(success | dock_entered), and the one-sided Fisher p against 0/60.
Then applies `runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md`'s rule:

    S5 (the external 30 % bar): C1 non-zero on >= 2 of 3 training seeds AND at least one
        C1 seed >= 8/60 (Fisher p < 0.05 vs 0/60)
    S4: C1 non-zero on one seed only, or the paired difference runs the right way in 2/3
    S3/B: indistinguishable or reversed -> the 0 % vs 73.3 % contrast was a seed lottery

Usage:
    python analyze_close_speed.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

EVAL = Path("runs/env_007/close_speed_test")

# arm key -> {training seed: run directory (relative to runs/env_007)}
ARMS = {
    "C0 (control_v1)": {
        0: "close_speed_test/C0_control/seed0",
        1: "close_speed_test/C0_control/seed1",
    },
    "C1 (+closing speed)": {
        0: "ablation_probe/probeD",                     # verified identical to C1
        1: "close_speed_test/C1_closing_speed/seed1",
        2: "close_speed_test/C1_closing_speed/seed2",
    },
}
# the reused runs also exist under their original names; both names may appear in the JSON
ALIASES = {
    "close_speed_test/C0_control/seed0": {"close_speed_test/C0_control/seed0"},
    "ablation_probe/probeD": {"ablation_probe/probeD"},
    "control_obs_only/train_1m2": {"control_obs_only/train_1m2"},
}


def fisher_one_sided(a, b, c, d):
    """P[X >= a] for X ~ Hypergeometric(a+b, c+d, a+c) — one-sided, table [[a,b],[c,d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = sum(comb(row1, k) * comb(n - row1, col1 - k)
                for k in range(max(0, col1 - (n - row1)), col1 + 1))
    p = 0.0
    for k in range(a, min(row1, col1) + 1):
        p += comb(row1, k) * comb(n - row1, col1 - k) / total
    return p


def load_records():
    out = {}
    for p in sorted(EVAL.glob("eval_block37000*.json")):
        for rec in json.loads(p.read_text(encoding="utf-8-sig")):
            out[rec["run_dir"].replace("\\", "/")] = rec
    return out


def find(records, key):
    for rd, rec in records.items():
        if rd.endswith(key):
            return rec
    return None


def main() -> None:
    records = load_records()
    print(f"loaded {len(records)} eval records\n")
    hdr = (f"{'arm':<22} {'seed':>4} {'success':>9} {'dock':>7} "
           f"{'P(succ|dock)':>13} {'mean_len':>9} {'return':>9}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for arm, seeds in ARMS.items():
        summary[arm] = {}
        for seed, key in sorted(seeds.items()):
            rec = find(records, key)
            if rec is None:
                print(f"{arm:<22} {seed:>4}   (not scored yet: {key})")
                continue
            s, n = rec["success"], rec["episodes"]
            dock = rec.get("dock_rate")
            cond = (rec["success"] / (dock * n)) if dock else float("nan")
            p = fisher_one_sided(s, n - s, 0, 60) if s > 0 else 1.0
            summary[arm][seed] = dict(success=s, n=n, dock=dock, cond=cond, p=p,
                                      mean_return=rec.get("mean_return"))
            print(f"{arm:<22} {seed:>4} {s:>4}/{n:<4} {dock:>7.2f} {cond:>13.3f} "
                  f"{rec.get('mean_episode_length', float('nan')):>9.1f} "
                  f"{rec.get('mean_return', float('nan')):>9.2f}   p={p:.4g}")

    c1 = summary.get("C1 (+closing speed)", {})
    c0 = summary.get("C0 (control_v1)", {})
    print()
    print("=== pre-registered verdict ===")
    if not c1:
        print("  C1 not scored yet — no verdict.")
        return
    nonzero = [s for s, v in c1.items() if v["success"] > 0]
    strong = [s for s, v in c1.items() if v["success"] >= 8]
    c0_total = sum(v["success"] for v in c0.values())
    c1_total = sum(v["success"] for v in c1.values())
    c1_per_seed = ", ".join("%d:%d" % (s, v["success"]) for s, v in sorted(c1.items()))
    c0_per_seed = ", ".join("%d:%d" % (s, v["success"]) for s, v in sorted(c0.items()))
    print(f"  C1 successes per seed : {{{c1_per_seed}}}   non-zero on {len(nonzero)}/{len(c1)} seeds")
    print(f"  C0 successes per seed : {{{c0_per_seed}}}   total {c0_total}")
    print(f"  C1 total {c1_total} vs C0 total {c0_total} over the same block")
    if len(c1) == 3:
        if len(nonzero) >= 2 and strong:
            verdict = "S5 HOLDS — a full-pipeline run is warrantable"
        elif nonzero:
            verdict = "S4 — directionally right but not separable; do NOT run the pipeline yet"
        else:
            verdict = "S3/B — indistinguishable from C0; the 0 % vs 73.3 % contrast was a seed lottery"
        print(f"  VERDICT: {verdict}")
    else:
        print(f"  partial: {len(c1)}/3 C1 seeds scored")
    p_all = fisher_one_sided(c1_total, 60 * len(c1) - c1_total,
                             c0_total, 60 * max(1, len(c0)) - c0_total) if c1_total else 1.0
    print(f"  pooled one-sided Fisher (C1 vs C0): p = {p_all:.4g}")


if __name__ == "__main__":
    main()
