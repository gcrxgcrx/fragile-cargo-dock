"""Is the ~0.6M short-training rung a usable *selector*?

Implements the pre-registered Branch B analysis in
`runs/env_007/LADDER_DECISION_PREREGISTRATION.md` (sections 3 and 3b):

  * per-candidate seed-to-seed dispersion at 0.6M (3 seeds),
  * Spearman rho between mean-0.6M outcome and the 1.2M outcome, across candidates,
  * the same for dock_rate and mean return,
  * the fixed decision bar: adopt the 0.6M rung iff rho(success) >= 0.60 AND the
    seed-to-seed spread is smaller than the between-candidate spread.

Candidate keys: the short-run names are `<cand>_s<seed>`; the long-run dir names are the
ones on disk, so a small alias table maps them onto the same key.

Usage
-----
    python rung_analysis.py \
        --short runs/env_007/rung_06m/eval_06m_block32000.json \
        --full  runs/env_007/rung_06m/eval_1m2_block32000.json \
        --out   runs/env_007/rung_06m/rung_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# long-run directory name -> candidate key used by the short runs
ALIASES = {"train_1m2": "control_v1"}
RHO_BAR = 0.60


def cand_key(name: str, long_run: bool = False) -> str:
    key = re.sub(r"_s\d+$", "", name)
    if long_run:
        key = ALIASES.get(key, key)
    return key


def spearman(x, y) -> float:
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
    ap.add_argument("--short", required=True, help="eval JSON of the 0.6M runs")
    ap.add_argument("--full", required=True, help="eval JSON of the 1.2M runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--short-label", default="0.6M",
                    help="budget label for the short runs, used in the printed table "
                         "(the script must not claim 0.6M when it is handed 1.0M data)")
    args = ap.parse_args()

    short_rows = json.loads(Path(args.short).read_text(encoding="utf-8"))
    full_rows = json.loads(Path(args.full).read_text(encoding="utf-8"))

    short: dict[str, list[dict]] = {}
    for r in short_rows:
        short.setdefault(cand_key(r["name"]), []).append(r)
    full: dict[str, dict] = {}
    for r in full_rows:
        full[cand_key(r["name"], long_run=True)] = r

    cands = [c for c in full if c in short]
    missing_short = [c for c in full if c not in short]
    missing_full = [c for c in short if c not in full]
    if missing_short:
        print(f"  [warn] no 0.6M runs yet for: {', '.join(sorted(missing_short))}")
    if missing_full:
        print(f"  [warn] no 1.2M reference for: {', '.join(sorted(missing_full))}")

    rows = []
    for c in sorted(cands):
        s = short[c]
        sr = np.array([r["success_rate"] for r in s], dtype=float)
        # sample sd needs n >= 2; with a single seed it is undefined
        sd = float(np.std(sr, ddof=1)) if len(sr) > 1 else float("nan")
        rows.append({
            "candidate": c,
            "n_seeds": len(s),
            "short_success_rate_mean": float(sr.mean()),
            "short_success_rate_min": float(sr.min()),
            "short_success_rate_max": float(sr.max()),
            "short_success_rate_sd": sd,
            "short_dock_rate_mean": float(np.mean([r["dock_rate"] for r in s])),
            "short_mean_return": float(np.mean([r["mean_return"] for r in s])),
            "full_success_rate": full[c]["success_rate"],
            "full_dock_rate": full[c]["dock_rate"],
            "full_mean_return": full[c]["mean_return"],
            # keep the per-seed numbers so a reader can see the raw spread
            "short_per_seed": [{"name": r["name"], "success": r["success"],
                                "episodes": r["episodes"], "dock_rate": r["dock_rate"],
                                "mean_return": r["mean_return"]} for r in sorted(s, key=lambda r: r["name"])],
        })

    print()
    print("=" * 104)
    print(f"{args.short_label} RUNG vs 1.2M REFERENCE  (fresh seeds "
          f"{full_rows[0]['seed_offset']}..{full_rows[0]['seed_offset'] + full_rows[0]['episodes'] - 1})")
    print("=" * 104)
    print(f"{'candidate':<14} {'success (' + args.short_label + ', 3 seeds)':>26} {'sd':>7} "
          f"{args.short_label + ' dock':>10} {'1.2M success':>13} {'1.2M dock':>10}")
    for r in rows:
        spread = f"{r['short_success_rate_min']*100:.1f}-{r['short_success_rate_max']*100:.1f}%"
        mean = f"{r['short_success_rate_mean']*100:.1f}%"
        print(f"{r['candidate']:<14} {mean + ' (' + spread + ')':>26} "
              f"{r['short_success_rate_sd']:>7.3f} {r['short_dock_rate_mean']:>10.3f} "
              f"{r['full_success_rate']*100:>12.1f}% {r['full_dock_rate']:>10.3f}")

    if not rows:
        print("\nnothing to compare yet")
        return

    out = {
        "seed_offset": full_rows[0]["seed_offset"],
        "episodes": full_rows[0]["episodes"],
        "rows": rows,
    }

    print()
    print("=" * 104)
    print(f"PER-CANDIDATE {args.short_label} SEED DISPERSION")
    print("=" * 104)
    for r in rows:
        per_seed = ", ".join(f"{p['name'].split('_')[-1]}={p['success']}/{p['episodes']}"
                             for p in r["short_per_seed"])
        print(f"  {r['candidate']:<14} {per_seed}")

    print()
    print("=" * 104)
    print(f"RANK AGREEMENT   (pre-registered bar: rho >= {RHO_BAR} and seed spread < between-candidate spread)")
    print("=" * 104)
    fs = [r["full_success_rate"] for r in rows]
    ss = [r["short_success_rate_mean"] for r in rows]
    fd = [r["full_dock_rate"] for r in rows]
    sd_ = [r["short_dock_rate_mean"] for r in rows]
    fr = [r["full_mean_return"] for r in rows]
    sr_ = [r["short_mean_return"] for r in rows]

    rho_success = spearman(ss, fs)
    rho_dock = spearman(sd_, fd)
    rho_return = spearman(sr_, fr)

    within = float(np.nanmean([r["short_success_rate_sd"] for r in rows]))
    between = float(np.std(ss, ddof=1)) if len(set(ss)) > 1 else float("nan")

    print(f"  rho({args.short_label} success, 1.2M success)      = {rho_success:>7.3f}   "
          f"({len(rows)} candidates)")
    print(f"  rho({args.short_label} dock_rate, 1.2M dock_rate)  = {rho_dock:>7.3f}")
    print(f"  rho({args.short_label} mean_return, 1.2M return)   = {rho_return:>7.3f}")
    print(f"  within-candidate seed sd (mean)      = {within:>7.3f}")
    print(f"  between-candidate sd of means        = {between:>7.3f}")

    if not np.isfinite(rho_success):
        verdict = "UNDEFINED: success has no variance on one side"
    elif rho_success >= RHO_BAR and np.isfinite(between) and within < between:
        verdict = (f"ADOPT: the {args.short_label} rung ranks candidates like 1.2M and beats "
                   "its own seed noise")
    elif rho_success >= RHO_BAR:
        verdict = "REJECT: ranks, but seed noise is not smaller than the between-candidate spread"
    else:
        verdict = (f"REJECT: the {args.short_label} rung does not rank candidates like 1.2M; "
                   "raise the rung")
    print(f"\n  VERDICT: {verdict}")

    out.update({
        "short_label": args.short_label,
        "rho_success": rho_success,
        "rho_dock_rate": rho_dock,
        "rho_mean_return": rho_return,
        "within_candidate_seed_sd": within,
        "between_candidate_sd": between,
        "rho_bar": RHO_BAR,
        "verdict": verdict,
    })
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
