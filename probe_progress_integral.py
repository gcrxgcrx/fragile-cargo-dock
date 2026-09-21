"""Is the shaping term *farmed*? Compare realised progress components against geometry.

The trigger for this script: `runs/env_007/prompt_ladder_v7/cand_01/training/training_summary.json`
(A=+4.28, the only v7 candidate that ever docked, 3/60). Its final-policy component table shows

    crate_to_dock_progress   episode_sum_mean = 2746.7   magnitude_share 0.98

The crate starts ~4.5 m from the dock and the dock tolerance is ~0.04 m, so the *net* distance
that can ever be closed is < 4.6 m. An incremental reward of 100 * (d_prev - d_now) can
therefore contribute at most ~460 per episode if it is a true potential difference — 2746.7 is
~6x that, and the raw per-step sum over 400 steps is 2516. In other words the term is being
collected over a closed loop, not over progress.

For every reward with a 1.2M `training_summary.json` on disk this script reports, from the
FINAL POLICY component evaluation:

    progress_sum   episode_sum_mean of every component whose name mentions progress/approach
    geom_bound     the largest net distance that component could be paid for
    ratio          progress_sum / (100 * geom_bound)   -- only meaningful for the 100x family
    top_share      the largest magnitude_share in the table
    active_rate    of that dominant component

and prints them sorted by the measured fresh-60 success, so the farmed arms can be compared
with the arms that actually work.

Usage:
    python probe_progress_integral.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- known fresh-60 outcomes (60-episode blocks, see SESSION_STATE.md) -------
KNOWN = {
    "repair_test/r09_progress_x50_stream": 23,
    "ridge_width/q02_progress_x100": 31,
    "ridge_width/q01_progress_x75": 0,
    "repair_test/r04_own_progress_x50": 0,
    "approach_ablation/p02_progress_x200": 0,
    "overshoot_ablation/o00_base": 16,
    "prompt_ladder_v7/cand_00": 0,
    "prompt_ladder_v7/cand_01": 3,
    "prompt_ladder_v7/cand_02": 0,
    "prompt_ladder_v7/cand_03": 0,
    "prompt_ladder_v7/cand_04": 0,
    "prompt_ladder_v7/cand_05": 0,
    "prompt_ladder_v7/cand_06": 0,
    "prompt_ladder_v7/cand_07": 0,
    "ladder_train/L2_cand_00": 0,
    "ladder_train/L2_cand_04": 0,
    "ladder_train/L2_cand_05": 0,
    "ladder_train/L2_cand_14": 0,
    "ladder_train/L0_cand_02": 0,
    "ablation_probe/probeD": 44,
    "ablation_probe/probeE": 39,
    "ablation_probe/probeA": 24,
    "control_obs_only/train_1m2": 0,
}

PROGRESS_WORDS = ("progress", "approach", "closing", "distance")
ROOT = Path("runs/env_007")


def find_summaries():
    """Find every 1.2M summary and derive its arm name.

    Layouts differ by experiment: `runs/env_007/<arm>/training_summary.json` (the
    ablation/repair arms) and `runs/env_007/<arm>/<cand>/training/training_summary.json`
    (the v7 ladder and the search runs). The KNOWN keys use the `<arm>` or
    `<arm>/<cand>` prefix, so strip a trailing `/training` too.
    """
    seen = set()
    for p in sorted(ROOT.rglob("training_summary.json")):
        rel = p.relative_to(ROOT).as_posix()
        arm = rel[: -len("/training_summary.json")]
        if arm.endswith("/training"):
            arm = arm[: -len("/training")]
        for key in (arm,):
            if key in KNOWN and key not in seen:
                seen.add(key)
                yield key, p


def final_table(summary):
    ext = summary.get("external_eval", {})
    return ext.get("final_policy_component_evaluation") or {}


def main() -> None:
    rows = []
    for arm, path in find_summaries():
        try:
            s = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {arm}: unreadable ({exc})")
            continue
        table = final_table(s)
        if not table:
            continue
        progress = 0.0
        progress_terms = []
        for name, d in table.items():
            if any(w in name.lower() for w in PROGRESS_WORDS):
                v = float(d.get("episode_sum_mean", 0.0))
                if abs(v) > 0:
                    progress_terms.append((name, v))
                    progress += v
        shares = [(n, float(d.get("magnitude_share", 0.0)),
                   float(d.get("active_rate", 0.0)),
                   float(d.get("episode_sum_mean", 0.0)))
                  for n, d in table.items() if d.get("magnitude_share")]
        top = max(shares, key=lambda t: t[1]) if shares else ("-", 0.0, 0.0, 0.0)
        rows.append(dict(arm=arm, success=KNOWN[arm], progress=progress,
                         terms="; ".join(f"{n}={v:.0f}" for n, v in progress_terms),
                         top_name=top[0], top_share=top[1], top_active=top[2],
                         top_sum=top[3]))

    rows.sort(key=lambda r: (-r["success"], -r["progress"]))
    hdr = (f"{'arm':<34} {'succ':>5} {'progress_sum':>12} {'top component':<34} "
           f"{'share':>6} {'act':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<34} {r['success']:>5} {r['progress']:>12.0f} "
              f"{r['top_name']:<34} {r['top_share']:>6.2f} {r['top_active']:>6.3f}")

    print()
    print("=== the farming signature ===")
    print("A true incremental term pays at most 100 x (net distance closed) per episode.")
    print("The crate starts about 4.5 m from the dock, so any progress_sum above ~500 is")
    print("collecting rewards over a closed loop rather than over progress.\n")
    for r in rows:
        if r["progress"] > 500:
            flag = "FARMED" if r["success"] == 0 else "farmed AND works"
            print(f"  {r['arm']:<34} succ={r['success']:>3}  progress_sum={r['progress']:>9.0f}"
                  f"   -> {flag}")
        elif r["progress"] > 0:
            print(f"  {r['arm']:<34} succ={r['success']:>3}  progress_sum={r['progress']:>9.0f}"
                  f"   -> bounded (consistent with a potential difference)")


if __name__ == "__main__":
    main()
